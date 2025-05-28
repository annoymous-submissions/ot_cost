# servers.py
"""
Server implementations for Federated Learning. Streamlined version.
Focuses on CPU-based aggregation and state management.
Uses run_clients helper and direct overrides for algorithms.
Manages best global model state directly.
"""
import copy
import numpy as np
import torch
from typing import Dict, Optional, List, Tuple, Any, Callable
# Get core types from helper
from helper import (MetricKey, TrainerConfig, SiteData, ModelState, # Import types
                   DiversityMixin, MetricsCalculator, infer_higher_is_better) # gpu_scope not strictly needed server-side

# Import client classes for type hints and instantiation in _create_client overrides
from clients import Client, FedProxClient, PFedMeClient, DittoClient

# =============================================================================
# == Base Server Class ==
# =============================================================================
class Server:
    """Base server class. Manages clients, runs rounds via step functions."""

    def __init__(self, config: TrainerConfig, globalmodelstate: ModelState):
        self.config = config
        self.server_device = torch.device('cpu') # Server always on CPU
        self.client_device_str = config.device # Target device for client computation
        self.clients: Dict[str, Client] = {}

        # Server holds the reference to the GLOBAL ModelState
        self.serverstate: ModelState = globalmodelstate
        # Ensure server's copy of the model is on CPU
        self.serverstate.model.cpu()
        
        # Initialize selection criterion values
        self.selection_criterion_key = config.selection_criterion_key
        self.selection_criterion_direction_overrides = config.selection_criterion_direction_overrides
        self.criterion_is_higher_better = infer_higher_is_better(
            self.selection_criterion_key, 
            self.selection_criterion_direction_overrides
        )
        
        # Server's own tracking of the best GLOBAL state encountered
        self.best_global_value_for_selection = float('-inf') if self.criterion_is_higher_better else float('inf')
        self.best_global_model_state_dict: Optional[Dict] = copy.deepcopy(self.serverstate.model.state_dict())

        self.server_type: str = "BaseServer"
        self.tuning: bool = False
        self.personal_model_required_by_server: bool = False # Set by set_server_type
        self.history: Dict[str, List] = {k: [] for k in [ # History for the current trial
            MetricKey.TRAIN_LOSSES, MetricKey.VAL_LOSSES, MetricKey.VAL_SCORES,
            MetricKey.TEST_LOSSES, MetricKey.TEST_SCORES
        ]}
        # Add tracking of client-specific metrics
        self.client_metrics: Dict[str, Dict[str, List]] = {}
        self.round_0_state_dict: Optional[Dict] = None # Captured after round 0 training


    def set_server_type(self, name: str, tuning: bool):
        """Sets the server type name and tuning status."""
        self.server_type = name
        self.tuning = tuning
        # Determine if server type requires personal models on clients
        self.personal_model_required_by_server = name in ['pfedme', 'ditto']

    def get_best_model_state_dict(self) -> Optional[Dict]:
        """Returns the state dict of the best global model found by the server (CPU)."""
        return self.best_global_model_state_dict

    def _create_client(self, clientdata: SiteData) -> Client:
        """Creates a base Client for FedAvg."""
        mc = MetricsCalculator(self.config.dataset_name)
        metric_fn = mc.calculate_metrics
        return Client(config=self.config, data=clientdata,
                      initial_global_state=self.serverstate, # Pass server's initial state
                      metric_fn=metric_fn,
                      personal_model=False)

    def add_client(self, clientdata: SiteData):
        """Creates and adds a client using the factory method."""
        if not isinstance(clientdata, SiteData): return
        try:
            client = self._create_client(clientdata=clientdata)
            self.clients[clientdata.site_id] = client
            self._update_client_weights()
        except Exception as e: print(f"ERROR creating/adding client {clientdata.site_id}: {e}")

    def _update_client_weights(self):
        """Calculates client weights based on num_samples."""
        if not self.clients: return
        total_samples = sum(getattr(client.data, 'num_samples', 0) for client in self.clients.values())
        default_weight = 1.0 / len(self.clients) if self.clients else 0.0
        for client in self.clients.values():
             client.data.weight = getattr(client.data, 'num_samples', 0) / total_samples if total_samples > 0 else default_weight

    def run_clients(self, step_fn: Callable[[Client], Any]) -> List[Tuple[str, Any]]:
        """
        Runs step_fn on all clients, with optional parallelism.
        
        Args:
            step_fn: Function to execute on each client
        
        Returns:
            List of (client_id, result) tuples in deterministic order
        """        
        results = []
        for client_id, client in self.clients.items():
            try:
                client_result = step_fn(client)
                results.append((client_id, client_result))
            except Exception as e: 
                print(f"ERROR step_fn client {client_id}: {e}")

        results.sort(key=lambda x: x[0]) 
        return results

    def train_round(self) -> None:
        """Runs one round of training and validation."""
        current_round = len(self.history.get(MetricKey.VAL_LOSSES, []))
        if current_round % 10 == 0:
                print(f"** Round {current_round + 1} **")
        use_personal = self.personal_model_required_by_server
        def train_val_step(client: Client) -> Dict:
            return client.train_and_validate(personal=use_personal)

        client_outputs = self.run_clients(train_val_step)

        # Aggregate Metrics & States
        round_metrics = {k: 0.0 for k in [MetricKey.TRAIN_LOSSES, MetricKey.VAL_LOSSES, MetricKey.VAL_SCORES]}
        states_for_agg: List[Dict[str, Any]] = []
        
        # Store client metrics for this round
        client_round_metrics = {}

        for client_id, output_dict in client_outputs:
            if not isinstance(output_dict, dict): continue
            weight = getattr(self.clients[client_id].data, 'weight', 0.0)
            round_metrics[MetricKey.TRAIN_LOSSES] += output_dict.get('train_loss', 0.0) * weight
            round_metrics[MetricKey.VAL_LOSSES] += output_dict.get('val_loss', float('inf')) * weight
            round_metrics[MetricKey.VAL_SCORES] += output_dict.get('val_score', 0.0) * weight
            
            # Store individual client metrics
            client_round_metrics[client_id] = {
                'train_loss': output_dict.get('train_loss', 0.0),
                'val_loss': output_dict.get('val_loss', float('inf')),
                'val_score': output_dict.get('val_score', 0.0),
                'weight': weight
            }
            
            if current_round % 10 == 0:
                print(f"Client {client_id}, weight {weight:2f} - Train Loss: {output_dict.get('train_loss', 0.0):4f}, "
                    f"Val Loss: {output_dict.get('val_loss', float('inf')):4f}, "
                    f"Val Score: {output_dict.get('val_score', 0.0):4f}")
            # Collect current state dicts only if NOT a personal training round for this server type
            if not use_personal and 'state_dict' in output_dict and output_dict['state_dict'] is not None:
                states_for_agg.append({'state_dict': output_dict['state_dict'], 'weight': weight})

        # Store client metrics for this round in the history
        if 'client_metrics' not in self.history:
            self.history['client_metrics'] = []
        self.history['client_metrics'].append(client_round_metrics)
        # Hooks
        self.after_step_hook(client_outputs)

        # Server Model Updates (if applicable)
        if not use_personal and states_for_agg:
            self.aggregate_models(states_for_agg) # Implemented by FLServer
            self.distribute_global_model(test=False) # Implemented by FLServer
        elif not use_personal:
            print(f"Warning: No client states for aggregation round {current_round + 1}.")

        # Capture Round 0 State (Based on current global model after potential aggregation)
        if current_round == 0 and self.round_0_state_dict is None:
            self.round_0_state_dict = self.serverstate.model.state_dict() # Already CPU

        # Track Metrics
        self.history[MetricKey.TRAIN_LOSSES].append(round_metrics[MetricKey.TRAIN_LOSSES])
        self.history[MetricKey.VAL_LOSSES].append(round_metrics[MetricKey.VAL_LOSSES])
        self.history[MetricKey.VAL_SCORES].append(round_metrics[MetricKey.VAL_SCORES])
        
        # Update Server's Best Global Model State based on selection criterion
        # Get the current value for the selection criterion
        current_value = round_metrics.get(self.selection_criterion_key)
        if current_value is not None:
            # Check if current value is better than previous best
            is_better = False
            if self.criterion_is_higher_better:
                # For metrics where higher is better (e.g., accuracy, F1 score)
                is_better = current_value > self.best_global_value_for_selection
            else:
                # For metrics where lower is better (e.g., loss, error rate)
                is_better = current_value < self.best_global_value_for_selection
            
            if is_better:
                # Log the update for clarity
                print(f"  Updating best global model: {self.selection_criterion_key} changed from "
                    f"{self.best_global_value_for_selection:.6f} to {current_value:.6f}")
                
                # Update best value and model state
                self.best_global_value_for_selection = current_value
                self.best_global_model_state_dict = copy.deepcopy(self.serverstate.model.state_dict())

    def test_global(self) -> None:
        """Tests the appropriate model (best global) on all clients."""
        test_personal = self.personal_model_required_by_server # Test personal model if server requires it

        # Distribute Best *Global* Model State First
        self.distribute_global_model(test=True) # Sends self.best_global_model_state_dict

        def test_step(client: Client) -> Tuple[float, float]:
            return client.test(personal=test_personal) # Client internally loads best state

        client_outputs = self.run_clients(test_step)

        # Aggregate Metrics
        round_test_loss, round_test_score = 0.0, 0.0
        
        # Store client test metrics
        client_test_metrics = {}
        
        for client_id, output_tuple in client_outputs:
            weight = getattr(self.clients[client_id].data, 'weight', 0.0)
            test_loss, test_score = output_tuple
            round_test_loss += test_loss * weight
            round_test_score += test_score * weight
            
            # Store individual client test metrics
            client_test_metrics[client_id] = {
                'test_loss': test_loss,
                'test_score': test_score,
                'weight': weight
            }
        print(f"Test Loss: {round_test_loss:.4f}, Test Score: {round_test_score:.4f}")
        # Store client test metrics in the history
        if 'client_test_metrics' not in self.history:
            self.history['client_test_metrics'] = {}
        self.history['client_test_metrics'] = client_test_metrics

        # Hooks
        self.after_step_hook(client_outputs)

        # Track Metrics
        self.history[MetricKey.TEST_LOSSES].append(round_test_loss)
        self.history[MetricKey.TEST_SCORES].append(round_test_score)

    # --- Hooks & Abstract Methods ---
    def after_step_hook(self, step_results: List[Tuple[str, Any]]): pass # For Mixins
    def aggregate_models(self, client_states_info: List[Dict[str, Any]]): pass
    def distribute_global_model(self, test: bool = False): pass


# =============================================================================
# == FL Server Base Class ==
# =============================================================================
class FLServer(Server):
    """Implements standard FedAvg aggregation and distribution (CPU-based)."""

    def aggregate_models(self, client_states_info: List[Dict[str, Any]]):
        """
        Performs FedAvg aggregation on ALL parameters, including BatchNorm statistics.
        This ensures that both trainable parameters and running statistics like
        mean and variance in batch normalization layers are properly aggregated.
        """
        global_model = self.serverstate.model  # Operate on server's model (CPU)
        global_state_dict = global_model.state_dict()
        
        # Initialize accumulator for all parameters in the state dict
        accumulated_state_dict = {}
        for key in global_state_dict:
            accumulated_state_dict[key] = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
        
        total_weight = 0.0
        with torch.no_grad():
            for info in client_states_info:
                state_dict = info.get('state_dict')
                weight = info.get('weight', 0.0)
                total_weight += weight
                
                # Accumulate each parameter by name, including BN stats
                for key in accumulated_state_dict.keys():
                    if key in state_dict:
                        client_param = state_dict[key]
                        if isinstance(client_param, torch.Tensor):
                            float_param = client_param.cpu().to(torch.float32)
                            accumulated_state_dict[key].add_(float_param, alpha=weight)
            
            # Process averaged parameters
            if total_weight > 0:
                for key in accumulated_state_dict.keys():
                    accumulated_state_dict[key].div_(total_weight)
                    # Update all parameters, including BN statistics
                    global_state_dict[key].copy_(accumulated_state_dict[key].to(global_state_dict[key].dtype))

    def distribute_global_model(self, test: bool = False):
        """Distributes the appropriate CPU global model state dictionary."""
        # Distribute current global model or server's tracked best global model
        state_dict_to_dist = self.get_best_model_state_dict() if test else self.serverstate.model.state_dict()
        try:
            for client in self.clients.values():
                client.set_model_state(state_dict_to_dist, test) # Client loads CPU state dict
        except Exception as e: print(f"Error distributing state_dict: {e}")


# =============================================================================
# == Specific Server Implementations ==
# =============================================================================

class FedAvgServer(DiversityMixin, FLServer):
    """Standard FedAvg with added diversity calculation."""
    def __init__(self, *args, **kwargs):
        FLServer.__init__(self, *args, **kwargs)
        DiversityMixin.__init__(self, *args, **kwargs)

    def _create_client(self, clientdata: SiteData) -> Client:
        """Creates a base Client for FedAvg."""
        mc = MetricsCalculator(self.config.dataset_name)
        metric_fn = mc.calculate_metrics
        return Client(config=self.config, data=clientdata,
                      initial_global_state=self.serverstate, # Pass server's initial state
                      metric_fn=metric_fn,
                      personal_model=False)

class FedProxServer(FLServer):
    """FedProx server."""
    def _create_client(self, clientdata: SiteData) -> FedProxClient:
        """Creates a FedProxClient."""
        mc = MetricsCalculator(self.config.dataset_name)
        metric_fn = mc.calculate_metrics
        return FedProxClient(config=self.config, data=clientdata,
                             initial_global_state=self.serverstate,
                             metric_fn=metric_fn,
                             personal_model=False) # FedProx client is not personalized

class PFedMeServer(FLServer):
    """pFedMe server."""
    def _create_client(self, clientdata: SiteData) -> PFedMeClient:
        """Creates a PFedMeClient."""
        mc = MetricsCalculator(self.config.dataset_name)
        metric_fn = mc.calculate_metrics
        return PFedMeClient(config=self.config, data=clientdata,
                            initial_global_state=self.serverstate,
                            metric_fn=metric_fn,
                            personal_model=True) # pFedMe client requires personal

class DittoServer(FLServer):
    """Ditto server."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _create_client(self, clientdata: SiteData) -> DittoClient:
        """Creates a DittoClient."""
        mc = MetricsCalculator(self.config.dataset_name)
        metric_fn = mc.calculate_metrics
        return DittoClient(config=self.config, data=clientdata,
                           initial_global_state=self.serverstate,
                           metric_fn=metric_fn,
                           personal_model=True) # Ditto client requires personal

    def train_round(self) -> None:
        """Ditto round: Global phase + Personal phase."""
        current_round = len(self.history.get(MetricKey.VAL_LOSSES, []))

        # --- 1. Global Model Update Phase ---
        def global_train_step(client: Client) -> Dict:
            return client.train_and_validate(personal=False)
        global_client_outputs = self.run_clients(global_train_step)
        global_states_for_agg = [{'state_dict': out.get('state_dict'), 'weight': getattr(self.clients[cid].data, 'weight', 0.0)}
                                for cid, out in global_client_outputs if isinstance(out, dict) and out.get('state_dict')]
        if global_states_for_agg:
            self.aggregate_models(global_states_for_agg)
            self.distribute_global_model(test=False)

        # --- 2. Personal Model Update Step ---
        def personal_train_step(client: Client) -> Dict:
            result = client.train_and_validate(personal=True)
            result.pop('state_dict', None) # Don't need personal state dict on server
            return result
        personal_client_outputs = self.run_clients(personal_train_step)

        # --- Aggregate and Record *Personal* Metrics ---
        personal_metrics = {k: 0.0 for k in [MetricKey.TRAIN_LOSSES, MetricKey.VAL_LOSSES, MetricKey.VAL_SCORES]}
        for cid, output_dict in personal_client_outputs:
            if not isinstance(output_dict, dict): continue
            weight = getattr(self.clients[cid].data, 'weight', 0.0)
            personal_metrics[MetricKey.TRAIN_LOSSES] += output_dict.get('train_loss', 0.0) * weight
            personal_metrics[MetricKey.VAL_LOSSES] += output_dict.get('val_loss', float('inf')) * weight
            personal_metrics[MetricKey.VAL_SCORES] += output_dict.get('val_score', 0.0) * weight

        self.history[MetricKey.TRAIN_LOSSES].append(personal_metrics[MetricKey.TRAIN_LOSSES])
        self.history[MetricKey.VAL_LOSSES].append(personal_metrics[MetricKey.VAL_LOSSES])
        self.history[MetricKey.VAL_SCORES].append(personal_metrics[MetricKey.VAL_SCORES])

        # --- Hooks ---
        self.after_step_hook(personal_client_outputs) # Hook after personal step

        # --- Capture Round 0 State ---
        if current_round == 0 and self.round_0_state_dict is None:
            try: self.round_0_state_dict = self.serverstate.model.state_dict()
            except: pass

        # --- Update Best *Global* Model State ---
        # Get the appropriate metric value from personal metrics
        current_value = personal_metrics.get(self.selection_criterion_key)
        if current_value is not None:
            # Check if current value is better than previous best
            is_better = False
            if self.criterion_is_higher_better:
                # For metrics where higher is better (e.g., accuracy, F1 score)
                is_better = current_value > self.best_global_value_for_selection
            else:
                # For metrics where lower is better (e.g., loss, error rate)
                is_better = current_value < self.best_global_value_for_selection
            
            if is_better:
                # Log the update for clarity
                print(f"  Updating Ditto best global model: {self.selection_criterion_key} changed from "
                    f"{self.best_global_value_for_selection:.6f} to {current_value:.6f}")
                
                # Update best value and model state
                self.best_global_value_for_selection = current_value
                # Save the *current global* model state when the personal metrics improve
                try: 
                    self.best_global_model_state_dict = copy.deepcopy(self.serverstate.model.state_dict())
                except Exception as e:
                    print(f"Warning: Failed to update best global model state: {e}")