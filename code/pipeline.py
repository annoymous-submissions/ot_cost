# pipeline.py
"""
Orchestrates FL experiments: manages loops over runs and costs,
delegates single trial execution, aggregates results into records,
and handles saving models/results. Streamlined version.
"""
import traceback
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union, Callable
import gc
# Import project modules
from directories import paths
dir_paths = paths()
ROOT_DIR = dir_paths.root_dir
DATA_DIR = dir_paths.data_dir
from configs import ALGORITHMS, DEVICE,REG_ALOGRITHMS
from helper import (set_seeds, get_parameters_for_dataset, get_model_instance, systematic_memory_cleanup, # Keep necessary imports
                    get_default_lr, get_default_reg, MetricKey, SiteData, ModelState, TrainerConfig) # Import types from helper
# Import necessary components
from servers import Server, FedAvgServer, FedProxServer, PFedMeServer, DittoServer # Import all server types
import models as ms
from data_processing import DataManager
from results_manager import ResultsManager, ExperimentType, TrialRecord # Import TrialRecord

# =============================================================================
# == Experiment Configuration ==
# =============================================================================
@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    dataset: str
    experiment_type: str
    num_clients: Optional[int] = None # Can be overridden by CLI

# =============================================================================
# == Single Run Executor Class ==
# =============================================================================
class SingleRunExecutor:
    """
    Handles the execution of a single Federated Learning run/trial.
    Creates model, server, clients, runs training/evaluation loop.
    Returns metrics history and model state dictionaries.
    """
    def __init__(self, dataset_name: str, default_params: Dict, device: torch.device):
        self.dataset_name = dataset_name
        self.default_params = default_params
        self.device = device # Target device for client computation

    def _create_model(self) -> Tuple[nn.Module, Union[nn.Module, Callable]]:
        """Creates model instance (on CPU initially)."""
        # Use the helper to get a fresh model instance
        model = get_model_instance(self.dataset_name)
        return model

    def _create_trainer_config(self, server_type: str, hyperparams: Dict, tuning: bool) -> TrainerConfig:
        """Creates the TrainerConfig."""
        lr = hyperparams.get('learning_rate')
        if lr is None: raise ValueError("Learning rate missing for TrainerConfig.")
        reg = hyperparams.get('reg_param')
        algo_params = {'reg_param': reg} if server_type in ['fedprox', 'pfedme', 'ditto'] and reg is not None else {}
        requires_personal_model = server_type in ['pfedme', 'ditto'] # Info for client init
        rounds = self.default_params.get('rounds_tune_inner') if tuning else self.default_params['rounds']
        max_parallel = self.default_params.get('max_parallel_clients', None) 
        use_weighted_loss = self.default_params.get('use_weighted_loss', False)
        
        # Get selection criteria from default_params
        selection_criterion_key = self.default_params.get('selection_criterion_key', 'val_losses')
        selection_criterion_direction_overrides = self.default_params.get('selection_criterion_direction_overrides', {})

        return TrainerConfig(
            dataset_name=self.dataset_name,
            device=str(self.device), 
            learning_rate=lr,
            batch_size=self.default_params['batch_size'], 
            epochs=self.default_params.get('epochs_per_round', 3),
            rounds=rounds, 
            requires_personal_model=requires_personal_model, 
            algorithm_params=algo_params,
            max_parallel_clients=max_parallel,
            use_weighted_loss=use_weighted_loss,
            selection_criterion_key=selection_criterion_key,
            selection_criterion_direction_overrides=selection_criterion_direction_overrides
        )


    def _create_server_instance(self, server_type: str, config: TrainerConfig, tuning: bool) -> Server:
        """Creates server instance with model on CPU."""
        model = self._create_model()
        # Initial global state for server (model is CPU)
        globalmodelstate = ModelState(model=model.cpu())

        server_mapping: Dict[str, type[Server]] = {
            'local': Server, # Assuming 'local' uses base Server or similar
            'fedavg': FedAvgServer,
            'fedprox': FedProxServer,
            'pfedme': PFedMeServer,
            'ditto': DittoServer
        }
        server_class = server_mapping.get(server_type)
        if server_class is None: raise ValueError(f"Unsupported server type: {server_type}")

        server = server_class(config=config, globalmodelstate=globalmodelstate)
        server.set_server_type(server_type, tuning)
        return server

    def _add_clients_to_server(self, server: Server, client_dataloaders: Dict) -> int:
        """Adds client data to the server. Server handles Client instantiation."""
        added_count = 0
        for client_id, loaders in client_dataloaders.items():
            try:
                train_loader, val_loader, test_loader = loaders
                clientdata = SiteData(site_id=client_id, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
                server.add_client(clientdata=clientdata)
                added_count += 1
            except Exception as e: print(f"ERROR adding client {client_id}: {e}") # Keep essential error logging
        return added_count

    def _train_and_evaluate(self, server: Server, rounds: int) -> Dict:
        """Runs the FL training/evaluation loop via server methods."""
        if not server.clients: return {'error': 'No clients available'}
        # Server internally manages history dict
        for round_num in range(rounds):
            try: server.train_round()
            except Exception as e:
                 metrics = getattr(server, 'history', {})
                 metrics['error'] = f"Error in train_round {round_num+1}: {e}"
                 return metrics # Return history accumulated so far + error
        if not server.tuning:
            try: server.test_global()
            except Exception as e:
                 metrics = getattr(server, 'history', {})
                 metrics['error_test'] = f'Error in test_global: {e}'
                 return metrics
        # Return final history accumulated by the server
        return getattr(server, 'history', {'error': 'Server history attribute missing.'})

    def execute_trial(self,
                    server_type: str,
                    hyperparams: Dict,
                    client_dataloaders: Dict,
                    tuning: bool
                    ) -> Tuple[Dict, Dict[str, Optional[Dict]]]:
        """
        Executes a single FL trial. Returns metrics history and model states.
        SingleRunExecutor has full responsibility for server lifecycle management.
        """
        server: Optional[Server] = None
        metrics: Dict = {}
        model_states: Dict[str, Optional[Dict]] = {'final': None, 'best': None, 'round0': None, 'error': None}

        try:
            trainer_config = self._create_trainer_config(server_type, hyperparams, tuning)
            # Pass tuning flag, learning rate implicitly handled by client via config
            server = self._create_server_instance(server_type, trainer_config, tuning)
            actual_clients_added = self._add_clients_to_server(server, client_dataloaders)
            if actual_clients_added == 0:
                metrics = {'error': 'No clients successfully added to server.'}
            else:
                metrics = self._train_and_evaluate(server, trainer_config.rounds)
                # Retrieve model states from server if trial succeeded
                if 'error' not in metrics:
                    model_states['best'] = server.get_best_model_state_dict()
                    model_states['round0'] = server.round_0_state_dict

        except Exception as e:
            err_msg = f"Executor setup/run failed: {e}"
            print(err_msg); traceback.print_exc() # Keep traceback for executor errors
            metrics['error'] = err_msg
            model_states['error'] = err_msg # Also store error marker with states
        finally:
            # SingleRunExecutor takes full responsibility for cleaning up the server
            if server:
                # First clean up each client's resources
                if hasattr(server, 'clients'):
                    for client_id, client in list(server.clients.items()):
                        # Clean client's dataloaders (these don't belong to us, just null the references)
                        if hasattr(client, 'data'):
                            if hasattr(client.data, 'train_loader'):
                                client.data.train_loader = None
                            if hasattr(client.data, 'val_loader'):
                                client.data.val_loader = None
                            if hasattr(client.data, 'test_loader'):
                                client.data.test_loader = None
                        
                        # Clean client's models/optimizers
                        if hasattr(client, 'global_state'):
                            if hasattr(client.global_state, 'optimizer'):
                                client.global_state.optimizer = None
                        
                        if hasattr(client, 'personal_state'):
                            if hasattr(client.personal_state, 'optimizer'):
                                client.personal_state.optimizer = None
                    
                    # Now clear and delete the clients dictionary
                    server.clients.clear()
                
                # Clean server state
                if hasattr(server, 'serverstate'):
                    server.serverstate.optimizer = None
                
                # Delete the server
                del server
                server = None

            # Comprehensive memory cleanup
            systematic_memory_cleanup()

        return metrics, model_states
# =============================================================================
# == Experiment Orchestrator Class ==
# =============================================================================
@dataclass
class RunMetadata:
    """Metadata for a single experiment run."""
    run_idx_total: int; seed_used: int
    cost_client_counts: Dict[Any, int]
    dataset_name: str = ""; num_target_clients: int = 0

@dataclass
class CostExecutionResult:
    """Results from executing all trials for a specific cost within one run."""
    cost: Any
    trial_records: List[TrialRecord]
    # No longer need to pass server state dicts from here

class Experiment:
    """Orchestrates FL experiments using a simplified, flatter structure."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.default_params = get_parameters_for_dataset(config.dataset)
        self.base_seed = self.default_params.get('base_seed', 42)
        self.num_target_clients = config.num_clients or self.default_params.get('default_num_clients', 2)
        # Initialize managers
        self.results_manager = ResultsManager(ROOT_DIR, config.dataset, self.num_target_clients)
        self.data_manager = DataManager(config.dataset, self.base_seed, DATA_DIR)
        self.single_run_executor = SingleRunExecutor(config.dataset, self.default_params, DEVICE)
        self.all_trial_records: List[TrialRecord] = [] # Main result storage

    def run_experiment(self, costs: List[Any]) -> List[TrialRecord]:
        """Main entry point."""
        experiment_type = self.config.experiment_type
        if experiment_type == ExperimentType.EVALUATION:
            self._execute_experiment_runs(experiment_type, costs, self._evaluate_cost_for_run)
        elif experiment_type in [ExperimentType.LEARNING_RATE, ExperimentType.REG_PARAM]:
            self._execute_experiment_runs(experiment_type, costs, self._tune_cost_for_run)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

        print(f"\n=== Experiment Orchestrator Finished ({experiment_type}) ===")
        return self.all_trial_records

    def _execute_experiment_runs(self, experiment_type: str, costs: List[Any], cost_execution_func: Callable):
        """
        Generic loop structure for executing multiple runs and costs.
        Experiment takes responsibility for client_dataloaders lifecycle.
        """
        self.all_trial_records, remaining_costs, completed_runs = \
            self.results_manager.get_experiment_status(experiment_type, costs, self.default_params, MetricKey)

        is_tuning = experiment_type != ExperimentType.EVALUATION
        target_runs_key = 'runs_tune' if is_tuning else 'runs'
        target_runs = self.default_params.get(target_runs_key, 1)

        if not remaining_costs and completed_runs >= target_runs: return # Already done

        remaining_runs_count = target_runs - completed_runs
        if remaining_runs_count <= 0 and remaining_costs:
            remaining_runs_count, completed_runs = target_runs, 0

        print(f"Orchestrator: Starting {remaining_runs_count} run(s) for '{experiment_type}' (Runs {completed_runs + 1} to {target_runs})...")

        # --- Run Loop ---
        for run_offset in range(remaining_runs_count):
            current_run_idx = completed_runs + run_offset # 0-based index
            current_seed = self.base_seed + current_run_idx
            print(f"\n--- Run {current_run_idx + 1}/{target_runs} (Seed: {current_seed}) ---")
            set_seeds(current_seed)

            run_records: List[TrialRecord] = []
            run_cost_client_counts: Dict[Any, int] = {}

            costs_to_run_this_iter = remaining_costs if completed_runs == 0 else costs

            # --- Cost Loop ---
            for cost in costs_to_run_this_iter:
                cost_results = None
                num_actual_clients = 0
                client_dataloaders = None  # Initialize to None
                
                try:
                    # Experiment is responsible for getting dataloaders
                    client_dataloaders = self.data_manager.get_dataloaders(
                        cost=cost, run_seed=current_seed, num_clients_override=self.config.num_clients
                    )
                    if not client_dataloaders: 
                        raise RuntimeError("No dataloaders.")
                        
                    num_actual_clients = len(client_dataloaders)
                    run_cost_client_counts[cost] = num_actual_clients
                    
                    # Execute trials for this cost (SingleRunExecutor manages server)
                    cost_results = cost_execution_func(cost, current_run_idx, current_seed, client_dataloaders, num_actual_clients)

                except Exception as e: # Catch errors in dataloading or cost execution
                    print(f"  ERROR processing cost {cost} in run {current_run_idx + 1}: {e}")
                    run_records.append(TrialRecord(cost=cost, run_idx=current_run_idx, server_type="N/A", error=f"Cost processing error: {e}"))
                    if num_actual_clients > 0: 
                        run_cost_client_counts[cost] = num_actual_clients # Record client count if possible
                finally:
                    # Cost-level cleanup - Experiment is responsible for client_dataloaders
                    if client_dataloaders:
                        for client_id, loaders in client_dataloaders.items():
                            # loaders is a tuple (train, val, test)
                            for loader in loaders:
                                if loader is not None and hasattr(loader, "_shutdown_workers"):
                                    loader._shutdown_workers()   # safe on all torch versions
                            client_dataloaders[client_id] = None   # drop reference to the tuple
                        client_dataloaders.clear()
                                        
                    # Force comprehensive memory cleanup
                    systematic_memory_cleanup()

                # Process cost results
                if cost_results and hasattr(cost_results, 'trial_records'):
                    run_records.extend(cost_results.trial_records)
                    # Explicitly clean up cost results to free memory
                    cost_results.trial_records = []
                    cost_results = None
            # --- End Cost Loop ---

            self.all_trial_records.extend(run_records) # Add records from this run

            # Save aggregated results after each full run with selection criteria in metadata
            run_meta = RunMetadata(
                run_idx_total=current_run_idx + 1, 
                seed_used=current_seed,
                cost_client_counts=run_cost_client_counts, 
                dataset_name=self.config.dataset,
                num_target_clients=self.num_target_clients
            )
            
            # Add selection criteria info to metadata
            metadata_dict = vars(run_meta)
            metadata_dict['selection_criterion_key'] = self.default_params.get('selection_criterion_key', 'val_losses')
            metadata_dict['selection_criterion_direction_overrides'] = self.default_params.get('selection_criterion_direction_overrides', {})
            
            self.results_manager.save_results(self.all_trial_records, experiment_type, metadata_dict)
            
            # Comprehensive memory cleanup at run level
            systematic_memory_cleanup()
    # --- Cost Processing Helpers ---

    def _tune_cost_for_run(self, cost: Any, run_idx: int, seed: int,
                        client_dataloaders: Dict, num_actual_clients: int
                        ) -> CostExecutionResult:
        """Executes all tuning trials for a single cost and run."""
        tuning_type = self.config.experiment_type
        trial_records: List[TrialRecord] = []
        # Determine tuning parameters
        if tuning_type == ExperimentType.LEARNING_RATE: param_key, try_vals_key, servers_key = 'learning_rate', 'learning_rates_try', 'servers_tune_lr'
        elif tuning_type == ExperimentType.REG_PARAM: param_key, try_vals_key, servers_key = 'reg_param', 'reg_params_try', 'servers_tune_reg'
        else: raise ValueError(f"Invalid tuning type: {tuning_type}")

        fixed_key = 'reg_param' if param_key == 'learning_rate' else 'learning_rate'
        fixed_val_func = get_default_reg if param_key == 'learning_rate' else get_default_lr
        fixed_val = fixed_val_func(self.config.dataset)
        try_vals = self.default_params.get(try_vals_key, [])
        servers_to_tune = self.default_params.get(servers_key, [])
        if not try_vals or not servers_to_tune: return CostExecutionResult(cost=cost, trial_records=[])

        # Loop over HPs and Servers
        for param_val in try_vals:
            for server_type in servers_to_tune:
                # For REG_PARAM tuning of personalized algorithms, use FedAvg's best LR
                if tuning_type == ExperimentType.REG_PARAM and server_type in REG_ALOGRITHMS:
                    # Get FedAvg's best learning rate
                    selection_criterion_key = self.default_params.get('selection_criterion_key', 'val_losses')
                    selection_criterion_direction_overrides = self.default_params.get('selection_criterion_direction_overrides', {})
                    fedavg_best_lr = self.results_manager.get_best_parameters(
                        ExperimentType.LEARNING_RATE, 
                        'fedavg', 
                        cost,
                        selection_criterion_key,
                        selection_criterion_direction_overrides
                    )
                    # Fallback to default if FedAvg's best LR not found
                    if fedavg_best_lr is None:
                        fedavg_best_lr = get_default_lr(self.config.dataset)
                    hp = {param_key: param_val, fixed_key: fedavg_best_lr}
                else:
                    hp = {param_key: param_val, fixed_key: fixed_val}
                    
                print(f"\n")
                print(f"========== Cost: {cost} | Server: {server_type:<10} | Run: {run_idx+1} | {param_key}: {param_val:.5f} ========== ", end="")
                print(f"\n")
                trial_metrics, _ = self.single_run_executor.execute_trial( # Ignore model states for tuning
                    server_type=server_type, hyperparams=hp,
                    client_dataloaders=client_dataloaders, tuning=True
                )
                record = TrialRecord(cost=cost, run_idx=run_idx, server_type=server_type,
                                    tuning_param_name=param_key, tuning_param_value=param_val,
                                    metrics=trial_metrics, error=trial_metrics.get('error'))
                trial_records.append(record)
        print(f"\n")
        return CostExecutionResult(cost=cost, trial_records=trial_records)

    def _evaluate_cost_for_run(self, cost: Any, run_idx: int, seed: int,
                        client_dataloaders: Dict, num_actual_clients: int
                        ) -> CostExecutionResult:
        """Executes only missing evaluation trials (servers) for a single cost and run.""" 
        trial_records: List[TrialRecord] = []
        
        # Get selection criteria from default params
        selection_criterion_key = self.default_params.get('selection_criterion_key', 'val_losses')
        selection_criterion_direction_overrides = self.default_params.get('selection_criterion_direction_overrides', {})

        # Check existing records to find which algorithms are already complete
        completed_algos = set()
        for record in self.all_trial_records:
            if record.cost == cost and record.run_idx == run_idx and record.error is None:
                completed_algos.add(record.server_type)
        
        # Only run algorithms that haven't been completed yet
        algorithms_to_run = [algo for algo in ALGORITHMS if algo not in completed_algos]
        
        if not algorithms_to_run:
            print(f"All algorithms already complete for cost {cost}, run {run_idx+1}")
            return CostExecutionResult(cost=cost, trial_records=[])
            
        print(f"Running missing algorithms for cost {cost}, run {run_idx+1}: {algorithms_to_run}")

        for server_type in algorithms_to_run:
            # For personalized algorithms, use FedAvg's best learning rate
            if server_type in REG_ALOGRITHMS:
                # Get FedAvg's best learning rate
                best_lr = self.results_manager.get_best_parameters(
                    ExperimentType.LEARNING_RATE, 
                    'fedavg', 
                    cost, 
                    selection_criterion_key,
                    selection_criterion_direction_overrides
                )
            else:
                # For local and fedavg, use their own best learning rate
                best_lr = self.results_manager.get_best_parameters(
                    ExperimentType.LEARNING_RATE, 
                    server_type, 
                    cost, 
                    selection_criterion_key,
                    selection_criterion_direction_overrides
                )
            
            # Get best reg param (specific to each algorithm)
            best_reg = self.results_manager.get_best_parameters(
                ExperimentType.REG_PARAM, 
                server_type, 
                cost,
                selection_criterion_key,
                selection_criterion_direction_overrides
            )
            
            # Use defaults if not found
            if best_lr is None: best_lr = get_default_lr(self.config.dataset)
            if best_reg is None and server_type in REG_ALOGRITHMS: best_reg = get_default_reg(self.config.dataset)
            eval_hyperparams = {'learning_rate': best_lr, 'reg_param': best_reg}
            print(f"\n")
            print(f"========== Cost: {cost} | Server: {server_type:<10} | Run: {run_idx+1} | LR: {best_lr:.5f} ========== ", end="")
            print(f"\n")
            # Execute trial
            trial_metrics, model_states = self.single_run_executor.execute_trial(
                server_type=server_type, hyperparams=eval_hyperparams,
                client_dataloaders=client_dataloaders, tuning=False
            )
            # Create record
            record = TrialRecord(cost=cost, run_idx=run_idx, server_type=server_type,
                                metrics=trial_metrics, error=trial_metrics.get('error') or model_states.get('error'))
            trial_records.append(record)

            # Save models immediately if successful (only FedAvg for now)
            if record.error is None and server_type == 'fedavg':
                for model_type, state_dict in model_states.items():
                    if model_type != 'error' and state_dict is not None:
                        self.results_manager.save_model_state(
                            state_dict, num_actual_clients, cost, seed, 'fedavg', model_type)
                        
        print(f"\n")
        # Return records, no need to return state dicts from here
        return CostExecutionResult(cost=cost, trial_records=trial_records)