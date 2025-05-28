# ot_data_manager.py
import os
import torch
import numpy as np
import logging
import traceback
from typing import Dict, Optional, Tuple, List, Union, Any
from abc import ABC, abstractmethod
from directories import paths
dir_paths = paths()
ROOT_DIR = dir_paths.root_dir
ACTIVATION_DIR = dir_paths.root_dir
DATA_DIR = dir_paths.data_dir
from configs import DEFAULT_PARAMS
import models as ms
from helper import set_seeds,  MetricKey, get_model_instance
from ot_utils import calculate_sample_loss, DEFAULT_EPS
from data_processing import DataManager as FLDataManager

# Import the new ResultsManager and related classes
from results_manager import ResultsManager, ExperimentType

# Configure module logger
logger = logging.getLogger(__name__)

# Abstract Base Class for Activation Extractors
class ActivationExtractor(ABC):
    """
    Abstract base class for extracting activations from models.
    Different models may require different extraction strategies.
    """
    
    @abstractmethod
    def extract(self, 
                model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                num_classes: int, 
                dataset_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract activations (features), probabilities, and labels from a model and dataloader.
        
        Args:
            model: Neural network model
            dataloader: DataLoader containing samples
            num_classes: Number of classes in the dataset
            dataset_name: Name of the dataset (for logging/info)
            
        Returns:
            Tuple of (features, probabilities, labels) tensors
        """
        pass

class HookBasedExtractor(ActivationExtractor):
    """
    Uses forward hooks to extract activations from the final layer of a neural network.
    Works for most standard neural network architectures.
    """
    
    def extract(self, 
                model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                num_classes: int, 
                dataset_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract activations using hooks on the final linear layer.
        
        Args:
            model: Neural network model
            dataloader: DataLoader containing samples
            num_classes: Number of classes in the dataset
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Tuple of (features, probabilities, labels) tensors
        """
        if not dataloader or len(dataloader.dataset) == 0:
            logger.warning(f"HookBasedExtractor ({dataset_name}): Empty dataloader provided.")
            return None, None, None
            
        # Find the final linear layer
        final_linear = self._find_final_linear_layer(model, num_classes)
        if final_linear is None:
            logger.warning(f"HookBasedExtractor ({dataset_name}): Could not find suitable final linear layer for model.")
            return None, None, None
            
        # Set model to eval mode and move to appropriate device
        device = next(model.parameters()).device
        model.eval()
        
        # Storage for activations
        all_pre_activations = []
        all_post_activations = []
        all_labels = []
        
        # Temporary storage for current batch
        current_batch_pre_acts = []
        current_batch_post_logits = []
        
        # Define hooks
        def pre_hook(module, input_data):
            inp = input_data[0] if isinstance(input_data, tuple) else input_data
            if isinstance(inp, torch.Tensor):
                current_batch_pre_acts.append(inp.detach())
                
        def post_hook(module, input_data, output_data):
            out = output_data if not isinstance(output_data, tuple) else output_data[0]
            if isinstance(out, torch.Tensor):
                current_batch_post_logits.append(out.detach())
                
        # Register hooks
        pre_handle = final_linear.register_forward_pre_hook(pre_hook)
        post_handle = final_linear.register_forward_hook(post_hook)
        
        # Process data through model
        try:
            with torch.no_grad():
                for batch_data in dataloader:
                    # Extract batch data appropriately
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                        batch_x, batch_y = batch_data[0], batch_data[1]
                    elif isinstance(batch_data, dict):
                        batch_x = batch_data.get('features')
                        batch_y = batch_data.get('labels')
                        if batch_x is None or batch_y is None:
                            logger.debug(f"HookBasedExtractor ({dataset_name}): Skipping batch due to missing 'features' or 'labels' in dict.")
                            continue
                    else:
                        logger.debug(f"HookBasedExtractor ({dataset_name}): Skipping batch due to unrecognized batch_data format.")
                        continue
                        
                    if batch_x.shape[0] == 0:
                        continue # Skip empty batches
                        
                    # Move data to device and reset storage
                    batch_x = batch_x.to(device)
                    current_batch_pre_acts.clear()
                    current_batch_post_logits.clear()
                    
                    # Forward pass to trigger hooks
                    _ = model(batch_x)
                    
                    # Check if hooks captured data
                    if not current_batch_pre_acts or not current_batch_post_logits:
                        logger.debug(f"HookBasedExtractor ({dataset_name}): Hooks did not capture data for a batch.")
                        continue
                        
                    # Process captured activations
                    pre_acts_batch = current_batch_pre_acts[0].cpu()
                    post_logits_batch = current_batch_post_logits[0].cpu()
                    
                    # Convert logits to probabilities based on task type
                    if num_classes == 1 and post_logits_batch.ndim == 2 and post_logits_batch.shape[1] == 1: # Binary classification typical output [N, 1]
                        post_probs_batch = torch.sigmoid(post_logits_batch).squeeze(-1) # Result is [N]
                    elif num_classes == 1 and post_logits_batch.ndim == 1 : # Binary classification already squeezed output [N]
                        post_probs_batch = torch.sigmoid(post_logits_batch) # Result is [N]
                    elif num_classes > 1 and post_logits_batch.ndim == 2 and post_logits_batch.shape[1] == num_classes: # Multiclass [N, K]
                        post_probs_batch = torch.softmax(post_logits_batch, dim=-1)
                    else: # Unexpected shape
                        logger.warning(f"HookBasedExtractor ({dataset_name}): Unexpected logits shape {post_logits_batch.shape} for num_classes {num_classes}. Cannot convert to probabilities.")
                        # Fallback or error handling might be needed here if this is critical
                        # For now, we'll let it fail during concatenation if shapes are inconsistent
                        post_probs_batch = post_logits_batch # Pass logits as is, downstream will fail or handle
                        
                    # Collect results
                    all_pre_activations.append(pre_acts_batch)
                    all_post_activations.append(post_probs_batch)
                    all_labels.append(batch_y.cpu().reshape(-1))  # Ensure 1D
        finally:
            # Always remove hooks
            pre_handle.remove()
            post_handle.remove()
            
        # Check if any data was collected
        if not all_pre_activations:
            logger.warning(f"HookBasedExtractor ({dataset_name}): No activations collected.")
            return None, None, None
            
        # Concatenate results
        try:
            final_h = torch.cat(all_pre_activations, dim=0)
            final_p = torch.cat(all_post_activations, dim=0)
            final_y = torch.cat(all_labels, dim=0)
            
            prob_shape_info = tuple(final_p.shape[1:]) if final_p is not None else "None"
            logger.info(f"HookBasedExtractor ({dataset_name}): Extracted {len(final_h)} samples. Feature shape {tuple(final_h.shape[1:])}, prob shape {prob_shape_info}")
            
            return final_h, final_p, final_y
            
        except Exception as e:
            logger.exception(f"HookBasedExtractor ({dataset_name}): Error concatenating activation results: {e}")
            return None, None, None
            
    def _find_final_linear_layer(self, model: torch.nn.Module, num_classes: int) -> Optional[torch.nn.Linear]:
        """
        Intelligently locate the final linear layer in a model using multiple strategies.
        (Implementation as previously provided)
        """
        # Strategy 1: Check common attribute names for the final layer
        common_names = ['output_layer', 'fc', 'linear', 'classifier', 'output', 'fc3', 'fc2', 'fc1']
        # Add model-specific known names if any
        if hasattr(model, 'classification_head_name'): # Hypothetical attribute
            common_names.insert(0, model.classification_head_name)

        for name in common_names:
            module = getattr(model, name, None)
            if isinstance(module, torch.nn.Linear):
                expected_out_features = 1 if num_classes == 2 else num_classes # For binary, output can be 1 or 2
                if module.out_features == num_classes or (num_classes == 2 and module.out_features == 1):
                    logger.debug(f"Found final linear layer by name: '{name}' with out_features={module.out_features}")
                    return module
            elif isinstance(module, torch.nn.Sequential) and len(module) > 0:
                last_layer = module[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    if last_layer.out_features == num_classes or (num_classes == 2 and last_layer.out_features == 1):
                        logger.debug(f"Found final linear layer in sequential module '{name}[-1]' with out_features={last_layer.out_features}")
                        return last_layer
        
        # Strategy 2: Find all linear layers and select the last one that matches output dimension
        linear_layers = []
        
        def collect_candidate_layers(m):
            if isinstance(m, torch.nn.Linear):
                linear_layers.append(m)
            for _, child in m.named_children(): # Iterate over named children
                collect_candidate_layers(child)
        
        collect_candidate_layers(model)
        
        if linear_layers:
            matching_layers = [layer for layer in linear_layers 
                            if layer.out_features == num_classes or 
                                (num_classes == 2 and layer.out_features == 1)] # Binary output can be 1
            
            if matching_layers:
                logger.debug(f"Found final linear layer by iterating all linear layers, matching num_classes={num_classes}. Using last one.")
                return matching_layers[-1]
            
            # Fallback for num_classes=1 (regression) or if no exact match for classification
            if num_classes == 1 and any(l.out_features == 1 for l in linear_layers):
                 # For regression (num_classes=1), often output is 1.
                 reg_layers = [l for l in linear_layers if l.out_features == 1]
                 logger.debug(f"Found regression-like (out_features=1) linear layer. Using last one.")
                 return reg_layers[-1]

            logger.warning(f"Could not find a linear layer strictly matching num_classes={num_classes}. Using the absolute last linear layer found (out_features={linear_layers[-1].out_features}). This might be incorrect.")
            return linear_layers[-1] # Return the very last linear layer if no better match
        
        logger.error("No linear layers found in the model.")
        return None


class RepVectorExtractor(ActivationExtractor):
    """
    Extracts activations using model-specific 'rep_vector' parameter.
    Designed for models like IXITiny that provide a special extraction mode.
    """
    
    def extract(self, 
                model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                num_classes: int, 
                dataset_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract activations using the model's rep_vector mode (special parameter).
        
        Args:
            model: Neural network model that supports rep_vector parameter
            dataloader: DataLoader containing samples
            num_classes: Number of classes in the dataset
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Tuple of (features, None, dummy_labels) - no probabilities are extracted
        """
        # Check if model's forward method has 'rep_vector' in its signature
        if not (hasattr(model, 'forward') and callable(model.forward) and 
                'rep_vector' in model.forward.__code__.co_varnames):
            logger.warning(f"RepVectorExtractor ({dataset_name}): Model does not support 'rep_vector' parameter in its forward method.")
            return None, None, None
        
        if not dataloader or len(dataloader.dataset) == 0:
            logger.warning(f"RepVectorExtractor ({dataset_name}): Empty dataloader provided.")
            return None, None, None
            
        # Set model to eval mode and move to appropriate device
        device = next(model.parameters()).device
        model.eval()
        
        # Set up storage for activations
        all_rep_vectors = []
        all_batch_sizes = []
        
        # Extract representations
        try:
            with torch.no_grad():
                for batch_data in dataloader:
                    # Extract batch data
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                        batch_x = batch_data[0]
                    elif isinstance(batch_data, dict):
                        batch_x = batch_data.get('features')
                        if batch_x is None:
                            logger.debug(f"RepVectorExtractor ({dataset_name}): Skipping batch due to missing 'features' in dict.")
                            continue
                    else:
                        logger.debug(f"RepVectorExtractor ({dataset_name}): Skipping batch due to unrecognized batch_data format.")
                        continue
                        
                    if batch_x.shape[0] == 0:
                        continue  # Skip empty batches
                        
                    # Move data to device
                    batch_x = batch_x.to(device)
                    
                    # Get representation directly using rep_vector=True
                    rep_vector = model(batch_x, rep_vector=True)
                    
                    # Collect results
                    all_rep_vectors.append(rep_vector.cpu())
                    all_batch_sizes.append(batch_x.shape[0])
        except Exception as e:
            logger.exception(f"RepVectorExtractor ({dataset_name}): Error in extraction loop: {e}")
            return None, None, None
            
        # Check if any data was collected
        if not all_rep_vectors:
            logger.warning(f"RepVectorExtractor ({dataset_name}): No representation vectors collected.")
            return None, None, None
            
        # Concatenate results
        try:
            combined_reps = torch.cat(all_rep_vectors, dim=0)
            total_samples = sum(all_batch_sizes)
            
            # For segmentation models like IXITiny, 'p_prob' is None
            # 'y' are dummy labels as actual segmentation masks are not used here
            dummy_labels = torch.zeros(total_samples, dtype=torch.long)
            
            logger.info(f"RepVectorExtractor ({dataset_name}): Extracted {len(combined_reps)} samples. Feature shape {tuple(combined_reps.shape[1:])}")
            
            return combined_reps, None, dummy_labels
            
        except Exception as e:
            logger.exception(f"RepVectorExtractor ({dataset_name}): Error concatenating representation vectors: {e}")
            return None, None, None


class OTDataManager:
    """
    Handles loading, caching, generation, and processing of activations
    and performance results using saved models.
    """
    def __init__(self, num_target_fl_clients: int, activation_dir: str = ACTIVATION_DIR, results_dir: str = None, loss_eps: float = DEFAULT_EPS):
        """
        Initializes the DataManager.
        
        Args:
            num_target_fl_clients (int): The target number of clients for FL experiment analysis
            activation_dir (str): Path to activation cache directory
            results_dir (str): Path to results directory (unused, retained for API compatibility)
            loss_eps (float): Epsilon for numerical stability in loss calculations
        """
        self.num_target_fl_clients = num_target_fl_clients
        self.activation_dir = activation_dir
        self.loss_eps = loss_eps
        os.makedirs(self.activation_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FL results manager will be initialized per-dataset
        self.results_manager = None
        
        logger.info(f"DataManager initialized targeting results for {self.num_target_fl_clients} FL clients on device {self.device}.")
        
        # Activation extractor instances map
        self._extractor_instances = {
            'hook_based': HookBasedExtractor(),
            'rep_vector': RepVectorExtractor()
            # New extractor types can be added here
        }
        self._default_extractor_type = 'hook_based'

    def _initialize_results_manager(self, dataset_name: str):
        """Initialize ResultsManager for a specific dataset."""
        if self.results_manager is None or self.results_manager.path_builder.dataset != dataset_name:
            self.results_manager = ResultsManager(
                root_dir=ROOT_DIR,
                dataset=dataset_name, 
                num_target_clients=self.num_target_fl_clients
            )

    def _get_model_path(self, dataset_name: str, fl_cost: Any, fl_seed: int, model_type: str = 'round0') -> str:
        """Gets path to saved model state dict for FL experiment."""
        self._initialize_results_manager(dataset_name)
        _, metadata = self.results_manager.load_results(ExperimentType.EVALUATION)
        num_clients_in_fl_run = self.num_target_fl_clients 
        
        if metadata and 'cost_client_counts' in metadata:
            cost_counts = metadata.get('cost_client_counts', {})
            if fl_cost in cost_counts:
                num_clients_in_fl_run = cost_counts[fl_cost]
        
        return self.results_manager.path_builder.get_model_save_path(
            num_clients_run=num_clients_in_fl_run,
            cost=fl_cost,
            seed=fl_seed,
            server_type='fedavg', # Assuming FedAvg server for model to be analyzed
            model_type=model_type 
        )

    def _load_model_for_activation_generation(self, dataset_name: str, fl_cost: Any, fl_seed: int, num_clients: int, model_type: str) -> Tuple[Optional[torch.nn.Module], Optional[Dict], int]:
        """
        Loads a saved model from FL experiment for activation generation.
        """
        logger.info(f"Loading model for activation generation: Dataset={dataset_name}, FL Cost={fl_cost}, FL Seed={fl_seed}, ModelType={model_type}")
        
        model_path = self._get_model_path(dataset_name, fl_cost, fl_seed, model_type)
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            return None, None, num_clients # Return target num_clients
            
        try:
            model_state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            logger.info(f"Loaded {model_type} model state from: {os.path.basename(model_path)}")
            
            
            model = get_model_instance(dataset_name)
            model.load_state_dict(model_state_dict)
            model.to(self.device)
            model.eval()
            
            
            fl_data_mgr = FLDataManager(dataset_name, DEFAULT_PARAMS[dataset_name]['base_seed'], DATA_DIR)
            dataloaders = fl_data_mgr.get_dataloaders(cost=fl_cost, run_seed=fl_seed, num_clients_override=num_clients)
            
            # Determine actual number of clients
            actual_num_dataloaders = len(dataloaders) if dataloaders else 0
            if actual_num_dataloaders != num_clients and actual_num_dataloaders > 0:
                logger.info(f"Dataloaders created for {actual_num_dataloaders} clients, differs from target {num_clients}.")
            
            return model, dataloaders, num_clients # Return target num_clients
            
        except Exception as e:
            logger.exception(f"Error loading model/dataloaders for {dataset_name}, {fl_cost}, {fl_seed}: {e}")
            return None, None, num_clients

    def get_performance(self, dataset_name: str, fl_cost: Any, fl_seed: int, aggregation_method: str = 'mean', metric: str = 'loss') -> Tuple[float, float]:
        """
        Loads final performance metrics from FL TrialRecords.
        """
        self._initialize_results_manager(dataset_name)
        all_records, _ = self.results_manager.load_results(ExperimentType.EVALUATION)
        
        if not all_records:
            logger.warning(f"No performance results found for {dataset_name}")
            return np.nan, np.nan
        
        base_seed = DEFAULT_PARAMS[dataset_name]['base_seed']
        fl_run_idx = fl_seed - base_seed
        
        # Filter records for the specific cost and run index
        cost_run_records = [r for r in all_records if r.cost == fl_cost and r.run_idx == fl_run_idx]
        
        if not cost_run_records:
            logger.warning(f"Cost {fl_cost} and run_idx {fl_run_idx} not found in results for {dataset_name}")
            return np.nan, np.nan
        
        local_score, fedavg_score = [], []
        if metric == 'loss':
            metric_key = MetricKey.TEST_LOSSES
        elif metric == 'score':
            metric_key = MetricKey.TEST_SCORES
        else:
            logger.warning(f"Unknown metric {metric}. Using 'loss'.")
            metric_key = MetricKey.TEST_LOSSES
            
        for record in cost_run_records:
            if not record.error and record.metrics.get(metric_key):
                final_metric = record.metrics[metric_key][-1]
                if record.server_type == 'local':
                    local_score.append(final_metric)
                elif record.server_type == 'fedavg':
                    fedavg_score.append(final_metric)
        
        agg_func = np.median if aggregation_method.lower() == 'median' else np.mean
        local_scores = float(agg_func(local_score)) if local_score else np.nan
        fedavg_scores = float(agg_func(fedavg_score)) if fedavg_score else np.nan
                
        logger.info(f"Performance for {dataset_name}, cost {fl_cost}, run_idx {fl_run_idx}: Local={local_scores if not np.isnan(local_scores) else 'NaN'}, FedAvg={fedavg_scores if not np.isnan(fedavg_scores) else 'NaN'}")
        return local_scores, fedavg_scores

    def _get_activation_cache_path(self, dataset_name: str, fl_cost: Any, fl_seed: int, client_id_1: Union[str, int], client_id_2: Union[str, int], loader_type: str, num_clients: int, model_type: str) -> str:
        """Constructs standardized path for activation cache files."""
        c1_str = str(client_id_1)
        c2_str = str(client_id_2)
        dataset_cache_dir = os.path.join(self.activation_dir, dataset_name, model_type)
        os.makedirs(dataset_cache_dir, exist_ok=True)
        
        # Format cost string consistently
        if isinstance(fl_cost, (int, float)):
            cost_str = f"{float(fl_cost):.4f}"
        else:
            cost_str = str(fl_cost).replace('/', '_') # Basic sanitization
            
        filename = f"activations_{dataset_name}_nc{num_clients}_cost_{cost_str}_seed{fl_seed}_c{c1_str}v{c2_str}_{loader_type}.pt"
        return os.path.join(dataset_cache_dir, filename)

    def _generate_activations(self, dataset_name: str, fl_cost: Any, fl_seed: int, client_id_1: Union[str, int], client_id_2: Union[str, int], num_clients: int, num_classes: Optional[int], loader_type: str, model_type: str) -> Optional[Tuple]:
        """
        Generates raw activations (h, p, y) for two clients from an FL model.
        """
        logger.debug(f"Attempting to generate raw activations for {dataset_name}, clients ({client_id_1}, {client_id_2}).")
        
        try:
            model, dataloaders_all, _ = self._load_model_for_activation_generation(
                dataset_name, fl_cost, fl_seed, num_clients, model_type
            )
            
            if model is None or dataloaders_all is None:
                logger.warning(f"Failed to load model or dataloaders for {dataset_name}, FL cost {fl_cost}, FL seed {fl_seed}.")
                return None
                
            h1_raw, p1_raw, y1_raw = self._extract_raw_activations_for_client(
                model, client_id_1, dataloaders_all, loader_type, num_classes, dataset_name
            )
            h2_raw, p2_raw, y2_raw = self._extract_raw_activations_for_client(
                model, client_id_2, dataloaders_all, loader_type, num_classes, dataset_name
            )
            
            # Check if essential data (h, y) was extracted for both clients
            if h1_raw is None or y1_raw is None or h2_raw is None or y2_raw is None:
                logger.warning(f"Essential raw activations (h or y) missing for one or both clients ({client_id_1}, {client_id_2}).")
                return None # Return None if critical data is missing
                
            return (h1_raw, p1_raw, y1_raw, h2_raw, p2_raw, y2_raw)
            
        except Exception as e:
            logger.exception(f"Error during raw activation generation for {dataset_name}: {e}")
            return None

    def get_activations(self, dataset_name: str, fl_cost: Any, fl_seed: int, client_id_1: Union[str, int], client_id_2: Union[str, int], num_classes: Optional[int], loader_type: str = 'val', force_regenerate: bool = False, model_type: str = 'round0', use_loss_weighting_hint: bool = False) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        """
        Gets processed activations for a specific client pair from an FL model.
        """
        self._initialize_results_manager(dataset_name)
        
        cid1_str = str(client_id_1)
        cid2_str = str(client_id_2)
        
        cache_path = self._get_activation_cache_path(
            dataset_name, fl_cost, fl_seed, 
            cid1_str, cid2_str, loader_type, self.num_target_fl_clients, model_type
        )
        
        raw_activations_tuple = None
        if not force_regenerate:
            raw_activations_tuple = self._load_activations_from_cache(cache_path)
            
        if raw_activations_tuple is None:
            logger.info(f"Cache {'miss' if not force_regenerate else 'bypass'} for {os.path.basename(cache_path)}. Generating raw activations.")
            raw_activations_tuple = self._generate_activations(
                dataset_name=dataset_name, fl_cost=fl_cost, fl_seed=fl_seed,
                client_id_1=cid1_str, client_id_2=cid2_str,
                num_clients=self.num_target_fl_clients, num_classes=num_classes,
                loader_type=loader_type, model_type=model_type
            )
            
            if raw_activations_tuple is not None:
                self._save_activations_to_cache(raw_activations_tuple, cache_path)
            else:
                logger.warning(f"Failed to generate raw activations for {dataset_name}, pair ({cid1_str}, {cid2_str}).")
                return None
                
        h1_r, p1_r, y1_r, h2_r, p2_r, y2_r = raw_activations_tuple
        
        logger.debug(f"Processing client data for pair ({cid1_str}, {cid2_str}) with use_loss_weighting_hint={use_loss_weighting_hint}")
        processed_data1 = self._process_client_data(
            h1_r, p1_r, y1_r, cid1_str, num_classes, use_loss_weighting_hint
        )
        processed_data2 = self._process_client_data(
            h2_r, p2_r, y2_r, cid2_str, num_classes, use_loss_weighting_hint
        )
        
        if processed_data1 is None or processed_data2 is None:
            logger.warning(f"Failed to process data for one or both clients ({cid1_str}, {cid2_str}).")
            return None
            
        return {cid1_str: processed_data1, cid2_str: processed_data2}
        

    def _load_activations_from_cache(self, path: str) -> Optional[Tuple]:
        """Loads activation data from cache with validation."""
        if not os.path.isfile(path):
            return None
        
        try:
            data = torch.load(path, map_location='cpu', weights_only = True) # Always load to CPU
            # Basic validation of cached structure
            if isinstance(data, tuple) and len(data) == 6:
                 # Further check if first element (h1) and fourth (h2) are not None,
                 # as these are essential. p and y can be None for some extractors.
                if data[0] is not None and data[3] is not None:
                    logger.info(f"Successfully loaded activations from cache: {os.path.basename(path)}")
                    return data
                else:
                    logger.warning(f"Cached data in {os.path.basename(path)} has None for essential features (h1 or h2).")
            else:
                logger.warning(f"Cached data in {os.path.basename(path)} has unexpected format.")
        except Exception as e:
            logger.warning(f"Failed loading activation cache {path}: {e}")
        
        return None

    def _save_activations_to_cache(self, data: Tuple, path: str) -> None:
        """Saves activation data to cache."""
        try:
            # Ensure all tensors are on CPU before saving
            cpu_data_list = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    cpu_data_list.append(item.cpu())
                else:
                    cpu_data_list.append(item) # Handles None or other non-tensor data
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(tuple(cpu_data_list), path)
            logger.info(f"Saved activations to cache: {os.path.basename(path)}")
        except Exception as e:
            logger.warning(f"Failed saving activation cache {path}: {e}")


    def _get_activation_extractor(self, dataset_name: str) -> ActivationExtractor:
        """
        Determines and returns the appropriate ActivationExtractor instance for the dataset.
        Relies on 'activation_extractor_type' in DEFAULT_PARAMS from configs.py.
        """
        dataset_params = DEFAULT_PARAMS.get(dataset_name, {})
        extractor_type_key = dataset_params.get('activation_extractor_type', self._default_extractor_type)
        
        extractor_instance = self._extractor_instances.get(extractor_type_key)
        if extractor_instance:
            logger.debug(f"Using '{extractor_type_key}' extractor for dataset '{dataset_name}'.")
            return extractor_instance
        else:
            logger.warning(f"Unknown extractor type '{extractor_type_key}' for dataset '{dataset_name}'. "
                           f"Falling back to default '{self._default_extractor_type}'.")
            return self._extractor_instances[self._default_extractor_type]

    def _extract_raw_activations_for_client(
        self, model: torch.nn.Module, client_id: Union[str, int], 
        dataloaders_all_clients: Dict, loader_type: str, 
        num_classes: int, dataset_name: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extracts raw activations (h, p, y) for a single client using the appropriate extractor.
        """
        client_id_str = str(client_id) # Normalize client_id to string
        logger.debug(f"Extracting raw activations for client {client_id_str}, dataset {dataset_name}, loader {loader_type}.")
        
        # Select the correct dataloader for the specific client
        client_dataloader_tuple = dataloaders_all_clients.get(client_id_str)
        if not client_dataloader_tuple: # Try integer key if string failed
            try:
                client_dataloader_tuple = dataloaders_all_clients.get(int(client_id))
            except ValueError:
                pass # client_id was not an int string
        
        if not client_dataloader_tuple:
            logger.warning(f"No dataloaders found for client {client_id_str} in provided dataloaders_all_clients.")
            return None, None, None
                
        loader_idx = {'train': 0, 'val': 1, 'test': 2}.get(loader_type.lower())
        if loader_idx is None or not isinstance(client_dataloader_tuple, (list, tuple)) or len(client_dataloader_tuple) <= loader_idx:
            logger.warning(f"Invalid loader structure or type '{loader_type}' for client {client_id_str}.")
            return None, None, None
                
        specific_dataloader = client_dataloader_tuple[loader_idx]
        if specific_dataloader is None or len(specific_dataloader.dataset) == 0:
            logger.info(f"Empty '{loader_type}' loader for client {client_id_str}.")
            # Return None for h, p, y so processing can identify this as no data
            return None, None, None 
            
        extractor = self._get_activation_extractor(dataset_name)
        return extractor.extract(model, specific_dataloader, num_classes, dataset_name)

    def _process_client_data(
        self, h_raw: Optional[torch.Tensor], 
        p_raw: Optional[torch.Tensor], 
        y_raw: Optional[torch.Tensor],
        client_id: str, 
        num_classes: int,
        use_loss_weighting_hint: bool = False
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        """
        Processes raw data for one client: validates probs, calculates loss & weights.
        """
        client_id_str = str(client_id)
        if h_raw is None or y_raw is None: # p_raw can be None (e.g. IXITiny)
            logger.info(f"Client {client_id_str}: Missing raw features (h) or labels (y). Cannot process.")
            return None # Cannot proceed without h and y
            
        # Ensure tensors are on CPU
        h_cpu = h_raw.cpu() if isinstance(h_raw, torch.Tensor) else torch.tensor(h_raw, device='cpu')
        y_cpu = y_raw.cpu().long() if isinstance(y_raw, torch.Tensor) else torch.tensor(y_raw, dtype=torch.long, device='cpu')
        p_prob_cpu = p_raw.cpu() if isinstance(p_raw, torch.Tensor) else None # p_raw could be None
        
        n_samples = y_cpu.shape[0]
        if n_samples == 0:
            logger.info(f"Client {client_id_str}: Zero samples after initial extraction.")
            return {'h': h_cpu, 'p_prob': p_prob_cpu, 'y': y_cpu, 'loss': torch.empty(0, device='cpu'), 'weights': torch.empty(0, device='cpu')}

        p_prob_validated = None
        if p_prob_cpu is not None:
            with torch.no_grad():
                p_prob_float = p_prob_cpu.float()
                if p_prob_float.ndim == 2 and p_prob_float.shape[0] == n_samples and p_prob_float.shape[1] == num_classes:
                    row_sums = p_prob_float.sum(dim=1)
                    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
                        logger.debug(f"Client {client_id_str}: Probabilities do not sum to 1. Normalizing.")
                        p_prob_float = p_prob_float / row_sums.unsqueeze(1).clamp(min=self.loss_eps)
                    p_prob_validated = torch.clamp(p_prob_float, 0.0, 1.0)
                elif num_classes == 2 and ((p_prob_float.ndim == 1 and p_prob_float.shape[0] == n_samples) or \
                                          (p_prob_float.ndim == 2 and p_prob_float.shape[0] == n_samples and p_prob_float.shape[1] == 1)):
                    p1 = torch.clamp(p_prob_float.view(-1), 0.0, 1.0)
                    p0 = 1.0 - p1
                    p_prob_validated = torch.stack([p0, p1], dim=1)
                else:
                    logger.warning(f"Client {client_id_str}: Cannot validate p_prob shape {p_prob_float.shape} for N={n_samples}, K={num_classes}.")
        
        loss_tensor = None
        if p_prob_validated is not None:
            loss_tensor = calculate_sample_loss(p_prob_validated, y_cpu, num_classes, self.loss_eps)
            if loss_tensor is None: # Log if calculate_sample_loss failed
                 logger.warning(f"Client {client_id_str}: Loss calculation failed despite valid p_prob.")
        
        weights_tensor = None
        if use_loss_weighting_hint and loss_tensor is not None and torch.isfinite(loss_tensor).all():
            loss_sum = loss_tensor.sum()
            if loss_sum > self.loss_eps:
                weights_tensor = loss_tensor / loss_sum
                logger.debug(f"Client {client_id_str}: Using loss-derived weights.")
            else: # Loss sum is too small, fall back to uniform
                logger.debug(f"Client {client_id_str}: Loss sum too small for loss-weighting, using uniform.")
                weights_tensor = torch.ones(n_samples, dtype=torch.float32, device='cpu') / n_samples
        else:
            if use_loss_weighting_hint and (loss_tensor is None or not torch.isfinite(loss_tensor).all()):
                logger.debug(f"Client {client_id_str}: Loss weighting requested but loss unavailable/invalid. Using uniform.")
            weights_tensor = torch.ones(n_samples, dtype=torch.float32, device='cpu') / n_samples
        
        return {
            'h': h_cpu, 
            'p_prob': p_prob_validated, 
            'y': y_cpu, 
            'loss': loss_tensor if loss_tensor is not None else torch.full((n_samples,), float('nan'), device='cpu'), # ensure loss tensor even if nan
            'weights': weights_tensor
        }

