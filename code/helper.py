# helper.py
"""
Core utility functions for the federated learning pipeline.
Streamlined version focusing on essential helpers.
REMOVED: translate_cost function.
"""
import os
import gc
import random
import numpy as np
import torch
from contextlib import contextmanager, suppress
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import directories
from typing import Dict, Optional, Tuple, List, Iterator, Any, Callable,  Union
from dataclasses import dataclass, field
import copy
from functools import partial
import sklearn.metrics as metrics
from losses import WeightedCELoss, ISICLoss, get_dice_loss, get_dice_score # Custom loss function
import torch.nn.functional as F

# Import global config directly
from configs import DEFAULT_PARAMS # Needed for config helpers

# --- Seeding ---
def set_seeds(seed_value: int = 42):
    """Sets random seeds for PyTorch, NumPy, and Python's random module."""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.benchmark = True
    
# --- Device Handling ---
@contextmanager
def gpu_scope():
    try: yield
    finally:
        with suppress(Exception):
            gc.collect()  # Collect Python objects first
            if torch.cuda.is_available():
                torch.cuda.synchronize()   # Wait for all CUDA kernels to finish
                torch.cuda.empty_cache()   # Release cached blocks
                torch.cuda.ipc_collect()   # Release IPC handles (e.g., from DataLoaders)

def move_to_device(batch: Any, device: torch.device) -> Any:
    """Moves a batch (tensor, list/tuple of tensors) to the specified device."""
    if isinstance(batch, (list, tuple)):
        return [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
    elif isinstance(batch, torch.Tensor): return batch.to(device)
    return batch

def systematic_memory_cleanup():
    """
    Performs systematic memory cleanup of Python and GPU resources.
    
    Args:
        is_gpu_active: Flag to indicate if GPU cleanup should be attempted
    """
    import gc
    gc.collect()  # Collect Python objects first
    if torch.cuda.is_available():
        torch.cuda.synchronize()   # Wait for all CUDA kernels to finish
        torch.cuda.empty_cache()   # Release cached blocks
        torch.cuda.ipc_collect()   # Release IPC handles (e.g., from DataLoaders

# --- Configuration Helpers ---
def get_parameters_for_dataset(dataset_name: str) -> Dict:
    """Retrieves the config dict for a dataset from global defaults."""
    params = DEFAULT_PARAMS.get(dataset_name)
    if params is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in configs.DEFAULT_PARAMS.")
    if 'dataset_name' not in params: params['dataset_name'] = dataset_name
    return params

def get_default_lr(dataset_name: str) -> float:
    """Gets the default learning rate from the dataset's config."""
    params = get_parameters_for_dataset(dataset_name)
    lr = params.get('default_lr')
    if lr is None: raise ValueError(f"'default_lr' not defined for '{dataset_name}'.")
    return lr

def get_default_reg(dataset_name: str) -> Optional[float]:
    """Gets the default regularization parameter (can be None)."""
    params = get_parameters_for_dataset(dataset_name)
    return params.get('default_reg_param')

def infer_higher_is_better(metric_key: str, direction_overrides: Optional[Dict[str, str]] = None) -> bool:
    """
    Infers whether a higher value is better for a given metric key.
    
    Args:
        metric_key: The metric key to check (e.g., 'val_losses', 'val_scores')
        direction_overrides: Optional dictionary of metric_key to direction ('higher' or 'lower')
        
    Returns:
        True if higher is better, False otherwise
    """
    # Check if we have an explicit override
    if direction_overrides and metric_key in direction_overrides:
        return direction_overrides[metric_key].lower() == 'higher'
    
    # Define known patterns for metric names
    KNOWN_PATTERNS = {
        'loss': 'lower', 
        'error': 'lower', 
        'mae': 'lower',
        'mse': 'lower',
        'rmse': 'lower',
        'accuracy': 'higher', 
        'score': 'higher', 
        'f1': 'higher', 
        'auc': 'higher', 
        'dice': 'higher',
        'precision': 'higher',
        'recall': 'higher'
    }
    
    # Check for known patterns in the metric key
    metric_key_lower = metric_key.lower()
    for pattern, direction in KNOWN_PATTERNS.items():
        if pattern in metric_key_lower:
            return direction.lower() == 'higher'
    
    # If no pattern matches, issue a warning and default to False (lower is better)
    print(f"WARNING: Could not determine if higher is better for metric '{metric_key}'. " 
          f"Defaulting to lower is better. Use direction_overrides for explicit control.")
    return False

# --- Configuration & Data Structures ---
def calculate_class_weights(dataset, num_classes):
    """
    Calculate class weights based on dataset label distribution.
    
    Args:
        dataset: PyTorch Dataset containing the training data
        num_classes: Total number of classes (fixed_classes in config)
    
    Returns:
        torch.Tensor: Class weights tensor of shape (num_classes,)
    """
    class_counts = torch.zeros(num_classes)
    
    # Iterate through dataset to count class occurrences
    for i in range(len(dataset)):
        _, label = dataset[i]
        if isinstance(label, torch.Tensor):
            label_idx = label.item() if label.numel() == 1 else label.argmax().item()
        else:
            label_idx = int(label)
            
        if 0 <= label_idx < num_classes:
            class_counts[label_idx] += 1
    
    # Handle classes with zero samples (avoid division by zero)
    min_count = class_counts[class_counts > 0].min().item() if torch.any(class_counts > 0) else 1
    # Replace zeros with a small fraction of the minimum count
    class_counts = torch.where(class_counts > 0, class_counts, torch.tensor(min_count * 0.1))
    
    # Calculate inverse frequency weights
    class_weights = 1.0 / class_counts
    
    # Normalize to make average weight = 1
    class_weights = class_weights * (num_classes / class_weights.sum())
    
    return class_weights

# --- Constants ---
class MetricKey:
    TRAIN_LOSSES = 'train_losses'; VAL_LOSSES = 'val_losses'; TEST_LOSSES = 'test_losses'
    TRAIN_SCORES = 'train_scores'; VAL_SCORES = 'val_scores'; TEST_SCORES = 'test_scores'

# Define Experiment types locally
class ExperimentType:
    LEARNING_RATE = 'learning_rate'
    REG_PARAM = 'reg_param'
    EVALUATION = 'evaluation' 
    OT_ANALYSIS = 'ot_analysis' 


@dataclass
class SiteData:
    """Client data and metadata."""
    site_id: str
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    weight: float = 1.0
    num_samples: int = 0

    def __post_init__(self):
        if self.num_samples == 0 and self.train_loader and hasattr(self.train_loader, 'dataset'):
            try: self.num_samples = len(self.train_loader.dataset)
            except: self.num_samples = 0

@dataclass
class TrainerConfig:
    """Training configuration."""
    dataset_name: str
    device: str # Target compute device string (e.g., 'cuda:0')
    learning_rate: float
    batch_size: int
    epochs: int = 1
    rounds: int = 1
    requires_personal_model: bool = False
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    max_parallel_clients: Optional[int] = None
    use_weighted_loss: bool = False
    selection_criterion_key: str = directories.SELECTION_CRITERION_KEY
    selection_criterion_direction_overrides: Optional[Dict[str, str]] = field(default_factory=dict)

@dataclass
class ModelState:
    """Holds state for one model: current weights, optimizer, best state."""
    model: nn.Module # Current model weights/arch (CPU)
    optimizer: Optional[optim.Optimizer] = None # Client creates and assigns
    selection_criterion_key: str = directories.SELECTION_CRITERION_KEY
    selection_criterion_direction_overrides: Optional[Dict[str, str]] = field(default_factory=dict)
    criterion_is_higher_better: bool = field(init=False)
    best_value_for_selection: float = field(init=False)
    best_model_state_dict: Optional[Dict] = field(init=False, default=None) # CPU state dict

    def __post_init__(self):
        """Initialize best state based on the initial model."""
        self.model.cpu()
        # Determine if higher is better for the selection criterion
        self.criterion_is_higher_better = infer_higher_is_better(
            self.selection_criterion_key, 
            self.selection_criterion_direction_overrides
        )
        
        # Initialize best value based on whether higher is better
        self.best_value_for_selection = float('-inf') if self.criterion_is_higher_better else float('inf')
        
        # Initialize best model state dict
        self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

    def update_best_state(self, current_metrics: Dict[str, float]) -> bool:
        """
        Updates best state if current metric value is better than previous best.
        
        Args:
            current_metrics: Dictionary of current metric values
            
        Returns:
            True if state was updated, False otherwise
        """
        if self.selection_criterion_key not in current_metrics:
            print(f"Warning: selection_criterion_key '{self.selection_criterion_key}' not found in metrics. "
                  f"Available keys: {list(current_metrics.keys())}. Skipping update.")
            return False
            
        current_value = current_metrics[self.selection_criterion_key]
        
        # Check if current value is better than previous best
        is_better = (self.criterion_is_higher_better and current_value > self.best_value_for_selection) or \
                    (not self.criterion_is_higher_better and current_value < self.best_value_for_selection)
        
        if is_better:
            self.best_value_for_selection = current_value
            # Ensure model is on CPU before getting state_dict
            self.model.cpu()
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            return True
        
        return False

    def get_best_model_state_dict(self) -> Optional[Dict]:
        """Returns the best recorded state dict (CPU)."""
        return self.best_model_state_dict

    def get_current_model_state_dict(self) -> Optional[Dict]:
        """Returns the current model state dict (CPU)."""
        self.model.cpu()
        # Simply return the CPU model's state dict which has unprefixed keys
        return self.model.state_dict()

    def load_current_model_state_dict(self, state_dict: Dict):
        """Loads state_dict into the current model (CPU)."""
        self.model.cpu()
        # No _orig_mod logic needed as self.model is the direct CPU instance
        self.model.load_state_dict(state_dict, strict=False)

    def load_best_model_state_dict_into_current(self):
        """Loads the best state into the current model if available."""
        if self.best_model_state_dict:
            self.load_current_model_state_dict(self.best_model_state_dict)
            return True
        return False

    def set_learning_rate(self, lr: float):
        """Updates the learning rate of the optimizer."""
        if self.optimizer:
             for param_group in self.optimizer.param_groups: param_group['lr'] = lr

# --- Minimal Training Manager ---
class TrainingManager:
    """Helper for device placement and batch preparation."""
    def __init__(self, compute_device_str: str):
        self.compute_device = torch.device(compute_device_str)
        self.cpu_device = torch.device('cpu')

    def prepare_batch(self, batch: Any, criterion: Union[nn.Module, Callable]) -> Optional[Tuple[Any, Any, Any]]:
        """Moves batch to compute device and handles labels."""
        if not isinstance(batch, (list, tuple)) or len(batch) < 2: return None
        batch_x, batch_y_orig = batch[0], batch[1]
        
        # Move input to device
        batch_x_dev = move_to_device(batch_x, self.compute_device)
        
        # Always move label tensor to the same device as inputs with non_blocking=True for efficiency
        batch_y_dev = batch_y_orig.to(self.compute_device, non_blocking=True) if isinstance(batch_y_orig, torch.Tensor) else batch_y_orig
        
        # Keep original labels on CPU for metrics calculation if needed
        batch_y_orig_cpu = batch_y_orig.cpu() if isinstance(batch_y_orig, torch.Tensor) else batch_y_orig

        # Process labels on device based on criterion type
        if isinstance(criterion, nn.CrossEntropyLoss) or isinstance(criterion, WeightedCELoss) or isinstance(criterion, ISICLoss):
            if batch_y_dev.ndim == 2 and batch_y_dev.shape[1] == 1: batch_y_dev = batch_y_dev.squeeze(1)
            batch_y_dev = batch_y_dev.long()
        elif callable(criterion) and criterion.__name__ == 'get_dice_loss':
            batch_y_dev = batch_y_dev.float()

        return batch_x_dev, batch_y_dev, batch_y_orig_cpu

def get_model_instance(dataset_name: str, **model_params) -> nn.Module:
    """Create a fresh model instance based on dataset_name.
    
    Args:
        dataset_name: Name of the dataset to determine model architecture
        **model_params: Optional parameters to pass to model constructor
        
    Returns:
        A new model instance on CPU
    """
    import models as ms  # Import here to avoid circular imports
    
    model_name_actual = 'Synthetic' if 'Synthetic_' in dataset_name else dataset_name
    model_class = getattr(ms, model_name_actual, None)
    if model_class is None:
        raise ValueError(f"Model class '{model_name_actual}' not found.")
    
    # Create model with optional parameters or default construction
    if model_params:
        return model_class(**model_params).cpu()
    else:
        return model_class().cpu()

# =============================================================================
# == Mixin for Diversity Calculation ==
# =============================================================================
class DiversityMixin:
    """Mixin class to calculate weight update divergence using Zhao et al. metric."""
    def __init__(self, *args, **kwargs):
        # Initialize history metrics for tracking
        getattr(self, 'history', {}).setdefault('weight_div', [])
        getattr(self, 'history', {}).setdefault('weight_orient', [])
        
        # Store global weights before training
        self.global_weights_before_training = None
        
        # Store original train_round method to patch it
        if hasattr(self, 'train_round'):
            self._original_train_round = self.train_round
            self.train_round = self._patched_train_round
    
    def _extract_weights(self, model) -> torch.Tensor:
        """Extract weights from a model as a flat tensor."""
        weights = [p.data.cpu().detach().clone().view(-1) for p in model.parameters() if p.requires_grad]
        if weights:
            return torch.cat(weights).float()
        return None
    
    def _patched_train_round(self) -> None:
        """Patched version of train_round that captures weights before training."""
        # Capture global weights before training
        if hasattr(self, 'serverstate') and hasattr(self.serverstate, 'model'):
            self.global_weights_before_training = self._extract_weights(self.serverstate.model)
        
        # Call original train_round
        return self._original_train_round()
    
    def _calculate_update_divergence(self, client_states):
        """Calculate weight update divergence between clients."""
        if self.global_weights_before_training is None or len(client_states) < 2:
            return np.nan, np.nan
        
        # Extract client updates
        client_updates = []
        for state_info in client_states:
            if 'state_dict' in state_info:
                from helper import get_model_instance
                # Create a temporary model with this state
                temp_model = get_model_instance(self.config.dataset_name)
                temp_model.load_state_dict(state_info['state_dict'])
                
                # Calculate update from global model
                client_weights = self._extract_weights(temp_model)
                client_update = client_weights - self.global_weights_before_training
                client_updates.append(client_update)
                
                # Clean up
                del temp_model
        
        # Calculate metrics between first two clients
        if len(client_updates) >= 2:
            # L2 norm of difference divided by norm of second client's update (Zhao metric)
            update_diff = client_updates[0] - client_updates[1]
            norm_diff = torch.norm(update_diff, p=2)
            norm_second = torch.norm(client_updates[1], p=2)
            
            # Calculate L2 divergence
            if norm_second > 1e-10:
                l2_div = (norm_diff / norm_second).item()
            else:
                l2_div = np.nan
                
            # Calculate cosine similarity
            norm_first = torch.norm(client_updates[0], p=2)
            if norm_first > 1e-10 and norm_second > 1e-10:
                cos_sim = torch.dot(client_updates[0], client_updates[1]) / (norm_first * norm_second)
                cos_sim = cos_sim.item()
                # Clamp to valid range
                cos_sim = max(min(cos_sim, 1.0), -1.0)
            else:
                cos_sim = np.nan
                
            return l2_div, cos_sim
        
        return np.nan, np.nan
    
    def after_step_hook(self, step_results: List[Tuple[str, Any]]):
        """Calculate divergence after client training."""
        # Check if this is a training step with state dicts
        is_training_step = any(isinstance(res, dict) and 'state_dict' in res for _, res in step_results)
        if not is_training_step:
            return
            
        # Extract client states from step_results
        client_states = []
        for client_id, output_dict in step_results:
            if isinstance(output_dict, dict) and 'state_dict' in output_dict:
                client_states.append({
                    'client_id': client_id,
                    'state_dict': output_dict['state_dict'],
                    'weight': getattr(self.clients[client_id].data, 'weight', 0.0) 
                })
        
        # Calculate divergence metrics
        if len(client_states) >= 2:
            l2_div, cos_sim = self._calculate_update_divergence(client_states)
            
            # Store metrics in history
            self.history['weight_div'].append(l2_div)
            self.history['weight_orient'].append(cos_sim)
            

class MetricsCalculator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        if 'Synthetic_' in dataset_name:
             self.dataset_key = 'Synthetic'
        else:
             self.dataset_key = dataset_name
        self.continuous_outcome = ['Weather']
        self.long_required = ['CIFAR', 'EMNIST', 'ISIC', 'Heart', 'Synthetic', 'Credit']
        self.tensor_metrics = ['IXITiny']
        
    def get_metric_function(self):
        """Returns appropriate metric function based on dataset."""
        metric_mapping = {
            'Synthetic': partial(metrics.f1_score, average='macro'),
            'Credit': partial(metrics.f1_score, average='macro'),
            'Weather': metrics.r2_score,
            'EMNIST': metrics.accuracy_score,
            'CIFAR': metrics.accuracy_score,
            'IXITiny': get_dice_score,
            'ISIC': metrics.balanced_accuracy_score,
            'Heart':metrics.balanced_accuracy_score
        }
        return metric_mapping[self.dataset_key]

    def process_predictions(self, labels, predictions):
        """Process model predictions based on dataset requirements."""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        if self.dataset_key in self.continuous_outcome:
            predictions = np.clip(predictions, -2, 2)
        elif self.dataset_key in self.long_required:
            predictions = predictions.argmax(axis=1)
            
        return labels, predictions

    def calculate_metrics(self, true_labels, predictions):
        """Calculate appropriate metric score."""
        true_labels, predictions_class = self.process_predictions(true_labels, predictions)
        metric_func = self.get_metric_function()
        if self.dataset_key in self.tensor_metrics:
            return metric_func(
                torch.tensor(true_labels, dtype=torch.float32),
                torch.tensor(predictions_class, dtype=torch.float32)
            )
        else:
            return metric_func(
                np.array(true_labels).reshape(-1),
                np.array(predictions_class).reshape(-1)
            )        