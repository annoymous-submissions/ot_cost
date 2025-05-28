# clients.py
"""
Client implementations for Federated Learning. Streamlined version.
Uses ModelState for state, TrainingManager for utils, direct overrides for algorithms.
"""
import copy
import torch
import gc
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List, Iterator, Any, Callable, Union
from configs import DEVICE
from helper import (gpu_scope, systematic_memory_cleanup, TrainerConfig, SiteData, ModelState, TrainingManager, 
                    MetricsCalculator, calculate_class_weights, get_parameters_for_dataset, get_model_instance)

from losses import ISICLoss, WeightedCELoss, get_dice_loss
from torch.amp import autocast, GradScaler
from functools import partial
# =============================================================================
# == Base Client Class ==
# =============================================================================
class Client:  # only the two changed methods are shown
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: TrainerConfig,
        data: SiteData,
        initial_global_state: ModelState,
        metric_fn: Callable,
        personal_model: bool = False,
    ):
        from helper import get_model_instance

        self.config = config
        self.data = data
        self.metric_fn = metric_fn
        self.site_id = data.site_id
        self.requires_personal_model = personal_model
        self.training_manager = TrainingManager(config.device)

        # loss
        self._initialize_criterion()

        # ---- GLOBAL MODEL (kept on CPU until train() is called) -------- #
        global_model = get_model_instance(self.config.dataset_name)
        global_model.load_state_dict(initial_global_state.model.state_dict())
        
        # Create global state with selection criteria from config
        self.global_state = ModelState(
            model=global_model.cpu(),
            selection_criterion_key=config.selection_criterion_key,
            selection_criterion_direction_overrides=config.selection_criterion_direction_overrides
        )
        self.global_state.optimizer = self._create_optimizer(self.global_state.model)
        # AMP scaler – enabled only when CUDA is present
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # ---- PERSONAL MODEL (optional) -------------------------------- #
        self.personal_state: Optional[ModelState] = None
        if self.requires_personal_model:
            personal_model = get_model_instance(self.config.dataset_name)
            personal_model.load_state_dict(initial_global_state.model.state_dict())
            
            # Create personal state with selection criteria from config
            self.personal_state = ModelState(
                model=personal_model.cpu(),
                selection_criterion_key=config.selection_criterion_key,
                selection_criterion_direction_overrides=config.selection_criterion_direction_overrides
            )
            self.personal_state.optimizer = self._create_optimizer(self.personal_state.model)

    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Creates optimizer for a given model instance."""
        wd = self.config.algorithm_params.get('weight_decay', 1e-4)
        lr = self.config.learning_rate

        # Get the appropriate module for parameters
        target_module = model._orig_mod if hasattr(model, '_orig_mod') else model
        
        # Create standard AdamW optimizer without fused option
        optimizer = optim.AdamW(
            target_module.parameters(),
            lr=lr,
            weight_decay=wd,
            eps=1e-8
        )
        
        return optimizer
        
    
    def _initialize_criterion(self):
        """Initialize criterion based on configuration."""

        dataset_params = get_parameters_for_dataset(self.config.dataset_name)
        
        num_classes = dataset_params.get('fixed_classes')
        criterion_type = dataset_params.get('criterion_type', 'CrossEntropyLoss')
        use_weighted_loss_flag = dataset_params.get('use_weighted_loss', False)
        
        if criterion_type == 'CrossEntropyLoss':
            if use_weighted_loss_flag and hasattr(self.data.train_loader, 'dataset'):
                if num_classes is None:
                    raise ValueError(f"'fixed_classes' must be defined in config for {self.config.dataset_name} when using weighted loss")
                # Calculate class weights
                class_weights = calculate_class_weights(self.data.train_loader.dataset, num_classes)
                # Pass weights directly to constructor
                self.criterion = WeightedCELoss(weights=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        elif criterion_type == 'ISICLoss':
            if num_classes is None:
                raise ValueError(f"'fixed_classes' must be defined in config for {self.config.dataset_name} when using ISICLoss")
            
            if hasattr(self.data.train_loader, 'dataset'):
                client_alpha = calculate_class_weights(self.data.train_loader.dataset, num_classes)
                self.criterion = ISICLoss(alpha=client_alpha)
            else:
                # Fallback if no dataset available (shouldn't happen)
                default_alpha = torch.tensor([1.0] * num_classes)
                self.criterion = ISICLoss(alpha=default_alpha)
    
        elif criterion_type == 'DiceLoss':
            self.criterion = get_dice_loss
        
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _get_state(self, personal: bool) -> ModelState:
        """Helper to get the correct state object."""
        state = self.personal_state if personal and self.requires_personal_model else self.global_state
        if state is None: raise RuntimeError(f"State {'personal' if personal else 'global'} not available.")
        return state
    
    
    def set_model_state(self, state_dict: Dict, test: bool = False):
        """Loads state dict into the client's global model state (CPU)."""
        self.global_state.load_current_model_state_dict(state_dict)


    def _train_batch(self, model: nn.Module, optimizer: optim.Optimizer, criterion: Union[nn.Module, Callable], batch_x: Any, batch_y: Any) -> float:
        """
        Performs a single training step. Assumes model, batch_x, batch_y on compute_device.
        Base implementation for FedAvg. Subclasses override this for algorithm logic.
        """
        # Determine if we should use AMP based on device
        use_amp = DEVICE == 'cuda' and batch_x.device.type == 'cuda'
        
        # More efficient memory usage with set_to_none=True
        optimizer.zero_grad(set_to_none=True)
        # Use autocast for forward pass when on GPU
        if use_amp:
            # CUDA + mixed precision
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Pure CPU (or explicit no-AMP) path
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        return loss.item()
    
    def _process_epoch(self, 
                    loader: DataLoader, 
                    model: nn.Module, 
                    is_training: bool, 
                    optimizer: Optional[optim.Optimizer] = None,
                    use_amp: bool = False) -> Tuple[float, List, List]:
        """
        Unified function to process one epoch for either training or evaluation.
        Efficiently handles GPU tensors and minimizes CPU transfers.
        """
        if is_training:
            model.train()
            if optimizer is None:
                raise ValueError("Optimizer required for training")
        else:
            model.eval()

        epoch_loss = 0.0
        num_batches = 0
        
        # For evaluation, keep tensors on device until the end
        device_outputs = []
        device_labels = []
        
        # For training, we don't collect outputs/labels, just track loss
        epoch_predictions_cpu, epoch_labels_cpu = [], []
        
        # Use inference_mode for evaluation - more efficient than no_grad
        context = torch.enable_grad() if is_training else torch.inference_mode()

        try:
            with context:
                for batch in loader:
                    prepared_batch = self.training_manager.prepare_batch(batch, self.criterion)
                    if prepared_batch is None: 
                        continue
                    batch_x_dev, batch_y_dev, batch_y_orig_cpu = prepared_batch
                    if is_training:
                        batch_loss = self._train_batch(model, optimizer, self.criterion, batch_x_dev, batch_y_dev)
                        epoch_loss += batch_loss
                    else:  # Evaluation
                        # Use autocast for eval when on GPU for consistency with training
                        with autocast('cuda', enabled=use_amp):
                            outputs = model(batch_x_dev)
                            loss = self.criterion(outputs, batch_y_dev)
                        epoch_loss += loss.item()
                        
                        # Store detached tensors on device
                        device_outputs.append(outputs.detach())
                        # Use device tensor directly
                        device_labels.append(batch_y_dev.detach() if torch.is_tensor(batch_y_dev) else batch_y_orig_cpu)

                    num_batches += 1
                    
                    # Free batch data explicitly
                    del prepared_batch, batch_x_dev, batch_y_dev, batch_y_orig_cpu
                    del batch

                # Only transfer to CPU once after all batches, when in evaluation mode
                if not is_training and device_outputs:
                    # Set minimal print options to avoid truncation issues in error messages
                    torch.set_printoptions(profile="minimal")
                    
                    # Concatenate on device first, then transfer to CPU once
                    if len(device_outputs) > 0:
                        if use_amp:
                            # GPU path - do single batch transfer
                            all_outputs = torch.cat(device_outputs, dim=0)
                            epoch_predictions_cpu = [all_outputs.cpu()]  # Single tensor in a list
                            
                            # Handle labels (might be on different devices)
                            if all(isinstance(tensor, torch.Tensor) and hasattr(tensor, 'device') and 
                                tensor.device.type == 'cuda' for tensor in device_labels):
                                all_labels = torch.cat([label for label in device_labels if torch.is_tensor(label)], dim=0)
                                epoch_labels_cpu = [all_labels.cpu()]  # Single tensor in a list
                            else:
                                # Fall back to per-tensor CPU transfer if needed
                                epoch_labels_cpu = [label.cpu() if torch.is_tensor(label) else label for label in device_labels]
                        else:
                            # CPU path - still batch transfers for memory efficiency
                            epoch_predictions_cpu = [output.cpu() for output in device_outputs]
                            epoch_labels_cpu = [label.cpu() if torch.is_tensor(label) else label for label in device_labels]
        finally:
            # Always clean up, even if exceptions occur
            if 'device_outputs' in locals():
                for output in device_outputs:
                    del output
                del device_outputs
            if 'device_labels' in locals():
                for label in device_labels:
                    if isinstance(label, torch.Tensor):
                        del label
                del device_labels
            if 'all_outputs' in locals():
                del all_outputs
            if 'all_labels' in locals():
                del all_labels
                
            # Force garbage collection
            systematic_memory_cleanup()

        avg_loss = epoch_loss / num_batches if num_batches > 0 else (0.0 if is_training else float('inf'))
        return avg_loss, epoch_predictions_cpu, epoch_labels_cpu
        # --- Public API ---

    def train_and_validate(self, personal: bool) -> Dict:
        """
        Runs local training epochs followed by a validation pass.
        Handles model compilation for GPU models and properly synchronizes
        weights between CPU and GPU versions.
        """
        # Wrap entire method in gpu_scope for better CUDA stream management
        with gpu_scope():
            # 1. Get the appropriate state based on personal flag
            state = self._get_state(personal)
            
            # 2. Determine device to use from configs.DEVICE
            use_gpu = DEVICE == 'cuda'
            
            # 3. Set up the model for training
            if use_gpu:
                # Move model to GPU
                model_on_gpu = state.model.to(self.training_manager.compute_device)
                
                operational_model = model_on_gpu
            else:
                # Use CPU model directly
                operational_model = state.model
            
            # Get optimizer from state
            optimizer = state.optimizer
            val_score = 0.0

            # 4. Training loop
            for _ in range(self.config.epochs):
                train_loss, _, _ = self._process_epoch(
                    loader=self.data.train_loader,
                    model=operational_model,
                    is_training=True,
                    optimizer=optimizer,
                    use_amp=use_gpu
                )

            # 5. Validation
            val_loss, preds_cpu, labels_cpu = self._process_epoch(
                loader=self.data.val_loader,
                model=operational_model,
                is_training=False,
                use_amp=use_gpu
            )

            if preds_cpu and labels_cpu:
                all_preds = torch.cat(preds_cpu, 0)
                all_labels = torch.cat(labels_cpu, 0)
                val_score = self.metric_fn(all_labels, all_preds)

            # 6. Sync weights back to CPU copy if using GPU
            if use_gpu:
                # Extract state dict without _orig_mod prefixes
                gpu_state_dict = (operational_model._orig_mod if hasattr(operational_model, "_orig_mod") 
                            else operational_model).state_dict()
                # Load into CPU model
                state.model.load_state_dict(gpu_state_dict)
                
                # Clean up GPU model
                operational_model = operational_model.cpu()
                torch.cuda.empty_cache()  # No need for availability check

            # 7. Track best model per client
            current_metrics = {
                'train_losses': train_loss,
                'val_losses': val_loss,
                'val_scores': val_score
            }
            state.update_best_state(current_metrics)

            # Return metrics and state dict
            return {
                'train_loss': train_loss,
                "val_loss": val_loss,
                "val_score": val_score,
                "state_dict": state.get_current_model_state_dict(),
            }

    def test(self, personal: bool) -> Tuple[float, float]:
        """
        Tests the model on test data.
        Handles both CPU and GPU models with proper state dict management.
        """
        # Wrap entire method in gpu_scope for better CUDA stream management
        with gpu_scope():
            # 1. Get correct state for testing
            state = self._get_state(personal)
            
            # 2. Get best state dictionary (on CPU)
            best_cpu_state_dict = state.get_best_model_state_dict()
            if best_cpu_state_dict is None:
                print(f"Warning: No best state available for testing (client: {self.site_id}, personal: {personal}).")
                return float('inf'), 0.0
            
            # 3. Determine device to use from configs.DEVICE
            use_gpu = DEVICE == 'cuda'
            
            test_loss, test_score = float('inf'), 0.0
            
            try:
                # 4. Create temporary model and load state
                temp_eval_model = get_model_instance(self.config.dataset_name)
                
                # Load state dict (handles both compiled and non-compiled cases)
                temp_eval_model.load_state_dict(best_cpu_state_dict)
                
                # 5. Move to appropriate device for evaluation
                if use_gpu:
                    model_for_eval = temp_eval_model.to(self.training_manager.compute_device)
                else:
                    model_for_eval = temp_eval_model
                
                # 6. Perform evaluation
                test_loss, test_predictions_cpu, test_labels_cpu = self._process_epoch(
                    loader=self.data.test_loader,
                    model=model_for_eval,
                    is_training=False,
                    use_amp=use_gpu
                )
                
                # Calculate test score
                if test_predictions_cpu and test_labels_cpu:
                    all_preds = torch.cat(test_predictions_cpu, dim=0)
                    all_labels = torch.cat(test_labels_cpu, dim=0)
                    test_score = self.metric_fn(all_labels, all_preds)
            
            finally:
                # 7. Clean up GPU resources
                if 'model_for_eval' in locals() and use_gpu:
                    model_for_eval = model_for_eval.cpu()
                    del model_for_eval
                
                if 'temp_eval_model' in locals():
                    del temp_eval_model
                    
                if use_gpu:
                    torch.cuda.empty_cache()  # No need for availability check
            
            # 8. Return test metrics
            return test_loss, test_score

# =============================================================================
# == Algorithm-Specific Client Implementations ==
# =============================================================================

class FedProxClient(Client):
    """FedProx Client: Overrides _train_batch to add proximal term."""
    def __init__(
        self,
        config: TrainerConfig,
        data: SiteData,
        initial_global_state: ModelState,
        metric_fn: Callable,
        personal_model: bool = False,
    ):
        # Set requires_personal_model to False for FedProx
        config.requires_personal_model = False
        # Call parent constructor with explicit parameters
        super().__init__(
            config=config,
            data=data,
            initial_global_state=initial_global_state,
            metric_fn=metric_fn,
            personal_model=False
        )
        self.reg_param = self.config.algorithm_params.get('reg_param', 1e-1)  # Mu
        self._initial_global_state_dict_cpu = self.global_state.get_current_model_state_dict()
        # Create GPU parameters dictionary for faster training
        self._initial_global_params_gpu = None

    def set_model_state(self, state_dict: Dict, test: bool = False):
        """Loads state and stores it as the reference for the prox term."""
        super().set_model_state(state_dict, test)
        self._initial_global_state_dict_cpu = copy.deepcopy(state_dict)
        # Clear GPU params cache on state update
        self._initial_global_params_gpu = None
        
    def _ensure_global_params_on_gpu(self):
        """Transfers global model parameters to GPU using a dictionary keyed by name for robust lookup."""
        if self._initial_global_params_gpu is None:
            # Create a dictionary mapping names to GPU tensors
            self._initial_global_params_gpu = {}
            for name, param_tensor in self._initial_global_state_dict_cpu.items():
                if isinstance(param_tensor, torch.Tensor):
                    self._initial_global_params_gpu[name] = param_tensor.to(self.training_manager.compute_device)

    def _train_batch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Union[nn.Module, Callable],
        batch_x: Any,
        batch_y: Any,
    ) -> float:
        """Single FedProx step with optional mixed-precision support."""

        from configs import DEVICE
        use_amp = DEVICE == 'cuda' and batch_x.device.type == 'cuda'
        device      = batch_x.device

        # ----- make sure the reference params live on the right device ----------
        if use_amp:                       # CUDA path
            self._ensure_global_params_on_gpu()

        optimizer.zero_grad(set_to_none=True)

        # ---------- forward & primary task loss ---------------------------------
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs   = model(batch_x)
                task_loss = criterion(outputs, batch_y)
        else:                                              # CPU / no-AMP
            outputs   = model(batch_x)
            task_loss = criterion(outputs, batch_y)

        # ---------- proximal term -----------------------------------------------
        param_module  = model._orig_mod if hasattr(model, "_orig_mod") else model
        proximal_term = torch.tensor(0.0, device=device)

        if use_amp and self._initial_global_params_gpu:
            for name, p_cur in param_module.named_parameters():
                if p_cur.requires_grad and name in self._initial_global_params_gpu:
                    p_ref = self._initial_global_params_gpu[name]
                    proximal_term += torch.sum((p_cur - p_ref) ** 2)
        else:
            for name, p_cur in param_module.named_parameters():
                if p_cur.requires_grad and name in self._initial_global_state_dict_cpu:
                    p_ref = self._initial_global_state_dict_cpu[name].to(device)
                    proximal_term += torch.sum((p_cur - p_ref) ** 2)

        total_loss = task_loss + 0.5 * self.reg_param * proximal_term

        # ---------- backward & optimiser step -----------------------------------
        if use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        return float(total_loss.detach())      


class PFedMeClient(Client):
    """pFedMe Client: Overrides train_and_validate for k-step inner loop."""
    def __init__(
        self,
        config: TrainerConfig,
        data: SiteData,
        initial_global_state: ModelState,
        metric_fn: Callable,
        personal_model: bool = True,
    ):
        # Set requires_personal_model to True for pFedMe
        config.requires_personal_model = True
        # Call parent constructor with explicit parameters
        super().__init__(
            config=config,
            data=data,
            initial_global_state=initial_global_state,
            metric_fn=metric_fn,
            personal_model=True
        )
        self.reg_param = self.config.algorithm_params.get('reg_param', 1e-1) 
        self.k_steps = self.config.algorithm_params.get('k_steps', 5)
        self._global_params_gpu = None  # Cache for global parameters on GPU
        self.outer_loop_factor = self.config.algorithm_params.get("outer_lr_factor", 30.0)

    def _prepare_global_params_gpu(self, device):
        """Creates or returns cached global parameters on the target device."""
        if self._global_params_gpu is None:
            # Get the appropriate module
            global_model = self.global_state.model
            # Transfer parameters to device
            self._global_params_gpu = [
                p.detach().clone().to(device)
                for p in global_model.parameters() if p.requires_grad
            ]
        return self._global_params_gpu

    def train_and_validate(self, personal: bool) -> Dict:
        """
        Custom pFedMe training implementation with k-step inner optimization.
        Reports average loss from last k-step of each batch.
        """
        # Wrap entire method in gpu_scope for better CUDA stream management
        with gpu_scope():
            if not personal:
                # Use standard implementation for global model
                return super().train_and_validate(personal=False)
            
            # Clear GPU params cache at the start of personal training to ensure fresh copy
            self._global_params_gpu = None
            
            # Get states
            personal_state = self._get_state(True)
            
            # Determine device based on configs
            from configs import DEVICE  # Import here to avoid circular imports
            use_gpu = DEVICE == 'cuda'
            
            # Set up models based on available compute resources
            if use_gpu:
                # Move model to GPU
                model = personal_state.model.to(self.training_manager.compute_device)
                device = self.training_manager.compute_device
                
                # Prepare global params on GPU
                global_params = self._prepare_global_params_gpu(device)
            else:
                # Use CPU models directly
                model = personal_state.model
                device = torch.device('cpu')
                global_params = None
            
            optimizer = personal_state.optimizer
            
            # Exit early if no data
            if not self.data.train_loader:
                return {
                    "val_loss": float('inf'),
                    "val_score": 0.0,
                    "state_dict": personal_state.get_current_model_state_dict()
                }
                
            model.train()
            criterion = self.criterion
            # Initialize accumulator for the last k-step loss
            epoch_last_k_step_loss = 0.0
            num_batches_processed = 0
            
            # Training loop
            for _ in range(self.config.epochs):
                for batch in self.data.train_loader:
                    prepared_batch = self.training_manager.prepare_batch(batch, criterion)
                    if prepared_batch is None: continue
                    batch_x_dev, batch_y_dev, _ = prepared_batch
                    
                    # Get target module for parameters
                    target_module = model._orig_mod if hasattr(model, "_orig_mod") else model
                    
                    # Get relevant parameters for proximal update
                    if global_params is None:
                        # CPU path - create on-the-fly
                        global_module = self.global_state.model
                        global_params = [p.detach().clone().to(device) for p in global_module.parameters() if p.requires_grad]
                    
                    # Variable to store the loss from the last k-step
                    last_k_step_loss = 0.0
                    
                    # K-step optimization
                    for k_idx in range(self.k_steps):
                        # Use autocast for forward pass on GPU
                        with autocast('cuda', enabled=use_gpu):
                            outputs = model(batch_x_dev)
                            loss = criterion(outputs, batch_y_dev)
                        
                        # Store the loss from the last k-step
                        if k_idx == self.k_steps - 1:
                            last_k_step_loss = loss.item()
                        
                        # Training step
                        optimizer.zero_grad(set_to_none=True)
                        
                        if use_gpu:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                    
                    # Proximal update step - use pre-transferred parameters
                    with torch.no_grad():
                        for param_personal, param_global in zip(target_module.parameters(), global_params):
                            if param_personal.requires_grad:
                                update_step = self.config.learning_rate * self.outer_loop_factor * self.reg_param * (param_personal - param_global)
                                param_personal.sub_(update_step)
                    
                    # Accumulate the last k-step loss
                    epoch_last_k_step_loss += last_k_step_loss
                    num_batches_processed += 1
                
            # Calculate average training loss for reporting
            train_loss = epoch_last_k_step_loss / num_batches_processed if num_batches_processed > 0 else float('inf')
                
            # Validation
            val_loss, preds_cpu, labels_cpu = self._process_epoch(
                loader=self.data.val_loader,
                model=model,
                is_training=False,
                use_amp=use_gpu
            )
            
            # Calculate validation score
            val_score = 0.0
            if preds_cpu and labels_cpu:
                all_preds = torch.cat(preds_cpu, 0)
                all_labels = torch.cat(labels_cpu, 0)
                val_score = self.metric_fn(all_labels, all_preds)
            
            # Sync weights back to CPU
            if use_gpu:
                # Get state dict without _orig_mod prefixes
                gpu_state_dict = (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()
                # Load into CPU model
                personal_state.model.load_state_dict(gpu_state_dict)
                
                # Clean up GPU model
                model = model.cpu()
                torch.cuda.empty_cache()  # No need for availability check
            
            # Track best model
            current_metrics = {
                'train_losses': train_loss,
                'val_losses': val_loss,
                'val_scores': val_score
            }
            personal_state.update_best_state(current_metrics)
            
            # Return results
            return {
                "val_loss": val_loss, 
                "val_score": val_score,
                "state_dict": personal_state.get_current_model_state_dict()
            }

class DittoClient(Client):
    """Ditto Client: Overrides _train_batch to modify gradients."""
    def __init__(
        self,
        config: TrainerConfig,
        data: SiteData,
        initial_global_state: ModelState,
        metric_fn: Callable,
        personal_model: bool = False,
    ):
        # Set requires_personal_model to True for Ditto
        config.requires_personal_model = True
        # Call parent constructor with explicit parameters
        super().__init__(
            config=config,
            data=data,
            initial_global_state=initial_global_state,
            metric_fn=metric_fn,
            personal_model=True
        )
        self.reg_param = self.config.algorithm_params.get('reg_param', 1e-1)  # Lambda
        # Cache for global model parameters on GPU
        self._global_params_gpu = None

    def _prepare_global_params_gpu(self, device):
        """Transfers global model parameters to GPU once."""
        if self._global_params_gpu is None:
            self._global_params_gpu = [
                p.detach().clone().to(device)
                for p in self.global_state.model.parameters() if p.requires_grad
            ]
        return self._global_params_gpu
    
    def _train_batch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: Union[nn.Module, Callable],
        batch_x: Any,
        batch_y: Any,
    ) -> float:
        """Ditto step — personal model gets regularised toward the global model."""

        from configs import DEVICE
        use_amp = DEVICE == 'cuda' and batch_x.device.type == 'cuda'

        is_personal   = (model is self.personal_state.model) or (
            hasattr(model, "_orig_mod") and model._orig_mod is self.personal_state.model
        )
        device = batch_x.device

        optimizer.zero_grad(set_to_none=True)

        # ---------- forward ------------------------------------------------------
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(batch_x)
                loss    = criterion(outputs, batch_y)
        else:
            outputs = model(batch_x)
            loss    = criterion(outputs, batch_y)

        # ---------- backward -----------------------------------------------------
        if use_amp:
            # scale & back-prop primary loss first
            self.scaler.scale(loss).backward()

            if is_personal:
                # unscale so we can add regularisation on raw grads
                self.scaler.unscale_(optimizer)

                target_module = (
                    model._orig_mod if hasattr(model, "_orig_mod") else model
                )
                global_params = self._prepare_global_params_gpu(device)

                with torch.no_grad():
                    for p_idx, p_personal in enumerate(target_module.parameters()):
                        if p_personal.grad is not None and p_idx < len(global_params):
                            p_global = global_params[p_idx]
                            p_personal.grad.add_(
                                self.reg_param * (p_personal.detach() - p_global)
                            )

                optimizer.step()      # grads are already unscaled
                self.scaler.update()
            else:
                self.scaler.step(optimizer)
                self.scaler.update()
        else:  # ------------------- CPU / no-AMP path ----------------------------
            loss.backward()

            if is_personal:
                target_module = model
                global_params = [
                    p.detach().clone()
                    for p in self.global_state.model.parameters()
                    if p.requires_grad
                ]
                with torch.no_grad():
                    for p_idx, p_personal in enumerate(target_module.parameters()):
                        if p_personal.grad is not None and p_idx < len(global_params):
                            p_personal.grad.add_(
                                self.reg_param * (p_personal.detach() - global_params[p_idx])
                            )

            optimizer.step()

        return float(loss.detach())

    def train_and_validate(self, personal: bool) -> Dict:
        """
        Reset GPU parameter cache at the start of training to ensure fresh global weights.
        Then delegate to the parent implementation.
        """
        if personal:
            # Clear GPU params cache at the start of training to ensure fresh copy
            self._global_params_gpu = None
            
        return super().train_and_validate(personal)