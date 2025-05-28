# ot_utils.py
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
import torch
import logging
import ot
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

# Configure module logger
logger = logging.getLogger(__name__)

# --- Utility Functions for Cost Matrices ---
from ot_configs import DEFAULT_EPS, DEFAULT_OT_REG, DEFAULT_OT_MAX_ITER


def normalize_cost_matrix(
    cost_matrix: Union[torch.Tensor, np.ndarray], 
    max_cost_val: float, 
    normalize_flag: bool, 
    eps_num: float = DEFAULT_EPS
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalizes cost matrix by dividing by max_cost_val if normalize_flag is True.
    
    Args:
        cost_matrix: Input cost matrix
        max_cost_val: Maximum cost value (scalar)
        normalize_flag: Whether to normalize or not
        eps_num: Small epsilon for numerical stability
        
    Returns:
        Normalized or original cost matrix
    """
    if normalize_flag and np.isfinite(max_cost_val) and max_cost_val > eps_num:
        return cost_matrix / max_cost_val
    elif normalize_flag and not np.isfinite(max_cost_val):
        logger.warning("Max cost is not finite, cannot normalize cost matrix.")
    return cost_matrix  # Return unnormalized


def prepare_ot_marginals(
    weights1: np.ndarray, 
    weights2: np.ndarray, 
    N: int, 
    M: int, 
    eps_num: float = DEFAULT_EPS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares marginal distributions for OT calculation with proper normalization.
    
    Args:
        weights1: Input weights for first distribution (can be loss-derived or uniform)
        weights2: Input weights for second distribution
        N: Number of samples in first distribution
        M: Number of samples in second distribution
        eps_num: Small epsilon for numerical stability
        
    Returns:
        Tuple of (a, b) normalized marginal distributions
    """
    weights1_np = np.asarray(weights1, dtype=np.float64)
    weights2_np = np.asarray(weights2, dtype=np.float64)
    
    # Renormalize marginals to ensure they sum to 1
    sum1 = weights1_np.sum()
    if not np.isclose(sum1, 1.0) and sum1 > eps_num:
        weights1_np /= sum1
    elif sum1 <= eps_num:
        logger.warning("First distribution weights sum to zero/negative, using uniform.")
        weights1_np = np.ones(N, dtype=np.float64) / max(1, N)
    
    sum2 = weights2_np.sum()
    if not np.isclose(sum2, 1.0) and sum2 > eps_num:
        weights2_np /= sum2
    elif sum2 <= eps_num:
        logger.warning("Second distribution weights sum to zero/negative, using uniform.")
        weights2_np = np.ones(M, dtype=np.float64) / max(1, M)
    
    return weights1_np, weights2_np


# --- OT Computation Wrapper ---
def compute_ot_cost(
    cost_matrix: Union[torch.Tensor, np.ndarray],
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    reg: float = DEFAULT_OT_REG,
    sinkhorn_thresh: float = 1e-3,
    sinkhorn_max_iter: int = DEFAULT_OT_MAX_ITER,
    eps_num: float = DEFAULT_EPS
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Computes OT cost using Sinkhorn algorithm from POT library.
    Handles input validation, NaN/Inf values, and marginal normalization.

    Args:
        cost_matrix: The (N, M) cost matrix (Tensor or Numpy array).
        a: Source marginal weights (N,). Defaults to uniform.
        b: Target marginal weights (M,). Defaults to uniform.
        reg: Entropy regularization term for Sinkhorn.
        sinkhorn_thresh: Stop threshold for Sinkhorn iterations.
        sinkhorn_max_iter: Max iterations for Sinkhorn.
        eps_num: Small epsilon for numerical stability checks.

    Returns:
        Tuple containing:
            - float: Computed OT cost (np.nan if failed).
            - np.ndarray or None: Transport plan (Gs).
    """
    if cost_matrix is None:
        logger.warning("OT computation skipped: Cost matrix is None.")
        return np.nan, None

    # Ensure cost matrix is a numpy float64 array
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix_np = cost_matrix.detach().cpu().numpy().astype(np.float64)
    elif isinstance(cost_matrix, np.ndarray):
        cost_matrix_np = cost_matrix.astype(np.float64)
    else:
        try:
            cost_matrix_np = np.array(cost_matrix, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Could not convert cost matrix to numpy array: {e}")
            return np.nan, None

    if cost_matrix_np.size == 0:
        N, M = cost_matrix_np.shape
        return 0.0, np.zeros((N, M))

    N, M = cost_matrix_np.shape
    if N == 0 or M == 0:
        return 0.0, np.zeros((N, M))

    # Handle non-finite values in cost matrix
    if not np.all(np.isfinite(cost_matrix_np)):
        max_finite_cost = np.nanmax(cost_matrix_np[np.isfinite(cost_matrix_np)])
        replacement_val = 1e6
        if np.isfinite(max_finite_cost):
            replacement_val = max(1.0, abs(max_finite_cost)) * 10.0
        cost_matrix_np[~np.isfinite(cost_matrix_np)] = replacement_val
        logger.warning(f"NaN/Inf detected in cost matrix. Replaced non-finite values with {replacement_val:.2e}.")

    # Prepare Marginals a and b
    if a is None:
        a = np.ones((N,), dtype=np.float64) / N
    else:
        a = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            a = np.ones_like(a) / max(1, len(a))
            logger.warning("NaN/Inf in marginal 'a'. Using uniform.")
        sum_a = a.sum()
        if sum_a <= eps_num:
            a = np.ones_like(a) / max(1, len(a))
            logger.warning("Marginal 'a' sums to zero or less. Using uniform.")
        elif not np.isclose(sum_a, 1.0):
            a /= sum_a

    if b is None:
        b = np.ones((M,), dtype=np.float64) / M
    else:
        b = np.asarray(b, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            b = np.ones_like(b) / max(1, len(b))
            logger.warning("NaN/Inf in marginal 'b'. Using uniform.")
        sum_b = b.sum()
        if sum_b <= eps_num:
            b = np.ones_like(b) / max(1, len(b))
            logger.warning("Marginal 'b' sums to zero or less. Using uniform.")
        elif not np.isclose(sum_b, 1.0):
            b /= sum_b

    # Compute OT using POT
    Gs = None
    ot_cost = np.nan
    try:
        cost_matrix_np_cont = np.ascontiguousarray(cost_matrix_np)

        # Try stabilized Sinkhorn first
        Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=reg, stopThr=sinkhorn_thresh,
                         numItermax=sinkhorn_max_iter, method='sinkhorn_stabilized',
                         warn=False, verbose=False)
        print(f"Stabilized Sinkhorn result: {Gs}")
        if Gs is None or np.any(np.isnan(Gs)):
            # Fallback to standard Sinkhorn
            if Gs is None: 
                logger.warning("Stabilized Sinkhorn failed. Trying standard Sinkhorn.")
            else: 
                logger.warning("Stabilized Sinkhorn resulted in NaN plan. Trying standard Sinkhorn.")
            Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=reg, stopThr=sinkhorn_thresh,
                             numItermax=sinkhorn_max_iter, method='sinkhorn',
                             warn=False, verbose=False)

        if Gs is None or np.any(np.isnan(Gs)):
            logger.warning("Sinkhorn computation failed (both stabilized and standard) or resulted in NaN plan.")
            Gs = None # Ensure Gs is None if calculation failed
        else:
            # Calculate OT cost
            ot_cost = np.sum(Gs * cost_matrix_np)
            if not np.isfinite(ot_cost):
                logger.warning(f"Calculated OT cost is not finite ({ot_cost}). Returning NaN.")
                ot_cost = np.nan

    except Exception as e:
        logger.warning(f"Error during OT computation (ot.sinkhorn): {e}")
        ot_cost = np.nan
        Gs = None

    return float(ot_cost), Gs
def compute_ot_cost(
    cost_matrix: Union[torch.Tensor, np.ndarray],
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    reg: float = DEFAULT_OT_REG,
    sinkhorn_thresh: float = 1e-3,
    sinkhorn_max_iter: int = DEFAULT_OT_MAX_ITER,
    eps_num: float = DEFAULT_EPS
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Computes OT cost using Sinkhorn algorithm from POT library.
    Handles input validation, NaN/Inf values, and marginal normalization.

    Args:
        cost_matrix: The (N, M) cost matrix (Tensor or Numpy array).
        a: Source marginal weights (N,). Defaults to uniform.
        b: Target marginal weights (M,). Defaults to uniform.
        reg: Entropy regularization term for Sinkhorn.
        sinkhorn_thresh: Stop threshold for Sinkhorn iterations.
        sinkhorn_max_iter: Max iterations for Sinkhorn.
        eps_num: Small epsilon for numerical stability checks.

    Returns:
        Tuple containing:
            - float: Computed OT cost (np.nan if failed).
            - np.ndarray or None: Transport plan (Gs).
    """
    if cost_matrix is None:
        logger.warning("OT computation skipped: Cost matrix is None.")
        return np.nan, None

    # Ensure cost matrix is a numpy float64 array
    if isinstance(cost_matrix, torch.Tensor):
        cost_matrix_np = cost_matrix.detach().cpu().numpy().astype(np.float64)
    elif isinstance(cost_matrix, np.ndarray):
        cost_matrix_np = cost_matrix.astype(np.float64)
    else:
        try:
            cost_matrix_np = np.array(cost_matrix, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Could not convert cost matrix to numpy array: {e}")
            return np.nan, None

    if cost_matrix_np.size == 0:
        N, M = cost_matrix_np.shape
        return 0.0, np.zeros((N, M))

    N, M = cost_matrix_np.shape
    if N == 0 or M == 0:
        return 0.0, np.zeros((N, M))

    # Handle non-finite values in cost matrix
    if not np.all(np.isfinite(cost_matrix_np)):
        max_finite_cost = np.nanmax(cost_matrix_np[np.isfinite(cost_matrix_np)])
        replacement_val = 1e6
        if np.isfinite(max_finite_cost):
            replacement_val = max(1.0, abs(max_finite_cost)) * 10.0
        cost_matrix_np[~np.isfinite(cost_matrix_np)] = replacement_val
        logger.warning(f"NaN/Inf detected in cost matrix. Replaced non-finite values with {replacement_val:.2e}.")

    # Prepare Marginals a and b
    if a is None:
        a = np.ones((N,), dtype=np.float64) / N
    else:
        a = np.asarray(a, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            a = np.ones_like(a) / max(1, len(a))
            logger.warning("NaN/Inf in marginal 'a'. Using uniform.")
        sum_a = a.sum()
        if sum_a <= eps_num:
            a = np.ones_like(a) / max(1, len(a))
            logger.warning("Marginal 'a' sums to zero or less. Using uniform.")
        elif not np.isclose(sum_a, 1.0):
            a /= sum_a

    if b is None:
        b = np.ones((M,), dtype=np.float64) / M
    else:
        b = np.asarray(b, dtype=np.float64)
        if not np.all(np.isfinite(b)):
            b = np.ones_like(b) / max(1, len(b))
            logger.warning("NaN/Inf in marginal 'b'. Using uniform.")
        sum_b = b.sum()
        if sum_b <= eps_num:
            b = np.ones_like(b) / max(1, len(b))
            logger.warning("Marginal 'b' sums to zero or less. Using uniform.")
        elif not np.isclose(sum_b, 1.0):
            b /= sum_b

    # NEW: Pre-scale the cost matrix to avoid numerical issues
    original_scale = cost_matrix_np.max()
    if original_scale > 1.0:  # If max cost is greater than 1.0, scale it down
        scaling_factor = 1.0 / original_scale
        cost_matrix_np = cost_matrix_np * scaling_factor
        logger.info(f"Pre-scaled cost matrix by factor {scaling_factor:.4f} to avoid numerical issues")
    
    # Convert to contiguous array for better performance
    cost_matrix_np_cont = np.ascontiguousarray(cost_matrix_np)
    
    # Compute OT using POT
    Gs = None
    ot_cost = np.nan
    
    # Determine if we need to adjust regularization based on cost matrix values
    cost_max = np.max(cost_matrix_np_cont)
    effective_reg = reg  # Start with specified reg parameter
    
    # Adjust regularization if costs are high
    if cost_max > 1.0:
        effective_reg = max(reg, cost_max / 100)  # Increase reg parameter proportionally
        logger.info(f"Adjusted regularization to {effective_reg:.6f} based on cost matrix values")
    
    try:
        # Try stabilized Sinkhorn first
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress numpy warnings
            Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=effective_reg, stopThr=sinkhorn_thresh,
                            numItermax=sinkhorn_max_iter, method='sinkhorn_stabilized',
                            warn=False, verbose=False)

        if Gs is None or np.any(np.isnan(Gs)):
            # Fallback to standard Sinkhorn
            if Gs is None: 
                logger.warning("Stabilized Sinkhorn failed. Trying standard Sinkhorn.")
            else: 
                logger.warning("Stabilized Sinkhorn resulted in NaN plan. Trying standard Sinkhorn.")
            
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress numpy warnings
                Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=effective_reg, stopThr=sinkhorn_thresh,
                                numItermax=sinkhorn_max_iter, method='sinkhorn',
                                warn=False, verbose=False)

        if Gs is None or np.any(np.isnan(Gs)):
            logger.warning("Both Sinkhorn methods failed. Trying with increased regularization.")
            # Try one more time with significantly higher regularization
            increased_reg = effective_reg * 5
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress numpy warnings
                Gs = ot.sinkhorn(a, b, cost_matrix_np_cont, reg=increased_reg, stopThr=sinkhorn_thresh * 10,
                                numItermax=sinkhorn_max_iter, method='sinkhorn_stabilized',
                                warn=False, verbose=False)
                
        if Gs is None or np.any(np.isnan(Gs)):
            # Final fallback: try an alternate algorithm (EMD for small problems)
            if N <= 100 and M <= 100:  # Only for reasonably small problems
                logger.warning("All Sinkhorn attempts failed. Trying exact EMD (may be slow).")
                try:
                    Gs = ot.emd(a, b, cost_matrix_np_cont)
                except Exception as e:
                    logger.warning(f"EMD computation failed: {e}")
                    Gs = None
            else:
                logger.warning("All Sinkhorn attempts failed. Problem too large for EMD fallback.")
                Gs = None

        if Gs is None or np.any(np.isnan(Gs)):
            logger.warning("All OT computation methods failed.")
            return np.nan, None
            
        # Calculate OT cost
        ot_cost = np.sum(Gs * cost_matrix_np_cont)
        
        # If we scaled the cost matrix, we need to re-scale the OT cost
        if original_scale > 1.0:
            ot_cost = ot_cost / scaling_factor
            
        if not np.isfinite(ot_cost):
            logger.warning(f"Calculated OT cost is not finite ({ot_cost}). Returning NaN.")
            ot_cost = np.nan

    except Exception as e:
        logger.warning(f"Error during OT computation: {e}")
        ot_cost = np.nan
        Gs = None

    return float(ot_cost), Gs

# --- Distance Metrics ---

def pairwise_euclidean_sq(X: Optional[torch.Tensor], Y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """ Calculates pairwise squared Euclidean distance: C[i,j] = ||X[i] - Y[j]||_2^2 """
    if X is None or Y is None or X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        logger.warning("pairwise_euclidean_sq: Invalid input shapes or None input.")
        return None
    try:
        # Using cdist is generally numerically stable and clear
        dist_sq = torch.cdist(X.float(), Y.float(), p=2).pow(2)
        return dist_sq
    except Exception as e:
        logger.warning(f"Error calculating squared Euclidean distance: {e}")
        return None

def calculate_label_emd(y1: Optional[torch.Tensor], y2: Optional[torch.Tensor], num_classes: int) -> float:
    """Calculates Earth Mover's Distance between label distributions."""
    if y1 is None or y2 is None: return np.nan
    try:
        y1_tensor = y1 if isinstance(y1, torch.Tensor) else torch.tensor(y1)
        y2_tensor = y2 if isinstance(y2, torch.Tensor) else torch.tensor(y2)
        n1, n2 = y1_tensor.numel(), y2_tensor.numel()
    except Exception as e:
        logger.warning(f"Error processing labels for EMD: {e}")
        return np.nan

    if n1 == 0 and n2 == 0: return 0.0
    if n1 == 0 or n2 == 0: return float(max(0.0, float(num_classes - 1)))

    y1_np = y1_tensor.detach().cpu().numpy().astype(int)
    y2_np = y2_tensor.detach().cpu().numpy().astype(int)
    class_values = np.arange(num_classes)

    hist1, _ = np.histogram(y1_np, bins=np.arange(num_classes + 1), density=False)
    hist2, _ = np.histogram(y2_np, bins=np.arange(num_classes + 1), density=False)

    sum1 = hist1.sum(); sum2 = hist2.sum()
    if sum1 == 0 or sum2 == 0: return float(max(0.0, float(num_classes - 1)))

    hist1_norm = hist1 / sum1; hist2_norm = hist2 / sum2;

    try:
        return float(wasserstein_distance(class_values, class_values, u_weights=hist1_norm, v_weights=hist2_norm))
    except ValueError as e:
        logger.warning(f"Wasserstein distance calculation failed: {e}")
        return np.nan

# --- Sample Management ---

def validate_samples_for_ot(
    features_dict: Dict[str, np.ndarray], 
    min_samples: int = 20,
    max_samples: int = 900
) -> Tuple[bool, Dict[str, np.ndarray]]:
    """
    Validates if clients have enough samples and samples down if exceeding maximum.
    
    Args:
        features_dict: Dictionary mapping client identifiers to feature arrays
        min_samples: Minimum number of samples required per client
        max_samples: Maximum number of samples to use per client (to improve efficiency)
        
    Returns:
        Tuple containing:
            - bool: Whether all clients have sufficient samples
            - Dict: Mapping of client identifiers to arrays of indices to use
    """
    indices_to_use = {}
    all_sufficient = True
    
    for client_id, features in features_dict.items():
        n_samples = len(features)
        
        # Check minimum threshold
        if n_samples < min_samples:
            logger.info(f"Client {client_id} has insufficient samples: {n_samples} < {min_samples}")
            all_sufficient = False
            indices_to_use[client_id] = np.arange(n_samples)  # Use all available samples
        # Check maximum threshold - only sample if we're strictly over the max
        elif n_samples > max_samples:
            # Sample down to max_samples
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            indices = rng.choice(n_samples, max_samples, replace=False)
            indices.sort()  # Keep original order
            indices_to_use[client_id] = indices
            
            # Log the sampling
            logger.info(f"Client {client_id} samples reduced: {n_samples} → {max_samples}")
        else:
            # Use all samples (already within limits)
            indices_to_use[client_id] = np.arange(n_samples)
    
    return all_sufficient, indices_to_use


def apply_sampling_to_data(
    tensors_dict: Dict[str, Optional[torch.Tensor]], 
    indices_dict: Dict[str, np.ndarray]
) -> Dict[str, Optional[torch.Tensor]]:
    """
    Apply sampling indices to multiple tensors consistently.
    
    Args:
        tensors_dict: Dictionary of tensors to sample (h, y, p_prob, weights, etc.)
        indices_dict: Dictionary mapping keys to indices arrays
        
    Returns:
        Dictionary with sampled tensors
    """
    sampled_tensors = {}
    for name, tensor in tensors_dict.items():
        if tensor is None:
            sampled_tensors[name] = None
            continue
            
        if name in indices_dict:
            indices = indices_dict[name]
            # Convert indices to tensor for indexing
            if isinstance(indices, np.ndarray):
                indices_tensor = torch.from_numpy(indices).long()
            else:
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                
            # Apply sampling
            sampled_tensors[name] = tensor[indices_tensor]
        else:
            # No sampling needed for this tensor
            sampled_tensors[name] = tensor
            
    return sampled_tensors

def calculate_sample_loss(
    p_prob: Optional[torch.Tensor], 
    y: Optional[torch.Tensor], 
    num_classes: int, 
    loss_eps: float = DEFAULT_EPS
) -> Optional[torch.Tensor]:
    """
    Calculates per-sample cross-entropy loss, with enhanced validation for multiclass.
    
    Args:
        p_prob: Predicted probability distribution of shape [N, K]
        y: Ground truth labels of shape [N]
        num_classes: Number of classes K
        loss_eps: Small epsilon value for numerical stability
        
    Returns:
        Tensor of per-sample losses of shape [N] or None if validation fails
    """
    if p_prob is None or y is None: 
        return None
        
    if not isinstance(p_prob, torch.Tensor): 
        p_prob = torch.tensor(p_prob)
    if not isinstance(y, torch.Tensor): 
        y = torch.tensor(y)

    try:
        # Ensure tensors are on CPU and correct dtype
        p_prob = p_prob.float().cpu()
        y = y.long().cpu()
        
        # Validate shapes and fix if needed for binary case
        if p_prob.ndim == 1 and num_classes == 2:
            # Convert [N] format to [N, 2] for binary case
            p_prob_1d = p_prob.view(-1)
            p1 = p_prob_1d.clamp(min=loss_eps, max=1.0 - loss_eps)
            p0 = 1.0 - p1
            p_prob = torch.stack([p0, p1], dim=1)
        elif p_prob.ndim == 2 and p_prob.shape[1] == 1 and num_classes == 2:
            # Convert [N, 1] format to [N, 2] for binary case
            p1 = p_prob.view(-1).clamp(min=loss_eps, max=1.0 - loss_eps)
            p0 = 1.0 - p1
            p_prob = torch.stack([p0, p1], dim=1)
            
        # Final shape validation
        if y.shape[0] != p_prob.shape[0] or p_prob.ndim != 2 or p_prob.shape[1] != num_classes:
            logger.warning(f"Loss calculation shape mismatch/invalid: P({p_prob.shape}), Y({y.shape}), K={num_classes}")
            return None
            
        # Ensure probabilities are valid (sum to 1, in [eps, 1-eps] range)
        row_sums = p_prob.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            logger.warning(f"Probabilities don't sum to 1 (min={row_sums.min().item():.4f}, "
                          f"max={row_sums.max().item():.4f}). Normalizing.")
            p_prob = p_prob / row_sums.unsqueeze(1).clamp(min=loss_eps)
            
        # Clamp probabilities for numerical stability
        p_prob = p_prob.clamp(min=loss_eps, max=1.0 - loss_eps)
        
        # Gather predicted probability for true class
        true_class_prob = p_prob.gather(1, y.view(-1, 1)).squeeze()
        
        # Calculate cross-entropy loss: -log(p[true_class])
        loss = -torch.log(true_class_prob)
        
        # Safety check for NaN/Inf values
        if not torch.isfinite(loss).all():
            logger.warning("Non-finite values in loss calculation. Replacing with large value.")
            loss = torch.where(torch.isfinite(loss), loss, torch.tensor(100.0, dtype=torch.float32))
            
    except Exception as e:
        logger.warning(f"Error during loss calculation: {e}")
        return None

    return torch.relu(loss)  # Ensure non-negative loss