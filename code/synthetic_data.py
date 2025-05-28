# synthetic_data.py
"""
Generates raw synthetic data (features and labels) as NumPy arrays.
Includes simple, controllable feature and concept shift mechanisms.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

# =============================================================================
# == Helper Functions for Shifts ==
# =============================================================================
def apply_feature_shift(X: np.ndarray,
                         delta: float,
                         kind: str = "mean",
                         cols: Optional[int] = None,
                         mu: float = 3.0,
                         sigma: float = 1.5,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Applies a feature distribution shift to the input data X with balanced direction.

    Args:
        X (np.ndarray): Input features (standard normal recommended).
        delta (float): Shift intensity parameter [0, 1].
        kind (str): Type of shift ('mean', 'scale', 'tilt', 'balanced_mean').
        cols (Optional[int]): Number of initial columns to affect. Defaults to min(10, n_features).
        mu (float): Mean shift factor (for kind='mean').
        sigma (float): Scale shift factor (for kind='scale').
        rng (Optional[np.random.Generator]): Random number generator for 'tilt'.

    Returns:
        np.ndarray: Modified feature array X.
    """
    n_samples, n_features = X.shape
    if cols is None:
        cols = min(10, n_features)  # Default to affect first 10 cols or fewer

    # mean shift - alternates shift direction for features
    if kind == "mean" and cols > 0:
        half_cols = cols // 2
        
        # First half of features shift positive
        if half_cols > 0:
            X[:, :half_cols] += delta * mu
            
        # Second half of features shift negative
        if cols - half_cols > 0:
            X[:, half_cols:cols] -= delta * mu
            
    # Balanced scale shift - some scale up, some scale down
    elif kind == "scale" and cols > 0:
        half_cols = cols // 2
        
        # First half scales up
        if half_cols > 0:
            X[:, :half_cols] *= (1.0 + delta * sigma)
            
        # Second half scales down (carefully to avoid division by zero)
        if cols - half_cols > 0:
            scale_down = 1.0 / (1.0 + delta * sigma) if delta * sigma < 0.9 else 0.1
            X[:, half_cols:cols] *= scale_down
            
        
    if rng is None:
        rng = np.random.default_rng()  # Use default if none provided
    # Generate a random orthogonal matrix Q via QR decomposition
    H = rng.standard_normal((n_features, n_features))
    Q, _ = np.linalg.qr(H)
    # Apply convex combination: (1-delta)*X + delta*(X @ Q)
    X[:] = (1.0 - delta) * X + delta * (X @ Q)

    return X

def apply_concept_shift(
    X: np.ndarray,
    gamma: float,
    option: str = "threshold",
    threshold_range_factor: float = 0.4,
    label_noise: float = 0.01,
    base_seed: int = 42,
    label_rule: str = "linear",
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Apply concept shift by adjusting the decision boundary.
    
    Parameters
    ----------
    X : np.ndarray
        Input features
    gamma : float
        Shift intensity parameter [0, 1]
    option : str
        Type of shift ('threshold' or 'rotation')
    threshold_range_factor : float
        Controls threshold shift range
    label_noise : float
        Probability of flipping labels
    base_seed : int
        Seed to ensure consistent model parameters
    label_rule : str
        Rule used for score generation
    rng : np.random.Generator, optional
        Random generator for label noise
        
    Returns
    -------
    y : np.ndarray
        Shifted binary labels
    """
    # Generate score using same function used in data generation
    score, params = generate_score(X, label_rule, base_seed)
    # Get the weight vector (for rotation method)
    w = params['w']
    if option == "threshold":
        # Shift the threshold based on gamma
        target_quantile = 0.5 + threshold_range_factor * (gamma)
        target_quantile = np.clip(target_quantile, 1e-6, 1.0 - 1e-6)
        threshold = np.quantile(score, target_quantile)
        y = (score > threshold).astype(int)
    elif option == "rotation":
        if X.shape[1] < 2:
            print("Warning: Rotation concept shift requires at least 2 features. Using baseline.")
            y = (score > 0).astype(int)
        else:
            # Determine how many features to rotate
            num_features = X.shape[1]
            num_rotation_pairs = num_features // 2  # How many pairs of features to rotate
            
            # Rotate weight vector across multiple feature planes
            angle = (gamma - 0.5) * np.pi * 0.6
            c, s = np.cos(angle), np.sin(angle)
            
            # Re-calculate score with rotated weights (only for linear rule)
            if label_rule == "linear":
                w_rot = w.copy()
                
                # Apply rotation to multiple feature pairs
                for i in range(num_rotation_pairs):
                    idx1, idx2 = i*2, i*2+1  # Pair features (0,1), (2,3), (4,5), etc.
                    
                    # Skip if we've reached the end of our features
                    if idx2 >= num_features:
                        break
                        
                    # Apply rotation to this feature pair
                    w1_orig, w2_orig = w[idx1], w[idx2]
                    w_rot[idx1] = c * w1_orig - s * w2_orig
                    w_rot[idx2] = s * w1_orig + c * w2_orig
                
                rot_score = X @ w_rot
                y = (rot_score > 0).astype(int)
            else:
                # For non-linear rules, fall back to threshold shifting
                print(f"Warning: Rotation not implemented for {label_rule}, using threshold shift.")
                target_quantile = 0.5 + threshold_range_factor * (gamma - 0.5)
                threshold = np.quantile(score, np.clip(target_quantile, 1e-6, 1.0 - 1e-6))
                y = (score > threshold).astype(int)
    
    else:
        print(f"Warning: Unknown concept option '{option}'. Using baseline.")
        y = (score > 0).astype(int)
    
    # Apply label noise
    if label_noise > 0:
        if rng is None:
            rng = np.random.default_rng(base_seed + 100)
        flip = rng.random(len(y)) < label_noise
        y[flip] = 1 - y[flip]
    
    return y


def _centre_and_scale(score: np.ndarray,
                      target_iqr: float = 2.0,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Shift the score distribution to median ≈ 0 and compress / stretch it
    so that its inter-quartile range equals `target_iqr`.

    Setting `target_iqr` smaller ⇒ more points sit near 0 ⇒ harder task.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) centre
    score = score - np.median(score)

    # 2) scale
    q1, q3 = np.quantile(score, [0.25, 0.75])
    iqr = max(np.abs(q3 - q1), 1e-9)          # avoid /0
    score = score * (target_iqr / iqr)

    # 3) optional small jitter so boundaries aren’t unnaturally sharp
    score += rng.normal(scale=0.02, size=score.shape)

    return score


def generate_score(X: np.ndarray, 
                  label_rule: str = "linear",
                  base_seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Generate scores for classification based on features X.
    
    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features)
    label_rule : str
        Rule to use for generating scores: 'linear', 'quadratic', 'piecewise', 'mlp'
    base_seed : int
        Seed for reproducible generation
        
    Returns
    -------
    score : np.ndarray
        Score values for each sample
    params : dict
        Dictionary of parameters used (w, W1, etc.) for later reuse
    """
    n_samples, n_features = X.shape
    base_rng = np.random.default_rng(base_seed)
    params = {}

    # Generate weight vector (store for all models)
    w = base_rng.standard_normal(n_features) 
    params['w'] = w
    if label_rule == "linear":
        lin_rng = np.random.default_rng(base_seed + 1)
        # Calculate feature importance scale factor
        importance = np.abs(w)
        importance_scale = importance / np.max(importance)  # Normalize to [0,1]

        # Inverse scaling - more important features get less noise
        noise_scale = 1.0 - (importance_scale) 

        # Generate feature-specific noise
        noise = lin_rng.standard_normal((n_samples, n_features)) * 2

        # Apply scaled noise
        # for i in range(n_features):
        #     X[:, i] += noise[:, i] * noise_scale[i]

        score = X @ w
    elif label_rule == "quadratic":
        k = min(5, n_features // 2)
        beta = 0.2
        params['beta'] = beta
        params['k'] = k
        score = X @ w + beta * (X[:, :k] * X[:, k:2*k]).sum(1)
        
    elif label_rule == "piecewise":
        raw = X @ w
        bins = np.digitize(raw, [-1., 0., 1.])  # 0-1-2
        score = bins.astype(float)
        params['bin_edges'] = [-1., 0., 1.]
        
    elif label_rule == "mlp":
        h = 100
        n_layers = 2
        params, forward_pass = create_multilayer_nn(n_features, h, n_layers, base_rng)
        score = forward_pass(X, params)
    else:
        raise ValueError(f"Unknown label_rule '{label_rule}'")
    
    score = _centre_and_scale(score, target_iqr=2.0, rng=base_rng)
    #score = np.sin(score)
    return score, params

def generate_synthetic_data(
    n_samples: int,
    n_features: int,
    label_noise: float = 0.0,
    base_seed: int = 42,
    **shift_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with consistent decision boundary."""
    # Create RNG and features
    base_rng = np.random.default_rng(base_seed)
    X = base_rng.standard_normal((n_samples, n_features)) / np.sqrt(n_features)
    #X = create_nonlinear_features(X, transformation_level='very_high')
    # Generate scores using the common function
    label_rule = shift_config.get("label_rule", "linear")
    score, _ = generate_score(X, label_rule, base_seed)
    # Convert scores to binary labels
    y = (score > 0).astype(int)
    
    # Apply label noise if specified
    if label_noise > 0:
        flip = base_rng.random(n_samples) < label_noise
        y[flip] = 1 - y[flip]
        
    return X.astype(np.float32), y.astype(np.int64)


def create_multilayer_nn(n_features, h, n_layers, base_rng):
    """
    Creates a multi-layer neural network with a specified number of hidden layers.

    Args:
        n_features: Number of input features.
        h: Number of neurons in each hidden layer.
        n_layers: Number of hidden layers.
        base_rng: A NumPy random number generator.

    Returns:
        A tuple containing:
        - params (dict): Dictionary of weight matrices and bias vectors.
        - forward_pass (function): A function that performs the forward pass
          through the network, taking input X and the params dictionary.
    """
    params = {}

    # Input layer to first hidden layer
    W1 = base_rng.standard_normal((n_features, h))
    b1 = base_rng.standard_normal(h)
    params['W1'] = W1
    params['b1'] = b1

    # Hidden layers
    for i in range(2, n_layers + 1):
        W = base_rng.standard_normal((h, h))
        b = base_rng.standard_normal(h)
        params[f'W{i}'] = W
        params[f'b{i}'] = b

    # Last hidden layer to output
    W_out = base_rng.standard_normal((h, 1))  # Assuming a single output for 'score'
    b_out = base_rng.standard_normal()
    params[f'W{n_layers + 1}'] = W_out
    params[f'b{n_layers + 1}'] = b_out

    def forward_pass(X, params):
        """
        Performs the forward pass through the neural network.

        Args:
            X: Input data (shape: (n_samples, n_features)).
            params: Dictionary of weight matrices and bias vectors.

        Returns:
            The output of the network (score).
        """
        hidden = np.tanh(X @ params['W1'] + params['b1'])
        for i in range(2, n_layers + 1):
            if i % 2 == 0:
                hidden = np.tanh(hidden @ params[f'W{i}'] + params[f'b{i}'])
            else:
                hidden = np.maximum(-0.1, hidden @ params[f'W{i}'] + params[f'b{i}'])
        score = hidden @ params[f'W{n_layers + 1}'] + params[f'b{n_layers + 1}']
        return score
    return params, forward_pass