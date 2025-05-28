"""
Configuration utilities for OT-based similarity analysis.
Defines OTConfig class and pre-defined configuration collections.
Includes both within-class and non-within-class configurations.
"""
import logging
from typing import Dict, Optional, Any, List

# Configure module logger
logger = logging.getLogger(__name__)
VERBOSE = True # Assuming VERBOSE is defined elsewhere or set directly. If not, set to True/False.

# --- Constants ---
DEFAULT_OT_REG = 0.001
DEFAULT_OT_MAX_ITER = 5000
DEFAULT_EPS = 1e-10


class OTConfig:
    """
    Configuration object for a single OT calculation run.
    Stores method type, name, and parameters.
    """
    # Define known method types to prevent typos
    KNOWN_METHOD_TYPES = {'direct_ot'}

    def __init__(self, method_type: str, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            method_type (str): The type of OT calculator
            name (str): A unique descriptive name for this configuration set
            params (dict, optional): Dictionary of parameters specific to the OT method. Defaults to {}.
        """
        if not isinstance(method_type, str) or not method_type:
            raise ValueError("method_type must be a non-empty string.")
        if method_type not in self.KNOWN_METHOD_TYPES:
            logger.warning(f"Unknown method_type '{method_type}'. Ensure a corresponding calculator exists in the factory.")
        self.method_type = method_type

        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string.")
        self.name = name

        self.params = params if params is not None else {}
        if not isinstance(self.params, dict):
            raise ValueError("params must be a dictionary.")

    def __repr__(self) -> str:
        return f"OTConfig(method_type='{self.method_type}', name='{self.name}', params={self.params})"


# ----- Predefined OT Configurations -----

# Parameters common to all direct_ot configurations, irrespective of within_class
ABSOLUTE_COMMON_PARAMS = {
    'use_loss_weighting': False,
    'normalize_cost': False,
    'min_samples': 20,
    'max_samples': 900,
    'verbose': VERBOSE,
    # Default OT parameters, can be overridden if needed
    'reg': DEFAULT_OT_REG,
    'max_iter': DEFAULT_OT_MAX_ITER,
    'compress_vectors': True, # Common for label distances
    'compression_threshold': 20,
    'compression_ratio': 5, # Example: retain 80% variance
}

# Base parameters for direct_ot configurations that ARE within_class_only
COMMON_WITHIN_CLASS_PARAMS = {
    **ABSOLUTE_COMMON_PARAMS,
    'within_class_only': True,
}

# Parameters specific to configurations using cosine feature distance and Hellinger label distance,
# and are also within_class_only.
COMMON_WC_COSINE_HELLINGER_PARAMS = {
    **COMMON_WITHIN_CLASS_PARAMS,
    'normalize_cost': True,
    'normalize_activations': True,
    'feature_distance': 'cosine',
    'label_distance': 'hellinger',
}

direct_ot_configs = [
    # Non-Within-Class Wasserstein configuration
    OTConfig(
        method_type='direct_ot',
        name='Direct_Wasserstein', # No WC_ prefix
        params={
            **ABSOLUTE_COMMON_PARAMS, # Base common parameters
            'normalize_activations': False,
            'feature_distance': 'euclidean',
            'label_distance': 'wasserstein_gaussian',
            'feature_weight': 1.0,
            'label_weight': 1.0,
            # 'within_class_only' is NOT set, so it defaults to False in the calculator
        }
    ),
    # Within-Class Wasserstein configuration
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Wasserstein', # WC for Within-Class
        params={
            **COMMON_WITHIN_CLASS_PARAMS, # Base within-class parameters
            'normalize_activations': False,
            'feature_distance': 'euclidean',
            'label_distance': 'wasserstein_gaussian',
            'feature_weight': 1.0,
            'label_weight': 1.0,
        }
    ),

    # Within-Class Hellinger-based configurations with varying feature/label weights
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_1:1',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 1.0,
            'label_weight': 1.0,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_2:1',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 2.0,
            'label_weight': 1.0,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_3:1',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 3.0,
            'label_weight': 1.0,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_1:2',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 1.0,
            'label_weight': 2.0,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_1:3',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 1.0,
            'label_weight': 3.0,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_1:4',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 1.0,
            'label_weight': 4.0,
        }
    ),

    # Within-Class Feature-only configurations
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_FeatureOnly_Cosine',
        params={
            **COMMON_WITHIN_CLASS_PARAMS, # Base within-class parameters
            'normalize_activations': True,
            'feature_distance': 'cosine',
            'label_distance': None, # No label distance component
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_FeatureOnly_Euclidean',
        params={
            **COMMON_WITHIN_CLASS_PARAMS, # Base within-class parameters
            'normalize_activations': False,
            'feature_distance': 'euclidean',
            'label_distance': None, # No label distance component
        }
    ),
]

subsampled_direct_ot_configs = [   
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_s100',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'max_samples': 100,
        }
    ),
        OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_s75',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'max_samples': 75,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_s50',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'max_samples':50,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_s25',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'max_samples': 25,
        }
    ),
]

reg_param_direct_ot_configs = [   
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_reg_0.005',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'reg': 0.005,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_reg_0.01',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'reg': 0.01,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_reg_0.05',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'reg': 0.05,
        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_0.1',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'reg':0.1,

        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_0.5',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'reg':0.5,

        }
    ),
    OTConfig(
        method_type='direct_ot',
        name='WC_Direct_Hellinger_4:1_1',
        params={
            **COMMON_WC_COSINE_HELLINGER_PARAMS,
            'feature_weight': 4.0,
            'label_weight': 1.0,
            'reg':1,

        }
    ),
]

all_configs = direct_ot_configs + subsampled_direct_ot_configs + reg_param_direct_ot_configs