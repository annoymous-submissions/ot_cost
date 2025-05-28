import torch
import torch.nn.functional as F
import numpy as np
import logging
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Type, List, Union # Added Type

# Import OTConfig from ot_configs
from ot_configs import OTConfig

# Import utilities from ot_utils.py
from ot_utils import (
    compute_ot_cost, pairwise_euclidean_sq,
    validate_samples_for_ot,
    prepare_ot_marginals, normalize_cost_matrix
)
from ot_configs import DEFAULT_EPS, DEFAULT_OT_REG, DEFAULT_OT_MAX_ITER
# Configure module logger
logger = logging.getLogger(__name__)

class OTCalculatorFactory:
    """ Factory class for creating OT calculator instances. """

    @classmethod
    def _get_calculator_map(cls) -> Dict[str, Type['BaseOTCalculator']]:
        """ Internal method to access the map, allowing definition after classes. """
        # Defined below BaseOTCalculator and its subclasses
        return {
            'direct_ot': DirectOTCalculator,
            # Register new calculator classes here
        }

    @staticmethod
    def create_calculator(config: OTConfig, client_id_1: str, client_id_2: str, num_classes: int) -> Optional['BaseOTCalculator']:
        """
        Creates an instance of the appropriate OT calculator based on the config.
        
        Args:
            config: OTConfig object with method_type and parameters
            client_id_1: First client identifier
            client_id_2: Second client identifier
            num_classes: Number of classes in the dataset
            
        Returns:
            An instance of the appropriate calculator or None if creation fails
        """
        calculator_map = OTCalculatorFactory._get_calculator_map()
        calculator_class = calculator_map.get(config.method_type)

        if calculator_class:
            try:
                instance = calculator_class(
                    client_id_1=client_id_1,
                    client_id_2=client_id_2,
                    num_classes=num_classes
                )
                return instance
            except Exception as e:
                logger.warning(f"Failed to instantiate calculator for config '{config.name}' (type: {config.method_type}): {e}")
                return None
        else:
            logger.warning(f"No calculator registered for method type '{config.method_type}' in config '{config.name}'. Skipping.")
            return None

# --- Base OT Calculator Class ---
class BaseOTCalculator(ABC):
    """
    Abstract Base Class for Optimal Transport similarity calculators.
    """
    def __init__(self, client_id_1: str, client_id_2: str, num_classes: int, **kwargs):
        self.client_id_1 = str(client_id_1)
        self.client_id_2 = str(client_id_2)
        self.num_classes = num_classes
        self.eps_num = kwargs.get('eps_num', DEFAULT_EPS) # Epsilon for numerical stability

        self.results: Dict[str, Any] = {}
        self.cost_matrices: Dict[str, Any] = {}
        self._reset_results() # Initialize result structures

    @abstractmethod
    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], data2: Dict[str, Optional[torch.Tensor]], params: Dict[str, Any]) -> None:
        """ Calculates the specific OT similarity metric. Must be implemented by subclasses. """
        pass

    @abstractmethod
    def _reset_results(self) -> None:
        """ Resets the internal results and cost matrix storage. """
        pass

    def get_results(self) -> Dict[str, Any]:
        """ Returns the calculated results. """
        return self.results

    def get_cost_matrices(self) -> Dict[str, Any]:
        """ Returns the computed cost matrices (if stored). """
        return self.cost_matrices

    def _preprocess_input(
        self, 
        data_client1: Dict[str, Optional[torch.Tensor]], 
        data_client2: Dict[str, Optional[torch.Tensor]], 
        required_keys: List[str]
    ) -> Optional[Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Optional[torch.Tensor]]]]:
        """
        Basic preprocessing for input data dictionaries from OTDataManager.
        Ensures required keys are present and tensors are on CPU.
        Relies on OTDataManager._process_client_data for detailed validation
        (e.g., probability formatting, loss calculation, weight generation).

        Args:
            data_client1: Processed data dictionary for client 1.
            data_client2: Processed data dictionary for client 2.
            required_keys: List of keys that must be non-None in both dictionaries.

        Returns:
            A tuple of (processed_data_client1, processed_data_client2) or None if validation fails.
        """
        if not isinstance(data_client1, dict) or not isinstance(data_client2, dict):
            logger.warning("Calculator _preprocess_input: Inputs must be dictionaries.")
            return None

        processed_data_c1 = {}
        processed_data_c2 = {}

        for client_idx, (data_in, data_out) in enumerate([(data_client1, processed_data_c1), 
                                                          (data_client2, processed_data_c2)]):
            client_name = self.client_id_1 if client_idx == 0 else self.client_id_2
            for key in data_in: # Iterate over all keys present in the input dict
                tensor = data_in.get(key)
                if tensor is None:
                    if key in required_keys:
                        logger.warning(f"Client {client_name}: Required input '{key}' is None.")
                        return None
                    data_out[key] = None
                    continue

                if isinstance(tensor, torch.Tensor):
                    data_out[key] = tensor.detach().cpu()
                else:
                    # This case should ideally not happen if OTDataManager prepares tensors
                    try:
                        data_out[key] = torch.tensor(tensor).cpu() 
                    except Exception as e:
                        logger.warning(f"Client {client_name}: Could not convert input '{key}' to CPU tensor: {e}")
                        return None
            
            # Final check for required keys after processing all available keys
            missing_keys = [k_req for k_req in required_keys if k_req not in data_out or data_out[k_req] is None]
            if missing_keys:
                logger.warning(f"Client {client_name}: Missing required keys {missing_keys} after processing.")
                return None
        
        # Basic check: Ensure 'h' features have same dimension if both present
        h1 = processed_data_c1.get('h')
        h2 = processed_data_c2.get('h')
        if h1 is not None and h2 is not None:
            if h1.ndim < 2 or h2.ndim < 2 : # Expect at least [N, D]
                 logger.warning(f"Feature dimensions are less than 2D (h1: {h1.shape}, h2: {h2.shape}).")
                 return None
            if h1.shape[0] == 0 or h2.shape[0] == 0: # No samples
                 logger.info(f"One or both clients have zero samples (h1: {h1.shape[0]}, h2: {h2.shape[0]}).")
                 # This is not an error for _preprocess_input, calculators handle it.
            elif h1.shape[1] != h2.shape[1]:
                 logger.warning(f"Feature dimension mismatch: h1 dim {h1.shape[1]} vs h2 dim {h2.shape[1]}.")
                 return None

        return processed_data_c1, processed_data_c2

# --- Concrete Calculator Implementations ---

class DirectOTCalculator(BaseOTCalculator):
    """
    Calculates direct OT cost between neural network activations with additional
    label distribution similarity using Hellinger distance.
    
    This method computes optimal transport using both the feature representations and
    the distributional properties of those representations, combining both to create
    a more comprehensive similarity metric.
    """
    LABEL_DISTANCE_FUNCTIONS = {
        'hellinger': '_hellinger_distance',
        'wasserstein_gaussian': '_wasserstein_gaussian_distance',
    }
    
    def _reset_results(self) -> None:
        """Initialize/reset the results dictionary and cost matrices."""
        self.results = {
            'direct_ot_cost': np.nan,
            'transport_plan': None,
            'feature_distance': None,
            'weighting_used': None,
            'feature_weight': np.nan,
            'label_costs': [],
            'label_distance': None,
        }
        self.cost_matrices = {
            'direct_ot': None,
            'feature_cost': None,
            'label_cost': None,
            'combined_cost': None
        }

    def calculate_similarity(self, data1: Dict[str, Optional[torch.Tensor]], 
                                data2: Dict[str, Optional[torch.Tensor]], 
                                params: Dict[str, Any]) -> None:
        """
        Calculate direct OT similarity with optional within-class matching.
        """
        # --- Common setup ---
        self._reset_results()
        
        # Extract parameters
        common_params = self._extract_params(params)
        
        # Store configuration in results
        self._store_config_in_results(common_params)
        
        # --- Data Validation and Preparation ---
        prepared_data = self._validate_and_prepare_data(data1, data2, common_params)
        
        if prepared_data is None:
            # Handle preprocessing failure with informative message
            missing_keys = "neural network activations ('h')"
            if common_params['within_class_only']:
                missing_keys += " and labels ('y')"
            elif common_params['label_distance'] is not None:
                missing_keys += f" and labels ('y' for {common_params['label_distance']})"
            logger.warning(f"DirectOT calculation requires {missing_keys}. Preprocessing failed or data missing. Skipping.")
            weight_type = "Loss" if common_params['use_loss_weighting'] else "Uniform"
            self.results['weighting_used'] = weight_type
            return
        
        # Check for empty datasets
        N, M = prepared_data['N'], prepared_data['M']
        if N == 0 or M == 0:
            logger.warning("DirectOT: One or both clients have zero samples. OT cost is 0.")
            self.results['direct_ot_cost'] = 0.0
            weight_type = "Loss" if common_params['use_loss_weighting'] else "Uniform"
            self.results['weighting_used'] = weight_type
            return
        
        # --- Build Cost Matrix ---
        if common_params['within_class_only']:
            # Handle the within-class case, which requires specialized processing
            self._calculate_similarity_within_class(
                prepared_data['h1'], prepared_data['h2'], 
                prepared_data['y1'], prepared_data['y2'], 
                prepared_data['w1'], prepared_data['w2'], 
                N, M, common_params
            )
            return
        
        cost_matrix = self._build_cost_matrix(prepared_data, common_params)
        
        if cost_matrix is None:
            # Cost matrix building failed
            logger.warning("DirectOT: Failed to build cost matrix. Skipping.")
            return
        
        # --- Prepare Marginals ---
        a, b, sample_info = self._prepare_marginals(prepared_data, common_params)
        
        if a is None or b is None:
            # Failed to prepare marginals
            return
        
        # --- Run OT Calculation ---
        self._run_ot(cost_matrix, a, b, sample_info, common_params)

    def _extract_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and return common parameters."""
        return {
            'verbose': params.get('verbose', False),
            'normalize_activations': params.get('normalize_activations', True),
            'normalize_cost': params.get('normalize_cost', True),
            'feature_distance': params.get('feature_distance', 'euclidean'),
            'use_loss_weighting': params.get('use_loss_weighting', False),
            'label_distance': params.get('label_distance', None),  # Only check for the new parameter
            'feature_weight': params.get('feature_weight', 2.0),
            'label_weight': params.get('label_weight', 1.0),
            'compress_vectors': params.get('compress_vectors', True),
            'compression_threshold': params.get('compression_threshold', 10),
            'compression_ratio': params.get('compression_ratio', 5),
            'reg': params.get('reg', DEFAULT_OT_REG),
            'max_iter': params.get('max_iter', DEFAULT_OT_MAX_ITER),
            'min_samples_threshold': params.get('min_samples', 20),
            'max_samples_threshold': params.get('max_samples', 900),
            'within_class_only': params.get('within_class_only', False),
        }
    
    def _store_config_in_results(self, params: Dict[str, Any]) -> None:
        """Store configuration in results dictionary."""
        self.results['feature_distance'] = params['feature_distance']
        self.results['label_distance'] = params['label_distance']
        self.results['feature_weight'] = params['feature_weight']
        self.results['label_weight'] = params['label_weight']
        self.results['within_class_only'] = params['within_class_only']
        
        
    def _validate_and_prepare_data(self, data1: Dict[str, Optional[torch.Tensor]], 
                              data2: Dict[str, Optional[torch.Tensor]],
                              params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validates and prepares input data for OT calculation.
        
        Args:
            data1: Dictionary of tensors for client 1
            data2: Dictionary of tensors for client 2
            params: Configuration parameters
            
        Returns:
            Dictionary of prepared data or None if validation fails
        """
        # Determine required keys based on params
        required_keys = ['h']
        if params['within_class_only'] or params['label_distance'] is not None:
            required_keys.append('y')
        if params['use_loss_weighting']:
            required_keys.append('weights')
        
        # Process inputs
        proc_data1, proc_data2 = self._preprocess_input(data1, data2, required_keys)
        
        if proc_data1 is None or proc_data2 is None:
            return None
                
        # Extract all data
        h1 = proc_data1['h']  
        h2 = proc_data2['h']  
        y1 = proc_data1.get('y')
        y2 = proc_data2.get('y')
        w1 = proc_data1.get('weights')
        w2 = proc_data2.get('weights')
        p1_prob = proc_data1.get('p_prob')
        p2_prob = proc_data2.get('p_prob')
        N, M = h1.shape[0], h2.shape[0]
        
        return {
            'h1': h1, 'h2': h2, 'y1': y1, 'y2': y2, 
            'w1': w1, 'w2': w2, 'p1_prob': p1_prob, 'p2_prob': p2_prob,
            'N': N, 'M': M
        }
    def _build_cost_matrix(self, prepared_data: Dict[str, Any], params: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Builds the cost matrix for OT calculation.
        
        Args:
            prepared_data: Dictionary of prepared data from _validate_and_prepare_data
            params: Configuration parameters
            
        Returns:
            Cost matrix tensor or None if building fails
        """
        # Extract data
        h1, h2 = prepared_data['h1'], prepared_data['h2']
        y1, y2 = prepared_data['y1'], prepared_data['y2']
        N, M = prepared_data['N'], prepared_data['M']
        
        # Normalize activations
        h1_norm, h2_norm = self._normalize_activations(h1, h2, params['normalize_activations'])
        
        # Compute feature cost matrix
        feature_cost_matrix, max_feature_cost = self._calculate_feature_cost(
            h1_norm, h2_norm, params['feature_distance'], params['normalize_activations']
        )
        
        if feature_cost_matrix is None:
            logger.warning(f"Failed to compute feature cost matrix with method: {params['feature_distance']}")
            return None
                
        # Normalize feature cost matrix if requested
        feature_cost_matrix = normalize_cost_matrix(
            feature_cost_matrix, max_feature_cost, params['normalize_cost'], self.eps_num
        )
                
        # Store None instead of the full feature cost matrix
        
        # Calculate label cost matrix if requested
        if params['label_distance'] is not None:
            label_cost_matrix = self._calculate_label_cost_matrix(
                h1, h2, y1, y2, N, M, feature_cost_matrix, params
            )
            
            # Combine feature and label costs
            cost_matrix = self._combine_cost_matrices(
                feature_cost_matrix, label_cost_matrix, 
                use_label_distance=(params['label_distance'] is not None), 
                feature_weight=params['feature_weight'], 
                label_weight=params['label_weight']
            )
        else:
            # Use only feature cost
            cost_matrix = feature_cost_matrix
        
        return cost_matrix
    
    def _prepare_marginals(self, prepared_data: Dict[str, Any], params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Prepares marginal distributions for OT calculation.
        
        Args:
            prepared_data: Dictionary of prepared data
            params: Configuration parameters
            
        Returns:
            Tuple of (a, b, sample_info) where sample_info contains effective dimensions after sampling
        """
        # Extract data
        h1, h2 = prepared_data['h1'], prepared_data['h2']
        w1, w2 = prepared_data['w1'], prepared_data['w2']
        N, M = prepared_data['N'], prepared_data['M']
        
        # Prepare weights for OT
        weight_type, full_w1, full_w2 = self._prepare_weights(
            w1, w2, N, M, params['use_loss_weighting']
        )
        
        self.results['weighting_used'] = weight_type
        
        # Validate sample counts and get sampling indices if needed
        features_dict = {
            "client1": h1.cpu().numpy(),
            "client2": h2.cpu().numpy()
        }
        all_sufficient, sample_indices = validate_samples_for_ot(
            features_dict, params['min_samples_threshold'], params['max_samples_threshold']
        )
        
        # If samples are insufficient, signal failure
        if not all_sufficient:
            logger.warning(f"DirectOT: One or both clients have insufficient samples (min={params['min_samples_threshold']}). Skipping.")
            return None, None, {'N_eff': 0, 'M_eff': 0}
        
        # Apply sampling to client 1 (rows)
        N_eff, M_eff = N, M
        sampled_w1, sampled_w2 = full_w1, full_w2
        
        if "client1" in sample_indices and len(sample_indices["client1"]) < N:
            sampled_w1 = full_w1[sample_indices["client1"]]
            N_eff = len(sample_indices["client1"])
            if params['verbose']:
                logger.info(f"Sampled client1 weights: {N_eff} from original {N}")
        
        # Apply sampling to client 2 (columns)
        if "client2" in sample_indices and len(sample_indices["client2"]) < M:
            sampled_w2 = full_w2[sample_indices["client2"]]
            M_eff = len(sample_indices["client2"])
            if params['verbose']:
                logger.info(f"Sampled client2 weights: {M_eff} from original {M}")
        
        # Prepare marginals with the sampled weights
        a, b = prepare_ot_marginals(sampled_w1, sampled_w2, N_eff, M_eff, self.eps_num)
        
        return a, b, {'N_eff': N_eff, 'M_eff': M_eff, 'sample_indices': sample_indices}

    def _run_ot(self, cost_matrix: torch.Tensor, a: np.ndarray, b: np.ndarray, 
            sample_info: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """
        Runs the OT calculation.
        
        Args:
            cost_matrix: Cost matrix
            a: Source marginal distribution
            b: Target marginal distribution
            sample_info: Information about sampling
            params: Configuration parameters
            
        Returns:
            True if OT calculation succeeded, False otherwise
        """
        # Apply sampling to cost matrix
        sampled_cost_matrix = cost_matrix
        sample_indices = sample_info['sample_indices']
        
        # Apply sampling to client 1 (rows)
        if "client1" in sample_indices and len(sample_indices["client1"]) < cost_matrix.shape[0]:
            indices1 = torch.from_numpy(sample_indices["client1"]).long()
            sampled_cost_matrix = sampled_cost_matrix[indices1]
            if params['verbose']:
                logger.info(f"Sampled client1 cost matrix rows: {len(indices1)} from original {cost_matrix.shape[0]}")
        
        # Apply sampling to client 2 (columns)
        if "client2" in sample_indices and len(sample_indices["client2"]) < cost_matrix.shape[1]:
            indices2 = torch.from_numpy(sample_indices["client2"]).long()
            sampled_cost_matrix = sampled_cost_matrix[:, indices2]
            if params['verbose']:
                logger.info(f"Sampled client2 cost matrix columns: {len(indices2)} from original {cost_matrix.shape[1]}")
        
        # Compute OT Cost
        ot_cost, transport_plan = compute_ot_cost(
            sampled_cost_matrix, a=a, b=b, reg=params['reg'], 
            sinkhorn_max_iter=params['max_iter'], eps_num=self.eps_num
        )
        
        self.results['direct_ot_cost'] = ot_cost
        self.results['transport_plan'] = None  # Don't store the full plan to save memory
        
        if params['verbose']:
            if np.isfinite(ot_cost):
                logger.info(f"  DirectOT Cost ({self.results['weighting_used']} weights): {ot_cost:.4f}")
            else:
                logger.info(f"  DirectOT Cost ({self.results['weighting_used']} weights): Failed")
                
            # Log label distances if available, regardless of verbosity
            if 'label_costs' in self.results and self.results['label_costs']:
                if params['label_distance'] is not None:
                    logger.info(f"  Label {params['label_distance']} distances by class pairs:")
                    for (label1, label2), dist in self.results['label_costs']:
                        logger.info(f"    Labels ({label1},{label2}): {dist:.4f}")
                else:
                    logger.info(f"  Using only feature distances ({params['feature_distance']})")
        
        return np.isfinite(ot_cost)
    def _calculate_similarity_within_class(self, h1, h2, y1, y2, w1, w2, N, M, params):
        """Calculate similarity with within-class matching."""
        verbose = params['verbose']
        
        # Store per-class results
        per_class_results = []
        total_weighted_ot_cost = 0.0
        total_weight = 0.0
        
        # Ensure labels are available
        if y1 is None or y2 is None:
            logger.warning("Within-class matching requires labels for both clients. Skipping.")
            self.results['weighting_used'] = 'Loss' if params['use_loss_weighting'] else 'Uniform'
            return
                
        # Convert to numpy for easier handling
        y1_np = y1.cpu().numpy()
        y2_np = y2.cpu().numpy()
        
        # Find unique classes in both clients
        unique_classes1 = set(np.unique(y1_np))
        unique_classes2 = set(np.unique(y2_np))
        shared_classes = sorted(unique_classes1.intersection(unique_classes2))
        
        if not shared_classes:
            logger.warning("No shared classes between clients. Cannot compute within-class OT.")
            self.results['direct_ot_cost'] = np.nan
            self.results['weighting_used'] = 'Loss' if params['use_loss_weighting'] else 'Uniform'
            self.results['shared_classes'] = []
            return
                
        if verbose:
            logger.info(f"Processing {len(shared_classes)} shared classes for within-class matching")
                
        # Process each class independently
        for class_label in shared_classes:
            class_result = self._process_single_class(h1, h2, y1, y2, w1, w2, N, M, 
                                                class_label, params)
            
            if class_result is not None:
                # Unpack results
                ot_cost_k, N_k, M_k = class_result
                
                # Determine weight for this class (average proportion of samples)
                weight_k = (N_k/N + M_k/M) / 2
                
                # Aggregate results
                total_weighted_ot_cost += ot_cost_k * weight_k
                total_weight += weight_k
                
                # Store per-class results with Hellinger distance if available
                class_result_dict = {
                    'class': int(class_label),
                    'ot_cost': float(ot_cost_k),
                    'weight': float(weight_k),
                    'samples_c1': int(N_k),
                    'samples_c2': int(M_k)
                }
                
                per_class_results.append(class_result_dict)
                
                if verbose:
                    logger.info(f"Class {class_label}: OT Cost = {ot_cost_k:.4f}, Weight = {weight_k:.4f}")
        
        # Calculate final weighted OT cost
        if total_weight > self.eps_num:
            final_ot_cost = total_weighted_ot_cost / total_weight
        else:
            logger.warning("No valid classes for OT calculation.")
            final_ot_cost = np.nan
                
        # Store results
        weighting_type_str = "Loss-Weighted" if params['use_loss_weighting'] else "Uniform"
        self.results['direct_ot_cost'] = final_ot_cost
        self.results['weighting_used'] = weighting_type_str
        self.results['shared_classes'] = shared_classes
        self.results['per_class_results'] = per_class_results
        

    def _process_single_class(self, h1, h2, y1, y2, w1, w2, N, M, class_label, params):
        """Process a single class for within-class OT calculation."""
        verbose = params['verbose']
        min_samples_threshold = params['min_samples_threshold']
        
        # Find indices for this class
        idx1_k = torch.where(y1 == class_label)[0]
        idx2_k = torch.where(y2 == class_label)[0]
        
        N_k = len(idx1_k)
        M_k = len(idx2_k)
        
        # Check if we have enough samples for this class
        if N_k < min_samples_threshold or M_k < min_samples_threshold:
            if verbose:
                logger.info(f"Class {class_label}: Not enough samples (C1:{N_k}, C2:{M_k}). Skipping.")
            return None
                
        # Extract class-specific data
        h1_k = h1[idx1_k]
        h2_k = h2[idx2_k]
        
        # Extract weights if using loss weighting
        w1_k = w1[idx1_k] if w1 is not None else None
        w2_k = w2[idx2_k] if w2 is not None else None
        
        # Normalize activations class-wise if requested
        h1_k_norm, h2_k_norm = self._normalize_activations(h1_k, h2_k, params['normalize_activations'])
        
        # Compute feature cost matrix for this class
        feature_cost_matrix_k, max_feature_cost_k = self._calculate_feature_cost(
            h1_k_norm, h2_k_norm, params['feature_distance'], params['normalize_activations']
        )
        
        if feature_cost_matrix_k is None:
            logger.warning(f"Failed to compute feature cost matrix for class {class_label}.")
            return None
                
        # Normalize feature cost matrix if requested
        feature_cost_matrix_k = normalize_cost_matrix(
            feature_cost_matrix_k, max_feature_cost_k, params['normalize_cost'], self.eps_num
        )
        
        # Initialize the final cost matrix with feature costs
        cost_matrix_k = feature_cost_matrix_k.clone()
        
        # Calculate label distance if requested for any distance method
        if params['label_distance'] is not None:
            # Convert to numpy for easier handling
            h1_k_np = h1_k.cpu().numpy()
            h2_k_np = h2_k.cpu().numpy()
            
            # Check if compression is needed
            vector_dim = h1_k_np.shape[1]
            was_compressed = False
            if params['compress_vectors'] and vector_dim >= params['compression_threshold']:
                if verbose:
                    logger.info(f"Class {class_label}: Compressing vectors from dimension {vector_dim} for {params['label_distance']} calculation")
                h1_comp, h2_comp = self._compress_vectors(h1_k_np, h2_k_np, params['compression_ratio'])
                was_compressed = True
            else:
                h1_comp, h2_comp = h1_k_np, h2_k_np
            
            # Get stats for each class
            stats1 = self._get_label_stats(h1_comp, np.zeros(h1_comp.shape[0]))  # All samples have the same class
            stats2 = self._get_label_stats(h2_comp, np.zeros(h2_comp.shape[0]))
            
            # Get the distance function
            distance_method_name = self.LABEL_DISTANCE_FUNCTIONS.get(params['label_distance'])
            if distance_method_name is None:
                logger.warning(f"Unknown label distance method '{params['label_distance']}'. Using Hellinger as fallback.")
                distance_method_name = self.LABEL_DISTANCE_FUNCTIONS['hellinger']
            
            # Get actual function reference from method name
            distance_func = getattr(self, distance_method_name)
            
            # Calculate distance if we have stats for both clients
            label_distance_k = None
            if 0 in stats1 and 0 in stats2:
                mu_1, sigma_1 = stats1[0]
                mu_2, sigma_2 = stats2[0]
                label_distance_k = distance_func(mu_1, sigma_1, mu_2, sigma_2)
            
            # If valid distance, combine with feature cost
            if label_distance_k is not None and np.isfinite(label_distance_k):
                # Normalize weights
                total_weight = params['feature_weight'] + params['label_weight']
                norm_feature_weight = params['feature_weight'] / total_weight
                norm_label_weight = params['label_weight'] / total_weight
                
                # Combine costs - multiply by whole matrix to maintain tensor shape
                cost_matrix_k = (norm_feature_weight * cost_matrix_k + 
                                norm_label_weight * label_distance_k)
                
                self.results['label_hellinger_weight'] = norm_label_weight
                # Record the label distance for this class pair
                self.results.setdefault('label_costs', []).append(((class_label, class_label), label_distance_k))
                
                if verbose:
                    compression_str = f"(compressed from {vector_dim})" if was_compressed else ""
                    logger.info(f"Class {class_label}: Combined cost matrix using feature cost (weight {norm_feature_weight:.2f}) "
                            f"and {params['label_distance']} distance {label_distance_k:.4f} {compression_str} (weight {norm_label_weight:.2f})")
        
        # Prepare weights for this class
        weighting_type_str, full_w1_k, full_w2_k = self._prepare_weights(
            w1_k, w2_k, N_k, M_k, params['use_loss_weighting']
        )
        
        # Check if sampling is needed within this class
        features_dict_k = {
            "client1": h1_k.cpu().numpy(),
            "client2": h2_k.cpu().numpy()
        }
        _, sample_indices_k = validate_samples_for_ot(
            features_dict_k, 2, params['max_samples_threshold']
        )
        
        # Sample and prepare for OT
        sampled_cost_matrix_k, a_k, b_k, N_k_eff, M_k_eff = self._apply_sampling_and_prepare_marginals(
            cost_matrix_k, full_w1_k, full_w2_k, sample_indices_k, N_k, M_k, verbose
        )
        # Compute OT cost for this class
        ot_cost_k, _ = compute_ot_cost(
            sampled_cost_matrix_k, a=a_k, b=b_k, reg=params['reg'], 
            sinkhorn_max_iter=params['max_iter'], eps_num=self.eps_num
        )
        
        if not np.isfinite(ot_cost_k):
            logger.warning(f"OT cost calculation failed for class {class_label}. Skipping.")
            return None
                
        return ot_cost_k, N_k, M_k


    def _normalize_activations(self, h1, h2, normalize_activations):
        """Normalize activations if requested."""
        if normalize_activations:
            h1_norm = F.normalize(h1.float(), p=2, dim=1, eps=self.eps_num)
            h2_norm = F.normalize(h2.float(), p=2, dim=1, eps=self.eps_num)
        else:
            h1_norm = h1.float()
            h2_norm = h2.float()
        return h1_norm, h2_norm

    def _calculate_feature_cost(self, h1_norm, h2_norm, distance_method, normalize_activations):
        """Calculate feature cost matrix based on distance method."""
        if distance_method == 'euclidean':
            feature_cost_matrix = torch.cdist(h1_norm, h2_norm, p=2)
            max_feature_cost = 2.0 if normalize_activations else float('inf')
        elif distance_method == 'cosine':
            cos_sim = torch.mm(h1_norm, h2_norm.t())
            feature_cost_matrix = 1.0 - cos_sim
            max_feature_cost = 2.0
        elif distance_method == 'squared_euclidean':
            feature_cost_matrix = pairwise_euclidean_sq(h1_norm, h2_norm)
            max_feature_cost = 4.0 if normalize_activations else float('inf')
        else:
            logger.warning(f"Unknown distance method: {distance_method}. Using euclidean.")
            feature_cost_matrix = torch.cdist(h1_norm, h2_norm, p=2)
            max_feature_cost = 2.0 if normalize_activations else float('inf')
            
        return feature_cost_matrix, max_feature_cost

    def _calculate_label_cost_matrix(self, h1, h2, y1, y2, N, M, feature_cost_matrix, params):
        """Calculate label cost matrix using the specified distance method."""
        # Initialize label cost matrix with same shape as feature cost matrix
        label_cost_matrix = torch.zeros_like(feature_cost_matrix)
        
        label_distance = params.get('label_distance')
        verbose = params.get('verbose')
        
        if label_distance is not None and y1 is not None and y2 is not None:
            try:                
                # Convert to numpy for easier handling
                h1_np = h1.cpu().numpy()
                h2_np = h2.cpu().numpy()
                y1_np = y1.cpu().numpy()
                y2_np = y2.cpu().numpy()
                
                # Check if compression is needed for high-dimensional vectors
                vector_dim = h1_np.shape[1]
                if params['compress_vectors'] and vector_dim > params['compression_threshold']:
                    if verbose:
                        logger.info(f"Compressing vectors from dimension {vector_dim} for {label_distance} calculation")
                    h1_comp, h2_comp = self._compress_vectors(h1_np, h2_np, params['compression_ratio'])
                else:
                    h1_comp, h2_comp = h1_np, h2_np
                
                # Get unique labels from both clients
                unique_labels1 = set(np.unique(y1_np))
                unique_labels2 = set(np.unique(y2_np))
                
                # Get the distance function using the strategy pattern
                distance_method_name = self.LABEL_DISTANCE_FUNCTIONS.get(label_distance)
                if distance_method_name is None:
                    logger.warning(f"Unknown label distance method '{label_distance}'. Using Hellinger as fallback.")
                    distance_method_name = self.LABEL_DISTANCE_FUNCTIONS['hellinger']
                
                # Get actual function reference from method name
                distance_func = getattr(self, distance_method_name)
                
                # Compute per-label statistics once (caching)
                stats1 = self._get_label_stats(h1_comp, y1_np)
                stats2 = self._get_label_stats(h2_comp, y2_np)
                
                # Calculate distances for each label pair
                label_pair_distances = {}
                
                for label1 in unique_labels1:
                    for label2 in unique_labels2:
                        if label1 in stats1 and label2 in stats2:  # Check if stats are available
                            # Get cached statistics
                            mu_1, sigma_1 = stats1[label1]
                            mu_2, sigma_2 = stats2[label2]
                            
                            # Calculate distance using the selected method
                            distance = distance_func(mu_1, sigma_1, mu_2, sigma_2)
                            
                            # Apply higher distance for different labels - 
                            # This is currently specific to Hellinger behavior
                            max_distance = 2.0 if label_distance == 'hellinger' else 1e3
                            if label_distance == 'hellinger':
                                if label1 != label2:
                                    if distance is not None:
                                        distance = 4 /(len(unique_labels1) + len(unique_labels2)) + distance
                                    else:
                                        distance = max_distance  # Default high distance for different labels
                                
                            if distance is not None:
                                label_pair_distances[(label1, label2)] = distance
                                # Store for reporting
                                self.results.setdefault('label_costs', []).append(((label1, label2), distance))
                            else:
                                # Use default values if distance calculation failed
                                if label1 != label2:
                                    label_pair_distances[(label1, label2)] = max_distance
                                    self.results.setdefault('label_costs', []).append(((label1, label2), 2))
                                else:
                                    label_pair_distances[(label1, label2)] = max_distance / 4
                                    self.results.setdefault('label_costs', []).append(((label1, label2), max_distance / 4))
                                if verbose:
                                    logger.warning(f"{label_distance} distance calculation failed for labels {label1},{label2}")
                        else:
                            # Not enough samples for distribution, use midpoint
                            if label1 != label2:
                                label_pair_distances[(label1, label2)] = max_distance
                                self.results.setdefault('label_costs', []).append(((label1, label2), max_distance))
                            else:
                                label_pair_distances[(label1, label2)] = max_distance / 4
                                self.results.setdefault('label_costs', []).append(((label1, label2), max_distance / 4))
                            if verbose:
                                logger.info(f"Not enough samples for labels {label1},{label2}.")
                
                # Fill the label cost matrix based on the calculated distances
                for i in range(N):
                    label_i = y1_np[i]
                    for j in range(M):
                        label_j = y2_np[j]
                        pair_key = (label_i, label_j)
                        if pair_key in label_pair_distances:
                            label_cost_matrix[i, j] = label_pair_distances[pair_key]
                
            except Exception as e:
                logger.warning(f"Label distance calculation failed: {e}")
                # Set all label costs to a neutral midpoint value
                label_cost_matrix.fill_(0.5)
        else:
            # Fill with neutral value if no label distance method specified
            label_cost_matrix.fill_(0.5)
                    
        return label_cost_matrix

    def _combine_cost_matrices(self, feature_cost_matrix, label_cost_matrix, use_label_distance: bool, feature_weight, label_weight):
        """Combine feature and label distance cost matrices."""
        if use_label_distance:
            # Normalize weights to sum to 1
            total_weight = feature_weight + label_weight
            norm_feature_weight = feature_weight / total_weight
            norm_label_weight = label_weight / total_weight
            # Combine costs
            combined_cost_matrix = (norm_feature_weight * feature_cost_matrix + 
                                norm_label_weight * label_cost_matrix)

            # Store the combined cost matrix
            self.cost_matrices['combined_cost'] = combined_cost_matrix.cpu().numpy()
            return combined_cost_matrix
        else:
            # Use only feature cost if label cost is not used
            self.cost_matrices['combined_cost'] = feature_cost_matrix.cpu().numpy()
            return feature_cost_matrix

    def _prepare_weights(self, w1, w2, N, M, use_loss_weighting):
        """Prepare weights for OT calculation."""
        if use_loss_weighting and w1 is not None and w2 is not None:
            full_w1 = w1.cpu().numpy()
            full_w2 = w2.cpu().numpy()
            weight_type = "Loss-Weighted"
        else:
            if use_loss_weighting: 
                logger.warning("Loss weighting requested but weights unavailable. Using uniform.")
            full_w1 = np.ones(N, dtype=np.float64) / N
            full_w2 = np.ones(M, dtype=np.float64) / M
            weight_type = "Uniform"
            
        return weight_type, full_w1, full_w2

    def _apply_sampling_and_prepare_marginals(self, cost_matrix, full_w1, full_w2, sample_indices, N, M, verbose):
        """Apply sampling to cost matrix and weights, then prepare marginals."""
        sampled_cost_matrix = cost_matrix
        sampled_w1, sampled_w2 = full_w1, full_w2
        N_eff, M_eff = N, M
        
        # Apply sampling to client 1 (rows)
        if "client1" in sample_indices and len(sample_indices["client1"]) < N:
            indices1 = torch.from_numpy(sample_indices["client1"]).long()
            sampled_cost_matrix = sampled_cost_matrix[indices1]
            sampled_w1 = full_w1[sample_indices["client1"]]
            N_eff = len(indices1)
            if verbose:
                logger.info(f"Sampled client1 cost matrix rows: {N_eff} from original {N}")
        
        # Apply sampling to client 2 (columns)
        if "client2" in sample_indices and len(sample_indices["client2"]) < M:
            indices2 = torch.from_numpy(sample_indices["client2"]).long()
            sampled_cost_matrix = sampled_cost_matrix[:, indices2]
            sampled_w2 = full_w2[sample_indices["client2"]]
            M_eff = len(indices2)
            if verbose:
                logger.info(f"Sampled client2 cost matrix columns: {M_eff} from original {M}")
        
        # Prepare marginals with the sampled weights
        a, b = prepare_ot_marginals(sampled_w1, sampled_w2, N_eff, M_eff, self.eps_num)
        
        return sampled_cost_matrix, a, b, N_eff, M_eff

    def _compress_vectors(self, X1: np.ndarray, X2: np.ndarray, 
                        compression_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress high-dimensional vectors using PCA.
        
        Args:
            X1: First set of vectors
            X2: Second set of vectors
            compression_ratio: Amount of variance to retain (0.0-1.0)
            
        Returns:
            Tuple of compressed vectors (X1_comp, X2_comp)
        """
        # Standardize the data
        scaler = StandardScaler()
        X1_scaled = scaler.fit_transform(X1)
        X2_scaled = scaler.transform(X2)  # Use same scaling for both

        # Apply PCA
        pca = PCA(n_components=compression_ratio)
        X1_comp = pca.fit_transform(X1_scaled)
        X2_comp = pca.transform(X2_scaled)  # Use same transformation for both
        
        return X1_comp, X2_comp

    def _get_label_stats(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Return {label: (μ, Σ)} dictionary of per-label statistics.
        
        Args:
            X: Feature matrix of shape [n_samples, n_features]
            y: Label vector of shape [n_samples]
            
        Returns:
            Dictionary mapping each label to its (mean, covariance) tuple
        """
        stats = {}
        for label in np.unique(y):
            indices = np.where(y == label)[0]
            if len(indices) >= 2:  # need at least 2 samples for covariance
                stats[label] = self._get_normal_params(X[indices])
        return stats
        
    def _get_normal_params(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance matrix for a set of vectors.
        
        Args:
            X: Input vectors (n_samples, n_features)
            
        Returns:
            Tuple of (mean vector, covariance matrix)
        """
        mu = np.mean(X, axis=0)
        sigma = np.cov(X, rowvar=False)
        # Add small regularization to ensure positive definiteness
        sigma += 1e-6 * np.eye(sigma.shape[0])
        return mu, sigma
        
    def _hellinger_distance(self, mu_1: np.ndarray, sigma_1: np.ndarray, 
                        mu_2: np.ndarray, sigma_2: np.ndarray) -> Optional[float]:
        """
        Calculate the Hellinger distance between two multivariate normal distributions.
        """
        try:
            # Add explicit shape checks
            if mu_1.shape != mu_2.shape or sigma_1.shape != sigma_2.shape:
                logger.warning(f"Shape mismatch: mu_1{mu_1.shape}, mu_2{mu_2.shape}, "
                            f"sigma_1{sigma_1.shape}, sigma_2{sigma_2.shape}")
                return None
                
            # Ensure positive definiteness through eigendecomposition
            s1_vals, s1_vecs = np.linalg.eigh(sigma_1)
            s2_vals, s2_vecs = np.linalg.eigh(sigma_2)
            
            # Reconstruct with only positive eigenvalues
            s1_vals = np.maximum(s1_vals, 1e-2)
            s2_vals = np.maximum(s2_vals, 1e-2)
            
            s1_recon = s1_vecs @ np.diag(s1_vals) @ s1_vecs.T
            s2_recon = s2_vecs @ np.diag(s2_vals) @ s2_vecs.T
            
            # Average covariance
            avg_sigma = (s1_recon + s2_recon) / 2
            
            # Calculate determinants
            det_s1 = np.linalg.det(s1_recon)
            det_s2 = np.linalg.det(s2_recon)
            det_avg_sigma = np.linalg.det(avg_sigma)
            # Avoid numerical issues with determinants
            if det_s1 <= 0 or det_s2 <= 0 or det_avg_sigma <= 0:
                return None
                
            # First term: determinant component
            term1 = (np.power(det_s1, 0.25) * np.power(det_s2, 0.25)) / np.sqrt(det_avg_sigma)
            
            # Second term: exponential component with mean difference
            diff_mu = mu_1 - mu_2
            inv_avg_sigma = np.linalg.inv(avg_sigma)
            term2 = np.exp(-0.125 * np.dot(diff_mu, np.dot(inv_avg_sigma, diff_mu)))
            
            # Final Hellinger distance
            distance = 1 - np.sqrt(term1 * term2)
            # Handle numerical issues
            if not np.isfinite(distance):
                return None
                
            return float(distance)
            
        except np.linalg.LinAlgError as e:
            # Handle linear algebra errors (singular matrices, etc)
            logger.warning(f"Linear algebra error in Hellinger calculation: {e}")
            return None
        except IndexError as e:
            # Handle indexing errors explicitly
            logger.warning(f"Indexing error in Hellinger calculation: {e}") 
            return None
        except Exception as e:
            # Handle other errors
            logger.warning(f"Hellinger distance calculation error: {e}")
            return None

    def _wasserstein_gaussian_distance(self, mu_1: np.ndarray, sigma_1: np.ndarray, 
                                    mu_2: np.ndarray, sigma_2: np.ndarray) -> Optional[float]:
        """
        Calculate the 2-Wasserstein distance between two multivariate Gaussian distributions.
        
        The formula is:
        W2^2(α, β) = ||μα-μβ||^2_2 + tr(Σα+Σβ-2(Σ^(1/2)_α Σβ Σ^(1/2)_α)^(1/2))
        
        For commuting covariance matrices, this simplifies to:
        W2^2(α, β) = ||μα-μβ||^2_2 + ||Σ^(1/2)_α - Σ^(1/2)_β||^2_F
        
        Args:
            mu_1: Mean vector of first distribution
            sigma_1: Covariance matrix of first distribution
            mu_2: Mean vector of second distribution
            sigma_2: Covariance matrix of second distribution
            
        Returns:
            Wasserstein distance or None if calculation fails
        """
        try:
            # Add explicit shape checks
            if mu_1.shape != mu_2.shape or sigma_1.shape != sigma_2.shape:
                logger.warning(f"Shape mismatch: mu_1{mu_1.shape}, mu_2{mu_2.shape}, "
                            f"sigma_1{sigma_1.shape}, sigma_2{sigma_2.shape}")
                return None
                    
            
            # Mean term (squared Euclidean distance between means)
            mean_term = np.sum((mu_1 - mu_2) ** 2)
            
            # Check if matrices might approximately commute (much faster calculation)
            # A and B commute if AB = BA
            commutation_error = np.linalg.norm(sigma_1 @ sigma_2 - sigma_2 @ sigma_1)
            matrices_commute = commutation_error < 1e-10
            
            if matrices_commute:
                # Use simplified formula for commuting matrices
                # W2^2 = ||μ1-μ2||^2 + ||Σ1^(1/2) - Σ2^(1/2)||^2_F
                
                # Ensure matrices are positive definite
                sigma1_pd = sigma_1 + 1e-6 * np.eye(sigma_1.shape[0])
                sigma2_pd = sigma_2 + 1e-6 * np.eye(sigma_2.shape[0])
                
                # Calculate matrix square roots
                sqrt_sigma1 = scipy.linalg.sqrtm(sigma1_pd)
                sqrt_sigma2 = scipy.linalg.sqrtm(sigma2_pd)
                
                # Frobenius norm of difference of square roots
                # ||A||_F^2 = Tr(A^T A) = sum of squared elements
                sqrt_diff = sqrt_sigma1 - sqrt_sigma2
                covariance_term = np.sum(sqrt_diff * sqrt_diff.conj())
                
                # Handle potential complex parts from numerical issues
                if np.iscomplexobj(covariance_term):
                    if np.abs(np.imag(covariance_term)).max() < 1e-10:
                        covariance_term = np.real(covariance_term)
                    else:
                        logger.warning("Complex value in Wasserstein distance (commuting case)")
                        return None
            else:
                # Use full formula for non-commuting matrices
                # W2^2 = ||μ1-μ2||^2 + Tr(Σ1 + Σ2 - 2(Σ1^(1/2) Σ2 Σ1^(1/2))^(1/2))
                
                # Ensure matrices are positive definite
                sigma1_pd = sigma_1 + 1e-6 * np.eye(sigma_1.shape[0])
                sigma2_pd = sigma_2 + 1e-6 * np.eye(sigma_2.shape[0])
                
                # Calculate sqrt(Σ1)
                sqrt_sigma1 = scipy.linalg.sqrtm(sigma1_pd)
                
                # Calculate sqrt(Σ1 Σ2 Σ1)
                inner_term = sqrt_sigma1 @ sigma2_pd @ sqrt_sigma1
                inner_sqrt = scipy.linalg.sqrtm(inner_term)
                
                # Full trace term
                covariance_term = np.trace(sigma1_pd) + np.trace(sigma2_pd) - 2 * np.trace(inner_sqrt)
                
                # Handle potential complex parts from numerical issues
                if np.iscomplexobj(covariance_term):
                    if np.abs(np.imag(covariance_term)).max() < 1e-10:
                        covariance_term = np.real(covariance_term)
                    else:
                        logger.warning("Complex value in Wasserstein distance (non-commuting case)")
                        return None
            
            # Combine terms and ensure non-negative
            wasserstein_squared = mean_term + covariance_term
            if wasserstein_squared < 0 and np.abs(wasserstein_squared) < 1e-10:
                # Handle slight negative values due to numerical issues
                wasserstein_squared = 0.0
                
            return float(np.sqrt(max(0, wasserstein_squared)))
                
        except Exception as e:
            logger.warning(f"Error in Wasserstein distance calculation: {e}")
            return None