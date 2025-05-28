# configs.py
"""
Central configuration file for the project.
Defines directory paths, constants, default hyperparameters,
data handling configurations, algorithm settings.
Streamlined version.
"""
import os
import torch
from directories import paths

# --- Global Settings ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
#torch.set_num_interop_threads(int(os.environ["OMP_NUM_THREADS"]))

N_WORKERS = 4 

# --- Core Directories ---
dir_paths = paths()
SELECTION_CRITERION_KEY = dir_paths.selection_criterion_key
DATA_DIR = dir_paths.data_dir
# --- Supported Algorithms ---
ALGORITHMS = ['local', 'fedavg', 'fedprox', 'pfedme', 'ditto'] # Add others as implemented
LR_ALGORITHMS = ['local', 'fedavg'] # Add others as implemented
REG_ALOGRITHMS = ['fedprox', 'pfedme', 'ditto'] # Add others as implemented
# --- Supported Datasets ---
DATASETS = [
    'Synthetic_Label', 'Synthetic_Feature', 'Synthetic_Concept',
    'Credit', 'EMNIST', 'CIFAR', 'ISIC', 'IXITiny',
]

# --- Common Configuration for Tabular-like Datasets ---
COMMON_TABULAR_PARAMS = dict(
    fixed_classes=2, # Default for binary classification, override for multiclass
    default_lr = 5e-3,
    learning_rates_try=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    default_reg_param=0.1,
    reg_params_try=[5, 2, 1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6],
    batch_size=32,
    epochs_per_round=3,
    rounds=50,
    rounds_tune_inner=30,
    runs=50,
    runs_tune=5,
    metric='F1', # Default metric for tabular
    base_seed=42,
    samples_per_client=300,
    default_num_clients=5,
    servers_tune_lr=LR_ALGORITHMS,
    servers_tune_reg = REG_ALOGRITHMS,
    partitioner_args={},
    max_parallel_clients=None,
    use_weighted_loss=False, # If True, client should use WeightedCELoss if criterion is CE
    shift_after_split=False,
    activation_extractor_type='hook_based',
    criterion_type="CrossEntropyLoss", # Default criterion
    source_args={}, # For raw data loading parameters
    selection_criterion_key= SELECTION_CRITERION_KEY, # Default for tabular: optimize for scores (F1, etc.)
    selection_criterion_direction_overrides={}, # Empty dict means use defaults based on key name,
    n_workers = 4
)

# --- Common Configuration for Image Datasets ---
COMMON_IMAGE_PARAMS = dict(
    fixed_classes=10, # Default for common image datasets like CIFAR/EMNIST
    default_lr=3e-3,
    learning_rates_try=[5e-3, 1e-3, 5e-4],
    default_reg_param=0.1,
    reg_params_try=[1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5],
    batch_size=96,
    epochs_per_round=3,
    rounds=50,
    rounds_tune_inner=20,
    runs=15,
    runs_tune=3,
    metric='Accuracy', # Default metric for image classification
    base_seed=42,
    default_num_clients=2,
    servers_tune_lr=LR_ALGORITHMS,
    servers_tune_reg = REG_ALOGRITHMS,
    partitioner_args={},
    max_parallel_clients=None,
    use_weighted_loss=False, # If True, client should use WeightedCELoss if criterion is CE
    shift_after_split=True, # Often true for image datasets with augmentation-like shifts
    activation_extractor_type='hook_based',
    criterion_type="CrossEntropyLoss", # Default criterion
    source_args={}, # For raw data loading parameters
    selection_criterion_key= SELECTION_CRITERION_KEY, # Default for image: optimize for accuracy
    selection_criterion_direction_overrides={}, # Empty dict means use defaults based on key name
    n_workers = 4
)


# --- Default Hyperparameters & Data Handling Configuration ---
DEFAULT_PARAMS = {

    # === Unified Synthetic Configurations ===
    'Synthetic_Label': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Label',
        'data_source': 'synthetic',
        'partitioning_strategy': 'dirichlet_indices',
        'dataset_class': 'SyntheticDataset',
        'source_args': {
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.1,
            'random_state': 42,
            'label_rule': 'mlp'
        },
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'Synthetic_Feature': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Feature',
        'data_source': 'synthetic',
        'partitioning_strategy': 'iid_indices',
        'shift_after_split': True,
        'dataset_class': 'SyntheticDataset',
        'source_args': {
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.01,
            'feature_shift_kind': 'mean',
            'feature_shift_cols': 14,
            'feature_shift_mu': 1.0,
            'label_rule': 'mlp',
        },
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'Synthetic_Concept': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Synthetic_Concept',
        'data_source': 'synthetic',
        'partitioning_strategy': 'iid_indices',
        'shift_after_split': True,
        'dataset_class': 'SyntheticDataset',
        'source_args': {
            'base_n_samples': 50000,
            'n_features': 14,
            'label_noise': 0.01,
            'random_state': 42,
            'concept_label_option': 'rotation',
            'concept_threshold_range_factor': 0.5,
            'label_rule': 'linear',
        },
        # criterion_type defaults to "CrossEntropyLoss"
    },

    # === Other Tabular Datasets ===
    'Credit': {
        **COMMON_TABULAR_PARAMS,
        'dataset_name': 'Credit',
        'data_source': 'credit_csv',
        'partitioning_strategy': 'iid_indices',
        'shift_after_split': True,
        'dataset_class': 'CreditDataset',
        'source_args': {
            'csv_path': os.path.join(DATA_DIR, 'Credit', 'creditcard.csv'),
            'drop_cols': ['Time'],
            'feature_shift_kind': 'mean',
            'feature_shift_cols': None,
            'feature_shift_mu': 1.0,
            'feature_shift_sigma': 1.5,
            'cols_percentage': 0.5,
        },
        # criterion_type defaults to "CrossEntropyLoss"
    },
    # === Image Datasets ===
    'CIFAR': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'CIFAR',
        'data_source': 'torchvision',
        'partitioning_strategy': 'iid_indices',
        'dataset_class': 'CIFARDataset',
        'source_args': {
            'dataset_name': 'CIFAR10',
            'feature_shift_kind': 'image',
            'max_rotation_angle': 45.0,
            'max_zoom': 0.3,
            'max_frequency': 1,
        },
        'samples_per_client': 5000,
        'batch_size': 512,
        'fixed_classes': 10,
        'default_lr': 5e-4, 
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'EMNIST': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'EMNIST',
        'data_source': 'torchvision',
        'partitioning_strategy': 'iid_indices',
        'dataset_class': 'EMNISTDataset',
        'source_args': {
            'dataset_name': 'EMNIST',
            'split': 'digits',
            'feature_shift_kind': 'image',
            'max_rotation_angle': 60.0,
            'max_zoom': 0.5,
            'max_frequency': 2,
        },
        'samples_per_client': 1000,
        'fixed_classes': 10, # Already in COMMON_IMAGE_PARAMS, explicit here
        'batch_size': 64,
        'default_lr': 5e-3, 
        # criterion_type defaults to "CrossEntropyLoss"
    },
    'ISIC': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'ISIC',
        'data_source': 'isic_paths',
        'partitioning_strategy': 'pre_split',
        'dataset_class': 'ISICDataset',
        'source_args': {
            'site_mappings': {'bcn_vmole': [[0], [1]], 'bcn_vmod': [[0], [2]], 'bcn_rose': [[0], [3]], 'bcn_msk': [[0], [4]], 'bcn_vienna': [[0], [5]],
                              'vmole_vmod': [[1], [2]], 'vmole_rose': [[1], [3]], 'vmole_msk': [[1], [4]], 'vmole_vienna': [[1], [5]], 'vmod_rose': [[2], [3]],
                              'vmod_msk': [[2], [4]], 'vmod_vienna': [[2], [5]], 'rose_msk': [[3], [4]], 'rose_vienna': [[3], [5]], 'msk_vienna': [[4], [5]]},
            'image_size': 200
        },
        'samples_per_client': 2000,
        'fixed_classes': 8,
        'metric': 'Balanced_accuracy',
        'use_weighted_loss': True, # This flag will be used by client
        'criterion_type': "ISICLoss", # Explicitly set criterion
        'batch_size': 128,
        'default_lr' : 5e-4,
        'selection_criterion_key': SELECTION_CRITERION_KEY,
        
    },
    
    'IXITiny': {
        **COMMON_IMAGE_PARAMS,
        'dataset_name': 'IXITiny',
        'data_source': 'ixi_paths',
        'partitioning_strategy': 'pre_split',
        'dataset_class': 'IXITinyDataset',
        'fixed_classes': None,  # Override for segmentation
        'default_lr': 1e-3,  # Specific learning rate
        'learning_rates_try': [1e-2, 5e-3, 1e-3],  # Custom learning rates
        'batch_size': 4,  # Smaller batch size for 3D volumes
        'rounds': 30,  # Fewer rounds
        'rounds_tune_inner': 10,  # Fewer tuning rounds
        'runs': 10,  # Fewer runs
        'metric': 'DICE',  # Segmentation metric
        'shift_after_split': False,  # Not applicable for pre-split
        'activation_extractor_type': 'rep_vector',  # Different extractor
        'criterion_type': "DiceLoss",  # Segmentation loss
        # Unique parameters for fixed train/test split
        'fixed_train_test_split': True,
        'metadata_path': 'metadata_tiny.csv',
        'id_column': 'Patient ID',
        'split_column': 'Split',
        'validation_from_train_size': 0.2,
        # Override source_args completely
        'source_args': {
            'site_mappings': {
                'guys_hh': [['Guys'], ['HH']],
                'iop_guys': [['IOP'], ['Guys']],
                'iop_hh': [['IOP'], ['HH']],
                'all': [['IOP'], ['HH'], ['Guys']]
            },
            'image_shape': (80, 48, 48)
        },
    },
}
# --- Dataset Costs / Experiment Parameters ---
DATASET_COSTS = {
    'Synthetic_Label': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    'Synthetic_Feature': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'Credit': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'EMNIST': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'CIFAR': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'Synthetic_Concept': [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
    'IXITiny': ['guys_hh', 'iop_guys', 'iop_hh', 'all'], 
    'ISIC': ['bcn_vmole','vmole_vmod', 'vmole_rose', 'vmole_msk', 'vmole_vienna','vmod_rose',], 
    # 'ISIC': ['bcn_vmole', 'bcn_vmod', 'bcn_rose', 'bcn_msk', 'bcn_vienna',
    #          'vmole_vmod', 'vmole_rose', 'vmole_msk', 'vmole_vienna','vmod_rose',
    #          'vmod_msk', 'vmod_vienna','rose_msk', 'rose_vienna','msk_vienna'], # Using string keys
}


DATASET_COSTS = {
    'Synthetic_Label': [1000.0, 10.0, 2.0, 1.0, 0.75, 0.5, 0.2, 0.1],
    'Synthetic_Feature': [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    'Credit': [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    'EMNIST': [0.0, 0.1, 0.25, 0.4, 0.6, 0.7, 0.9],
    'CIFAR': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'Synthetic_Concept': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'IXITiny': ['guys_hh', 'iop_guys', 'iop_hh'],
    'ISIC': ['bcn_vmole','vmole_vmod', 'vmole_rose', 'vmole_msk', 'vmole_vienna','vmod_rose',], 
    # 'ISIC': ['bcn_vmole', 'bcn_vmod', 'bcn_rose', 'bcn_msk', 'bcn_vienna',
    #          'vmole_vmod', 'vmole_rose', 'vmole_msk', 'vmole_vienna','vmod_rose',
    #          'vmod_msk', 'vmod_vienna','rose_msk', 'rose_vienna','msk_vienna'], # Using string keys
}