# data_loading.py
"""
Handles loading initial, raw data from various sources (files, torchvision, generation).
Returns basic data structures like NumPy arrays, lists of paths, or torchvision Dataset objects.
Simplified version with reduced complexity and more consistent pattern handling.
"""
import os
import glob
import hashlib
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10, EMNIST
from torchvision import transforms
from typing import Dict, Tuple, Any, Optional, List, Union, Callable

# Import the synthetic data generator function
from synthetic_data import generate_synthetic_data

# =============================================================================
# == Raw Data Loader Functions ==
# =============================================================================


# data_loading.py
def load_synthetic_raw(dataset_name, source_args, cost_key, base_seed, **_):                   
    X, y = generate_synthetic_data(
        n_samples=source_args['base_n_samples'],
        n_features=source_args['n_features'],
        label_noise=source_args.get('label_noise', 0.0),
        base_seed=base_seed,               # one RNG
        label_rule=source_args.get('label_rule', 'linear')
    )
    return X, y


def load_torchvision_raw(source_args: dict,
                         data_dir: str,
                         base_seed: int,
                         transform_config: Optional[dict] = None, # Keep signature, but ignore
                         cost_key: Optional[Any] = None) -> Tuple[Any, Any]: # Returns raw torchvision dataset objects
    """
    Loads raw torchvision datasets (train/test) WITHOUT applying initial transforms.
    Transforms will be handled later by the final Dataset class.
    """
    tv_dataset_name = source_args.get('dataset_name')
    split = source_args.get('split', 'digits')  # Default for EMNIST

    # Dataset mapping - downloads raw data only
    dataset_mapping = {
        'CIFAR10': lambda: (
            CIFAR10(root=data_dir, train=True, download=True, transform=None), # transform=None
            CIFAR10(root=data_dir, train=False, download=True, transform=None) # transform=None
        ),
        'EMNIST': lambda: (
            EMNIST(root=data_dir, split=split, train=True, download=True, transform=None), # transform=None
            EMNIST(root=data_dir, split=split, train=False, download=True, transform=None) # transform=None
        )
    }

    loader_fn = dataset_mapping.get(tv_dataset_name)
    if loader_fn:
        # Return the tuple of (train_dataset, test_dataset)
        return loader_fn()
    else:
        raise ValueError(f"Torchvision loader not configured for: {tv_dataset_name}")


def load_credit_raw(source_args: dict,
                    cost_key: Any,
                    data_dir: str,
                    base_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Loads raw Credit Card Fraud data from CSV."""
    csv_path = source_args['csv_path']
    df = pd.read_csv(csv_path)
    
    # Drop specified columns
    drop_cols = source_args.get('drop_cols', [])
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    
    # Extract labels and features
    labels_np = df['Class'].values.astype(np.int64)
    features_np = df.drop(columns=['Class']).values.astype(np.float32)
    
    return features_np, labels_np

def load_isic_paths_raw(client_num: int,
                       cost_key: Any,
                       source_args: dict,
                       data_dir: str,
                       base_seed: int) -> Tuple[List[str], np.ndarray]:
    """Loads ISIC image paths and labels for a specific client's assigned site(s)."""
    # Get root directory and site assignments
    root_dir = source_args.get('data_dir', data_dir)
    site_map = source_args['site_mappings']
    site_indices = site_map[cost_key][client_num - 1]
    
    # Ensure site_indices is a list
    if not isinstance(site_indices, list):
        site_indices = [site_indices]
    
    # Image directory path
    img_dir_path = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed')
    img_files_list, labels_list = [], []
    
    # Process each site
    for site_idx in site_indices:
        csv_path = os.path.join(root_dir, f'site_{site_idx}_files_used.csv')
        if not os.path.exists(csv_path):
            continue
            
        try:
            # Load file paths and labels from CSV
            files_df = pd.read_csv(csv_path)
            
            # Create full image paths and get labels
            img_files_list.extend([
                os.path.join(img_dir_path, f"{stem}.jpg") 
                for stem in files_df['image']
            ])
            labels_list.extend(files_df['label'].values)
        except Exception:
            continue
    
    return img_files_list, np.array(labels_list).astype(np.int64)


def load_ixi_paths_raw(client_num: int,
                      cost_key: Any,
                      source_args: dict,
                      data_dir: str,
                      base_seed: int)-> Tuple[List[str], List[str]]:
    """Loads IXI image and label file paths for a specific client's assigned site(s)."""
    # Get root directory and site assignments
    root_dir = source_args.get('data_dir', data_dir)
    sites_map = source_args['site_mappings']
    site_names = sites_map[cost_key][client_num - 1]
    
    # Ensure site_names is a list
    if not isinstance(site_names, list):
        site_names = [site_names]
    
    # Directory paths
    img_dir = os.path.join(root_dir, 'flamby/image')
    lbl_dir = os.path.join(root_dir, 'flamby/label')
    
    # Get all image and label files
    all_img_files = glob.glob(os.path.join(img_dir, '*.nii.gz'))
    all_lbl_files = glob.glob(os.path.join(lbl_dir, '*.nii.gz'))
    # Helper functions to extract ID and site
    def get_file_info(filepath):
        basename = os.path.basename(filepath)
        base_id = basename.split('_')[0]
        # Extract site name
        known_sites = ['Guys', 'HH', 'IOP']
        site = None
        for part in basename.split('.')[0].split('-'):
            if part in known_sites:
                site = part
                break
        return base_id, site
    
    # Filter files by site and create ID-to-path mappings
    image_dict = {}
    label_dict = {}
    
    for img_file in all_img_files:
        img_id, img_site = get_file_info(img_file)
        if img_site in site_names:
            image_dict[img_id] = img_file
            
    for lbl_file in all_lbl_files:
        lbl_id, lbl_site = get_file_info(lbl_file)
        if lbl_site in site_names:
            label_dict[lbl_id] = lbl_file

    # Find common IDs to ensure alignment
    common_ids = sorted(list(set(image_dict.keys()) & set(label_dict.keys())))
    
    # Create aligned lists of image and label files
    aligned_image_files = [image_dict[id_] for id_ in common_ids]
    aligned_label_files = [label_dict[id_] for id_ in common_ids]
    
    return aligned_image_files, aligned_label_files

# =============================================================================
# == Loader Factory ==
# =============================================================================

def get_loader(source_name: str) -> Callable:
    """Factory function to get the appropriate raw data loader function."""
    loaders = {
        'synthetic': load_synthetic_raw,
        'torchvision': load_torchvision_raw,
        'credit_csv': load_credit_raw,
        'isic_paths': load_isic_paths_raw,
        'ixi_paths': load_ixi_paths_raw
    }
    
    loader = loaders.get(source_name)
    if loader:
        return loader
    else:
        raise ValueError(f"Unknown data source name: '{source_name}'")