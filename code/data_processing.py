# data_processing.py
"""
Contains DataManager for orchestrating data loading/partitioning/processing,
and DataPreprocessor for client-level splitting and Dataset instantiation.
Streamlined version with simplified code flow and reduced conditional complexity.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, Tuple, Any, Optional, List, Union, Callable
from sklearn.model_selection import train_test_split
import random
import copy
# Project Imports
from directories import paths
dir_paths = paths()
DATA_DIR = dir_paths.data_dir
from configs import N_WORKERS
from helper import get_parameters_for_dataset
from data_loading import get_loader
from data_partitioning import get_partitioner
from synthetic_data import apply_feature_shift, apply_concept_shift
# Import all final Dataset classes
from data_sets import (SyntheticDataset, CreditDataset, EMNISTDataset,
                      CIFARDataset, ISICDataset, IXITinyDataset)
import re
# =============================================================================
# == Data Splitting Functions (Internal Helpers) ==
# =============================================================================

def _split_data(X: Union[np.ndarray, List], y: Union[np.ndarray, List],
                test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
               ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    """Splits data arrays/lists into train, val, test using simple random split."""
    num_samples = len(X)
    indices = np.arange(num_samples)
    if num_samples < 5: # Handle small datasets
        empty_X = np.array([], dtype=X.dtype).reshape(0, *X.shape[1:]) if isinstance(X, np.ndarray) and X.ndim > 1 else np.array([], dtype=getattr(X, 'dtype', None))
        empty_y = np.array([], dtype=y.dtype) if isinstance(y, np.ndarray) else []
        if isinstance(X, list): empty_X = []
        return (X, y), (empty_X, empty_y), (empty_X, empty_y)

    idx_train_val, idx_test = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 1e-6 else 0.0
    
    if len(idx_train_val) < 2 or relative_val_size <= 0 or relative_val_size >= 1.0:
        idx_train, idx_val = idx_train_val, np.array([], dtype=int)
    else:
        idx_train, idx_val = train_test_split(idx_train_val, test_size=relative_val_size, random_state=seed + 1, shuffle=True)

    if isinstance(X, np.ndarray):
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
        X_test, y_test = X[idx_test], y[idx_test]
    elif isinstance(X, list):
        X_train, X_val, X_test = [X[i] for i in idx_train], [X[i] for i in idx_val], [X[i] for i in idx_test]
        if isinstance(y, np.ndarray): 
            y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
        else: 
            y_train, y_val, y_test = [y[i] for i in idx_train], [y[i] for i in idx_val], [y[i] for i in idx_test]
    else: 
        raise TypeError(f"Unsupported data type for splitting: {type(X)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _split_indices(indices: List[int], test_size: float = 0.2, val_size: float = 0.2, seed: int = 42
                  ) -> Tuple[List[int], List[int], List[int]]:
    """Splits a list of indices into train, validation, and test index lists."""
    num_samples = len(indices)
    np_indices = np.array(indices)
    
    if num_samples < 5: 
        return indices, [], [] 
        
    idx_train_val, idx_test = train_test_split(np_indices, test_size=test_size, random_state=seed, shuffle=True)
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 1e-6 else 0.0
    
    if len(idx_train_val) < 2 or relative_val_size <= 0 or relative_val_size >= 1.0:
        idx_train, idx_val = idx_train_val, np.array([], dtype=int)
    else:
        idx_train, idx_val = train_test_split(idx_train_val, test_size=relative_val_size, random_state=seed + 1, shuffle=True)
        
    return idx_train.tolist(), idx_val.tolist(), idx_test.tolist()

def _extract_patient_id_from_filename(filename):
    """Extract patient ID from IXI filename patterns like 'IXI002-...' -> 2.
    CAN BE EXTENDED FOR OTHER FORMATS."""
    match = re.search(r'IXI0*(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None
# =============================================================================
# == Data Preprocessor Class (Client-Level Processing) ==
# =============================================================================
class DataPreprocessor:
    """Handles client-level train/val/test splitting and Dataset instantiation."""
    def __init__(self, dataset_config: dict, batch_size: int):
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.dataset_name = dataset_config.get('dataset_name', 'UnknownDataset')
        self.dataset_class_name = self.dataset_config.get('dataset_class')
        if not self.dataset_class_name:
             raise ValueError(f"Config for '{self.dataset_name}' missing 'dataset_class'.")

        
    def _get_dataset_instance(self, data_args: dict, split_type: str, **extra_kwargs):
        """Instantiates the correct Dataset class based on config name."""
        common_args = {'split_type': split_type, 'dataset_config': self.dataset_config}
        common_args.update(extra_kwargs)
        try:
            # Dataset class mapping
            dataset_classes = {
                'SyntheticDataset': SyntheticDataset,
                'CreditDataset': CreditDataset,
                'EMNISTDataset': EMNISTDataset,
                'CIFARDataset': CIFARDataset,
                'ISICDataset': ISICDataset,
                'IXITinyDataset': IXITinyDataset
            }
            
            # Get the right dataset class constructor
            dataset_class = dataset_classes.get(self.dataset_class_name)
            if dataset_class is None:
                raise ValueError(f"Dataset class '{self.dataset_class_name}' not supported")
                
            # Create dataset with appropriate args
            if self.dataset_class_name in ['SyntheticDataset', 'CreditDataset']:
                return dataset_class(**data_args)
            else:
                return dataset_class(**data_args, **common_args)
                
        except Exception as e:
             print(f"Error instantiating Dataset '{self.dataset_class_name}' for split '{split_type}': {e}")
             return None

    def preprocess_client_data(self, client_id: str, client_data_bundle: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Processes raw data bundle for one client into DataLoaders."""
        # Extract data type and content from bundle
        data_type = client_data_bundle.get('type')
        raw_data = client_data_bundle.get('data')
        client_dataset_args = {k.split('-')[-1]: v for k, v in client_data_bundle.items() if k not in ['type', 'data'] and client_id in k}
        
        # Special handling for pre-split paths (IXITiny with fixed splits)
        if data_type == 'pre_split_paths':
            train_data = raw_data.get('train')
            test_data = raw_data.get('test')
            
            # Further split train into train/val
            val_size = self.dataset_config.get('validation_from_train_size', 0.2)
            train_paths, train_labels = train_data
            
            if len(train_paths) > 5:  # Minimum size for meaningful split
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    train_paths, train_labels, test_size=val_size,
                )
                val_data = (val_paths, val_labels)
            else:
                # Not enough training data for validation split
                val_data = ([], [])
            # Create datasets
            train_dataset = self._get_dataset_instance({'image_paths': train_paths, 'label_paths': train_labels}, 'train') if train_paths else None
            val_dataset = self._get_dataset_instance({'image_paths': val_paths, 'label_paths': val_labels}, 'val') if val_data[0] else None
            test_dataset = self._get_dataset_instance({'image_paths': test_data[0], 'label_paths': test_data[1]}, 'test') if test_data[0] else None
        
        
        if data_type == 'subset':
            
            indices, base_data = raw_data.get('indices', []), raw_data.get('base_data')
            train_indices, val_indices, test_indices = _split_indices(indices)
            
            # Handle torch dataset subset
            if isinstance(base_data, torch.utils.data.Dataset):
                train_dataset = Subset(base_data, train_indices) if train_indices else None
                val_dataset = Subset(base_data, val_indices) if val_indices else None
                test_dataset = Subset(base_data, test_indices) if test_indices else None
            
            # Handle tuple of torch datasets (e.g., torchvision train/test datasets)
            elif isinstance(base_data, tuple) and all(isinstance(d, torch.utils.data.Dataset) for d in base_data):
                # Use the first dataset (typically train) for all splits
                angle = client_dataset_args.get('rotation_angle', 0.0)
                zoom = client_dataset_args.get('zoom', 0.0)
                freq = client_dataset_args.get('frequency', 0.0)
                trans_args = {'angle':angle, 'zoom':zoom, 'frequency':freq}
                train_dataset = self._get_dataset_instance({'base_tv_dataset': base_data[0],'indices' :train_indices},
                                                            'train', **trans_args)
                val_dataset   =self._get_dataset_instance({'base_tv_dataset': base_data[0],'indices' :val_indices},
                                                           'val', **trans_args)
                test_dataset  = self._get_dataset_instance({'base_tv_dataset': base_data[0],'indices' :test_indices}, 
                                                           'test', **trans_args)
            
            elif isinstance(base_data, tuple) and len(base_data) == 2 and isinstance(base_data[0], np.ndarray):
                base_X, base_y = base_data
                train_data = (base_X[train_indices], base_y[train_indices]) if train_indices else None
                val_data = (base_X[val_indices], base_y[val_indices]) if val_indices else None
                test_data = (base_X[test_indices], base_y[test_indices]) if test_indices else None
                
                # Create actual datasets
                train_dataset = self._get_dataset_instance({'X_np': train_data[0], 'y_np': train_data[1]}, 'train') if train_data else None
                val_dataset = self._get_dataset_instance({'X_np': val_data[0], 'y_np': val_data[1]}, 'val') if val_data else None
                test_dataset = self._get_dataset_instance({'X_np': test_data[0], 'y_np': test_data[1]}, 'test') if test_data else None
                
        elif data_type == 'direct':
            X, y = raw_data.get('X'), raw_data.get('y')
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = _split_data(X, y)
            
            # Create datasets
            train_dataset = self._get_dataset_instance({'X_np': X_train, 'y_np': y_train}, 'train') if len(X_train) > 0 else None
            val_dataset = self._get_dataset_instance({'X_np': X_val, 'y_np': y_val}, 'val') if len(X_val) > 0 else None
            test_dataset = self._get_dataset_instance({'X_np': X_test, 'y_np': y_test}, 'test') if len(X_test) > 0 else None
            
        elif data_type == 'paths':
            X_paths, y_data = raw_data.get('X_paths'), raw_data.get('y_data')
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = _split_data(X_paths, y_data)
            
            # Determine if we're dealing with paths for both X and y
            is_y_paths = isinstance(y_data, list)
            y_key = 'label_paths' if is_y_paths else 'labels_np'
            
            # Create datasets
            train_args = {'image_paths': X_train, y_key: y_train}
            val_args = {'image_paths': X_val, y_key: y_val}
            test_args = {'image_paths': X_test, y_key: y_test}
            
            train_dataset = self._get_dataset_instance(train_args, 'train',) if len(X_train) > 0 else None
            val_dataset = self._get_dataset_instance(val_args, 'val',) if len(X_val) > 0 else None
            test_dataset = self._get_dataset_instance(test_args, 'test') if len(X_test) > 0 else None


        # Create DataLoaders with appropriate settings
        g_train = torch.Generator()
        g_train.manual_seed(int(torch.initial_seed()))
        n_workers = self.dataset_config.get('num_workers', N_WORKERS)
        if n_workers <= 1:
            pin_mem = False   
        else:
            pin_mem = True
        persistent_work = n_workers > 0
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,  generator=g_train,
            num_workers=n_workers, pin_memory=pin_mem, drop_last=False,
            persistent_workers=persistent_work) if train_dataset else DataLoader([])
            
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=n_workers, pin_memory=pin_mem,
            persistent_workers=persistent_work) if val_dataset else DataLoader([])
            
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=n_workers, pin_memory=pin_mem,
            persistent_workers=persistent_work) if test_dataset else DataLoader([])
        
        return train_loader, val_loader, test_loader

# =============================================================================
# == Data Manager Class (Orchestrator) ==
# =============================================================================
class DataManager:
    """Orchestrates dataset loading, partitioning, and preprocessing."""
    def __init__(self, dataset_name: str, base_seed: int, data_dir_root: str):
        self.dataset_name = dataset_name
        self.base_seed = base_seed
        self.data_dir_root = data_dir_root
        self.config = get_parameters_for_dataset(dataset_name)
        self.data_source_type = self.config['data_source']
        self.partitioning_strategy = self.config['partitioning_strategy']
        self.batch_size = self.config['batch_size']
        self.py_random_sampler = random.Random(base_seed + 100)

    def _create_client_bundle(self, loaded_data):
        """Helper to convert loaded data to a standardized client bundle format."""
        # Create appropriate bundle based on data type
        if isinstance(loaded_data[0], list):
            bundle = {'type': 'paths', 'data': {'X_paths': loaded_data[0], 'y_data': loaded_data[1]}}
        elif isinstance(loaded_data[0], np.ndarray):
            bundle = {'type': 'direct', 'data': {'X': loaded_data[0], 'y': loaded_data[1]}}
        elif isinstance(loaded_data[0], torch.utils.data.Dataset):
            bundle = {'type': 'torchvision_raw', 'data': loaded_data}
        else:
            print(f"Warning: Could not determine bundle type for data: {type(loaded_data[0])}")
            return None
        return bundle
    
    def _prepare_loader_args(self, cost, client_num=None, num_clients=None):
        """Prepare common loader arguments with dataset-specific adjustments."""
        args = {'source_args': self.config.get('source_args', {}), 'cost_key': cost, 'base_seed' : self.base_seed}
        # Add dataset-specific arguments
        if self.data_source_type == 'synthetic':
            args['dataset_name'] = self.dataset_name
            if num_clients is not None:
                args['num_clients'] = num_clients
        elif self.data_source_type == 'torchvision':
            args['data_dir'] = self.data_dir_root
        else:
            args['data_dir'] = os.path.join(self.data_dir_root, self.dataset_name)
            
        # Add client number if specified
        if client_num is not None:
            args['client_num'] = client_num
        return args

    def _extract_partitioner_input(self, data):
        """Extract data needed for the partitioner based on the data format."""
        # For torchvision datasets
        data_type = type(data[0]).__module__
        if data_type.startswith('torchvision.datasets'):
        # Handle both tensor and list targets
            tr_data = data[0]
            if isinstance(tr_data.targets, torch.Tensor):
                partitioner_input = tr_data.targets.numpy()
            else:
                partitioner_input = np.array(tr_data.targets) 
            return partitioner_input, len(tr_data)
        # For tuple data (features, labels)
        elif isinstance(data, tuple) and len(data) == 2:
            return data[1], len(data[0])  # labels, num_samples
            
        raise ValueError(f"Cannot extract partitioner input from data type: {type(data)}")
    
    def _sample_data_for_client(self,
                                data_or_indices: Union[Tuple[np.ndarray, np.ndarray], Tuple[List[str], Union[np.ndarray, List[str]]], List[int]],
                                target_size: Optional[int]
                               ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[List[str], Union[np.ndarray, List[str]]], List[int]]:
        if target_size is None or target_size <= 0:
            return data_or_indices # No sampling requested

        current_size = 0
        is_indices = False
        is_numpy = False
        is_paths = False
        # Determine type and current size
        if isinstance(data_or_indices, list) and all(isinstance(i, int) for i in data_or_indices):
            is_indices = True
            current_size = len(data_or_indices)
        elif isinstance(data_or_indices, tuple) and len(data_or_indices) == 2:
            if isinstance(data_or_indices[0], np.ndarray):
                is_numpy = True
                current_size = len(data_or_indices[0])
            elif isinstance(data_or_indices[0], list):
                is_paths = True
                current_size = len(data_or_indices[0])

        # Perform sampling if needed
        if current_size > target_size:
            # Use the instance's sampler for reproducibility across calls within a run
            sampled_indices = self.py_random_sampler.sample(range(current_size), target_size)
            sampled_indices.sort() # Optional: Keep original order for numpy/paths

            if is_indices:
                original_indices = np.array(data_or_indices) # Convert to numpy for easy indexing
                return original_indices[sampled_indices].tolist()
            elif is_numpy:
                X, y = data_or_indices
                return X[sampled_indices], y[sampled_indices]
            elif is_paths:
                X_paths, y_data = data_or_indices
                X_sampled = [X_paths[i] for i in sampled_indices]
                y_sampled = [y_data[i] for i in sampled_indices] if isinstance(y_data, list) else y_data[sampled_indices]
                return X_sampled, y_sampled
            else:
                 print("Warning: Could not determine data type for sampling.")
                 return data_or_indices # Return original if type unknown
        else:
            # Keep all data if already at or below target size
            return data_or_indices

    def get_dataloaders(self, cost: Any, run_seed: int, num_clients_override: Optional[int] = None
                        ) -> Dict[str, Tuple[DataLoader, DataLoader, DataLoader]]:
        """Gets DataLoaders for all clients for a specific run configuration."""
        num_clients = num_clients_override or self.config.get('default_num_clients', 2)
        loader_func = get_loader(self.data_source_type)
        partitioner_func = get_partitioner(self.partitioning_strategy)
        client_final_data_bundles = {} # Store bundles AFTER potential sampling
        target_samples_per_client = self.config.get('samples_per_client')
        source_args = self.config.get('source_args', {})
        apply_shift = self.config.get("shift_after_split", False)
        
        # --- Seed the instance sampler for this specific run/cost ---
        # Ensures sampling is deterministic per run, even if DataManager is reused
        self.py_random_sampler.seed(run_seed + 101 + hash(str(cost)))

        # Generic handling for datasets with fixed train/test splits
        if self.config.get('fixed_train_test_split', False):
            # Get metadata path - either from a specific library location or data directory
            metadata_path = self.config.get('metadata_path')
            if not metadata_path:
                print(f"Warning: fixed_train_test_split is True but no metadata_path provided for {self.dataset_name}")
                return {}
                
            # Try common locations for the metadata file
            metadata_locations = [
                # Dataset-specific library location if applicable
                os.path.join(os.path.dirname(sys.modules.get(f'flamby.datasets.fed_{self.dataset_name.lower()}', None).__file__), 
                            'metadata', metadata_path) if f'flamby.datasets.fed_{self.dataset_name.lower()}' in sys.modules else None,
                # General data directory
                os.path.join(DATA_DIR, self.dataset_name, metadata_path)
            ]
            
            metadata_file = None
            for location in metadata_locations:
                if location and os.path.exists(location):
                    metadata_file = location
                    break
                    
            if not metadata_file:
                print(f"Warning: Metadata file {metadata_path} not found for {self.dataset_name}")
                return {}
                
            # Load metadata from CSV
            try:
                metadata_df = pd.read_csv(metadata_file)
                id_col = self.config.get('id_column', 'Patient ID')
                split_col = self.config.get('split_column', 'Split')
                
                # Create mapping of ID to train/test split
                id_splits = {row[id_col]: row[split_col] for _, row in metadata_df.iterrows()}
                
                # Handle client site mappings based on the cost parameter
                center_mappings = source_args.get('site_mappings', {}).get(cost, [])
                
                # For each client (center/site)
                for client_idx, center_ids in enumerate(center_mappings):
                    client_id = f"client_{client_idx+1}"
                    
                    # Load data for this client from data loader
                    loader_args = self._prepare_loader_args(cost, client_idx+1, num_clients)
                    loaded_data = loader_func(**loader_args)
                    
                    # Extract file paths and labels based on data format
                    if isinstance(loaded_data[0], list):
                        file_paths, labels = loaded_data
                    else:
                        print(f"Warning: Unsupported data format for fixed split in {self.dataset_name}")
                        continue
                    
                    # Split into train/test based on metadata IDs
                    train_paths, train_labels = [], []
                    test_paths, test_labels = [], []
                    
                    for path, label in zip(file_paths, labels):
                        # Extract ID using a function appropriate for the dataset
                        item_id = _extract_patient_id_from_filename(path)
                        if item_id is not None:
                            split = id_splits.get(item_id, 'train')  # Default to train if not in metadata
                            if split.lower() == 'train':
                                train_paths.append(path)
                                train_labels.append(label)
                            else:  # Assume anything not 'train' is 'test'
                                test_paths.append(path)
                                test_labels.append(label)
                    
                    # Apply sampling if needed
                    train_data = (train_paths, train_labels)
                    test_data = (test_paths, test_labels)
                    
                    sampled_train = self._sample_data_for_client(train_data, target_samples_per_client)
                    
                    # Store in a specialized pre-split bundle format
                    client_final_data_bundles[client_id] = {
                        'type': 'pre_split_paths',
                        'data': {
                            'train': sampled_train,
                            'test': test_data
                        }
                    }
                
                # Process the bundles and return dataloaders
                preprocessor = DataPreprocessor(self.config, self.batch_size)
                client_dataloaders = {}
                
                for client_id, bundle in client_final_data_bundles.items():
                    try:
                        dataloaders = preprocessor.preprocess_client_data(client_id, bundle)
                        if dataloaders[0] and hasattr(dataloaders[0], 'dataset') and len(dataloaders[0].dataset) > 0:
                            client_dataloaders[client_id] = dataloaders
                        else:
                            print(f"Warning: Client {client_id} has no training samples after preprocessing, skipping.")
                    except Exception as e:
                        print(f"Error preprocessing data for client {client_id}: {e}")
                        
                return client_dataloaders
                    
            except Exception as e:
                print(f"Error loading metadata for {self.dataset_name}: {e}")

        base_data_for_partitioning, all_labels = None, None
        if self.partitioning_strategy == 'pre_split':
            #print(f"Loading pre-split data for {num_clients} clients...")
            for client_idx in range(1, num_clients + 1):
                client_id = f"client_{client_idx}"
                try:
                    # 1. Load raw data for the client
                    loader_args = self._prepare_loader_args(cost, client_idx, num_clients)
                    # For Synthetic_Feature/Concept, this now loads base_n_samples with client-specific shift/seed
                    loaded_data = loader_func(**loader_args) # e.g., (X_np, y_np) or (paths, labels)
                    # 2. Sample the loaded data down to target size (if applicable)
                    sampled_data = self._sample_data_for_client(loaded_data, target_samples_per_client)

                    # 3. Create the final bundle from sampled data
                    bundle = self._create_client_bundle(sampled_data)
                    if bundle and sampled_data[0] is not None and len(sampled_data[0]) > 0: # Check if sampling resulted in data
                        client_final_data_bundles[client_id] = bundle
                    else:
                        print(f"Warning: Client {client_id} has no data after loading/sampling.")

                except Exception as e:
                    print(f"Error loading/sampling pre-split data for client {client_id}: {e}")
                    # Optionally continue or raise

        else: # Handle datasets requiring partitioning (Dirichlet, IID)
            #print(f"Loading and partitioning base data ({self.partitioning_strategy})...")
            try:
                # 1. Load base data
                loader_args = self._prepare_loader_args(cost)
                base_data_for_partitioning = loader_func(**loader_args)
                
                # Extract labels correctly ONCE for both partitioning and display
                partitioner_input, num_samples = self._extract_partitioner_input(base_data_for_partitioning)
                all_labels = copy.deepcopy(partitioner_input)
                
                # 2. Run partitioner with the extracted labels
                partitioner_kwargs = {**self.config.get('partitioner_args', {})}
                if self.partitioning_strategy == 'dirichlet_indices':
                    partitioner_kwargs['alpha'] = float(cost)
                client_indices_full = partitioner_func(partitioner_input, num_clients, seed=run_seed, **partitioner_kwargs)
        
                # 3. Sample the *indices* for each client down to target size
                client_indices_final = {}
                for client_id_idx, indices_list in client_indices_full.items():
                    sampled_indices = self._sample_data_for_client(indices_list, target_samples_per_client)
                    if sampled_indices: # Only add if sampling resulted in indices
                        client_indices_final[client_id_idx] = sampled_indices

                # 4. Apply shifts BEFORE creating client bundles
                # -------------------------------- shift-after-split ------------------------
                if apply_shift and base_data_for_partitioning:
                    if isinstance(base_data_for_partitioning[0], np.ndarray):
                        # Unpack the shared pool so we can mutate in-place
                        X_pool, y_pool = base_data_for_partitioning
    
                    for c_idx, idx_list in client_indices_final.items():
                        # Same spacing rule for all datasets
                        symmetric_shift = (2 * c_idx / max(num_clients - 1, 1)) - 1
                        gamma_i = float(cost) * symmetric_shift
                        
                        # Get source arguments from config
                        source_args = self.config.get('source_args', {})
                        
                        if 'feature_shift_kind' in source_args:
                            if source_args.get('feature_shift_kind') in ['mean','scale', 'tilt']:
                                # Determine the number of columns to affect
                                n_features = X_pool.shape[1]
                                cols = source_args.get("feature_shift_cols")
                                
                                # For Credit dataset or if cols is None, calculate based on percentage
                                if cols is None:
                                    cols_percentage = source_args.get("cols_percentage", 0.5)
                                    cols = int(n_features * cols_percentage)
                                
                                # Apply feature shift with standardized parameters
                                X_pool[idx_list] = apply_feature_shift(
                                    X_pool[idx_list],
                                    delta=gamma_i,
                                    kind=source_args.get("feature_shift_kind", "mean"),
                                    cols=cols,
                                    mu=source_args.get("feature_shift_mu", 1.0),
                                    sigma=source_args.get("feature_shift_sigma", 1.5),
                                    rng=np.random.default_rng(self.base_seed),
                                )
                            elif source_args.get('feature_shift_kind') in ['image']:
                                max_angle = source_args.get('max_rotation_angle', 0.0)
                                angle = max_angle * gamma_i
                                source_args[f'client_{c_idx+1}-rotation_angle'] = angle

                                max_zoom = source_args.get('max_zoom', 0.0)
                                zoom = max_zoom * gamma_i
                                source_args[f'client_{c_idx+1}-zoom'] = zoom

                                max_freq = source_args.get('max_frequency', 0.0)
                                freq = max_freq * gamma_i
                                source_args[f'client_{c_idx+1}-frequency'] = freq

                        # Apply concept shift if specifie
                        elif 'concept_label_option' in source_args:
                            # Concept shift remains unchanged
                            y_pool[idx_list] = apply_concept_shift(
                                X_pool[idx_list],
                                gamma=gamma_i,
                                option=source_args.get("concept_label_option", "threshold"),
                                threshold_range_factor=source_args.get("concept_threshold_range_factor", 0.5),
                                label_noise=source_args.get("label_noise", 0.0),
                                base_seed=self.base_seed,
                                label_rule=source_args.get("label_rule", "linear"),
                                rng=np.random.default_rng(self.base_seed)
                            )
                    
                    # Update the base data with shifted values
                    try:
                        base_data_for_partitioning = (X_pool, y_pool)
                        all_labels = y_pool  # Update labels reference
                    except NameError:
                        pass
    
                # ---------------------------------------------------------------------------

                # 5. NOW create client bundles using the shifted data
                for client_id_idx, final_indices in client_indices_final.items():
                    client_id = f"client_{client_id_idx+1}"
                    client_final_data_bundles[client_id] = {
                        'type': 'subset',
                        'data': {'indices': final_indices, 'base_data': base_data_for_partitioning},
                        **source_args
                    }

            except Exception as e:
                print(f"Error partitioning or sampling data: {e}")
                raise

        #print_class_dist(client_final_data_bundles, all_labels)
        
        # --- Process Final Bundles into DataLoaders ---
        preprocessor = DataPreprocessor(self.config, self.batch_size)
        client_dataloaders = {}
        for client_id, bundle in client_final_data_bundles.items():
            try:
                # preprocess_client_data performs the train/val/test split on the (already sampled) data
                dataloaders = preprocessor.preprocess_client_data(client_id, bundle)
                if dataloaders[0] and hasattr(dataloaders[0], 'dataset') and len(dataloaders[0].dataset) > 0:
                    client_dataloaders[client_id] = dataloaders
                else:
                    print(f"Warning: Client {client_id} has no training samples after preprocessing, skipping.")
            except Exception as e:
                print(f"Error preprocessing data for client {client_id}: {e}")

        return client_dataloaders


def print_class_dist(client_final_data_bundles, all_labels=None):
    print("\n--- Client class distribution (after sampling) ---")
    for client_id in sorted(client_final_data_bundles):
        bundle = client_final_data_bundles[client_id]

        # Extract label array y_client for this client
        if bundle.get("type") == "subset":                    
            indices = bundle["data"]["indices"]
            # Use the pre-extracted labels instead of trying to unpack base_data
            if all_labels is not None:
                y_client = all_labels[indices]  
            else:
                # Fallback for pre-split case
                try:
                    _, base_y = bundle["data"]["base_data"]
                    y_client = base_y[indices]
                except Exception as e:
                    print(f"  {client_id}: Unable to extract labels: {e}")
                    continue
        else:                                                 
            # pre-split path remains unchanged
            payload = bundle["data"]                          
            if isinstance(payload, dict):                   
                y_client = payload.get("y", [])
            else:                                            
                _, y_client = payload

        # Build pretty distribution string and print
        uniq, cnts = np.unique(y_client, return_counts=True)
        dist = ", ".join(f"Class {u}: {c}" for u, c in zip(uniq, cnts))
        print(f"  {client_id}: {len(y_client)} samples ({dist})")

    print("-" * 50)