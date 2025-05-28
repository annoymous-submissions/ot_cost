# data_sets.py
"""
Defines final Dataset classes for use with PyTorch DataLoaders.
Each class takes raw, client-specific, split data (NumPy arrays or paths)
and handles appropriate transformations and tensor conversions internally.
Streamlined version: No base classes, no external scaling (except Heart internal),
simplified SyntheticDataset. Assumes necessary libraries are installed.
"""
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from typing import Dict, Tuple, Any, Optional, List, Union, Callable

# Assume necessary libraries are installed
import nibabel as nib
from monai.transforms import (LoadImaged, Resized, NormalizeIntensityd,EnsureChannelFirstd,
                              AsDiscreted, ToTensord, Compose)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

# =============================================================================
# == Final Dataset Wrapper Classes ==
# =============================================================================

class SyntheticDataset(TorchDataset):
    """Final Dataset wrapper for all synthetic data. Converts NumPy to Tensor."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray):
        self.features = X_np
        self.labels = y_np.astype(np.int64)

    def __len__(self) -> int: return len(self.features) # Use features array length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.labels[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor


class CreditDataset(TorchDataset):
    """Final Dataset wrapper for Credit Card data."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray):
        self.features = X_np
        self.labels = y_np.astype(np.int64)

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features[idx]
        label = self.labels[idx]
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor


class HeartDataset(TorchDataset):
    """Final Dataset wrapper for Heart Disease data. Handles internal scaling."""
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray, dataset_config: dict, **kwargs): # Accept kwargs
        self.features_unscaled = X_np
        self.labels = y_np.astype(np.int64)
        source_args = dataset_config.get('source_args', {})
        self.feature_names = source_args.get('feature_names', [])
        self.cols_to_scale = source_args.get('cols_to_scale', [])
        self.scale_values = source_args.get('scale_values', {})
        # Pre-calculate indices for efficiency
        self.scale_indices = [self.feature_names.index(col) for col in self.cols_to_scale if col in self.feature_names and col in self.scale_values]
        self.means = {col: self.scale_values[col][0] for col in self.cols_to_scale if col in self.scale_values}
        self.std_devs = {col: max(np.sqrt(self.scale_values[col][1]), 1e-9) for col in self.cols_to_scale if col in self.scale_values} # Avoid div by zero

    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_unscaled = self.features_unscaled[idx]
        label = self.labels[idx]
        feature_scaled = feature_unscaled.copy().astype(np.float32)
        for col_idx in self.scale_indices:
            col_name = self.feature_names[col_idx]
            feature_scaled[col_idx] = (feature_scaled[col_idx] - self.means[col_name]) / self.std_devs[col_name]

        feature_tensor = torch.tensor(feature_scaled, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feature_tensor, label_tensor


class _BaseImgDS(TorchDataset):
    MEAN_STD   = None          # override
    TRAIN_AUG  = None          # list[transform]  – override
    RESIZE_TO  = None          # (H,W) – EMNIST only

    def __init__(self,
                 split_type        : str,
                 X_np              : np.ndarray = None,
                 y_np              : np.ndarray = None,
                 base_tv_dataset   = None,
                 indices           = None,
                 **trans_args):
        self.is_train  = split_type == "train"
        self.trans_args = trans_args
        if X_np is not None:                      # -------- NumPy path
            self.mode = "numpy"
            self.images = X_np
            self.labels = y_np.astype(np.int64)
        elif base_tv_dataset is not None:         # -------- torchvision path
            self.mode   = "tv"
            self.base   = base_tv_dataset
            self.indices = indices
        else:
            raise ValueError("Provide either (X_np,y_np) or (base_tv_dataset,indices)")

        self.transform = self._build_transform()

    # -----------------------------------------------------------
    def _build_transform(self):
        mean, std = self.MEAN_STD
        t = []

        # add ToPIL *only* for NumPy / tensor inputs
        if self.mode == "numpy":
            t.append(transforms.ToPILImage())

        z = self.trans_args.get('zoom', 0.0)      # e.g. +0.2 → 1.2×, −0.2 → 0.8×
        if abs(z) > 1e-3:
            scale = 1.0 + z
            # Replace Lambda with picklable DeterministicAffineZoom
            t.append(DeterministicAffineZoom(scale_factor=scale, fill_value=100))

        if self.RESIZE_TO is not None:
            t.append(transforms.Resize(self.RESIZE_TO))

        r = self.trans_args.get('angle', 0.0)
        if abs(r) > 1e-6:
            t.append(transforms.RandomAffine(
                    degrees=(r, r)))
            
        # --- frequency filter -------------------------------------------------
        f = self.trans_args.get('frequency', 0.0)
        if abs(f) > 1e-3:
            # Replace Lambda with appropriate picklable transform class
            # We'll use a dataset-specific check here
            if isinstance(self, EMNISTDataset):
                t.append(EMNISTFrequencyFilter(delta=f))
            elif isinstance(self, CIFARDataset):
                t.append(CIFARImageTransform(delta=f))

        if self.is_train and self.TRAIN_AUG:
            t.extend(self.TRAIN_AUG)

        t.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return transforms.Compose(t)

    # -----------------------------------------------------------
    def __len__(self):
        return len(self.images) if self.mode == "numpy" else len(self.indices)

    def __getitem__(self, idx):
        if self.mode == "numpy":
            img, label = self.images[idx], int(self.labels[idx])
        else:  # torchvision
            base_idx = self.indices[idx]
            img, label = self.base[base_idx]

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# Add these top-level callable transform classes at the beginning of the file
class DeterministicAffineZoom(torch.nn.Module):
    """Applies a deterministic zoom using TF.affine."""
    def __init__(self, scale_factor: float, fill_value: int = 100):
        super().__init__()
        self.scale_factor = scale_factor
        self.fill_value = fill_value # Can be a single int or a tuple for RGB

    def forward(self, img: Image.Image) -> Image.Image:
        return TF.affine(img, angle=0, translate=(0, 0),
                         scale=self.scale_factor, shear=(0, 0), fill=self.fill_value)


class EMNISTFrequencyFilter(torch.nn.Module):
    """Applies the EMNIST-specific frequency filter."""
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, img: Image.Image) -> Image.Image:
        return EMNISTDataset.freq_filter_static(img, self.delta)


class CIFARImageTransform(torch.nn.Module):
    """Applies CIFAR-specific image transformations."""
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, img: Image.Image) -> Image.Image:
        img = CIFARDataset.color_jitter_filter_static(img, self.delta)
        img = CIFARDataset.shear_filter_static(img, self.delta)
        return img

# ------------------------ concrete datasets --------------------
class EMNISTDataset(_BaseImgDS):
    MEAN_STD  = ((0.1307,), (0.3081,))
    TRAIN_AUG = [transforms.RandomRotation((-0, 0))]
    RESIZE_TO = (28, 28)
    
    def img_transform(self, img: Image.Image, delta: float) -> Image.Image:
        return self.freq_filter(img, delta)

    def freq_filter(self, img: Image.Image, delta: float) -> Image.Image:
        # Use the static version for implementation
        return EMNISTDataset.freq_filter_static(img, delta)
    
    @staticmethod
    def freq_filter_static(img: Image.Image, delta: float) -> Image.Image:
        # delta>0 high-pass, delta<0 low-pass
        if abs(delta) < 1e-3:
            return img
        arr = np.array(img, np.float32)
        fft = np.fft.fftshift(np.fft.fft2(arr, axes=(0,1)))
        h, w = arr.shape[:2]; r = int(min(h,w)*0.1*abs(delta))
        y,x = np.ogrid[-h//2:h//2, -w//2:w//2]
        mask = x**2+y**2 <= r*r
        if delta>0:  fft[mask] *= 0.3          # high-pass
        else:        fft[~mask] *= 0.3         # low-pass
        filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fft)))
        return Image.fromarray(filtered.clip(0,255).astype(np.uint8))


class CIFARDataset(_BaseImgDS):
    MEAN_STD  = ((0.4914, 0.4822, 0.4465),
                 (0.2470, 0.2435, 0.2616))
    TRAIN_AUG = [transforms.RandomCrop(32, padding=4, padding_mode="reflect")]
    RESIZE_TO = None
    
    def img_transform(self, img: Image.Image, delta: float) -> Image.Image:
        # Keep original function, but use static implementations
        return self.shear_filter(self.color_jitter_filter(img, delta), delta)
    
    def color_jitter_filter(self, img: Image.Image, delta: float) -> Image.Image:
        # Use static implementation
        return CIFARDataset.color_jitter_filter_static(img, delta)
    
    @staticmethod
    def color_jitter_filter_static(img: Image.Image, delta: float) -> Image.Image:
        """
        Apply a deterministic color adjustment based on delta.
        Aims to simulate parts of ColorJitter's behavior deterministically.
        """
        if abs(delta) < 1e-3: # no-op for tiny delta
            return img

        # Define maximum impact scales for each parameter when |delta|=1.
        BRIGHTNESS_STRENGTH = 0.4
        CONTRAST_STRENGTH = 0.4
        SATURATION_STRENGTH = 0.4
        HUE_STRENGTH = 0.2

        # Calculate deterministic adjustment factors
        brightness_factor = max(0.0, 1.0 + delta * BRIGHTNESS_STRENGTH)
        contrast_factor = max(0.0, 1.0 + delta * CONTRAST_STRENGTH)
        saturation_factor = max(0.0, 1.0 + delta * SATURATION_STRENGTH)
        hue_factor = torch.clamp(torch.tensor(delta * HUE_STRENGTH), -0.5, 0.5).item()

        # Apply transformations sequentially
        img_transformed = TF.adjust_brightness(img, brightness_factor)
        img_transformed = TF.adjust_contrast(img_transformed, contrast_factor)
        img_transformed = TF.adjust_saturation(img_transformed, saturation_factor)
        if abs(hue_factor) > 1e-6: # Only apply hue if it's a meaningful shift
            img_transformed = TF.adjust_hue(img_transformed, hue_factor)
        
        return img_transformed
    
    def shear_filter(self, img: Image.Image, delta: float) -> Image.Image:
        # Use static implementation
        return CIFARDataset.shear_filter_static(img, delta)
    
    @staticmethod
    def shear_filter_static(img: Image.Image, delta: float) -> Image.Image:
        """
        Apply a deterministic shear transformation based on a signed delta.
        """
        if abs(delta) < 1e-3: # no-op for tiny delta
            return img

        # --- Tunable Parameters ---
        if delta > 0:
            MAX_SHEAR_X_DEGREES = 0.0 # e.g., up to 20 degrees shear
            MAX_SHEAR_Y_DEGREES = 40.0 # e.g., up to 10 degrees shear
        else:
            MAX_SHEAR_X_DEGREES = 40.0 # e.g., up to 20 degrees shear
            MAX_SHEAR_Y_DEGREES = 00.0 # e.g., up to 10 degrees shear
        # --- End Tunable Parameters ---

        shear_x_degrees = delta * MAX_SHEAR_X_DEGREES
        shear_y_degrees = delta * MAX_SHEAR_Y_DEGREES
        f = 200 + 2 * shear_y_degrees if delta > 0 else 200 + 2 * shear_x_degrees
        img_transformed = TF.affine(img, 
                                angle=0, 
                                translate=(0, 0), 
                                scale=1.0, 
                                shear=(shear_x_degrees, shear_y_degrees),
                                fill = (f,f,f) # fill color for shear
                                )
        
        return img_transformed



class ISICDataset(TorchDataset):
    """Final ISIC dataset wrapper (expects image paths, labels). Uses Albumentations."""
    def __init__(self, image_paths: List[str], labels_np: np.ndarray, split_type: str, dataset_config: dict):
        self.image_paths = image_paths
        self.labels = labels_np.astype(np.int64)
        self.is_train = (split_type == 'train')
        self.sz = dataset_config.get('source_args', {}).get('image_size', 200)
        self.transform = self._get_transform()

    def _get_transform(self) -> Callable:
        """Creates appropriate transforms for training or validation/testing."""     
        mean, std = (0.585, 0.500, 0.486), (0.229, 0.224, 0.225)
        # Transforms applied to both training and validation/test sets at the end
        common_end = [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2()
        ]
        
        if self.is_train:
            # Training-specific augmentations
            aug_list = [
                A.RandomScale(scale_limit=0.07, p=0.5),
                A.Rotate(limit=50, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}, p=0.3),
                A.RandomCrop(height=self.sz, width=self.sz, p=1.0),
                A.CoarseDropout(max_holes=4, max_height=8, max_width=8, p=0.3)
            ]
            # Add the common transforms
            aug_list.extend(common_end)
        else:
            # Validation/test augmentations (simpler)
            aug_list = [
                A.CenterCrop(height=self.sz, width=self.sz, p=1.0),
                *common_end
            ]
        
        return A.Compose(aug_list)


    def __len__(self) -> int: return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.image_paths[idx], self.labels[idx]
        try:
            image = np.array(Image.open(image_path).convert('RGB'))
            transformed = self.transform(image=image)
            image_tensor = transformed['image'].float()
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor
        except Exception as e: raise RuntimeError(f"Failed processing ISIC sample {idx}: {image_path}") from e


class IXITinyDataset(TorchDataset):
    """Final IXITiny dataset wrapper (expects image/label paths). Uses MONAI."""
    def __init__(self, image_paths: List[str], label_paths: List[str], split_type: str, dataset_config: dict):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.is_train = (split_type == 'train')
        common_shape = dataset_config.get('source_args', {}).get('image_shape', (80, 48, 48))
        self.transform = self._get_transform(common_shape)

    def _get_transform(self, spatial_size) -> Callable:
        keys = ["image", "label"]
        all_transforms = [
            # 1. Load image and label from paths.
            LoadImaged(keys=keys, image_only=True, ensure_channel_first=False, reader="NibabelReader"),
            EnsureChannelFirstd(keys=keys), # Result: (C, H, W, D) array, C=1 for grayscale

            # 2. Resize both image and label to the common_shape.
            Resized(keys=keys, spatial_size=spatial_size, mode=("bilinear", "nearest")),

            # 3. Normalize the intensity of the image.
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            # 4. Convert the label to a one-hot representation.
            AsDiscreted(keys="label", to_onehot=2),

            # 5. Convert image and label arrays to PyTorch Tensors.
            ToTensord(keys=keys)
        ]
        return Compose(all_transforms)

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_dict = {"image": self.image_paths[idx], "label": self.label_paths[idx]}
        try:
            transformed_dict = self.transform(data_dict)
            image_tensor = transformed_dict['image']
            label_tensor = transformed_dict['label']
            if hasattr(image_tensor, 'as_tensor'): # Check if it's a MetaTensor
                image_tensor = image_tensor.as_tensor()
            if hasattr(label_tensor, 'as_tensor'):
                label_tensor = label_tensor.as_tensor()
            return image_tensor.float(), label_tensor.float()
        except Exception as e: raise RuntimeError(f"Failed processing IXI sample {idx}: {self.image_paths[idx]}") from e

