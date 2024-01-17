from sklearn.preprocessing import StandardScaler
from torch.utils.data  import DataLoader, random_split, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torchvision import transforms
import nibabel as nib
import torchio as tio
from abc import ABC, abstractmethod
from PIL import Image

global ROOT_DIR
ROOT_DIR = ''
DATASET_TYPES_TABULAR = {'Synthetic', 'Credit', 'Weather'}
DATASET_TYPES_IMAGE = {'CIFAR', 'EMNIST', 'IXITiny', 'ISIC'}
CONTINUOUS_OUTCOME = {'Weather'}
LARGE_TEST_SET = {'Synthetic', 'Credit', 'Weather', 'CIFAR', 'EMNIST'}
torch.manual_seed(1)
np.random.seed(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset_handler(dataset_name):
    if dataset_name in DATASET_TYPES_TABULAR:
        return TabularDatasetHandler(dataset_name)
    elif dataset_name in DATASET_TYPES_IMAGE:
        return ImageDatasetHandler(dataset_name)


class AbstractDatasetHandler(ABC):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    @abstractmethod
    def preprocess_data(self, X, y):
        pass

class TabularDatasetHandler(AbstractDatasetHandler):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.scaler = StandardScaler()
        self.scaler_label = StandardScaler()
    
    def preprocess_data(self, data, size, fit_transform = False):
        X, y = data
        if fit_transform:
            ##preprocess on single dataset
            self.scaler.fit(X[:size])
            X_tensor = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            if self.dataset_name in CONTINUOUS_OUTCOME:
                y = y.reshape(-1,1)
                self.scaler_label.fit(y[:size])
                y_tensor = torch.tensor(self.scaler_label.transform(y), dtype=torch.float32)
            else:
                y_tensor = torch.tensor(y, dtype=torch.float32)    
        else:
            X_tensor = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
            if self.dataset_name in CONTINUOUS_OUTCOME:
                y = y.reshape(-1,1)
                y_tensor = torch.tensor(self.scaler_label.transform(y), dtype=torch.float32)   
            else:
                y_tensor = torch.tensor(y, dtype=torch.float32)    
        return TensorDataset(X_tensor, y_tensor)
    
class ImageDatasetHandler(AbstractDatasetHandler):
    def preprocess_data(self, data, size = None, fit_transform = False):
        X, y = data
        if self.dataset_name in ['EMNIST', 'CIFAR']:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            if self.dataset_name == 'EMNIST':
                    if fit_transform:
                       transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)), 
                                            transforms.Resize((28, 28)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,)) ])
                    else:
                        transform = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((28, 28)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,)) ])
            
            elif self.dataset_name == 'CIFAR':
                    if fit_transform:
                        transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    else:
                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            X_tensor = torch.stack([transform(image) for image in X_tensor])
            return TensorDataset(X_tensor, y_tensor)
        elif self.dataset_name in ['IXITiny']:
            return IXITinyDataset(data)
        elif self.dataset_name in ['ISIC']:
            return ISICDataset(data)
            
class IXITinyDataset(Dataset):
    def __init__(self, data, size = None, transform=None):
        image_paths, label_paths = data
        self.image_paths = image_paths
        landmarks = tio.HistogramStandardization.train(
                        image_paths,
                        output_path=f'{ROOT_DIR}/data/IXITiny/landmarks.npy')
        self.label_paths = label_paths
        self.transform_image = tio.Compose([
                            tio.ToCanonical(),
                            tio.Resample(4),
                            tio.CropOrPad((48, 60, 48)),
                            #tio.HistogramStandardization({'mri': landmarks}),
                            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                            tio.OneHot()
                        ])
        
        self.transform_label = tio.Compose([
                            tio.ToCanonical(),
                            tio.Resample(4),
                            tio.CropOrPad((48, 60, 48)),
                            tio.OneHot()
                        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        image = torch.tensor(nib.load(image_path).get_fdata(), dtype=torch.float).unsqueeze(0)
        label = torch.tensor(nib.load(label_path).get_fdata(), dtype=torch.float).unsqueeze(0)

        image = self.transform_image(image)
        label = self.transform_label(label)
        return image, label
    
class ISICDataset(Dataset):
    def __init__(self, data, size = None, transform=None):
        image_paths, labels = data
        sz = 200
        mean=(0.585, 0.500, 0.486)
        std=(0.229, 0.224, 0.225)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else transforms.Compose([transforms.ToTensor(),
                transforms.CenterCrop(sz),
                transforms.Normalize(mean, std)
                ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype = torch.int64)
        return image, label
    

class DataPreprocessor:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.handler = get_dataset_handler(self.dataset)

    def preprocess(self, X, y):
        train_data, val_data, test_data, size = self.split(X, y) 
        return self.create_dataloaders(train_data, val_data, test_data, size)
    
    def preprocess_joint(self, X1, y1, X2, y2):
        train_data, val_data, test_data, size = self.split_joint(X1, y1, X2, y2) 
        return self.create_dataloaders(train_data, val_data, test_data, size)

    def split(self, X, y, test_size=0.2, val_size = 0.2):
        if self.dataset in LARGE_TEST_SET :
            test_size = 0.6 #Make large test set so test metrics are more stable (larger sample size added)
        else:
            test_size = 0.2
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size = test_size, random_state=np.random.RandomState(42))
        X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size = val_size, random_state=np.random.RandomState(42))
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), X_train.shape[0]
    
    def split_joint(self, X1, y1, X2, y2):
        (X_train1, y_train1), (X_val1, y_val1), (X_test, y_test), size = self.split(X1, y1)
        (X_train2, y_train2), (X_val2, y_val2), (_, _), _ = self.split(X2, y2)
        X_train = np.concatenate((X_train1, X_train2), axis = 0)
        y_train = np.concatenate((y_train1, y_train2), axis = 0)
        X_val = np.concatenate((X_val1, X_val2), axis = 0)
        y_val = np.concatenate((y_val1, y_val2), axis = 0)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), size

    def create_dataloaders(self, train_data, val_data, test_data, size):
        train_data = self.handler.preprocess_data(train_data, size, fit_transform= True)
        val_data = self.handler.preprocess_data(val_data, size, fit_transform= False)
        test_data = self.handler.preprocess_data(test_data, size, fit_transform= False)
        if DEVICE == 'cuda':
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor = 2)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle = False, pin_memory=True, num_workers=4, prefetch_factor = 2)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle = False, pin_memory=True, num_workers=4, prefetch_factor = 2)
        else:
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle = False)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle = False)
        return train_loader, val_loader, test_loader