Synthetic_LR_dict = {0.03:{'single': 5e-3, 'joint': 5e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.10:{'single': 5e-3, 'joint': 5e-2, 'federated': 1e-1, 'pfedme':5e-2, 'ditto':1e-1},
            0.20:{'single': 1e-2, 'joint': 1e-1, 'federated': 1e-1, 'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'single': 5e-2, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.40:{'single': 5e-2, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.50:{'single': 5e-2, 'joint': 3e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1}}

Synthetic_OPTIM_dict = {0.03:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.10:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.20:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.30:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'SGD'},
            0.40:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'SGD'},
            0.50:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'SGD'}}

Credit_LR_dict = {0.12:{'single': 5e-3, 'joint': 5e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.23:{'single': 5e-3, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'single': 1e-2, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.40:{'single': 5e-2, 'joint': 1e-3, 'federated': 5e-3, 'pfedme':5e-2, 'ditto':1e-1}}

Credit_OPTIM_dict = {0.12:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.23:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.30:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.40:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'}}

Weather_LR_dict = {0.11:{'single': 3e-3, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.19:{'single': 5e-3, 'joint': 5e-2, 'federated': 1e-1, 'pfedme':1e-1, 'ditto':1e-1},
            0.30:{'single': 1e-2, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':5e-3, 'ditto':5e-2},
            0.40:{'single': 1e-2, 'joint': 1e-3, 'federated': 2e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.48:{'single': 1e-2, 'joint': 1e-3, 'federated': 5e-2, 'pfedme':1e-2, 'ditto':3e-2}}

Weather_OPTIM_dict = {0.11:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'ADM', 'ditto':'ADM'},
            0.19:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'ADM', 'ditto':'ADM'},
            0.30:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'ADM', 'ditto':'ADM'},
            0.40:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'ADM', 'ditto':'ADM'},
            0.48:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'ADM', 'ditto':'ADM'}}

EMNIST_LR_dict = {0.11:{'single': 5e-2, 'joint': 1e-2, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':5e-2},
            0.19:{'single': 5e-3, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':5e-2},
            0.25:{'single': 1e-2, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':5e-2, 'ditto':1e-1},
            0.34:{'single': 1e-2, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':5e-2, 'ditto':5e-2},
            0.39:{'single': 1e-2, 'joint': 5e-3, 'federated': 3e-2, 'pfedme':5e-2, 'ditto':5e-2}}

EMNIST_OPTIM_dict = {0.11:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.19:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.25:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.34:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.39:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'}}

CIFAR_LR_dict = {0.08:{'single': 1e-3, 'joint': 5e-4, 'federated': 1e-2, 'pfedme':1e-2, 'ditto':1e-2},
            0.21:{'single': 1e-3, 'joint': 5e-3, 'federated': 1e-2, 'pfedme':5e-2, 'ditto':5e-3},
            0.30:{'single': 5e-4, 'joint': 5e-2, 'federated': 1e-2, 'pfedme':5e-2, 'ditto':5e-2},
            0.38:{'single': 5e-4, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-2}}

CIFAR_OPTIM_dict = {0.08:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.21:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'ADM'},
            0.30:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'ADM'},
            0.38:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'ADM'}}

IXITiny_LR_dict = {0.08:{'single': 1e-1, 'joint': 1e-1, 'federated': 5e-2, 'pfedme':1e-1, 'ditto':1e-1},
            0.28:{'single': 1e-1, 'joint': 1e-1, 'federated': 1e-1, 'pfedme':5e-2, 'ditto':1e-1},
            0.30:{'single': 1e-1, 'joint': 1e-2, 'federated': 1e-1, 'pfedme':5e-2, 'ditto':1e-1}}

IXITiny_OPTIM_dict = {0.08:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.28:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'SGD'},
            0.30:{'single': 'ADM', 'joint': 'ADM', 'federated': 'SGD', 'pfedme':'SGD', 'ditto':'SGD'}}

ISIC_LR_dict = {0.06:{'single': 5e-4, 'joint': 5e-3, 'federated': 1e-1, 'pfedme':5e-3, 'ditto':1e-2},
            0.15:{'single': 5e-3, 'joint': 5e-3, 'federated': 1e-2, 'pfedme':5e-3, 'ditto':1e-1},
            0.19:{'single': 5e-3, 'joint': 5e-3, 'federated': 1e-2, 'pfedme':5e-3, 'ditto':1e-1},
            0.25:{'single': 5e-3, 'joint': 1e-2, 'federated': 1e-2, 'pfedme':1e-2, 'ditto':1e-2},
            0.3:{'single': 5e-3, 'joint': 5e-3, 'federated': 5e-2, 'pfedme':1e-2, 'ditto':1e-2},}

ISIC_OPTIM_dict = {0.06:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.15:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.19:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.25:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'},
            0.3:{'single': 'ADM', 'joint': 'ADM', 'federated': 'ADM', 'pfedme':'ADM', 'ditto':'ADM'}}