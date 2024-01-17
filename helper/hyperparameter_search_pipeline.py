global ROOT_DIR
ROOT_DIR = ''

import pandas as pd
import sys
import os
import numpy as np
import torch
import torch.nn as nn
sys.path.append(f'{ROOT_DIR}/code/helper')
import data_preprocessing as dp
import trainers as tr
import models_helper as mh
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import importlib
import pickle
importlib.reload(dp)
importlib.reload(tr)
importlib.reload(mh)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TABULAR = ['Synthetic', 'Credit', 'Weather']
CLASS_ADJUST = ['EMNIST', 'CIFAR'] # as each dataset cost has different labels

def check_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU(s) available.")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # To check GPU memory allocation (useful to determine if it's in use)
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert bytes to GB
            allocated_mem = torch.cuda.memory_allocated(i) / 1e9
            cached_mem = torch.cuda.memory_reserved(i) / 1e9

            print(f"  Total Memory: {total_mem:.2f} GB")
            print(f"  Allocated Memory: {allocated_mem:.2f} GB")
            print(f"  Cached Memory: {cached_mem:.2f} GB")
check_gpu()

def loadData(DATASET, DATA_DIR, data_num, cost):
    if DATASET in TABULAR:
        if DATASET == 'Synthetic':
            ##load data
            X = pd.read_csv(f'{DATA_DIR}/data_{data_num}_{cost:.2f}.csv', sep = ' ', names = [i for i in range(13)])
            X = X.sample(800)
        elif DATASET == 'Credit':
            X = pd.read_csv(f'{DATA_DIR}/data_{data_num}_{cost:.2f}.csv', sep = ' ', names = [i for i in range(29)])
            X = X.sample(800,replace=True)
        elif DATASET == 'Weather':
            X = pd.read_csv(f'{DATA_DIR}/data_{data_num}_{cost:.2f}.csv', sep = ' ', names = [i for i in range(124)])
            X = X.sample(n=1600)
        y = X.iloc[:,-1]
        X = X.iloc[:,:-1]
        return X.values, y.values
    elif DATASET in CLASS_ADJUST:
        ##load data
        data = np.load(f'{DATA_DIR}/data_{data_num}_{cost:.2f}.npz')
        ##get X and label
        X = data['data']
        y = data['labels']
        class_size = 250
        ind = sample_per_class(y, class_size)
        X_sample =  X[ind]
        y_sample = y[ind]
        unique_labels = np.unique(y_sample)
        mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y_sample_mapped = np.vectorize(mapping.get)(y_sample)

        return X_sample, y_sample_mapped
    elif DATASET == 'IXITiny':
        sites = {0.08: [['Guys'], ['HH']],
             0.28: [['IOP'], ['Guys']],
             0.30: [['IOP'], ['HH']]}
        site_names = sites[cost][data_num-1]

        image_dir = os.path.join(DATA_DIR, 'flamby/image')
        label_dir = os.path.join(DATA_DIR, 'flamby/label')
        image_files = []
        label_files = []
        for name in site_names:
                image_files.extend([f'{image_dir}/{file}' for file in os.listdir(image_dir) if name in file])
                label_files.extend([f'{label_dir}/{file}'  for file in os.listdir(label_dir) if name in file])
        image_files, label_files = align_image_label_files(image_files, label_files)
        return np.array(image_files), np.array(label_files)
    elif DATASET == 'ISIC':
        dataset_pairings = {0.06: (2,2), 0.15:(2,0), 0.19:(2,3), 0.25:(2,1), 0.3:(1,3)}
        site = dataset_pairings[cost][data_num-1]
        files = pd.read_csv(f'{DATA_DIR}/site_{site}_files_used.csv', nrows = 2000)
        image_files = [f'{DATA_DIR}/ISIC_2019_Training_Input_preprocessed/{file}.jpg' for file in files['image']]
        labels = files['label'].values
        return np.array(image_files), labels

#FOR EMNIST AND CIFAR
def sample_per_class(labels, class_size = 500):
  df = pd.DataFrame({'labels': labels})
  df_stratified = df.groupby('labels').apply(lambda x: x.sample(class_size, replace=False))
  ind = df_stratified.index.get_level_values(1)
  return ind

#FOR ISIC AND IXITiny
def get_common_name(full_path):
    return os.path.basename(full_path).split('_')[0]

def align_image_label_files(image_files, label_files):
    labels_dict = {get_common_name(path): path for path in label_files}
    images_dict = {get_common_name(path): path for path in image_files}
    common_keys = sorted(set(labels_dict.keys()) & set(images_dict.keys()))
    sorted_labels = [labels_dict[key] for key in common_keys]
    sorted_images = [images_dict[key] for key in common_keys]
    return sorted_images, sorted_labels

def set_parameters_for_dataset(DATASET):
    if DATASET == 'Synthetic':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'  
        EPOCHS = 300
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 2000
        DATASET = 'Synthetic'
        METRIC_TEST = 'F1'

    elif DATASET == 'Credit':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 300
        WARMUP_STEPS = EPOCHS // 15 
        BATCH_SIZE = 2000
        DATASET = 'Credit'
        METRIC_TEST = 'F1'

    elif DATASET == 'Weather':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 300
        WARMUP_STEPS = EPOCHS // 15 
        BATCH_SIZE = 4000
        METRIC_TEST = 'R2'

    elif DATASET == 'EMNIST':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 500
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 5000
        METRIC_TEST = 'Accuracy'

    elif DATASET == 'CIFAR':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 500
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 64
        METRIC_TEST = 'Accuracy'

    elif DATASET == 'IXITiny':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}' 
        EPOCHS = 100
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 12
        METRIC_TEST = 'DICE'
    
    elif DATASET == 'ISIC':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 500
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 128
        METRIC_TEST = 'Balanced_accuracy'
    if DATASET in ['IXITiny', 'ISIC']:
        RUNS = 1 # Small number of runs to test
    elif DATASET in ['EMNIST']:
        RUNS = 10
    elif DATASET in ['CIFAR']:
        RUNS = 5
    elif DATASET in ['Synthetic', 'Credit', 'Weather']:
        RUNS = 100
    return DATA_DIR, EPOCHS, WARMUP_STEPS, BATCH_SIZE, RUNS, METRIC_TEST

def createModel(DATASET, architecture, c, WARMUP_STEPS, LR, OPTIM):
    if DATASET == 'Synthetic':
        model = mh.Synthetic()
        criterion = nn.BCELoss()
    elif DATASET == 'Credit':
        model = mh.Credit()
        criterion = nn.BCELoss()
    elif DATASET == 'Weather':
        model = mh.Weather()
        criterion = nn.MSELoss()
    elif DATASET == 'EMNIST':
        with open(f'{ROOT_DIR}/data/{DATASET}/CLASSES', 'rb') as f:
            classes_used = pickle.load(f)
            if architecture == 'single':
                CLASSES = len(classes_used[c][0])
            else:
                CLASSES = len(set(classes_used[c][0] + classes_used[c][1]))
        model = mh.EMNIST(CLASSES)
        criterion = nn.CrossEntropyLoss()
    elif DATASET == 'CIFAR':
        with open(f'{ROOT_DIR}/data/{DATASET}/CLASSES', 'rb') as f:
            classes_used = pickle.load(f)
            if architecture == 'single':
                CLASSES = len(classes_used[c][0])
            else:
                CLASSES = len(set(classes_used[c][0] + classes_used[c][1]))
        model = mh.CIFAR(CLASSES)
        model = nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
    elif DATASET == 'IXITiny':
        model = mh.IXITiny()
        model = nn.DataParallel(model)
        criterion = get_dice_loss
    elif DATASET == 'ISIC':
        model = mh.ISIC()
        model = nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)
    
    #Different optimizers for federated and non federated
    if OPTIM == 'ADM':
        optimizer = torch.optim.AdamW(model.parameters(), lr = LR, amsgrad = True, betas = (0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.9, weight_decay = 1e-4)
    exp_scheduler = ExponentialLR(optimizer, gamma=0.9)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / WARMUP_STEPS))
    lr_scheduler = (warmup_scheduler, exp_scheduler)
    
    return model, criterion, optimizer, lr_scheduler
          
def get_dice_loss(output, target, SPATIAL_DIMENSIONS = (2, 3, 4), epsilon=1e-9):
    p0 = output
    g0 = target
    p1 = 1 - p0
    g1 = 1 - g0
    tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
    fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
    fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
    num = 2 * tp
    denom = 2 * tp + fp + fn + epsilon
    dice_score = num / denom
    return torch.mean(1 - dice_score)

def modelRuns(DATASET, c):
    DATA_DIR, EPOCHS, WARMUP_STEPS, BATCH_SIZE, RUNS, METRIC_TEST = set_parameters_for_dataset(DATASET)
    #excuse the laziness ;)
    scores = {'single':{'ADM':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}, 'SGD':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}}, 
              'joint':{'ADM':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}, 'SGD':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}},
              'federated':{'ADM':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}, 'SGD':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}}, 
              'pfedme':{'ADM':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}, 'SGD':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}},
              'ditto':{'ADM':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}, 'SGD':{1e-1:[], 5e-2:[], 1e-2:[], 5e-3:[], 5e-4:[]}}}
    X1, y1 = loadData(DATASET, DATA_DIR, 1, c)
    X2, y2 = loadData(DATASET, DATA_DIR, 2, c)
    LR_try = [1e-1, 5e-2, 1e-2, 5e-3, 5e-4]
    OPTIM_try = ['ADM', 'SGD']
    for OPTIM in OPTIM_try:
        for LR in LR_try:
            for _ in range(RUNS):
                #single
                arch = 'single'
                model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS, LR, OPTIM)
                dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
                site_1 = dataloader.preprocess(X1, y1)
                trainer = tr.ModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
                trainer.set_loader(site_1)
                score, test_loss, val_loss, train_loss = trainer.run()
                scores[arch][OPTIM][LR].append(score)

                #joint
                arch = 'joint'
                model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS, LR, OPTIM)
                dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
                site_joint = dataloader.preprocess_joint(X1, y1, X2, y2)
                trainer = tr.ModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
                trainer.set_loader(site_joint)
                score, test_loss, val_loss, train_loss = trainer.run()
                scores[arch][OPTIM][LR].append(score)

                #federated
                arch = 'federated'
                model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS, LR, OPTIM)
                dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
                site_1 = dataloader.preprocess(X1, y1)
                site_2 = dataloader.preprocess(X2, y2)
                trainer = tr.FederatedModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
                trainer.set_loader(site_1, site_2)
                score, test_loss, val_loss, train_loss = trainer.run()
                scores[arch][OPTIM][LR].append(score)
                
                #pfedme
                arch = 'pfedme'
                model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS, LR, OPTIM)
                dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
                site_1 = dataloader.preprocess(X1, y1)
                site_2 = dataloader.preprocess(X2, y2)
                trainer = tr.FederatedModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE, pfedme = True)
                trainer.set_loader(site_1, site_2)
                score, test_loss, val_loss, train_loss = trainer.run()
                scores[arch][OPTIM][LR].append(score)
                #ditto
                arch = 'ditto'
                model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS, LR, OPTIM)
                dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
                site_1 = dataloader.preprocess(X1, y1)
                site_2 = dataloader.preprocess(X2, y2)
                trainer = tr.DittoModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
                trainer.set_loader(site_1, site_2)
                score, test_loss, val_loss, train_loss = trainer.run()
                scores[arch][OPTIM][LR].append(score)
    return scores

def runAnalysis(DATASET, costs):
    results_scores = {}
    if f'{DATASET}_hyperparameter_search.pkl' in os.listdir(f'{ROOT_DIR}/results/'):
        with open(f'{ROOT_DIR}/results/{DATASET}_hyperparameter_search.pkl', 'rb') as f:
            results_scores = pickle.load(f)
    costs = list(set(costs)- set(list(results_scores.keys())))
    
    for c in costs:
        results_scores[c] =  modelRuns(DATASET, c)
        with open(f'{ROOT_DIR}/results/{DATASET}_hyperparameter_search.pkl', 'wb') as f:
            pickle.dump(results_scores, f)
    return results_scores
