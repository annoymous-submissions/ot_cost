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
import hyperparameters as hp
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
import importlib
import pickle
importlib.reload(dp)
importlib.reload(tr)
importlib.reload(mh)
importlib.reload(hp)

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
            X = X.sample(800, replace=True)
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
        RUNS = 500
        DATASET = 'Synthetic'
        METRIC_TEST = 'F1'

    elif DATASET == 'Credit':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 300
        WARMUP_STEPS = EPOCHS // 15 
        BATCH_SIZE = 2000
        RUNS = 500
        DATASET = 'Credit'
        METRIC_TEST = 'F1'

    elif DATASET == 'Weather':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 300
        WARMUP_STEPS = EPOCHS // 15 
        BATCH_SIZE = 4000
        RUNS = 500
        METRIC_TEST = 'R2'

    elif DATASET == 'EMNIST':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 500
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 5000
        RUNS = 50
        METRIC_TEST = 'Accuracy'

    elif DATASET == 'CIFAR':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 500
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 256
        RUNS = 20
        METRIC_TEST = 'Accuracy'

    elif DATASET == 'IXITiny':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}' 
        EPOCHS = 100
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 12
        RUNS = 3
        METRIC_TEST = 'DICE'
    
    elif DATASET == 'ISIC':
        DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'
        EPOCHS = 500
        WARMUP_STEPS = EPOCHS // 15
        BATCH_SIZE = 128
        RUNS = 3
        METRIC_TEST = 'Balanced_accuracy'
    return DATA_DIR, EPOCHS, WARMUP_STEPS, BATCH_SIZE, RUNS, METRIC_TEST

def createModel(DATASET, architecture, c, WARMUP_STEPS):
    if DATASET == 'Synthetic':
        LR_dict = hp.Synthetic_LR_dict[c]
        OPTIM_dict = hp.Synthetic_OPTIM_dict[c]
        model = mh.Synthetic()
        criterion = nn.BCELoss()
    elif DATASET == 'Credit':
        LR_dict = hp.Credit_LR_dict[c]
        OPTIM_dict = hp.Credit_OPTIM_dict[c]
        model = mh.Credit()
        criterion = nn.BCELoss()
    elif DATASET == 'Weather':
        LR_dict = hp.Weather_LR_dict[c]
        OPTIM_dict = hp.Weather_OPTIM_dict[c]
        model = mh.Weather()
        criterion = nn.MSELoss()
    elif DATASET == 'EMNIST':
        LR_dict = hp.EMNIST_LR_dict[c]
        OPTIM_dict = hp.EMNIST_OPTIM_dict[c]
        with open(f'{ROOT_DIR}/data/{DATASET}/CLASSES', 'rb') as f:
            classes_used = pickle.load(f)
            if architecture == 'single':
                CLASSES = len(classes_used[c][0])
            else:
                CLASSES = len(set(classes_used[c][0] + classes_used[c][1]))
        model = mh.EMNIST(CLASSES)
        criterion = nn.CrossEntropyLoss()
    elif DATASET == 'CIFAR':
        LR_dict = hp.CIFAR_LR_dict[c]
        OPTIM_dict = hp.CIFAR_OPTIM_dict[c]
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
        LR_dict = hp.IXITiny_LR_dict[c]
        OPTIM_dict = hp.IXITiny_OPTIM_dict[c]
        model = mh.IXITiny()
        model = nn.DataParallel(model)
        criterion = get_dice_loss
    elif DATASET == 'ISIC':
        LR_dict = hp.ISIC_LR_dict[c]
        OPTIM_dict = hp.ISIC_OPTIM_dict[c]
        model = mh.ISIC()
        model = nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)

    if OPTIM_dict[architecture] == 'ADM':
        LR = LR_dict[architecture]
        optimizer = torch.optim.AdamW(model.parameters(), lr = LR, amsgrad = True, betas = (0.9, 0.999))
    elif OPTIM_dict[architecture] == 'SGD':
        LR = LR_dict[architecture]
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

def modelRuns(DATASET, c, scores, test_losses, val_losses, train_losses, gradient_diversity):
    DATA_DIR, EPOCHS, WARMUP_STEPS, BATCH_SIZE, RUNS, METRIC_TEST = set_parameters_for_dataset(DATASET)
    #scores = {'single':[], 'joint':[], 'federated':[], 'pfedme':[], 'ditto':[]}
    #test_losses = {'single':[], 'joint':[], 'federated':[], 'pfedme':[], 'ditto':[]}
    #val_losses = {'single':[], 'joint':[], 'federated':[], 'pfedme':[], 'ditto':[]}
    #train_losses = {'single':[], 'joint':[], 'federated':[], 'pfedme':[], 'ditto':[]}
    #gradient_diversity = {'metric':[], 'cosine': []}
    X1, y1 = loadData(DATASET, DATA_DIR, 1, c)
    X2, y2 = loadData(DATASET, DATA_DIR, 2, c)

    for _ in range(RUNS):
        #single
        arch = 'single'
        model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS)
        dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
        site_1 = dataloader.preprocess(X1, y1)
        trainer = tr.ModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
        trainer.set_loader(site_1)
        score, test_loss, val_loss, train_loss = trainer.run()
        scores[arch].append(score), test_losses[arch].append(test_loss), val_losses[arch].append(val_loss), train_losses[arch].append(train_loss) 

        #joint
        arch = 'joint'
        model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS)
        dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
        site_joint = dataloader.preprocess_joint(X1, y1, X2, y2)
        trainer = tr.ModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
        trainer.set_loader(site_joint)
        score, test_loss, val_loss, train_loss = trainer.run()
        scores[arch].append(score), test_losses[arch].append(test_loss), val_losses[arch].append(val_loss), train_losses[arch].append(train_loss) 

        #federated
        arch = 'federated'
        model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS)
        dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
        site_1 = dataloader.preprocess(X1, y1)
        site_2 = dataloader.preprocess(X2, y2)
        trainer = tr.FederatedModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
        trainer.set_loader(site_1, site_2)
        score, test_loss, val_loss, train_loss = trainer.run()
        scores[arch].append(score), test_losses[arch].append(test_loss), val_losses[arch].append(val_loss), train_losses[arch].append(train_loss) 
        #gradient_diversity['metric'] = trainer.gradient_diversity_metric
        #gradient_diversity['cosine'] = trainer.gradient_diversity_cosine

        #pfedme
        arch = 'pfedme'
        model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS)
        dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
        site_1 = dataloader.preprocess(X1, y1)
        site_2 = dataloader.preprocess(X2, y2)
        trainer = tr.FederatedModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE, pfedme = True)
        trainer.set_loader(site_1, site_2)
        score, test_loss, val_loss, train_loss = trainer.run()
        scores[arch].append(score), test_losses[arch].append(test_loss), val_losses[arch].append(val_loss), train_losses[arch].append(train_loss) 

        #ditto
        arch = 'ditto'
        model, criterion, optimizer, lr_scheduler = createModel(DATASET, arch,c, WARMUP_STEPS)
        dataloader = dp.DataPreprocessor(DATASET, BATCH_SIZE)
        site_1 = dataloader.preprocess(X1, y1)
        site_2 = dataloader.preprocess(X2, y2)
        trainer = tr.DittoModelTrainer(EPOCHS, WARMUP_STEPS, DATASET, model, criterion, optimizer, lr_scheduler, DEVICE)
        trainer.set_loader(site_1, site_2)
        score, test_loss, val_loss, train_loss = trainer.run()
        scores[arch].append(score), test_losses[arch].append(test_loss), val_losses[arch].append(val_loss), train_losses[arch].append(train_loss) 

    return scores, train_losses, val_losses, test_losses, gradient_diversity

def runAnalysis(DATASET, costs):
    results_scores = {}
    results_train_losses = {}
    results_val_losses = {}
    results_test_losses = {}
    gradient_diversities = {}

    if f'{DATASET}_scores_full.pkl' in os.listdir(f'{ROOT_DIR}/results/{DATASET}'):
        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_scores_full.pkl', 'rb') as f:
            results_scores = pickle.load(f)

        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_train_losses_full.pkl', 'rb') as f:
            results_train_losses = pickle.load(f)
        
        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_val_losses_full.pkl', 'rb') as f:
            results_val_losses = pickle.load(f)

        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_test_losses_full.pkl', 'rb') as f:
            results_test_losses = pickle.load(f)

        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_gradient_diversities_full.pkl', 'rb') as f:
                gradient_diversities = pickle.load(f)

    #costs = list(set(costs)- set(list(results_scores.keys())))

    for c in costs:
        results_scores[c], results_train_losses[c], results_val_losses[c], results_test_losses[c], gradient_diversities[c] =  modelRuns(DATASET, c, results_scores[c], results_train_losses[c], results_val_losses[c], results_test_losses[c], gradient_diversities[c])

        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_scores_full.pkl', 'wb') as f:
            pickle.dump(results_scores, f)
        
        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_train_losses_full.pkl', 'wb') as f:
            pickle.dump(results_train_losses, f)

        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_val_losses_full.pkl', 'wb') as f:
            pickle.dump(results_val_losses, f)
        
        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_test_losses_full.pkl', 'wb') as f:
            pickle.dump(results_test_losses, f)
        
        with open(f'{ROOT_DIR}/results/{DATASET}/{DATASET}_gradient_diversities_full.pkl', 'wb') as f:
            pickle.dump(gradient_diversities, f)

    return results_scores, results_train_losses, results_val_losses, results_test_losses
