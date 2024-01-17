global ROOT_DIR
ROOT_DIR = ''
RESULTS_DIR = f'{ROOT_DIR}/results/sampleSize'

import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(f'{ROOT_DIR}/code/helper/')
sys.path.append(f'{ROOT_DIR}/code/Synthetic/')
import OTCost as ot
import importlib
importlib.reload(ot)
from sklearn.preprocessing import normalize, StandardScaler
from emnist import extract_training_samples
from torch.utils.data import Dataset
import pickle
import random
import os
import concurrent.futures

private = False
try:
    CPU = int(os.environ.get('SLURM_CPUS_PER_TASK'))
except (TypeError, ValueError):
    CPU = 5

def wrangle_results(DATASET, results, save):
    plt.clf()
    df = pd.DataFrame(results).T
    df = df.sort_index(axis=1)
    baseline = df.iloc[:,[-1]]  
    print(baseline.iloc[:,0].values)    
    for col in df.columns[:-1]:
        print(df[col].values) 
        sns.lineplot(x = baseline.iloc[:,0].values, y = df[col].values, label = col)
    final_cost = float(baseline.values.max())
    first_cost = float(baseline.values.min())
    plt.plot(np.linspace(first_cost - 1e-2, final_cost + 1e-2, 100), np.linspace(first_cost - 1e-2, final_cost + 1e-2, 100), linestyle = '--', alpha = 0.5, label = 'Perfect agreement', color = 'black')
    plt.xlabel('Full dataset OT cost', fontsize = 14)
    plt.ylabel(f'Sampled dataset OT cost', fontsize = 14)
    plt.legend(title = 'Sample size', fontsize = 12, title_fontsize = 12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save:
        df.to_csv(f'{RESULTS_DIR}/{DATASET}_sample.csv')
        plt.savefig(f'{RESULTS_DIR}/{DATASET}_sample.pdf')
    plt.show()

def loadData(name, size):
    ##load data
    X = pd.read_csv(f'{ROOT_DIR}/data/Synthetic/{name}.csv', sep = ' ', names = [i for i in range(13)])
    ##merge
    X = X.sample(frac = 1)
    X = X.iloc[:size]
    ##get X and label
    y = X.iloc[:,-1]
    X = X.iloc[:,:-1]
    return X.values,y.values

def dictionaryCreater(X1, y1, X2, y2):
    ##wrangle to dictionary for OT cost calculation
    data, label = {"1": X1, "2": X2}, {"1": np.array(y1).reshape(1,-1)[0], "2": np.array(y2).reshape(1,-1)[0]}
    return data, label


def compute_synthetic_cost(DATASET, c, size):
    ## load data
    name1, name2 = f'data_1_{"{:.2f}".format(c)}', f'data_2_{"{:.2f}".format(c)}'
    X1, y1 = loadData(name1, size = size)
    X2, y2 = loadData(name2, size = size)

    data, label = dictionaryCreater(X1, y1, X2, y2)
    OTCost_label = ot.OTCost(DATASET, data, label)
    cost = OTCost_label.calculate_ot_cost()
    return c, size, round(cost, 2)

def synthetic():
    importlib.reload(ot)
    DATASET = 'Synthetic'
    cs = [0.03, 0.10, 0.20, 0.30, 0.40, 0.50]
    sizes = [200, 400, 600, 800, 999]
    results = {cost:{num:0 for num in sizes} for cost in cs }

    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU) as executor:
        futures = [executor.submit(compute_synthetic_cost, DATASET, c, size) for c in cs for size in sizes]
        
        for future in concurrent.futures.as_completed(futures):
            c, size, cost = future.result()
            results[c][size] = cost
    wrangle_results(DATASET, results, save = True)

def compute_credit_cost(args):
    def loadData(name):
        ##load data
        data = pd.read_csv(f'{ROOT_DIR}/data/Credit/{name}.csv', sep = ' ', names = [i for i in range(29)])
        ##get X and label
        y = data.iloc[:,-1]
        X = data.iloc[:,:-1]
        return X,y

    def dictionaryCreater(X1, y1, X2, y2):
        ##wrangle to dictionary for OT cost calculation
        data_, label = {"1": X1, "2": X2}, {"1": np.array(y1).reshape(1,-1)[0], "2": np.array(y2).reshape(1,-1)[0]}
        data= {"1" : normalize(data_['1'], axis = 1, norm = 'l2'), "2" : normalize(data_['2'], axis = 1, norm = 'l2')}
        return data, data_, label

    def sampler(X, y, size):
        return X.iloc[:size,:], y.iloc[:size] 
    DATASET, c, size = args
    
    results = {}
    results[c] = {}

    name1, name2 = f'data_1_{"{:.2f}".format(c)}', f'data_2_{"{:.2f}".format(c)}'
    X1, y1 = loadData(name1)
    X2, y2 = loadData(name2)

    X1, y1 = sampler(X1, y1, size)
    X2, y2 = sampler(X2, y2, size)

    data, data_, label = dictionaryCreater(X1, y1, X2, y2)
    OTCost_label = ot.OTCost(DATASET, data, label)
    cost = OTCost_label.calculate_ot_cost()

    results[c][size] = round(cost, 2)
    return results


def credit():
    DATASET = 'Credit'
    importlib.reload(ot)
    cs = [0.12, 0.23, 0.30, 0.40]
    sizes = [200, 400, 600, 800, 999]
    all_args = [(DATASET, c, size) for c in cs for size in sizes]
    results = {cost:{num:0 for num in sizes} for cost in cs }
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU) as executor:
        for arg, res in zip(all_args, executor.map(compute_credit_cost, all_args)):
            _,cost,num = arg
            results[cost][num] = res[cost][num]

    wrangle_results(DATASET, results, save = True)

def compute_weather_cost(args):
    def extractData(df, climates,n = 2000):
        df = df[df['climate'].isin(climates)]
        ind = np.random.choice(df.shape[0], n)
        X = df.iloc[ind, 6:]
        y = df.iloc[ind, 5]
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized, y.values

    def dictionaryCreator(df, climates):
        ##wrangle to dictionary for OT cost calculation
        X1, y1 = extractData(df, climates[0],n = 4000)  
        X2, y2 = extractData(df, climates[1],n = 4000)  
        data, label = {"1": X1, "2": X2}, {"1": y1, "2": y2}
        return data, label

    def sampler(data, label, num = 500):
        data_, label_  = {}, {}
        for i in data:
            idx = np.random.choice(np.arange(data[i].shape[0]), num, replace=False)
            data_[i] = data[i][idx]
            label_[i] = label[i][idx]
        return data_, label_
    ## load data
    ##load dataset
    df = pd.read_csv(f'{ROOT_DIR}/data/Weather/shifts_canonical_train.csv', nrows = 20000)
    df_snow = pd.read_csv(f'{ROOT_DIR}/data/Weather/shifts_canonical_eval_out.csv', nrows = 5000)
    df = pd.concat([df, df_snow])
    df.dropna(inplace = True)
    
    DATASET, climate, ac, num = args
    results = {}
    results[ac] = {}
    data, label = dictionaryCreator(df, climate)
    data_, label_ = sampler(data, label, num=num)
    OTCost_label = ot.OTCost(DATASET, data_, label_)
    cost = OTCost_label.calculate_ot_cost()
    results[ac][num] = round(cost, 2)
    return results


def weather():
    DATASET = 'Weather'
    costs = [0.11, 0.19, 0.3, 0.4, 0.48]
    climates = [[['tropical', 'mild temperate'],['tropical', 'mild temperate']],
                [['tropical', 'mild temperate'], ['dry', 'mild temperate']],
                [['tropical', 'mild temperate'], ['dry']],
                [['tropical', 'mild temperate'], ['snow', 'dry']],
                [['tropical', 'mild temperate'],['snow']]]
    nums = [200, 400, 600, 800, 999]
    all_args = [(DATASET,climate, ac, num) for climate, ac in zip(climates, costs) for num in nums]
    results = {cost:{num:0 for num in nums} for cost in costs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU) as executor:
        for arg, res in zip(all_args, executor.map(compute_weather_cost, all_args)):
            _, _, cost,num = arg
            results[cost][num] = res[cost][num]
    wrangle_results(DATASET, results, save = True)

def compute_emnist_cost(args):
    ##load dataset
    images_full, labels_full = extract_training_samples('byclass')

    def getIndices(indices, size):
        indices_use = []
        ## loop through and pull indices
        for ind in indices:
            indices_use.extend(np.where(np.isin(labels_full, ind) == True)[0][:size])
        return indices_use

    def pull_labels(images, labels, indices, size):
        ##get indices for x, i
        ind_1 =  getIndices(indices[0], size)
        ind_2 =  getIndices(indices[1], size)

        ##pull data and labels
        X1 = images[ind_1] / 255
        X2 = images[ind_2] / 255
        y1 = labels[ind_1]
        y2 = labels[ind_2]

        return {"1": X1, "2": X2}, {"1":y1, "2":y2}

    def sampler(data, label, num):
        data_, label_  = {}, {}
        for i in data:
            idx = np.random.choice(np.arange(data[i].shape[0]), num, replace=False)
            data_[i] = data[i][idx]
            label_[i] = label[i][idx].reshape(1,-1)[0]
            data_[i] = data_[i].reshape((num, 28*28))
        return data_, label_
    

    DATASET, ind, ac, num = args
    results = {}
    results[ac] = {}
    try:
        importlib.reload(ot)
        data, label = pull_labels(images_full, labels_full, ind, size=1000)
        data_, label_ = sampler(data, label, num=num)
        OTCost_label = ot.OTCost(DATASET, data_, label_)
        cost = OTCost_label.calculate_ot_cost()
        results[ac][num] = round(cost, 2)
        return results
    except Exception as e:
        print(f"Error encountered for index: {args[1]}\n, num: {args[3]}\n,\n cost: {args[0]}.\n Error message: {str(e)}")
        raise


def emnist():
    DATASET = 'EMNIST'
    importlib.reload(ot)
    costs = [0.11,0.19,0.25,0.34,0.39]
    indices =  [
                [[x for x in range(10)], [x for x in range(10)]],
                [[x for x in range(10)] + [12,24,18,28], [x for x in range(10)] + [x + 26 for x in [12,24,18,28]]],
                [[x for x in range(10)] + [11,12,13,14,16,24,18,28], [x for x in range(10)] + [x + 26 for x in [11,12,13,14,16,24,18,28]]],
                [[x for x in range(10)] + [x for x in range(10, 25)], [x for x in range(10)] + [x for x in range(36, 51)]],
                [[x for x in range(10)] +[x for x in range(10, 35)], [x for x in range(10)] +[x for x in range(36, 61)]]
                ]

    nums = [600, 800, 999]
    all_args = [(DATASET, ind, ac, num) for ind, ac in zip(indices, costs) for num in nums]
    results = {cost:{num:0 for num in nums} for cost in costs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU) as executor:
        for arg, res in zip(all_args, executor.map(compute_emnist_cost, all_args)):
            _, _, cost,num = arg
            results[cost][num] = res[cost][num]
    wrangle_results(DATASET, results, save = True)



def compute_cifar_cost(args):  
    DATASET = 'CIFAR'
    DATA_DIR = f'{ROOT_DIR}/data/CIFAR'
    n_emb = 1000
    class EmbeddedImagesDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image, embedding, label, coarse_label = self.data[idx]
            return image, embedding, label, coarse_label
        
    with open(f'{DATA_DIR}/cifar_{n_emb}_emb.pkl', 'rb') as f:
        data= pickle.load(f)
    def extract_by_labels(dataset, target_labels, label_type = 'fine'):
        extracted_data = []
        for image, embedding, fine_label, coarse_label in dataset:
            if label_type == 'fine':
                label = fine_label
            else:
                label = coarse_label
            if label in target_labels:
                extracted_data.append((image, embedding, fine_label, coarse_label))
        return EmbeddedImagesDataset(extracted_data)

    def get_datasets(dataset, labels_extract, label_type = 'fine'):
        d1 = extract_by_labels(dataset, labels_extract[0], label_type)
        d2 = extract_by_labels(dataset, labels_extract[1], label_type)
        return d1, d2

    def sampler(dataset, num_samples):
        indices = random.sample(range(len(dataset)), num_samples)
        sampled_data = [dataset[i] for i in indices]
        embs = np.array([entry[1] for entry in sampled_data])
        label = np.array([entry[2] for entry in sampled_data])
        return embs, label

    ind, ac, num = args
    results = {}
    results[ac] = {}

    importlib.reload(ot)
    d1, d2 = get_datasets(data, ind)
    x1, y1 = sampler(d1, num)
    x2, y2 = sampler(d2, num)
    data_ = {'1': x1, '2': x2}
    label_ = {'1': y1, '2': y2}
    OTCost_label = ot.OTCost(DATASET, data_, label_)
    cost = OTCost_label.calculate_ot_cost()
    results[ac][num] = round(cost, 2)
    return results

def cifar():
    DATASET = 'CIFAR'
    importlib.reload(ot)
    costs = [0.08, 0.21, 0.30, 0.38]
    indices =  [
                [[x for x in range(10)], [x for x in range(10)]],
                [[11,98,29,73, 78, 49, 97, 51, 55, 92], [11,98,29,73, 78, 49, 42, 83, 72, 82]],
                [[11,50,78,1,92, 78, 49, 97, 55, 16, 14], [11, 36, 29, 73, 82, 78, 49, 42, 12, 23, 51]],
                [[11,50,78,8,92,2,49,98,89,3], [17, 36, 30, 73, 83,28, 34, 42, 10, 20]]
                ]

    nums = [200, 400, 600, 800, 999]
    all_args = [(ind, ac, num) for ind, ac in zip(indices, costs) for num in nums]
    results = {cost:{num:0 for num in nums} for cost in costs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU) as executor:
        for arg, res in zip(all_args, executor.map(compute_cifar_cost, all_args)):
            _, cost,num = arg
            results[cost][num] = res[cost][num]
    wrangle_results(DATASET, results, save = True)

def compute_ixitiny_cost(args):
    DATASET = 'IXITiny'
    DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'

    files = os.listdir(f'{DATA_DIR}/embedding')
    sites = ['Guys', 'HH', 'IOP']
    site_samples = {}
    site_embeddings = {}
    for site in sites:
        site_samples[site] = [file for file in files if site in file]

    for site, files in site_samples.items():
        embeddings = [np.load(f'{DATA_DIR}/embedding/{file}') for file in files]
        site_embeddings[site] = np.array(embeddings) 

    ind, ac, num = args
    results = {}
    results[ac] = {}

    importlib.reload(ot)
    X1, X2 = site_embeddings[ind[0]][:num], site_embeddings[ind[1]][:num]
    data = {'1': X1, '2': X2}
    label = {'1': np.ones(X1.shape[0]), '2':np.ones(X2.shape[0])}
    OTCost_label = ot.OTCost(DATASET, data, label)
    cost = OTCost_label.calculate_ot_cost()
    results[ac][num] = round(cost, 2)
    return results

def ixitiny():
    DATASET = 'IXITiny'
    importlib.reload(ot)
    costs = [0.08, 0.21, 0.30, 0.38]
    indices =  [['Guys', 'HH'],
                ['Guys', 'IOP'],
                ['HH', 'IOP']]

    nums = [20, 50, 100, 200, 400]
    all_args = [(ind, ac, num) for ind, ac in zip(indices, costs) for num in nums]
    results = {cost:{num:0 for num in nums} for cost in costs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU) as executor:
        for arg, res in zip(all_args,executor.map(compute_ixitiny_cost, all_args)):
            _, cost,num = arg
            results[cost][num] = res[cost][num]
    wrangle_results(DATASET, results, save = True)

def compute_isic_cost(args):
    DATASET = 'ISIC'
    DATA_DIR = f'{ROOT_DIR}/data/{DATASET}'

    ##Load labels
    labels = pd.read_csv(f'{DATA_DIR}/ISIC_2019_Training_GroundTruth.csv')
    def create_category(row):
        for idx, value in enumerate(row):
            if value == 1:
                return idx
        return None

    labels['label'] = labels.apply(create_category, axis=1) - 1
    labels = labels[['image', 'label']]
    labels.set_index('image', inplace = True)

    NUM_SAMPLES = 2000
    files = os.listdir(f'{DATA_DIR}/embedding')
    sites = [i for i in range(6)]
    site_samples = {}
    site_embeddings = {}
    site_labels = {}
    for site in sites:
        sites_files = [file for file in files if f'center_{site}' in file]
        sites_files = np.random.choice(sites_files, size = NUM_SAMPLES)
        site_samples[site] = sites_files
        names = [f.split(f'center_{site}_')[-1].split('.npy')[0] for f in sites_files]
        labels_site = labels.loc[names]
        site_labels[site] = labels_site['label'].values

    for site, files in site_samples.items():
        embeddings = [np.load(f'{DATA_DIR}/embedding/{file}') for file in files]
        site_embeddings[site] = np.array(embeddings)

    def create_dictionaries(site_embeddings, site_labels, sites, NUM_SAMPLES = 500):
        data = {'1': site_embeddings[sites[0]][:NUM_SAMPLES], '2' :site_embeddings[sites[1]][:NUM_SAMPLES]}
        labels = {'1': site_labels[sites[0]][:NUM_SAMPLES], '2': site_labels[sites[1]][:NUM_SAMPLES]}
        data, labels = remove_rare_labels(data, labels, min_count = 30)
        return data, labels

    #Function is needed as estimating label cost with fewer data points leads to degeneracy
    def remove_rare_labels(data, labels, min_count):
        for key in labels:
            unique_labels, counts = np.unique(labels[key], return_counts=True)
            labels_to_remove = unique_labels[counts <= min_count]
            mask = np.isin(labels[key], labels_to_remove, invert=True)
            labels[key] = labels[key][mask]
            data[key] = data[key][mask]
        return data, labels
    ind, ac, num = args
    results = {}
    results[ac] = {}
    importlib.reload(ot)
    if ind[0] != ind[1]:
        data, labels = create_dictionaries(site_embeddings, site_labels, ind, num)
        ISIC_OTCost_label = ot.OTCost(DATASET, data, labels)
        cost = ISIC_OTCost_label.calculate_ot_cost()
    else:
        data = {'1': site_embeddings[ind[0]][:num], '2': site_embeddings[ind[0]][num:num*2]}
        labels = {'1': site_labels[ind[0]][:num], '2': site_labels[ind[0]][num:num*2]}
        data, labels = remove_rare_labels(data, labels, min_count=30)
        OTCost_label = ot.OTCost(DATASET, data, labels)
        cost = OTCost_label.calculate_ot_cost()

    results[ac][num] = round(cost, 2)
    return results

def isic():
    DATASET = 'ISIC'
    importlib.reload(ot)
    site_pairs = [(2,2), (2,0), (2,3), (2,1), (1,3)]
    costs = [0.06, 0.15, 0.19, 0.25, 0.3]
    nums = [200, 400, 600, 800, 999]

    all_args = [(ind, ac, num) for ind, ac in zip(site_pairs, costs) for num in nums]
    results = {cost:{num:0 for num in nums} for cost in costs}
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU) as executor:
        for arg, res in zip(all_args, executor.map(compute_isic_cost, all_args)):
            _, cost,num = arg
            results[cost][num] = res[cost][num]
    wrangle_results(DATASET, results, save = True)

def main():
    #synthetic()
    #credit()
    #weather()
    emnist()
    #cifar()
    #ixitiny()
    #isic()
    
if __name__ == "__main__":
    main()
