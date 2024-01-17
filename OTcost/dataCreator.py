import pandas as pd
import numpy as np
import sys
import random
from sklearn.utils import shuffle

global ROOT_DIR
ROOT_DIR = ''
DATA_DIR = f'{ROOT_DIR}/data/Synthetic'

def dataset_creator(lv_pos, lv_neg, dataset, total_cases, pos_share, label_switch, coef_sep = 0.85, switch = False, seed = 1):
    np.random.seed(seed)
    random.seed(seed)
    coefs = 12
    
    ##pos_cases
    pos_cases = int(np.floor(total_cases * pos_share))
    values_pos = np.zeros((pos_cases, coefs))
    label_pos = np.zeros((pos_cases, 1))
    std_scale = 1
    for i in range(pos_cases):
        ##generate random LVs based on overall risk and LV's
        risk =  np.random.normal(1, 0.4)
        std = np.random.normal(0,0.5, 7)
        risk = risk + std
        lv_pos_person = np.array([c[0]*risk[i] if np.random.binomial(1,coef_sep, 1)[0] == 1 else c[1]*risk[i] for i,c in enumerate(zip(lv_pos, lv_neg))])
        if dataset == 1:
        ## generate variables from LV
            values_pos[i]=   variables_from_lv(lv_pos_person, dataset)
        else:
        ## generate variables from LV (different relationships)
            values_pos[i]=  variables_from_lv(lv_pos_person, dataset)

        label = 1
        ##label switch
        if np.random.binomial(1,label_switch, 1)[0] == 0:
            label = 0
        label_pos[i] = label

    ##neg_cases
    neg_cases = total_cases - pos_cases
    values_neg = np.zeros((neg_cases, coefs))
    label_neg = np.zeros((neg_cases, 1))
    for i in range(neg_cases):
        ##generate random LVs based on overall risk and LV's
        risk =  np.random.normal(1, 0.5)
        std = np.random.normal(0,1, 7)
        risk = risk + std
        lv_neg_person = np.array([c[0]*risk[i] if np.random.binomial(1,coef_sep, 1)[0] == 1 else c[1]*risk[i] for i,c in enumerate(zip(lv_neg, lv_pos))])
        if dataset == 1:
            ## generate variables from LV
            values_neg[i]=  variables_from_lv(lv_neg_person, dataset)

        else:
        ## generate variables from LV (different relationships)
            values_neg[i]=  variables_from_lv(lv_neg_person, dataset)
        
        label = 0
        ##label switch
        if np.random.binomial(1,label_switch, 1)[0] == 0:
            label = 1
        label_neg[i] = label
        
    values = np.concatenate((values_pos, values_neg))
    labels = np.concatenate((label_pos, label_neg))
    table = np.concatenate((values, labels), axis =1 )
    np.random.shuffle(table)
    return table

def non_iid_creator(frac, total_cases = 800):
    ##create datasets CLASSIFICATION
    ##latent variables
    lv_pos1 = np.array([0.5, 0.5, 0.5, 0.55])
    lv_neg1 = np.array([0.9, 0.25, 0.2, 0.2])

    lv_pos2 = np.array([-0.2, -0.4, -0.1, 0.1, -0.1, 0.2, -0.2])
    lv_neg2 = np.array([-0.2, -0.3, -0.2, 0.2, -0.1, 0.2, -0.2]) 
    
    label_switch = 0.9
    coef_sep = 0.9
    pos_share = 0.5
    
    ##d1
    dataset = 1
    data_1 = dataset_creator(lv_pos1, lv_neg1, dataset, total_cases, pos_share, label_switch, coef_sep, seed = 1)
    X1, y1 =  data_1[:,:-1], data_1[:, -1]
    X1, y1 = shuffle(X1, y1)

    data_1b = dataset_creator(lv_pos1, lv_neg1, dataset, total_cases, pos_share,label_switch,  coef_sep, seed = 2)
    X1b, y1b = data_1b[:,:-1], data_1b[:, -1]
    
    ##d2
    dataset = 2
    data_2 = dataset_creator(lv_pos2, lv_neg2, dataset, total_cases, pos_share, label_switch, coef_sep, seed = 1)
    X2, y2 =  data_2[:,:-1], data_2[:, -1]
    
    
    ##mix datasets for 2nd dataset
    a = int(np.floor(total_cases * frac))
    X2, y2 = np.concatenate((X2[:a],X1b[a:])), np.concatenate((y2[:a],y1b[a:]))
    X2, y2 = shuffle(X2, y2)
    
    data, label = {}, {}
    data['1'], label['1'] = X1, y1
    data['2'], label['2'] = X2, y2
    return data, label

def variables_from_lv(lv, dataset):
    if dataset == 1:
        rvs = np.array([np.sin(lv[1]), 
                        np.exp(lv[0]/2),
                        lv[3]*lv[1], 
                        np.random.exponential(np.exp(lv[2])), 
                        np.random.normal(lv[2], 0.5) * np.random.normal(lv[0], 0.5), 
                        np.random.normal(lv[1], 1), 
                        np.random.normal(lv[3],0.5) * np.random.normal(lv[2], 0.05), 
                        lv[2]*lv[3],
                        np.random.normal(lv[1], 0.1), 
                        np.random.normal(lv[2], 0.1),
                        np.random.normal(lv[1], 0.1), 
                        np.random.normal(-2, 1)])
    else:
         rvs = np.array([np.exp(lv[0])+ np.sin(lv[0]), 
                        -np.random.beta(lv[2]**2, lv[1]**2),
                        np.sin(lv[4]) - np.random.beta(lv[4]**2, lv[5]**2),
                        np.random.normal(lv[3]), 
                        np.random.poisson(abs(lv[4])) + np.random.beta(abs(lv[3]), 1), 
                        50*np.power(lv[2],2), 
                        3**(lv[5]*lv[6]),
                        np.random.normal(lv[5]+lv[3], 1),
                        np.random.gamma(abs(lv[2])), 
                        np.random.exponential(lv[2]**2), 
                        -np.random.normal(lv[5], 0.1), 
                        np.random.normal(1.7, 0.7)])
    return rvs

def deterministicDatasetCreator(a0, a1, nfeatures, noise, switch = 20, size = 200):
    ## create 2 data points exact opposite of a
    b0 = -a0
    b1 = -a1
    
    ##create full dataset with points adding apprropriate noise - #=label
    Xa1 = np.repeat(a1, size, axis = 0) + np.random.normal(0, noise[1], size = nfeatures*size).reshape(-1,nfeatures)
    Xa0 = np.repeat(a0, size, axis = 0) + np.random.normal(0, noise[1], size = nfeatures*size).reshape(-1,nfeatures)
    Xb1 = np.repeat(b1, size, axis = 0) + np.random.normal(noise[0], noise[1], size = nfeatures*size).reshape(-1,nfeatures)
    Xb0 = np.repeat(b0, size, axis = 0) + np.random.normal(noise[0], noise[1], size = nfeatures*size).reshape(-1,nfeatures)

    ##concatenate data
    X1 = np.concatenate((Xa1,Xa0), axis = 0)
    y1 = np.concatenate((np.ones(size), np.zeros(size)), axis = 0)
    X2 = np.concatenate((Xb1,Xb0), axis = 0)
    y2 = np.concatenate((np.ones(size), np.zeros(size)), axis = 0)

    ##switch some labels
    y1[0:switch], y1[-switch:-1] = 0, 1
    y2[0:switch*2], y2[-switch*2:-1] = 0, 1

    data = {'1': X1, '2': X2}
    label = {'1': y1, '2': y2}
    return data, label


def saveDataset(X,y, name):
    d1= np.concatenate((X, y.reshape(-1,1)), axis=1)
    np.savetxt(f'{DATA_DIR}/{name}.csv',d1)
    return


def dataset_creator2D(lv_pos, lv_neg, dataset, total_cases, pos_share, label_switch, coef_sep = 0.85, switch = False, seed = 1):
    np.random.seed(seed)
    random.seed(seed)
    coefs = 2
    
    ##pos_cases
    pos_cases = int(np.floor(total_cases * pos_share))
    values_pos = np.zeros((pos_cases, coefs))
    label_pos = np.zeros((pos_cases, 1))
    std_scale = 1
    for i in range(pos_cases):
        ##generate random LVs based on overall risk and LV's
        risk =  np.random.normal(1, 0.4)
        std = np.random.normal(0,0.5, 7)
        risk = risk + std
        lv_pos_person = np.array([c[0]*risk[i] if np.random.binomial(1,coef_sep, 1)[0] == 1 else c[1]*risk[i] for i,c in enumerate(zip(lv_pos, lv_neg))])
        if dataset == 1:
        ## generate variables from LV
            values_pos[i]=   variables_from_lv2D(lv_pos_person, dataset)
        else:
        ## generate variables from LV (different relationships)
            values_pos[i]=  variables_from_lv2D(lv_pos_person, dataset)

        label = 1
        ##label switch
        if np.random.binomial(1,label_switch, 1)[0] == 0:
            label = 0
        label_pos[i] = label

    ##neg_cases
    neg_cases = total_cases - pos_cases
    values_neg = np.zeros((neg_cases, coefs))
    label_neg = np.zeros((neg_cases, 1))
    for i in range(neg_cases):
        ##generate random LVs based on overall risk and LV's
        risk =  np.random.normal(1, 0.5)
        std = np.random.normal(0,1, 7)
        risk = risk + std
        lv_neg_person = np.array([c[0]*risk[i] if np.random.binomial(1,coef_sep, 1)[0] == 1 else c[1]*risk[i] for i,c in enumerate(zip(lv_neg, lv_pos))])
        if dataset == 1:
            ## generate variables from LV
            values_neg[i]=  variables_from_lv2D(lv_neg_person, dataset)

        else:
        ## generate variables from LV (different relationships)
            values_neg[i]=  variables_from_lv2D(lv_neg_person, dataset)
        
        label = 0
        ##label switch
        if np.random.binomial(1,label_switch, 1)[0] == 0:
            label = 1
        label_neg[i] = label
        
    values = np.concatenate((values_pos, values_neg))
    labels = np.concatenate((label_pos, label_neg))
    table = np.concatenate((values, labels), axis =1 )
    np.random.shuffle(table)
    return table

def non_iid_creator2D(frac, total_cases = 500):
    ##create datasets CLASSIFICATION
    ##latent variables
    lv_pos1 = np.array([0.5, 0.5])
    lv_neg1 = np.array([0.9, 0.25])

    lv_pos2 = np.array([-0.2, -0.2])
    lv_neg2 = np.array([-0.2, -0.2])  
    
    label_switch = 0.9
    coef_sep = 0.9
    pos_share = 0.5
    
    ##d1
    dataset = 1
    data_1 = dataset_creator2D(lv_pos1, lv_neg1, dataset, total_cases, pos_share, label_switch, coef_sep, seed = 1)
    X1, y1 =  data_1[:,:-1], data_1[:, -1]
    X1, y1 = shuffle(X1, y1)

    data_1b = dataset_creator2D(lv_pos1, lv_neg1, dataset, total_cases, pos_share,label_switch,  coef_sep, seed = 2)
    X1b, y1b = data_1b[:,:-1], data_1b[:, -1]
    
    ##d2
    dataset = 2
    data_2 = dataset_creator2D(lv_pos2, lv_neg2, dataset, total_cases, pos_share, label_switch, coef_sep, seed = 1)
    X2, y2 =  data_2[:,:-1], data_2[:, -1]
    
    
    ##mix datasets for 2nd dataset
    a = int(np.floor(total_cases * frac))
    X2, y2 = np.concatenate((X2[:a],X1b[a:])), np.concatenate((y2[:a],y1b[a:]))
    X2, y2 = shuffle(X2, y2)
    
    data, label = {}, {}
    data['1'], label['1'] = X1, y1
    data['2'], label['2'] = X2, y2
    return data, label

def variables_from_lv2D(lv, dataset):
    if dataset == 1:
        rvs = np.array([1.2*lv[0], 
                        2.1*lv[1]])
    else:
         rvs = np.array([-(lv[0]**2) - 3,
                        0.3*lv[1] - 3])
    return rvs