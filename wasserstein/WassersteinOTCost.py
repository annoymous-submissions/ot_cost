#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.warnings.filterwarnings('ignore', category=RuntimeWarning)
from itertools import product
import ot
from sklearn.preprocessing import normalize
import sympy as sym
from scipy.linalg import eigh, sqrtm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TABULAR =  {'Synthetic', 'Credit', 'Weather'}
IMAGE = {'CIFAR', 'EMNIST', 'IXITiny', 'ISIC'}

class OTCost:
    def __init__(self, dataset, data, label, private=False, lam=1e-3):
        self.dataset = dataset
        self.data = data
        self.label = label
        self.private = private
        self.lam = lam
        self.label_costs = []
        self.feature_costs = []
    
    def feature_cost(self, i, index_1, index_2):
        if not self.private:
            part_X1 = self.data['1'][index_1]
            part_X2 = self.data['2'][index_2]
            vector_dim = part_X1.shape[1]
            if vector_dim > 1000:
                part_X1, part_X2 = compress_vector((-1,1), part_X1, part_X2)
            feature_cost = ot.dist(part_X1, part_X2) # EUCLIDEAN DISTANCE
            self.feature_costs.append((i, np.percentile(feature_cost, 10), np.percentile(feature_cost, 25), np.percentile(feature_cost, 50), np.percentile(feature_cost, 75), np.percentile(feature_cost, 90)))
        else:
            feature_cost = privateDotproduct(self.data, index_1, index_2)
        return feature_cost
    
    def label_cost(self, i, index_1, index_2):
        part_X1 = self.data['1'][index_1]
        part_X2 = self.data['2'][index_2]
        
        ## Very large embeddings give degenerate covariance matrics, compression stops this
        compressed_X1, compressed_X2 = compress_vector(i, part_X1, part_X2)
        mu_1, sigma_1 = get_normal_params(compressed_X1)
        mu_2, sigma_2 = get_normal_params(compressed_X2)

        label_cost = label_distance(mu_1, sigma_1, mu_2, sigma_2)
        while label_cost == None:
            #repeat with smaller number of PC's
            n_components = compressed_X1.shape[1]
            partway = n_components  - n_components // 8
            compressed_X1 = compressed_X1[:,:partway]
            compressed_X2 = compressed_X2[:,:partway]
            mu_1, sigma_1 = get_normal_params(compressed_X1)
            mu_2, sigma_2 = get_normal_params(compressed_X2)
            label_cost = label_distance(mu_1, sigma_1, mu_2, sigma_2)
        self.label_costs.append((i, label_cost))
        return label_cost

    def total_cost(self):
        n1, n2 = self.data['1'].shape[0], self.data['2'].shape[0]
        costs_all = np.zeros((n1,n2))
        label_perm = list(product(set(self.label['1']), set(self.label['2']), repeat = 1))
        for i in label_perm:
            ## pull data with desired labels
            index_1 = np.argwhere(self.label['1'] == i[0]).reshape(1,-1)[0]
            index_2 = np.argwhere(self.label['2'] == i[1]).reshape(1,-1)[0] 
            feat_cost = self.feature_cost(i, index_1, index_2)              
            label_cost = self.label_cost(i, index_1, index_2)
            normalization_param = 3

            cost = feat_cost + label_cost
            cost /= normalization_param
            #Fill cost matrix
            for idx1_val, idx1_index in enumerate(index_1):
                for idx2_val, idx2_index in enumerate(index_2):
                    costs_all[idx1_index, idx2_index] = cost[idx1_val, idx2_val]
        self.costs_all = costs_all
        return
            
    def calculate_ot_cost(self):
        self.total_cost()
        #Stability param ensures the algorithm works with the epsilon in the algorithm
        costs_stable = self.costs_all / (self.costs_all.mean()*5)
        a, b = np.ones((costs_stable.shape[0])) / costs_stable.shape[0], np.ones((costs_stable.shape[1])) / costs_stable.shape[1]
        self.Gs = ot.bregman.sinkhorn_stabilized(a, b, costs_stable, self.lam, stopThr=1e-6, numItermax=20000, warn = True, verbose=False)
        ot_cost = (self.Gs * self.costs_all).sum() * (self.costs_all.mean()*5)
        print(f'cost: {"{:.2f}".format(ot_cost)}')
        return ot_cost


def get_normal_params(part_data):
    mu = np.mean(part_data, axis=0)
    sigma = np.cov(part_data, rowvar=False)
    return mu, sigma

def label_distance(mu_1, sigma_1, mu_2, sigma_2):
    mu_diff = mu_1 - mu_2
    euclidean_distance = np.linalg.norm(mu_diff)**2
    sigma_a_sqrt = sqrtm(sigma_1)
    sigma_b_sqrt = sqrtm(sigma_2)
    frobenius_norm = np.linalg.norm(sigma_a_sqrt - sigma_b_sqrt, 'fro')**2
    cost = euclidean_distance + frobenius_norm
    if np.isnan(cost):
        cost = None
        print('Degenerate vector, reducing dimension')
    return cost

def compress_vector(i, data_1, data_2):
    #compression doesnt violate privacy as its fit on one dataset only
    # Standardize the datasets
    scaler = StandardScaler()
    scaled_data_1 = scaler.fit_transform(data_1)
    scaled_data_2 = scaler.transform(data_2)

    # Apply PCA transformation,
    vr = 5 if i[0] == i[1] else 0.8
    pca = PCA(n_components=vr)
    compressed_1 = pca.fit_transform(scaled_data_1)
    compressed_2 = pca.transform(scaled_data_2)

    return compressed_1, compressed_2

################## PRIVATE DOT PRODUCT  ################## 

def prepData(data, index_i, index_j):

    ## transpose datasets
    X1 = data['1'].T
    X2 = data['2'].T

    ##select the correct indeices for the labels
    X1 = X1[:, index_i]
    X2 = X2[:, index_j]
    ## normalise vecotrs on l2 norm

    X1 = normalize(X1, norm = 'l2', axis = 0)
    X2 = normalize(X2, norm = 'l2', axis = 0)
    n = X1.shape[0]
    ## select large prime
    p = sym.nextprime(n)

    return X1, X2, n, p

def generatorMatrix(k,p):
    V = np.vander(np.arange(1,k//2 + 1), p - 1, increasing = True) % p
    G = np.array(sym.Matrix(V).rref(pivots = False), dtype = float)
    #G = rref(V, tol = 1.0e-12)
    remaining_cols = G.shape[1] - (p-1-G.shape[1])
    G_ = G[:,:remaining_cols]
    
    ##Break down matrix G and G^-1 into A,B,C,D
    A = G_.T
    D = np.hstack((G_[:,k//2:].T, G_[:,:k//2] - 2* G_[:,:k//2]))
    ##B1 = I, B2 = -I + A2
    B = np.vstack((np.eye(k//2), np.eye(k//2)- 2*np.eye(k//2) + A[k//2:]))
    ##C1 = I-A2, C2 = I
    C =  np.hstack((np.eye(k//2) - A[k//2:], np.eye(k//2)))
    return A, B, C, D

def multipartyComp(X, M):
    X1, X2 = X
    A, B, C, D = M

    ##party a
    X1_a = np.dot(X1.T, A)
    X1_b = np.dot(X1.T, B)

    ##party b
    X2_a = np.dot(C, X2)
    X2_b = np.dot(D, X2)

    ##join
    V1 = np.dot(X1_a, X2_a)
    V2 = np.dot(X1_b, X2_b)

    return V1, V2

def privateDotproduct(data, index_i, index_j):
    X1, X2, n, p = prepData(data, index_i, index_j)
    A, B, C, D = generatorMatrix(n, p)
    X, M = (X1, X2), (A, B, C, D)
    V1, V2 = multipartyComp(X, M)
    dot_product = V1 + V2
    return 1 - dot_product