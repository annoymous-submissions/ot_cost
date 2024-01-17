global ROOT_DIR
ROOT_DIR = ''
global DATA_DIR
DATA_DIR = f'{ROOT_DIR}/data/ISIC'

import sys
import os
sys.path.append(f'{ROOT_DIR}/code/ISIC/')
sys.path.append(f'{ROOT_DIR}/code/ISIC/vgg_ae')
sys.path.append(f'{ROOT_DIR}/code/helper/')
import numpy as np
import pandas as pd
import vgg
import dataset
import importlib
importlib.reload(dataset)
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader as dl
from collections import OrderedDict
from multiprocessing import Pool

BATCH_SIZE = 8


def load_model():
    ## Pretrained
    configs = vgg.get_configs('vgg16')
    model = vgg.VGGAutoEncoder(configs)
    checkpoint = torch.load(f'{ROOT_DIR}/code/ISIC/vgg_ae/imagenet-vgg16.pth', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module.' in key:
            name = key[7:] # remove 'module.' prefix
        else:
            name = key 
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict)
    return model

def extract_embedding(model, image):
    B = image.shape[0]
    with torch.no_grad():
        image = image.transpose(2,1)
        embedding = model(image, embedding = True)
        return embedding.reshape(B,-1).detach()

def create_embedding(loader, center):
    labels_list = []
    names_list = []
    model = load_model()
    for images, labels, paths in loader:
        embeddings = extract_embedding(model, images)
        image_names = [p.split('/')[-1].split('.')[0] for p in paths]
        names_list.extend(image_names)
        emb_paths = [f'{DATA_DIR}/embedding/center_{center}_{image_name}' for image_name in image_names]
        for i in range(len(emb_paths)):
            emb_save = embeddings[i].numpy()
            emb_path = emb_paths[i]
            np.save(emb_path, emb_save)
        labels_list.append(labels)   
    all_labels = (torch.cat(labels_list, dim=0)).numpy()
    labels_df = pd.DataFrame({
                    "Name": names_list,
                    "Label": all_labels
                })
    labels_df.to_csv(f'{DATA_DIR}/center_{center}_labels.csv', index = False)
    return

def main(i):
    train_data = dataset.FedIsic2019(center = i, train=True, pooled = False, data_path=DATA_DIR)
    val_data = dataset.FedIsic2019(center = i, train=False, pooled = False, data_path=DATA_DIR)
    train_loader = dl(train_data, batch_size = BATCH_SIZE, shuffle = False)
    val_loader = dl(val_data, batch_size = BATCH_SIZE, shuffle = False)
    create_embedding(train_loader, center = i)
    create_embedding(val_loader, center = i)

if __name__ == '__main__':
    cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 5))
    centers = range(6)
    with Pool(cpu) as pool:
            results = pool.map(main, centers)