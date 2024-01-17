global ROOT_DIR
ROOT_DIR = ''
global DATA_DIR
DATA_DIR = f'{ROOT_DIR}/data/ISIC'

import sys
import json
import os
import torch
sys.path.append(f'{ROOT_DIR}/code/ISIC/')
import torch.nn.functional as F
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader as dl
from torch.optim.lr_scheduler import ExponentialLR
import copy
from multiprocessing import Pool


BATCH_SIZE = 128
LR = 5e-2
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, n_emb):
        super(Autoencoder, self).__init__()
        self.n_emb = n_emb
        self.bottleneck_dim = (16,25,25)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 16, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(16),
)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(self.bottleneck_dim[0] * self.bottleneck_dim[1] * self.bottleneck_dim[2], self.n_emb)
)
        self.expand = nn.Sequential(nn.Linear(self.n_emb, self.bottleneck_dim[0] * self.bottleneck_dim[1] * self.bottleneck_dim[2]))

        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(32),      
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(32),   
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(16),   
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1), nn.LeakyReLU(0.1), nn.BatchNorm2d(16), 
            nn.Conv2d(16, 3, 3, padding=1),    
            nn.Sigmoid()
)
    

    def forward(self, x, get_embedding=False):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        embedding = self.bottleneck(x_flat)
        if get_embedding:
            return embedding
        x = self.expand(embedding)
        x = x.view(x.size(0), self.bottleneck_dim[0], self.bottleneck_dim[1], self.bottleneck_dim[2])
        x = self.decoder(x)
        return x



def train_autoencoder(n_emb):
    # Initialize
    model = Autoencoder(n_emb)
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    # Load data
    train_data = dataset.FedIsic2019(train=True, pooled = True, data_path=DATA_DIR)
    

    val_data = dataset.FedIsic2019(train=False, pooled = True, data_path=DATA_DIR)
    

    # Early stopping parameters
    patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')

    # Loss tracking
    train_losses = []
    val_losses = []

    subset_batches_train = 25
    subset_batches_val = 8

    # Training loop
    for epoch in range(EPOCHS):
        #inside loop as i subset the training per epoch
        train_loader = dl(train_data, batch_size = BATCH_SIZE, shuffle = True)
        val_loader = dl(val_data, batch_size = BATCH_SIZE, shuffle = True)
        
        model.train()
        
        # Training step
        train_loss = 0.0
        for i, (image, label) in enumerate(train_loader):
            if i>= subset_batches_train:
                break
            image = image.transpose(2,1)
            image = image.to(DEVICE) 
            optimizer.zero_grad()
            reconstructed = model(image)
            loss = criterion(reconstructed, image)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        lr_scheduler.step()   
        train_losses.append(train_loss / subset_batches_train)
        
        # Validation step
        model.eval()
        if epoch % 10 == 0:
            val_loss = 0.0
            with torch.no_grad():
                for i, (image, label) in enumerate(val_loader):
                    if i >= subset_batches_val:
                        break
                    image = image.transpose(2,1)
                    image = image.to(DEVICE)
                    reconstructed = model(image)
                    loss = criterion(reconstructed, image)
                    
                    val_loss += loss.item()
                    
            val_loss /= subset_batches_val
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_model = copy.deepcopy(model)
                best_val_loss = val_loss
                early_stopping_counter = 0
                best_model.to('cpu')
                torch.save(best_model.state_dict(), f'{ROOT_DIR}/data/ISIC/model_checkpoint_{n_emb}.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping")
                    break
    
    best_model.to('cpu')
    torch.save(best_model.state_dict(), f'{ROOT_DIR}/data/ISIC/model_checkpoint_{n_emb}.pth')
    return train_losses, val_losses


def main(n_emb):
    return n_emb, train_autoencoder(n_emb)

if __name__ == '__main__':
    n_embs = [512, 1024, 2048, 4096]
    cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', 5))
    if DEVICE == 'cpu':
        with Pool(cpu) as pool:
            results = pool.map(main, n_embs)
    else:
        results = []
        for n_emb in n_embs:
            results.append(main(n_emb))
        
    losses = {}
    for n_emb, loss in results:
        losses[n_emb] = loss
    with open(f'{ROOT_DIR}/data/ISIC/losses.json', 'w') as f:
        json.dump(losses, f)
    print("Losses saved to file.")
    