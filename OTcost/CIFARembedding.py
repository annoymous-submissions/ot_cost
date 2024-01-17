import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import os
import torch
import torch.nn as nn
import torchvision.models as models

ROOT_DIR = ''
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Download data
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR100(root=f'{ROOT_DIR}/data/CIFAR/', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR100(root=f'{ROOT_DIR}/data/CIFAR/', train=False, download=True, transform=transform)

#Load data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, n_emb):
        super(Autoencoder, self).__init__()
        self.n_emb = n_emb
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
        )
        
        self.bottleneck = nn.Sequential(
            nn.Linear(16 * 8 * 8, self.n_emb),
            nn.ReLU()
        )

        self.expand = nn.Sequential(
            nn.Linear(self.n_emb, 16 * 8 * 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 3, 3, padding=1),
                    nn.Sigmoid())
        
    def forward(self, x, get_embedding=False):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        embedding = self.bottleneck(x_flat)
        x = self.expand(embedding)
        x = x.view(x.size(0), 16, 8, 8)
        x = self.decoder(x)
        if get_embedding:
            return embedding
        return x


#Train autoencoder
def train_autoencoder(n_emb):
    model = Autoencoder(n_emb)
    if f'model_checkpoint_{n_emb}_2.pth' in os.listdir(f'{ROOT_DIR}/data/CIFAR/'):
        state_dict = torch.load(f'{ROOT_DIR}/data/CIFAR/model_checkpoint_{n_emb}_2.pth')
        model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = np.inf
    patience = 10
    no_improvement_count = 0 

    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        train_loss_sum = 0
        num_batches = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            train_loss = criterion(outputs, inputs)
            train_loss_sum += train_loss.item()
            num_batches += 1
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        avg_train_loss = train_loss_sum / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        with torch.no_grad():
            val_loss_sum = 0
            num_batches = 0
            for inputs, _ in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                val_loss = criterion(outputs, inputs)
                val_loss_sum += val_loss.item()
                num_batches += 1
        avg_val_loss = val_loss_sum / num_batches if num_batches > 0 else 0
        val_losses.append(avg_val_loss)
        if epoch % 10 == 0:
            print(avg_train_loss, avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            model.to('cpu')
            torch.save(model.state_dict(), f'{ROOT_DIR}/data/CIFAR/model_checkpoint_{n_emb}.pth')
            model.to(DEVICE)
            best_val_loss = avg_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Stopping early after {patience} epochs without improvement.")
                break
    return best_val_loss, train_losses, val_losses

def main():
    print(DEVICE)
    n_embs = [250, 500, 750, 1000, 2000]
    losses = {}
    for n_emb in n_embs:
        losses[n_emb] = train_autoencoder(n_emb)
    with open(f'{ROOT_DIR}/data/CIFAR/losses_2.json', 'w') as f:
        json.dump(losses, f)
    print("Losses saved to file.")

if __name__ == '__main__':
    main()
    