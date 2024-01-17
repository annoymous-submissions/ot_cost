import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchio as tio
from pathlib import Path
import copy
from unet import UNet
import torch.nn.init as init

ROOT_DIR = ''
BATCH_SIZE = 64
LR = 5e-2
EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4

def load_data():
    training_split_ratio = 0.9
    dataset_dir = Path(f'{ROOT_DIR}/data/IXITiny')
    images_dir = dataset_dir / 'image'
    labels_dir = dataset_dir / 'label'
    image_paths = sorted(images_dir.glob('*.nii.gz'))
    label_paths = sorted(labels_dir.glob('*.nii.gz'))

    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
            brain=tio.LabelMap(label_path),
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    training_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(4),
        tio.CropOrPad((48, 60, 48)),
        tio.RandomMotion(p=0.2),
        tio.RandomBiasField(p=0.3),
        tio.RandomNoise(p=0.5),
        tio.RandomFlip(),
        tio.OneOf({
            tio.RandomAffine(): 0.8,
            tio.RandomElasticDeformation(): 0.2,
        }),
        tio.OneHot(),
    ])

    validation_transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(4),
        tio.CropOrPad((48, 60, 48)),
        tio.OneHot(),
    ])

    num_subjects = len(dataset)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

    training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

    training_batch_size = 8
    validation_batch_size = training_batch_size

    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=training_batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=validation_batch_size,
        )

    return train_loader, val_loader

class Autoencoder(nn.Module):
    def __init__(self, n_emb):
        super(Autoencoder, self).__init__()
        self.n_emb = n_emb
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2 * 2, self.n_emb),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.n_emb, 32 * 2 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (32, 2, 2, 2)),
            nn.ConvTranspose3d(32, 64, kernel_size=(2,3,2), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(3,2,3), stride=2, padding=1, output_padding=1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        init.zeros_(m.bias)

    def forward(self, x, embedding = False):
        x = self.encoder(x)
        if embedding:
            return x
        x = self.decoder(x)
        return x

def get_dice_score(output, target, epsilon=1e-9):
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
    return dice_score

def get_dice_loss(output, target):
    return 1 - get_dice_score(output, target)


def train_autoencoder(n_emb):
    # Initialize
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        upsampling_type='linear',
        padding=True,
        activation='PReLU',
    )
    checkpoint = torch.load(f'{ROOT_DIR}/data/IXITiny/whole_images_epoch_5.pth')
    model.load_state_dict(checkpoint['weights'])
    model.to(DEVICE)
    ae = Autoencoder(n_emb)
    ae = ae.to(DEVICE)
    criterion = get_dice_loss
    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)
    
    # Load data
    train_loader, val_loader = load_data()
    # Early stopping parameters
    patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')

    # Loss tracking
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        ae.train()
        
        # Training step
        train_loss = 0.0
        for input_tensor in train_loader:
            image = input_tensor['mri']['data'].float()
            image = image.to(DEVICE) 
            optimizer.zero_grad()
            
            #MODEL
            skip_connections, encoding = model.encoder(image)
            encoding = model.bottom_block(encoding)
            
            ### WITHOUT AE
            x = model.decoder(skip_connections, encoding)
            logits = model.classifier(x)
            
            ### WITH AE
            encoding_ = ae(encoding, embedding=False)
            x_ae = model.decoder(skip_connections, encoding_)
            logits_ae = model.classifier(x_ae)
            #OUTPUT
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            probabilities_ae = F.softmax(logits_ae, dim=CHANNELS_DIMENSION)
            
            losses = criterion(probabilities, probabilities_ae)
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_losses.append(train_loss / len(train_loader))
        
        # Validation step
        model.eval()
        ae.eval()
        
        val_loss = 0.0
        with torch.no_grad():
            for input_tensor in val_loader:
                image = input_tensor['mri']['data'].float()
                image = image.to(DEVICE)
                skip_connections, encoding = model.encoder(image)
                encoding = model.bottom_block(encoding)
                
                ### WITHOUT AE
                x = model.decoder(skip_connections, encoding)
                logits = model.classifier(x)
                
                ### WITH AE
                encoding_ = ae(encoding, embedding=False)
                x_ae = model.decoder(skip_connections, encoding_)
                logits_ae = model.classifier(x_ae)
                
                #OUTPUT
                probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
                probabilities_ae = F.softmax(logits_ae, dim=CHANNELS_DIMENSION)
                    
                losses = criterion(probabilities, probabilities_ae)
                
                val_loss += losses.mean().item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss / len(train_loader)}, Validation Loss: {val_loss}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_model = copy.deepcopy(ae)
            best_val_loss = val_loss
            early_stopping_counter = 0
        elif val_loss < min(val_losses[-5:]):
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping")
                break
    
    best_model.to('cpu')
    torch.save(best_model.state_dict(), f'{ROOT_DIR}/data/IXITiny/model_checkpoint_{n_emb}.pth')
    return train_losses, val_losses


def main():
    print(DEVICE)

    n_embs = [256, 512, 1024, 2048, 4096]
    losses = {}
    for n_emb in n_embs:
        losses[n_emb] = train_autoencoder(n_emb)
    with open(f'{ROOT_DIR}/data/IXITiny/losses.json', 'w') as f:
        json.dump(losses, f)
    print("Losses saved to file.")

if __name__ == '__main__':
    main()
    