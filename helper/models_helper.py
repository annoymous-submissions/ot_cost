global ROOT_DIR
ROOT_DIR = ''

import pandas as pd
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.append(f'{ROOT_DIR}/code/helper')
import data_preprocessing as dp
import trainers as tr
import pipeline as pp
import importlib
import torch.nn.functional as F
from torchvision import models
from unet import UNet
importlib.reload(dp)
importlib.reload(tr)
importlib.reload(pp)
from unet import UNet
from torchvision import models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Synthetic(torch.nn.Module):
    def __init__(self):
            super(Synthetic, self).__init__()
            self.input_size = 12
            self.hidden_size = [18, 6]
            self.fc = torch.nn.Sequential(nn.Linear(self.input_size, self.hidden_size[0]),
                                            nn.ReLU(),
                                            nn.Dropout(0.3),
                                            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size[1], 1))
            self.sigmoid = torch.nn.Sigmoid()
            for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
            output = self.fc(x)
            output = self.sigmoid(output)
            return output
    
class Credit(torch.nn.Module):
    def __init__(self):
            super(Credit, self).__init__()
            self.input_size = 28
            self.hidden_size  = [56,56,28]
            self.fc = nn.Sequential(
                    nn.Linear(self.input_size, self.hidden_size[0]),
                    nn.BatchNorm1d(self.hidden_size[0]),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                    nn.BatchNorm1d(self.hidden_size[1]),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.hidden_size[1], self.hidden_size[2]),
                    nn.BatchNorm1d(self.hidden_size[2]),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size[2], 1)
            )
            for layer in self.fc:
                if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        nn.init.constant_(layer.bias, 0)
            self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
            output = self.fc(x)
            output = self.sigmoid(output)
            return output
    
class Weather(torch.nn.Module):
    def __init__(self):
            super(Weather, self).__init__()
            self.input_size = 123
            self.hidden_size  = [123,123,50]
            self.fc = nn.Sequential(
                    nn.Linear(self.input_size, self.hidden_size[0]),
                    nn.BatchNorm1d(self.hidden_size[0]),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                    nn.BatchNorm1d(self.hidden_size[1]),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(self.hidden_size[1], self.hidden_size[2]),
                    nn.BatchNorm1d(self.hidden_size[2]),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size[2], 1)
            )
            for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                            nn.init.constant_(layer.bias, 0)
    def forward(self, x):
            output = self.fc(x)
            return output

class EMNIST(nn.Module):
    def __init__(self, CLASSES):
        super(EMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, CLASSES)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

class CIFAR(nn.Module):
    def __init__(self, CLASSES):
        super(CIFAR, self).__init__()
        
        self.resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_ftrs, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, CLASSES)
        )
        for layer in self.resnet.fc:
                if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        nn.init.constant_(layer.bias, 0)
    def forward(self, x):
        x = self.resnet(x)
        return x
    
class IXITiny(nn.Module):
    def __init__(self):
        super(IXITiny, self).__init__()
        self.CHANNELS_DIMENSION = 1
        self.SPATIAL_DIMENSIONS = 2, 3, 4

        self.model = UNet(
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
        checkpoint = torch.load(f'{ROOT_DIR}/data/IXITiny/whole_images_epoch_5.pth', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['weights'])

        for name, param in self.named_parameters():
                param.requires_grad = True

    def forward(self, x):
        logits = self.model(x)
        probabilities = F.softmax(logits, dim=self.CHANNELS_DIMENSION)
        return probabilities
    
    def initialize_weights(self):
        if isinstance(self.classifier, nn.Conv3d):
            nn.init.xavier_normal_(self.classifier.weight.data)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias.data, 0)

class ISIC(nn.Module):
    def __init__(self):
        super(ISIC, self).__init__()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        for _, param in self.efficientnet.named_parameters():
            param.requires_grad = True
        self.efficientnet.classifier.fc = nn.Linear(1280, 8)

    def forward(self, x):
        logits = self.efficientnet(x)
        return logits

    def initialize_weights(self):
        nn.init.xavier_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, 0)

        nn.init.xavier_normal_(self.features.weight.data)
        if self.features.bias is not None:
            nn.init.constant_(self.features.bias.data, 0)