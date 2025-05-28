
from directories import paths
import torch.nn.functional as F
import torch.nn as nn
import torch
from unet import UNet
from torchvision.models import resnet18

dir_paths = paths()
ROOT_DIR = dir_paths.root_dir

class L2Normalize(torch.nn.Module):
    def __init__(self, dim=1):
        super(L2Normalize, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
    

class Synthetic(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Synthetic, self).__init__()
        self.input_size = 14
        self.hidden_size = 14
        
        self.fc = torch.nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 2)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = x.squeeze(1)
        return self.fc(x)

class Credit(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(Credit, self).__init__()
        self.input_size = 29
        self.hidden_size = [10]
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size[0]),
            nn.LayerNorm(self.hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size[0], 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        x = x.squeeze(1)
        return self.fc(x)

class EMNIST(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(EMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.norm1 = nn.LayerNorm(256)
        self.fc = nn.Linear(256, 120)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.norm2 = nn.LayerNorm(120)
        self.fc1 = nn.Linear(120, 20)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.norm3 = nn.LayerNorm(20)
        self.fc2 = nn.Linear(20, 10)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.norm1(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.norm2(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout2(out)
        
        out = self.norm3(out)
        out = self.fc2(out)
        return out


# class CIFAR(nn.Module):
#     def __init__(self, dropout_rate=0.25):
#         super(CIFAR, self).__init__()

#         # Convolutional Block 1
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # Input: 3x32x32
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 32x16x16
#         self.dropout_conv1 = nn.Dropout2d(p=dropout_rate * 0.5) # Optional: Spatial Dropout

#         # Convolutional Block 2
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Input: 32x16x16
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64x8x8
#         self.dropout_conv2 = nn.Dropout2d(p=dropout_rate * 0.5) # Optional: Spatial Dropout

#         # Convolutional Block 3
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Input: 64x8x8
#         self.relu3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128x4x4

#         # Fixed flattened size for 32x32 input
#         self.flattened_size = 128 * 4 * 4

#         # Fully Connected Layers
#         self.ln1 = nn.LayerNorm(self.flattened_size)
#         self.fc1 = nn.Linear(self.flattened_size, 256)
#         self.relu4 = nn.ReLU()
#         self.dropout_fc1 = nn.Dropout(p=dropout_rate)

#         self.ln2 = nn.LayerNorm(256)
#         self.fc2 = nn.Linear(256, 20) # Output layer for 10 CIFAR-10 classes
#         self.relu5 = nn.ReLU()
#         self.dropout_fc2 = nn.Dropout(p=dropout_rate)


#         self.ln3 = nn.LayerNorm(20)
#         self.fc3 = nn.Linear(20, 10) # Output layer for 10 CIFAR-10 classes

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#             elif isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # Conv Block 1
#         out = self.conv1(x)
#         out = self.relu1(out)
#         out = self.pool1(out)
#         out = self.dropout_conv1(out) # Apply dropout after pooling

#         # Conv Block 2
#         out = self.conv2(out)
#         out = self.relu2(out)
#         out = self.pool2(out)
#         out = self.dropout_conv2(out) # Apply dropout after pooling

#         # Conv Block 3
#         out = self.conv3(out)
#         out = self.relu3(out)
#         out = self.pool3(out)

#         # Flatten
#         out = out.view(out.size(0), -1) # Flatten the feature map

#         # FC Block 1
#         out = self.ln1(out) 
#         out = self.fc1(out)
#         out = self.relu4(out)
#         out = self.dropout_fc1(out)

#         # FC Block 2
#         out = self.ln2(out)
#         out = self.fc2(out)
#         out = self.relu5(out)
#         out = self.dropout_fc2(out)

#         # Output Layer
#         out = self.ln3(out) 
#         out = self.fc3(out)
#         return out
    
class CIFAR(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(CIFAR, self).__init__()
        
        # Custom scaling layer
        class Mul(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale
            def forward(self, x):
                return x * self.scale
        
        # Initial block with 2x2 kernel (no padding)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
            nn.GELU(),
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, padding='same', bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64), 
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(64), 
            nn.GELU(),
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding='same', bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256), 
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256), 
            nn.GELU(),
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256), 
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(256), 
            nn.GELU(),
        )
        
        # Pooling and scaling
        self.pool = nn.MaxPool2d(3)
        
        # Calculate the flattened size after pooling - for 32x32 input
        self.flattened_size = 256  # After multiple pooling layers and final MaxPool(3)
        
        # FC layers with LayerNorm from original implementation
        self.ln1 = nn.LayerNorm(self.flattened_size)
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.relu4 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)

        self.ln2 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 20)
        self.relu5 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)

        self.ln3 = nn.LayerNorm(20)
        self.fc3 = nn.Linear(20, 10)  # Output layer for 10 CIFAR-10 classes
        
        self._initialize_weights()
    
    def forward(self, x):
        # Convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Pooling
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers with LayerNorm
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout_fc1(x)

        x = self.ln2(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout_fc2(x)

        x = self.ln3(x)
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class IXITiny(nn.Module):
    def __init__(self):
        super(IXITiny, self).__init__()
        self.CHANNELS_DIMENSION = 1
        self.SPATIAL_DIMENSIONS = (2, 3, 4)
        self.load = False # pretrained
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
        
        # --- Determine bottleneck channels ---
        try:
            self.bottleneck_channels = self.model.bottom_block.conv2.conv_layer.out_channels
        except AttributeError:
            self.bottleneck_channels = 64 # fallback value
            print(f"Could not dynamically determine bottleneck channels, using hardcoded {self.bottleneck_channels}")


        # --- Add layers for representation extraction ---
        # Global Average Pooling for 3D feature maps
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # LayerNorm
        self.representation_layernorm = nn.LayerNorm(self.bottleneck_channels, elementwise_affine=False)

        # --- Load pretrained weights ---
        try:
            if self.load:
                checkpoint = torch.load(
                    f'{ROOT_DIR}/data/IXITiny/whole_images_epoch_5.pth',
                    map_location=torch.device('cpu')
                )
                self.model.load_state_dict(checkpoint['weights'])
        except Exception as e:
            print(f"Could not load pre-trained weights: {e}. Model will use initial weights.")


        # Enable gradient computation for all parameters
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, rep_vector = False):
        # Manually pass data through UNet components to extract bottleneck features
        skip_connections, features_for_bottom = self.model.encoder(x)
        # 2. Bottom block (Bottleneck)
        bottleneck_output_3d = self.model.bottom_block(features_for_bottom)
        # --- Representation Extraction ---
        pooled_representation = self.global_avg_pool(bottleneck_output_3d)

        flattened_representation = pooled_representation.view(pooled_representation.size(0), -1)

        self.representation_vector = self.representation_layernorm(flattened_representation)
        # --- End Representation Extraction ---
        if rep_vector:
            return self.representation_vector

        # 3. Decoder

        decoder_output = self.model.decoder(skip_connections, bottleneck_output_3d)

        # 4. Classifier (final 1x1x1 conv in UNet)
        logits = self.model.classifier(decoder_output)

        # 5. Softmax
        probabilities = F.softmax(logits, dim=self.CHANNELS_DIMENSION)

        return probabilities

class ISIC(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(ISIC, self).__init__()
        
        # Load pre-trained EfficientNet
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        
        # Freeze all layers except the final blocks
        # EfficientNet has 8 blocks (0-7) - we'll unfreeze only blocks 6 and 7 
        # plus the head (the final classifier section)
        for name, param in self.efficientnet.named_parameters():
            # Freeze all parameters by default
            param.requires_grad = False
            
            # Unfreeze only the last two blocks and the classifier
            if any(x in name for x in ['blocks.6', 'blocks.7', 'classifier']):
                param.requires_grad = True
        
        # Get the input features for the final fully connected layer
        self.features_dim = self.efficientnet.classifier.fc.in_features # This is 1280 for B0
        
        # Remove the original classifier
        self.efficientnet.classifier = nn.Identity()
        
        # Create new fully connected layers with LayerNorm similar to CIFAR model
        self.ln1 = nn.LayerNorm(self.features_dim)
        self.fc1 = nn.Linear(self.features_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.ln2 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        # Corrected dimensions for ln3 and fc3
        self.ln3 = nn.LayerNorm(64) # Expects 64 features from fc2/dropout2
        self.fc3 = nn.Linear(64, 8)  # Takes 64 features, outputs 8 classes
        
        self._initialize_weights()
    
    def forward(self, x):
        # Extract features using EfficientNet backbone
        features = self.efficientnet(x) # Output shape: [batch_size, 1280, H', W']
        
        # Apply Global Average Pooling
        # Input: [batch_size, 1280, H', W'], Output: [batch_size, 1280, 1, 1]
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Flatten the features
        # Input: [batch_size, 1280, 1, 1], Output: [batch_size, 1280]
        flattened_features = torch.flatten(pooled_features, 1)
        
        # Apply new FC layers with LayerNorm
        out = self.ln1(flattened_features) # ln1 receives [batch_size, 1280]
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.ln2(out) # ln2 receives [batch_size, 256]
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out) # dropout2 outputs [batch_size, 64]
        
        out = self.ln3(out) # ln3 (corrected) receives [batch_size, 64]
        out = self.fc3(out) # fc3 (corrected) receives [batch_size, 64]
        
        return out
    
    def _initialize_weights(self):
        # Initialize weights for FC layers
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def count_trainable_params(self):
        """Count and print the number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
        
        # Print breakdown by module
        for name, module in self.named_children():
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if module_params > 0:
                print(f"  - {name}: {module_params:,} trainable parameters")
        
        return trainable_params