"""
Deep learning models for low light image enhancement and denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class DenseBlock(nn.Module):
    """Dense block for feature extraction"""
    
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, 1, 1),
                nn.Dropout2d(0.2)
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class LowLightEnhancementNet(nn.Module):
    """
    Main network for low light image enhancement and denoising
    Simplified U-Net architecture with attention mechanisms
    """
    
    def __init__(self, in_channels=3, out_channels=3, num_filters=64, num_blocks=8):
        super(LowLightEnhancementNet, self).__init__()
        
        # Encoder
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, 7, 1, 3),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks with fixed channel progression
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attention = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        encoder_channels = [num_filters, num_filters*2, num_filters*4, num_filters*8]
        
        for i, channels in enumerate(encoder_channels):
            # Residual block
            self.encoder_blocks.append(ResidualBlock(
                encoder_channels[i-1] if i > 0 else num_filters, 
                channels
            ))
            
            # Attention
            self.encoder_attention.append(CBAM(channels))
            
            # Downsample (except for last layer)
            if i < len(encoder_channels) - 1:
                self.downsample.append(nn.Conv2d(channels, channels, 3, 2, 1))
        
        # Bottleneck
        bottleneck_channels = num_filters * 8
        self.bottleneck = nn.Sequential(
            ResidualBlock(bottleneck_channels, bottleneck_channels),
            CBAM(bottleneck_channels),
            ResidualBlock(bottleneck_channels, bottleneck_channels)
        )
        
        # Decoder with matching channel progression
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        decoder_channels = [num_filters*4, num_filters*2, num_filters, num_filters]
        
        for i, channels in enumerate(decoder_channels):
            # Upsample
            self.upsample.append(nn.ConvTranspose2d(
                bottleneck_channels if i == 0 else decoder_channels[i-1],
                channels, 2, 2
            ))
            
            # Skip connection adjustment
            if i < len(encoder_channels) - 1:
                skip_channels = encoder_channels[-(i+2)]  # Corresponding encoder channels
                self.skip_connections.append(nn.Conv2d(skip_channels, channels, 1))
            
            # Attention
            self.decoder_attention.append(CBAM(channels))
            
            # Residual block
            self.decoder_blocks.append(ResidualBlock(channels, channels))
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, 1, 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, out_channels, 1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x = self.initial_conv(x)
        encoder_features = []
        
        for i in range(4):
            x = self.encoder_blocks[i](x)
            x = self.encoder_attention[i](x)
            encoder_features.append(x)
            if i < 3:  # Don't downsample the last encoder layer
                x = self.downsample[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i in range(4):
            # Upsample
            x = self.upsample[i](x)
            
            # Skip connection
            if i < len(encoder_features) - 1:  # Skip connection for first 3 decoder layers
                skip_feature = encoder_features[-(i+2)]  # Get corresponding encoder feature
                skip_feature = self.skip_connections[i](skip_feature)  # Adjust channels
                x = x + skip_feature
            
            # Attention
            x = self.decoder_attention[i](x)
            
            # Residual block
            x = self.decoder_blocks[i](x)
        
        # Final output
        x = self.final_conv(x)
        
        return x

class Discriminator(nn.Module):
    """
    Discriminator for adversarial training (optional)
    """
    
    def __init__(self, in_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # First layer
            nn.Conv2d(in_channels, num_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Hidden layers
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(num_filters * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class VGGFeatureExtractor(nn.Module):
    """
    VGG feature extractor for perceptual loss
    """
    
    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pre-trained VGG19
        import torchvision
        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.features = nn.ModuleList()
        self.feature_layers = feature_layers
        
        for i, layer in enumerate(vgg):
            if i in feature_layers:
                self.features.append(layer)
            if i == max(feature_layers):
                break
    
    def forward(self, x):
        features = []
        for layer in self.features:
            x = layer(x)
            features.append(x)
        return features

def create_model(config):
    """Create and return the model"""
    model = LowLightEnhancementNet(
        in_channels=config.NUM_CHANNELS,
        out_channels=config.NUM_CHANNELS,
        num_filters=config.NUM_FILTERS,
        num_blocks=config.NUM_BLOCKS
    )
    
    return model

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
