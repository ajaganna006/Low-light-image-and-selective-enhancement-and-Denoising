"""
Simplified DEDUNet: Dual-Enhancing Dense-UNet for Low-Light Image Enhancement
A working implementation with all the key features but simplified architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DFCAttention(nn.Module):
    """
    Decoupled Fully Connection (DFC) Attention Module
    """
    def __init__(self, channels, reduction=16):
        super(DFCAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # Spatial attention
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Decoupled connection
        self.decoupled_conv = nn.Conv2d(channels, channels, 1, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        # Spatial attention
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.conv_spatial(torch.cat([avg_spatial, max_spatial], dim=1))
        
        # Apply attention
        x_att = x * channel_att * spatial_att
        
        # Decoupled connection
        x_decoupled = self.decoupled_conv(x_att)
        
        return x + x_decoupled


class DenseBlock(nn.Module):
    """
    Dense Block with skip connections
    """
    def __init__(self, in_channels, growth_rate=16, num_layers=3):
        super(DenseBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate))
        
        self.layers = nn.ModuleList(layers)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def _make_dense_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, 3, 1, 1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = [x]
        
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        out = torch.cat(features, dim=1)
        out = self.bottleneck(out)
        
        return out


class DualEnhancementBlock(nn.Module):
    """
    Dual Enhancement Block for brightness enhancement and noise reduction
    """
    def __init__(self, channels):
        super(DualEnhancementBlock, self).__init__()
        
        # Brightness enhancement branch
        self.brightness_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # For brightness adjustment
        )
        
        # Noise reduction branch
        self.noise_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # For noise suppression
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        brightness_att = self.brightness_branch(x)
        noise_att = self.noise_branch(x)
        
        # Apply attention
        brightness_enhanced = x * brightness_att
        noise_reduced = x * (1 - noise_att)
        
        # Fusion
        fused = torch.cat([brightness_enhanced, noise_reduced], dim=1)
        out = self.fusion(fused)
        
        return out + x


class DEDUNetSimple(nn.Module):
    """
    Simplified DEDUNet for Low-Light Image Enhancement
    """
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super(DEDUNetSimple, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            DenseBlock(base_channels * 2, growth_rate=16),
            DFCAttention(base_channels * 2),
            nn.MaxPool2d(2)
        )
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            DenseBlock(base_channels * 4, growth_rate=16),
            DFCAttention(base_channels * 4),
            nn.MaxPool2d(2)
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            DenseBlock(base_channels * 8, growth_rate=16),
            DFCAttention(base_channels * 8),
            nn.MaxPool2d(2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, 1, 1),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            DenseBlock(base_channels * 16, growth_rate=16),
            DFCAttention(base_channels * 16),
            DualEnhancementBlock(base_channels * 16)
        )
        
        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 3, 2, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 3, 2, 1, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, 2, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        # Dual enhancement blocks in decoder
        self.dual_enhance4 = DualEnhancementBlock(base_channels * 8)
        self.dual_enhance3 = DualEnhancementBlock(base_channels * 4)
        self.dual_enhance2 = DualEnhancementBlock(base_channels * 2)
        self.dual_enhance1 = DualEnhancementBlock(base_channels)
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Bottleneck
        bottleneck = self.bottleneck(e4)
        
        # Decoder with skip connections
        d4 = self.decoder4(bottleneck)
        d4 = self.dual_enhance4(d4)
        
        d3 = self.decoder3(d4)
        d3 = self.dual_enhance3(d3)
        
        d2 = self.decoder2(d3)
        d2 = self.dual_enhance2(d2)
        
        d1 = self.decoder1(d2)
        d1 = self.dual_enhance1(d1)
        
        # Final output
        out = self.final_conv(d1)
        
        return out


def create_dedunet_simple(in_channels=3, out_channels=3, base_channels=64):
    """
    Create simplified DEDUNet model instance
    """
    model = DEDUNetSimple(in_channels, out_channels, base_channels)
    return model


def count_parameters(model):
    """
    Count the number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = create_dedunet_simple()
    print(f"DEDUNet Simple Model Created!")
    print(f"Total Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

