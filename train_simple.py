#!/usr/bin/env python3
"""
Simplified training script for Low Light Enhancement
Works with minimal dependencies
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2
import random
from tqdm import tqdm

# Simple model architecture
class SimpleEnhancementNet(nn.Module):
    """Simplified U-Net for low light enhancement"""
    
    def __init__(self, in_channels=3, out_channels=3):
        super(SimpleEnhancementNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 512)
        
        # Decoder
        self.dec4 = self._conv_block(1024, 256)  # 512 + 512 from skip connection
        self.dec3 = self._conv_block(512, 128)   # 256 + 256 from skip connection
        self.dec2 = self._conv_block(256, 64)    # 128 + 128 from skip connection
        self.dec1 = self._conv_block(128, 64)    # 64 + 64 from skip connection
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        
        # Final output
        out = self.final(d1)
        return torch.tanh(out)  # Output in range [-1, 1]

def create_simple_dataset(num_samples=100, size=(256, 256)):
    """Create a simple synthetic dataset"""
    print("Creating synthetic dataset...")
    
    # Create directories
    os.makedirs("data/train/low", exist_ok=True)
    os.makedirs("data/train/normal", exist_ok=True)
    os.makedirs("data/val/low", exist_ok=True)
    os.makedirs("data/val/normal", exist_ok=True)
    
    # Generate synthetic images
    for i in tqdm(range(num_samples), desc="Generating images"):
        # Create a random pattern
        normal = np.random.randint(50, 200, (size[0], size[1], 3), dtype=np.uint8)
        
        # Add some structure
        cv2.rectangle(normal, (50, 50), (200, 200), (255, 255, 255), -1)
        cv2.circle(normal, (128, 128), 30, (0, 0, 0), -1)
        cv2.putText(normal, f"IMG{i:03d}", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Create low light version
        low_light = create_low_light_version(normal)
        
        # Save images
        split = "train" if i < num_samples * 0.8 else "val"
        cv2.imwrite(f"data/{split}/normal/img_{i:03d}.png", normal)
        cv2.imwrite(f"data/{split}/low/img_{i:03d}.png", low_light)
    
    print(f"✓ Created {num_samples} image pairs")

def create_low_light_version(image, gamma_range=(2.0, 3.5), brightness_range=(0.1, 0.4)):
    """Create low light version of image"""
    # Random gamma correction
    gamma = random.uniform(*gamma_range)
    low_light = np.power(image / 255.0, gamma) * 255.0
    
    # Random brightness reduction
    brightness = random.uniform(*brightness_range)
    low_light = low_light * brightness
    
    # Add noise
    noise = np.random.normal(0, 10, low_light.shape)
    low_light = np.clip(low_light + noise, 0, 255)
    
    return low_light.astype(np.uint8)

def load_image_pair(low_path, normal_path, size=(256, 256)):
    """Load and preprocess image pair"""
    low = cv2.imread(low_path)
    normal = cv2.imread(normal_path)
    
    # Resize
    low = cv2.resize(low, size)
    normal = cv2.resize(normal, size)
    
    # Convert to RGB
    low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
    normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    low = (low / 255.0) * 2 - 1
    normal = (normal / 255.0) * 2 - 1
    
    # Convert to tensor
    low = torch.from_numpy(low).permute(2, 0, 1).float()
    normal = torch.from_numpy(normal).permute(2, 0, 1).float()
    
    return low, normal

def train_epoch(model, train_files, device, optimizer, criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for low_path, normal_path in tqdm(train_files, desc="Training"):
        low, normal = load_image_pair(low_path, normal_path)
        low = low.unsqueeze(0).to(device)
        normal = normal.unsqueeze(0).to(device)
        
        optimizer.zero_grad()
        enhanced = model(low)
        loss = criterion(enhanced, normal)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_files)

def validate_epoch(model, val_files, device, criterion):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for low_path, normal_path in tqdm(val_files, desc="Validation"):
            low, normal = load_image_pair(low_path, normal_path)
            low = low.unsqueeze(0).to(device)
            normal = normal.unsqueeze(0).to(device)
            
            enhanced = model(low)
            loss = criterion(enhanced, normal)
            total_loss += loss.item()
    
    return total_loss / len(val_files)

def save_sample_results(model, val_files, device, epoch):
    """Save sample results"""
    model.eval()
    os.makedirs("results", exist_ok=True)
    
    with torch.no_grad():
        # Get first validation sample
        low_path, normal_path = val_files[0]
        low, normal = load_image_pair(low_path, normal_path)
        low = low.unsqueeze(0).to(device)
        normal = normal.unsqueeze(0).to(device)
        
        enhanced = model(low)
        
        # Convert back to images
        low_img = ((low.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        enhanced_img = ((enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        normal_img = ((normal.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
        
        # Create comparison
        comparison = np.hstack([low_img, enhanced_img, normal_img])
        cv2.imwrite(f"results/epoch_{epoch:03d}.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

def main():
    """Main training function"""
    print("Simple Low Light Enhancement Training")
    print("=" * 40)
    
    # Create dataset
    create_simple_dataset(num_samples=200)
    
    # Get file lists
    train_files = []
    val_files = []
    
    for split in ["train", "val"]:
        low_dir = f"data/{split}/low"
        normal_dir = f"data/{split}/normal"
        
        if os.path.exists(low_dir) and os.path.exists(normal_dir):
            low_files = sorted([f for f in os.listdir(low_dir) if f.endswith('.png')])
            normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith('.png')])
            
            for low_file, normal_file in zip(low_files, normal_files):
                low_path = os.path.join(low_dir, low_file)
                normal_path = os.path.join(normal_dir, normal_file)
                
                if split == "train":
                    train_files.append((low_path, normal_path))
                else:
                    val_files.append((low_path, normal_path))
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleEnhancementNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 20
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss = train_epoch(model, train_files, device, optimizer, criterion)
        
        # Validate
        if val_files:
            val_loss = validate_epoch(model, val_files, device, criterion)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "best_model_simple.pth")
                print("✓ New best model saved!")
        else:
            print(f"Train Loss: {train_loss:.4f}")
        
        # Save sample results
        if val_files and epoch % 5 == 0:
            save_sample_results(model, val_files, device, epoch)
    
    # Save final model
    torch.save(model.state_dict(), "final_model_simple.pth")
    print("\n✓ Training completed!")
    print("Models saved:")
    print("  - best_model_simple.pth")
    print("  - final_model_simple.pth")
    print("  - results/ (sample images)")

if __name__ == "__main__":
    main()
