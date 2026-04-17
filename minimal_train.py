"""
Minimal training script for low light image enhancement
Works with available data and creates a simple model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

class SimpleDataset(Dataset):
    """Simple dataset for training"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = self._load_pairs()
    
    def _load_pairs(self):
        """Load image pairs"""
        pairs = []
        
        # Look for low and normal directories
        low_dir = os.path.join(self.data_dir, "low")
        normal_dir = os.path.join(self.data_dir, "normal")
        
        if os.path.exists(low_dir) and os.path.exists(normal_dir):
            low_images = sorted(os.listdir(low_dir))
            normal_images = sorted(os.listdir(normal_dir))
            
            for low_img, normal_img in zip(low_images, normal_images):
                if (low_img.lower().endswith(('.png', '.jpg', '.jpeg')) and 
                    normal_img.lower().endswith(('.png', '.jpg', '.jpeg'))):
                    pairs.append({
                        'low': os.path.join(low_dir, low_img),
                        'normal': os.path.join(normal_dir, normal_img)
                    })
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load images
        low_img = cv2.imread(pair['low'])
        normal_img = cv2.imread(pair['normal'])
        
        # Convert to RGB
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256
        low_img = cv2.resize(low_img, (256, 256))
        normal_img = cv2.resize(normal_img, (256, 256))
        
        # Convert to tensors and normalize to [-1, 1]
        low_tensor = torch.from_numpy(low_img).permute(2, 0, 1).float() / 255.0 * 2 - 1
        normal_tensor = torch.from_numpy(normal_img).permute(2, 0, 1).float() / 255.0 * 2 - 1
        
        return low_tensor, normal_tensor

class MinimalEnhancementNet(nn.Module):
    """Minimal enhancement network"""
    
    def __init__(self):
        super(MinimalEnhancementNet, self).__init__()
        
        # Simple encoder-decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for low_light, normal in pbar:
        low_light = low_light.to(device)
        normal = normal.to(device)
        
        optimizer.zero_grad()
        enhanced = model(low_light)
        loss = criterion(enhanced, normal)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for low_light, normal in pbar:
            low_light = low_light.to(device)
            normal = normal.to(device)
            
            enhanced = model(low_light)
            loss = criterion(enhanced, normal)
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def save_sample_results(model, dataloader, device, epoch, output_dir):
    """Save sample results"""
    model.eval()
    
    with torch.no_grad():
        for low_light, normal in dataloader:
            low_light = low_light.to(device)
            normal = normal.to(device)
            
            enhanced = model(low_light)
            
            # Convert to images
            def tensor_to_image(tensor):
                img = (tensor + 1) / 2
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).cpu().numpy()
                return (img * 255).astype(np.uint8)
            
            # Create comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(tensor_to_image(low_light[0]))
            axes[0].set_title('Low Light')
            axes[0].axis('off')
            
            axes[1].imshow(tensor_to_image(enhanced[0]))
            axes[1].set_title('Enhanced')
            axes[1].axis('off')
            
            axes[2].imshow(tensor_to_image(normal[0]))
            axes[2].set_title('Target')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'sample_epoch_{epoch}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            break

def main():
    parser = argparse.ArgumentParser(description='Minimal Training for Low Light Enhancement')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Training data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='minimal_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = SimpleDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("No training data found!")
        print("Please ensure you have low/ and normal/ directories in your data folder")
        return
    
    print(f"Found {len(dataset)} training pairs")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # Create model
    model = MinimalEnhancementNet().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)
        
        # Train
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, dataloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print("New best model saved!")
        
        # Save sample results
        if epoch % 2 == 0:
            save_sample_results(model, dataloader, device, epoch, args.output_dir)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Best model: {os.path.join(args.output_dir, 'best_model.pth')}")

if __name__ == '__main__':
    main()
