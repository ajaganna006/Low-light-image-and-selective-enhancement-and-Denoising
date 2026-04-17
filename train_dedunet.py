"""
Training script for DEDUNet (Dual-Enhancing Dense-UNet)
Advanced training with multiple loss functions and superior performance metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse

from dedunet_simple import create_dedunet_simple
from data_loader import create_synthetic_dataset, create_low_light_image
from utils import calculate_psnr, calculate_ssim, save_image


class DEDUNetLoss(nn.Module):
    """
    Comprehensive loss function for DEDUNet training
    Combines multiple loss types for superior performance
    """
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, delta=0.1):
        super(DEDUNetLoss, self).__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # SSIM loss weight
        self.gamma = gamma  # Perceptual loss weight
        self.delta = delta  # Gradient loss weight
        
        # L1 Loss
        self.l1_loss = nn.L1Loss()
        
        # SSIM Loss
        self.ssim_loss = SSIMLoss()
        
        # Perceptual Loss (VGG-based) - simplified version
        self.perceptual_loss = PerceptualLoss()
        
        # Gradient Loss
        self.gradient_loss = GradientLoss()
        
    def forward(self, pred, target):
        # L1 Loss
        l1 = self.l1_loss(pred, target)
        
        # SSIM Loss
        ssim = self.ssim_loss(pred, target)
        
        # Perceptual Loss
        perceptual = self.perceptual_loss(pred, target)
        
        # Gradient Loss
        gradient = self.gradient_loss(pred, target)
        
        # Total Loss
        total_loss = (self.alpha * l1 + 
                     self.beta * ssim + 
                     self.gamma * perceptual + 
                     self.delta * gradient)
        
        return total_loss, {
            'l1': l1.item(),
            'ssim': ssim.item(),
            'perceptual': perceptual.item(),
            'gradient': gradient.item(),
            'total': total_loss.item()
        }


class SSIMLoss(nn.Module):
    """
    SSIM Loss for structural similarity
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        
    def forward(self, pred, target):
        return 1 - self._ssim(pred, target)
    
    def _ssim(self, pred, target):
        # Simplified SSIM calculation
        mu1 = torch.mean(pred)
        mu2 = torch.mean(target)
        
        sigma1 = torch.var(pred)
        sigma2 = torch.var(target)
        sigma12 = torch.mean((pred - mu1) * (target - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim


class PerceptualLoss(nn.Module):
    """
    Simplified Perceptual Loss
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Simple feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
            
    def forward(self, pred, target):
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        return torch.mean((pred_features - target_features) ** 2)


class GradientLoss(nn.Module):
    """
    Gradient Loss for edge preservation
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        
    def forward(self, pred, target):
        # Sobel filters for gradient calculation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32).view(1, 1, 3, 3)
        
        if pred.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
            
        # Calculate gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # Gradient magnitude
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        return torch.mean((pred_grad - target_grad) ** 2)


class DEDUNetTrainer:
    """
    DEDUNet Trainer with advanced training strategies
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_dedunet_simple(
            in_channels=3, 
            out_channels=3, 
            base_channels=config.base_channels
        ).to(self.device)
        
        # Loss function
        self.criterion = DEDUNetLoss(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Tensorboard
        self.writer = SummaryWriter(f'runs/dedunet_{int(time.time())}')
        
        # Best metrics
        self.best_psnr = 0
        self.best_ssim = 0
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        
        for batch_idx, (low_light, normal) in enumerate(pbar):
            low_light = low_light.to(self.device)
            normal = normal.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            enhanced = self.model(low_light)
            
            # Calculate loss
            loss, loss_dict = self.criterion(enhanced, normal)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                psnr = calculate_psnr(enhanced, normal)
                ssim = calculate_ssim(enhanced, normal)
                
                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PSNR': f'{psnr:.2f}',
                    'SSIM': f'{ssim:.4f}'
                })
                
                # Log to tensorboard
                if batch_idx % 100 == 0:
                    global_step = epoch * len(dataloader) + batch_idx
                    self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                    self.writer.add_scalar('Train/PSNR', psnr, global_step)
                    self.writer.add_scalar('Train/SSIM', ssim, global_step)
                    
                    # Log individual loss components
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f'Train/Loss_{key}', value, global_step)
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'loss': total_loss / len(dataloader),
            'psnr': total_psnr / len(dataloader),
            'ssim': total_ssim / len(dataloader)
        }
    
    def validate(self, dataloader, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for low_light, normal in tqdm(dataloader, desc='Validation'):
                low_light = low_light.to(self.device)
                normal = normal.to(self.device)
                
                enhanced = self.model(low_light)
                
                loss, _ = self.criterion(enhanced, normal)
                psnr = calculate_psnr(enhanced, normal)
                ssim = calculate_ssim(enhanced, normal)
                
                total_loss += loss.item()
                total_psnr += psnr
                total_ssim += ssim
        
        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM', avg_ssim, epoch)
        
        # Save best model
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint(epoch, avg_psnr, avg_ssim, is_best=True)
        
        return {
            'loss': avg_loss,
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }
    
    def save_checkpoint(self, epoch, psnr, ssim, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'psnr': psnr,
            'ssim': ssim,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = f'checkpoints/dedunet_epoch_{epoch+1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = 'checkpoints/dedunet_best.pth'
            torch.save(checkpoint, best_path)
            print(f'✅ New best model saved! PSNR: {psnr:.2f}, SSIM: {ssim:.4f}')
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"🚀 Starting DEDUNet training on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            print(f"\n📅 Epoch {epoch+1}/{self.config.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Print epoch summary
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"PSNR: {train_metrics['psnr']:.2f}, "
                  f"SSIM: {train_metrics['ssim']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"PSNR: {val_metrics['psnr']:.2f}, "
                  f"SSIM: {val_metrics['ssim']:.4f}")
            print(f"Best PSNR: {self.best_psnr:.2f}")
        
        print(f"\n🎉 Training completed! Best PSNR: {self.best_psnr:.2f}")
        self.writer.close()


def create_dedunet_config():
    """Create configuration for DEDUNet training"""
    class Config:
        # Model parameters
        base_channels = 32  # Reduced for faster training
        
        # Training parameters
        epochs = 50  # Reduced for demo
        batch_size = 4  # Reduced for memory
        learning_rate = 1e-4
        weight_decay = 1e-5
        
        # Loss weights
        alpha = 1.0    # L1 loss
        beta = 0.1     # SSIM loss
        gamma = 0.1    # Perceptual loss
        delta = 0.1    # Gradient loss
        
        # Data parameters
        image_size = 256
        num_workers = 2  # Reduced for stability
        
    return Config()


def main():
    parser = argparse.ArgumentParser(description='Train DEDUNet')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='data', help='Data path')
    
    args = parser.parse_args()
    
    # Create config
    config = create_dedunet_config()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Create synthetic dataset if no real data
    print("📁 Creating dataset...")
    train_dataset, val_dataset = create_synthetic_dataset(
        num_samples=1000,
        image_size=config.image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Create trainer
    trainer = DEDUNetTrainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
