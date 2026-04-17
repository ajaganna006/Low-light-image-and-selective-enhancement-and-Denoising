"""
Training script for low light image enhancement and denoising
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

from config import Config
from data_loader import create_data_loaders, create_synthetic_dataset
from models import create_model, count_parameters
from losses import create_loss_function
from utils import save_checkpoint, load_checkpoint, calculate_metrics, save_images

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (low_light, normal) in enumerate(pbar):
        low_light = low_light.to(device)
        normal = normal.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        enhanced = model(low_light)
        
        # Calculate loss
        loss_dict = criterion(enhanced, normal)
        loss = loss_dict['total']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0
            loss_components[key] += value.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'L1': f'{loss_dict["l1"].item():.4f}',
            'SSIM': f'{loss_dict["ssim"].item():.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % Config.LOG_INTERVAL == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'Train/{key.capitalize()}', value.item(), global_step)
    
    # Calculate average losses
    avg_loss = total_loss / len(train_loader)
    avg_components = {key: value / len(train_loader) for key, value in loss_components.items()}
    
    return avg_loss, avg_components

def validate_epoch(model, val_loader, criterion, device, epoch, writer):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    loss_components = {}
    metrics = {'psnr': 0, 'ssim': 0}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')
        
        for batch_idx, (low_light, normal) in enumerate(pbar):
            low_light = low_light.to(device)
            normal = normal.to(device)
            
            # Forward pass
            enhanced = model(low_light)
            
            # Calculate loss
            loss_dict = criterion(enhanced, normal)
            loss = loss_dict['total']
            
            # Update statistics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value.item()
            
            # Calculate metrics
            batch_metrics = calculate_metrics(enhanced, normal)
            for key, value in batch_metrics.items():
                metrics[key] += value
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PSNR': f'{batch_metrics["psnr"]:.2f}',
                'SSIM': f'{batch_metrics["ssim"]:.3f}'
            })
    
    # Calculate average losses and metrics
    avg_loss = total_loss / len(val_loader)
    avg_components = {key: value / len(val_loader) for key, value in loss_components.items()}
    avg_metrics = {key: value / len(val_loader) for key, value in metrics.items()}
    
    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    for key, value in avg_components.items():
        writer.add_scalar(f'Val/{key.capitalize()}', value, epoch)
    for key, value in avg_metrics.items():
        writer.add_scalar(f'Val/{key.upper()}', value, epoch)
    
    return avg_loss, avg_components, avg_metrics

def train_model(config, resume_from=None):
    """Main training function"""
    
    # Create directories
    config.create_dirs()
    
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader, val_loader = create_data_loaders(config)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic dataset...")
        create_synthetic_dataset(config, num_samples=1000)
        train_loader, val_loader = create_data_loaders(config)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function and optimizer
    criterion = create_loss_function(config)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Create tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(config.LOG_DIR, f"run_{timestamp}"))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_psnr = 0
    
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS-1}")
        print("-" * 50)
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        if epoch % config.VAL_INTERVAL == 0:
            val_loss, val_components, val_metrics = validate_epoch(
                model, val_loader, criterion, device, epoch, writer
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics['psnr'],
                    os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                )
                print(f"New best model saved! PSNR: {best_psnr:.2f}")
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val PSNR: {val_metrics['psnr']:.2f}")
            print(f"Val SSIM: {val_metrics['ssim']:.3f}")
        
        # Save checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, epoch, best_psnr,
                os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Save sample images
        if epoch % config.SAVE_INTERVAL == 0:
            save_sample_images(model, val_loader, device, epoch, config)
    
    # Save final model
    save_checkpoint(
        model, optimizer, config.NUM_EPOCHS-1, best_psnr,
        os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    )
    
    print("Training completed!")
    writer.close()

def save_sample_images(model, val_loader, device, epoch, config):
    """Save sample images during training"""
    model.eval()
    
    with torch.no_grad():
        # Get a batch of validation images
        for low_light, normal in val_loader:
            low_light = low_light.to(device)
            normal = normal.to(device)
            
            # Generate enhanced images
            enhanced = model(low_light)
            
            # Save images
            save_images(
                low_light[:4], enhanced[:4], normal[:4],
                os.path.join(config.RESULTS_DIR, f'samples_epoch_{epoch}.png')
            )
            break

def main():
    parser = argparse.ArgumentParser(description='Train Low Light Enhancement Model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Start training
    train_model(config, resume_from=args.resume)

if __name__ == '__main__':
    main()
