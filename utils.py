"""
Utility functions for low light image enhancement and denoising
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms

def save_checkpoint(model, optimizer, epoch, best_psnr, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_psnr
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint

def calculate_psnr(pred, target):
    """Calculate PSNR between predicted and target images"""
    # Convert to numpy arrays
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Convert from [-1, 1] to [0, 1] range if needed
    if pred_np.min() < 0:
        pred_np = (pred_np + 1) / 2
    if target_np.min() < 0:
        target_np = (target_np + 1) / 2
    
    # Clamp values
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # Calculate PSNR
    mse = np.mean((pred_np - target_np) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(pred, target):
    """Calculate SSIM between predicted and target images"""
    # Convert to numpy arrays
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Convert from [-1, 1] to [0, 1] range if needed
    if pred_np.min() < 0:
        pred_np = (pred_np + 1) / 2
    if target_np.min() < 0:
        target_np = (target_np + 1) / 2
    
    # Clamp values
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # Calculate SSIM for each channel
    ssim_values = []
    for i in range(pred_np.shape[1]):  # For each channel
        ssim_val = ssim(pred_np[0, i], target_np[0, i], data_range=1.0)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def calculate_metrics(pred, target):
    """Calculate PSNR and SSIM metrics"""
    # Convert to numpy arrays
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Convert from [-1, 1] to [0, 1] range
    pred_np = (pred_np + 1) / 2
    target_np = (target_np + 1) / 2
    
    # Clamp values
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    batch_size = pred_np.shape[0]
    psnr_values = []
    ssim_values = []
    
    for i in range(batch_size):
        # Calculate PSNR
        psnr_val = psnr(target_np[i].transpose(1, 2, 0), pred_np[i].transpose(1, 2, 0), data_range=1.0)
        psnr_values.append(psnr_val)
        
        # Calculate SSIM
        ssim_val = ssim(target_np[i].transpose(1, 2, 0), pred_np[i].transpose(1, 2, 0), 
                       multichannel=True, data_range=1.0)
        ssim_values.append(ssim_val)
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values)
    }

def save_images(low_light, enhanced, target, filepath, num_images=4):
    """Save comparison images"""
    # Convert tensors to numpy arrays
    low_light = low_light.detach().cpu().numpy()
    enhanced = enhanced.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    # Convert from [-1, 1] to [0, 1] range
    low_light = (low_light + 1) / 2
    enhanced = (enhanced + 1) / 2
    target = (target + 1) / 2
    
    # Clamp values
    low_light = np.clip(low_light, 0, 1)
    enhanced = np.clip(enhanced, 0, 1)
    target = np.clip(target, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 4, 12))
    
    for i in range(num_images):
        # Low light image
        axes[0, i].imshow(low_light[i].transpose(1, 2, 0))
        axes[0, i].set_title('Low Light')
        axes[0, i].axis('off')
        
        # Enhanced image
        axes[1, i].imshow(enhanced[i].transpose(1, 2, 0))
        axes[1, i].set_title('Enhanced')
        axes[1, i].axis('off')
        
        # Target image
        axes[2, i].imshow(target[i].transpose(1, 2, 0))
        axes[2, i].set_title('Target')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    # Convert from [-1, 1] to [0, 1] range
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

def image_to_tensor(image_path, size=(256, 256)):
    """Convert image to tensor"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize(size, Image.LANCZOS)
    
    # Convert to tensor
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image)
    
    # Convert to [-1, 1] range
    tensor = tensor * 2 - 1
    
    return tensor

def enhance_image(model, image_path, device, output_path=None):
    """Enhance a single image"""
    model.eval()
    
    # Load and preprocess image
    tensor = image_to_tensor(image_path).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced = model(tensor)
    
    # Convert to PIL Image
    enhanced_image = tensor_to_image(enhanced.squeeze(0))
    
    if output_path:
        enhanced_image.save(output_path)
    
    return enhanced_image

def create_comparison_grid(image_paths, model, device, output_path):
    """Create a grid comparing original and enhanced images"""
    model.eval()
    
    num_images = len(image_paths)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 4, 8))
    
    for i, image_path in enumerate(image_paths):
        # Load original image
        original = Image.open(image_path).convert('RGB')
        
        # Enhance image
        enhanced = enhance_image(model, image_path, device)
        
        # Display images
        axes[0, i].imshow(original)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(enhanced)
        axes[1, i].set_title('Enhanced')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_image_statistics(image_path):
    """Calculate basic statistics of an image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for some statistics
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    stats = {
        'mean_brightness': np.mean(gray),
        'std_brightness': np.std(gray),
        'min_brightness': np.min(gray),
        'max_brightness': np.max(gray),
        'histogram': np.histogram(gray, bins=256, range=(0, 256))[0]
    }
    
    return stats

def visualize_histogram(image_path, output_path=None):
    """Visualize image histogram"""
    stats = calculate_image_statistics(image_path)
    
    plt.figure(figsize=(10, 6))
    plt.hist(stats['histogram'], bins=256, range=(0, 256), alpha=0.7)
    plt.title('Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

def apply_gamma_correction(image, gamma):
    """Apply gamma correction to image"""
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    return cv2.LUT(image, table)

def create_test_dataset(config, num_samples=50):
    """Create a test dataset for evaluation"""
    import glob
    
    # Look for images in the data directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(config.DATA_ROOT, '**', ext), recursive=True))
    
    if not image_paths:
        print("No images found in data directory.")
        return
    
    # Create test directory
    test_dir = config.TEST_DIR
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy random images to test directory
    import random
    test_images = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    for i, img_path in enumerate(test_images):
        # Copy image to test directory
        img_name = f"test_{i:03d}.png"
        img = Image.open(img_path).convert('RGB')
        img.save(os.path.join(test_dir, img_name))
    
    print(f"Created test dataset with {len(test_images)} images")

def benchmark_model(model, test_loader, device):
    """Benchmark model performance"""
    model.eval()
    
    total_psnr = 0
    total_ssim = 0
    total_inference_time = 0
    num_samples = 0
    
    with torch.no_grad():
        for low_light, normal in test_loader:
            low_light = low_light.to(device)
            normal = normal.to(device)
            
            # Measure inference time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            enhanced = model(low_light)
            end_time.record()
            
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            
            # Calculate metrics
            metrics = calculate_metrics(enhanced, normal)
            
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            total_inference_time += inference_time
            num_samples += low_light.size(0)
    
    # Calculate averages
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    avg_inference_time = total_inference_time / num_samples
    
    print(f"Benchmark Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.3f}")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'inference_time': avg_inference_time
    }
