"""
Evaluation script for low light image enhancement and denoising
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from data_loader import create_data_loaders
from models import create_model
from utils import load_checkpoint, calculate_metrics, save_images, benchmark_model

def evaluate_model(model, test_loader, device, config):
    """Evaluate model on test dataset"""
    model.eval()
    
    total_metrics = {'psnr': 0, 'ssim': 0}
    all_predictions = []
    all_targets = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, (low_light, normal) in enumerate(tqdm(test_loader)):
            low_light = low_light.to(device)
            normal = normal.to(device)
            
            # Forward pass
            enhanced = model(low_light)
            
            # Calculate metrics
            batch_metrics = calculate_metrics(enhanced, normal)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                total_metrics[key] += value
            
            # Store predictions and targets for detailed analysis
            all_predictions.append(enhanced.cpu())
            all_targets.append(normal.cpu())
    
    # Calculate average metrics
    avg_metrics = {key: value / len(test_loader) for key, value in total_metrics.items()}
    
    print(f"Evaluation Results:")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average SSIM: {avg_metrics['ssim']:.3f}")
    
    return avg_metrics, all_predictions, all_targets

def analyze_results(predictions, targets, config, num_samples=10):
    """Analyze and visualize results"""
    print("Analyzing results...")
    
    # Concatenate all predictions and targets
    all_pred = torch.cat(predictions, dim=0)
    all_target = torch.cat(targets, dim=0)
    
    # Calculate per-image metrics
    psnr_values = []
    ssim_values = []
    
    for i in range(len(all_pred)):
        pred = all_pred[i:i+1]
        target = all_target[i:i+1]
        metrics = calculate_metrics(pred, target)
        psnr_values.append(metrics['psnr'])
        ssim_values.append(metrics['ssim'])
    
    # Statistics
    print(f"\nDetailed Analysis:")
    print(f"PSNR - Mean: {np.mean(psnr_values):.2f}, Std: {np.std(psnr_values):.2f}")
    print(f"PSNR - Min: {np.min(psnr_values):.2f}, Max: {np.max(psnr_values):.2f}")
    print(f"SSIM - Mean: {np.mean(ssim_values):.3f}, Std: {np.std(ssim_values):.3f}")
    print(f"SSIM - Min: {np.min(ssim_values):.3f}, Max: {np.max(ssim_values):.3f}")
    
    # Create visualizations
    create_metric_plots(psnr_values, ssim_values, config)
    
    # Save sample results
    save_sample_results(all_pred[:num_samples], all_target[:num_samples], config)

def create_metric_plots(psnr_values, ssim_values, config):
    """Create plots for metric distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # PSNR histogram
    ax1.hist(psnr_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('PSNR (dB)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('PSNR Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(psnr_values), color='red', linestyle='--', 
                label=f'Mean: {np.mean(psnr_values):.2f}')
    ax1.legend()
    
    # SSIM histogram
    ax2.hist(ssim_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('SSIM')
    ax2.set_ylabel('Frequency')
    ax2.set_title('SSIM Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(ssim_values), color='red', linestyle='--', 
                label=f'Mean: {np.mean(ssim_values):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'metric_distributions.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()

def save_sample_results(predictions, targets, config):
    """Save sample results for visual inspection"""
    # Create dummy low light images (we don't have them in evaluation)
    low_light = torch.zeros_like(predictions)
    
    save_images(
        low_light, predictions, targets,
        os.path.join(config.RESULTS_DIR, 'evaluation_samples.png'),
        num_images=min(8, len(predictions))
    )

def compare_with_baselines(model, test_loader, device, config):
    """Compare with baseline methods"""
    print("Comparing with baseline methods...")
    
    model.eval()
    baseline_results = {}
    
    with torch.no_grad():
        for batch_idx, (low_light, normal) in enumerate(tqdm(test_loader)):
            low_light = low_light.to(device)
            normal = normal.to(device)
            
            # Our model
            enhanced = model(low_light)
            our_metrics = calculate_metrics(enhanced, normal)
            
            # Baseline: Gamma correction
            gamma_corrected = apply_gamma_correction_baseline(low_light, gamma=2.2)
            gamma_metrics = calculate_metrics(gamma_corrected, normal)
            
            # Baseline: Histogram equalization
            hist_eq = apply_histogram_equalization_baseline(low_light)
            hist_metrics = calculate_metrics(hist_eq, normal)
            
            # Accumulate results
            for method, metrics in [('our_model', our_metrics), 
                                  ('gamma_correction', gamma_metrics),
                                  ('histogram_equalization', hist_metrics)]:
                if method not in baseline_results:
                    baseline_results[method] = {'psnr': 0, 'ssim': 0}
                for key, value in metrics.items():
                    baseline_results[method][key] += value
    
    # Calculate averages
    for method in baseline_results:
        for key in baseline_results[method]:
            baseline_results[method][key] /= len(test_loader)
    
    # Print comparison
    print("\nBaseline Comparison:")
    print("-" * 50)
    for method, metrics in baseline_results.items():
        print(f"{method.replace('_', ' ').title()}:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.3f}")
        print()
    
    return baseline_results

def apply_gamma_correction_baseline(image, gamma=2.2):
    """Apply gamma correction as baseline"""
    # Convert from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    
    # Apply gamma correction
    corrected = torch.pow(image, 1.0 / gamma)
    
    # Convert back to [-1, 1]
    return corrected * 2 - 1

def apply_histogram_equalization_baseline(image):
    """Apply histogram equalization as baseline"""
    # Convert from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    
    # Convert to numpy for histogram equalization
    image_np = image.cpu().numpy()
    batch_size, channels, height, width = image_np.shape
    
    equalized = np.zeros_like(image_np)
    
    for i in range(batch_size):
        for c in range(channels):
            # Apply histogram equalization
            img = (image_np[i, c] * 255).astype(np.uint8)
            equalized_img = cv2.equalizeHist(img)
            equalized[i, c] = equalized_img / 255.0
    
    # Convert back to tensor and [-1, 1] range
    equalized = torch.from_numpy(equalized).to(image.device)
    return equalized * 2 - 1

def main():
    parser = argparse.ArgumentParser(description='Evaluate Low Light Enhancement Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default=None, help='Path to test directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--compare_baselines', action='store_true', help='Compare with baseline methods')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    if args.test_dir:
        config.TEST_DIR = args.test_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    # Create directories
    config.create_dirs()
    
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    
    # Create test data loader
    try:
        _, test_loader = create_data_loaders(config)
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Please ensure test data is available or create synthetic data first.")
        return
    
    # Evaluate model
    metrics, predictions, targets = evaluate_model(model, test_loader, device, config)
    
    # Analyze results
    analyze_results(predictions, targets, config)
    
    # Compare with baselines if requested
    if args.compare_baselines:
        compare_with_baselines(model, test_loader, device, config)
    
    # Benchmark model
    benchmark_results = benchmark_model(model, test_loader, device)
    
    print("\nEvaluation completed!")
    print(f"Results saved to: {config.RESULTS_DIR}")

if __name__ == '__main__':
    main()
