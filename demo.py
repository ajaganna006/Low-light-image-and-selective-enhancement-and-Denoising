"""
Demo script for Low Light Image Enhancement
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from config import Config
from models import create_model
from utils import load_checkpoint, enhance_image
from data_loader import create_low_light_image
from data_loader import create_synthetic_dataset

def create_demo_images():
    """Create demo images for testing"""
    print("Creating demo images...")
    
    # Create a simple test image
    test_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    # Add some patterns
    for i in range(0, 256, 32):
        test_image[i:i+16, :] = [200, 150, 100]
        test_image[:, i:i+16] = [100, 200, 150]
    
    # Add some text-like patterns
    cv2.putText(test_image, "DEMO", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(test_image, "IMAGE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Save normal image
    normal_path = "demo_normal.png"
    cv2.imwrite(normal_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    
    # Create low light version
    low_light = create_low_light_image(test_image, gamma_range=(2.0, 3.0), brightness_range=(0.1, 0.3))
    low_light_path = "demo_low_light.png"
    cv2.imwrite(low_light_path, cv2.cvtColor(low_light, cv2.COLOR_RGB2BGR))
    
    print(f"✓ Demo images created:")
    print(f"  Normal: {normal_path}")
    print(f"  Low light: {low_light_path}")
    
    return normal_path, low_light_path

def demo_without_model():
    """Demo basic image processing without trained model"""
    print("\nDemo: Basic Image Processing")
    print("-" * 30)
    
    # Create demo images
    normal_path, low_light_path = create_demo_images()
    
    # Load images
    normal = cv2.imread(normal_path)
    low_light = cv2.imread(low_light_path)
    
    # Apply basic enhancement techniques
    # 1. Gamma correction
    gamma_corrected = apply_gamma_correction(low_light, gamma=2.2)
    
    # 2. Histogram equalization
    hist_eq = apply_histogram_equalization(low_light)
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = apply_clahe(low_light)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Normal')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(low_light, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Low Light')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Gamma Correction')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Histogram Equalization')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(clahe, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('CLAHE')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_basic_enhancement.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Basic enhancement demo completed")
    print("  Results saved to: demo_basic_enhancement.png")

def apply_gamma_correction(image, gamma):
    """Apply gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_histogram_equalization(image):
    """Apply histogram equalization"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_clahe(image):
    """Apply CLAHE"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def demo_with_model():
    """Demo with trained model (if available)"""
    print("\nDemo: Deep Learning Enhancement")
    print("-" * 35)
    
    # Check if model checkpoint exists
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("⚠ No trained model found. Please train a model first:")
        print("  python train.py")
        return
    
    # Load model
    config = Config()
    device = torch.device(config.DEVICE)
    model = create_model(config)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Create demo images
    normal_path, low_light_path = create_demo_images()
    
    # Enhance with model
    enhanced = enhance_image(model, low_light_path, device)
    
    # Load images for comparison
    normal = Image.open(normal_path)
    low_light = Image.open(low_light_path)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(low_light)
    axes[0].set_title('Low Light Input')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title('Deep Learning Enhanced')
    axes[1].axis('off')
    
    axes[2].imshow(normal)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_deep_learning_enhancement.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Deep learning enhancement demo completed")
    print("  Results saved to: demo_deep_learning_enhancement.png")

def demo_synthetic_dataset():
    """Demo synthetic dataset creation"""
    print("\nDemo: Synthetic Dataset Creation")
    print("-" * 35)
    
    config = Config()
    
    # Create a small synthetic dataset
    print("Creating synthetic dataset...")
    create_synthetic_dataset(config, num_samples=5)
    
    print("✓ Synthetic dataset created")
    print("  Check the data/train/ directory for generated images")

def main():
    """Main demo function"""
    print("Low Light Image Enhancement - Demo")
    print("=" * 40)
    
    # Demo 1: Basic image processing
    demo_without_model()
    
    # Demo 2: Synthetic dataset creation
    demo_synthetic_dataset()
    
    # Demo 3: Deep learning enhancement (if model available)
    demo_with_model()
    
    print("\n" + "=" * 40)
    print("Demo completed!")
    print("\nGenerated files:")
    print("  - demo_normal.png")
    print("  - demo_low_light.png") 
    print("  - demo_basic_enhancement.png")
    print("  - demo_deep_learning_enhancement.png (if model available)")
    print("\nTo train your own model:")
    print("  python train.py")

if __name__ == "__main__":
    main()
