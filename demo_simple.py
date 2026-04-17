#!/usr/bin/env python3
"""
Simple demo script for Low Light Enhancement
Works with minimal dependencies
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def create_demo_images():
    """Create demo images for testing"""
    print("Creating demo images...")
    
    # Create a test image with patterns
    size = (256, 256)
    normal = np.ones((size[0], size[1], 3), dtype=np.uint8) * 128
    
    # Add some patterns
    cv2.rectangle(normal, (50, 50), (200, 200), (255, 255, 255), -1)
    cv2.circle(normal, (128, 128), 40, (0, 0, 0), -1)
    cv2.putText(normal, "DEMO", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(normal, "IMAGE", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Add some noise and texture
    noise = np.random.randint(0, 50, normal.shape, dtype=np.uint8)
    normal = cv2.add(normal, noise)
    
    # Save normal image
    normal_path = "demo_normal.png"
    cv2.imwrite(normal_path, cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
    
    # Create low light version
    low_light = create_low_light_version(normal)
    low_light_path = "demo_low_light.png"
    cv2.imwrite(low_light_path, cv2.cvtColor(low_light, cv2.COLOR_RGB2BGR))
    
    print(f"✓ Demo images created:")
    print(f"  Normal: {normal_path}")
    print(f"  Low light: {low_light_path}")
    
    return normal_path, low_light_path

def create_low_light_version(image, gamma=2.5, brightness=0.2):
    """Create low light version of image"""
    # Gamma correction
    low_light = np.power(image / 255.0, gamma) * 255.0
    
    # Brightness reduction
    low_light = low_light * brightness
    
    # Add noise
    noise = np.random.normal(0, 8, low_light.shape)
    low_light = np.clip(low_light + noise, 0, 255)
    
    return low_light.astype(np.uint8)

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
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def demo_basic_enhancement():
    """Demo basic image processing techniques"""
    print("\nDemo: Basic Image Enhancement Techniques")
    print("=" * 45)
    
    # Create demo images
    normal_path, low_light_path = create_demo_images()
    
    # Load images
    normal = cv2.imread(normal_path)
    low_light = cv2.imread(low_light_path)
    
    # Apply different enhancement techniques
    gamma_corrected = apply_gamma_correction(low_light, gamma=2.2)
    hist_eq = apply_histogram_equalization(low_light)
    clahe = apply_clahe(low_light)
    
    # Create comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row
    axes[0, 0].imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Normal', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(low_light, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Low Light Input', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Gamma Correction', fontsize=12)
    axes[0, 2].axis('off')
    
    # Bottom row
    axes[1, 0].imshow(cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Histogram Equalization', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(clahe, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('CLAHE', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_basic_enhancement.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Basic enhancement demo completed")
    print("  Results saved to: demo_basic_enhancement.png")

def demo_with_model():
    """Demo with trained model (if available)"""
    print("\nDemo: Deep Learning Enhancement")
    print("=" * 35)
    
    # Check if model exists
    model_path = "best_model_simple.pth"
    if not os.path.exists(model_path):
        print("⚠ No trained model found.")
        print("To train a model, run: py train_simple.py")
        return
    
    try:
        import torch
        from train_simple import SimpleEnhancementNet
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleEnhancementNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print("✓ Model loaded successfully")
        
        # Create demo images
        normal_path, low_light_path = create_demo_images()
        
        # Enhance with model
        from inference_simple import enhance_image
        enhanced = enhance_image(model, low_light_path, device)
        
        # Load images for comparison
        normal = cv2.imread(normal_path)
        low_light = cv2.imread(low_light_path)
        
        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(low_light, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Low Light Input')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced)
        axes[1].set_title('Deep Learning Enhanced')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('demo_deep_learning_enhancement.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✓ Deep learning enhancement demo completed")
        print("  Results saved to: demo_deep_learning_enhancement.png")
        
    except ImportError:
        print("⚠ PyTorch not available. Install with: py install_packages_simple.py")
    except Exception as e:
        print(f"✗ Error with model demo: {e}")

def main():
    """Main demo function"""
    print("Simple Low Light Image Enhancement - Demo")
    print("=" * 45)
    
    # Demo 1: Basic image processing
    demo_basic_enhancement()
    
    # Demo 2: Deep learning enhancement (if model available)
    demo_with_model()
    
    print("\n" + "=" * 45)
    print("Demo completed!")
    print("\nGenerated files:")
    print("  - demo_normal.png")
    print("  - demo_low_light.png") 
    print("  - demo_basic_enhancement.png")
    print("  - demo_deep_learning_enhancement.png (if model available)")
    print("\nTo train your own model:")
    print("  py train_simple.py")
    print("\nTo enhance your own images:")
    print("  py inference_simple.py --input your_image.jpg --output enhanced.jpg")

if __name__ == "__main__":
    main()
