"""
Quick Demo for Low Light Image Enhancement
A simple, working demo that showcases the project capabilities
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def create_test_image():
    """Create a test image for demonstration"""
    # Create a colorful test image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add some patterns
    img[50:100, 50:100] = [255, 100, 100]  # Red square
    img[150:200, 150:200] = [100, 255, 100]  # Green square
    img[100:150, 100:150] = [100, 100, 255]  # Blue square
    
    # Add some text
    cv2.putText(img, "TEST", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "IMAGE", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

def create_low_light_version(image, gamma=2.5, brightness=0.3):
    """Create a low light version of the image"""
    # Apply gamma correction to darken
    low_light = np.power(image / 255.0, gamma) * 255.0
    
    # Reduce brightness
    low_light = low_light * brightness
    
    # Add some noise
    noise = np.random.normal(0, 10, low_light.shape)
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

def apply_brightness_contrast(image, alpha=1.5, beta=30):
    """Apply brightness and contrast adjustment"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_image_comprehensive(image):
    """Apply comprehensive enhancement"""
    # 1. Gamma correction
    gamma_corrected = apply_gamma_correction(image, gamma=2.2)
    
    # 2. Brightness and contrast
    bright_contrast = apply_brightness_contrast(gamma_corrected, alpha=1.3, beta=20)
    
    # 3. CLAHE
    clahe_result = apply_clahe(bright_contrast)
    
    return clahe_result

def create_comparison_grid(original, low_light, enhanced, output_path):
    """Create a comparison grid"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original methods
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(low_light, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Low Light Input')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Enhanced Output')
    axes[0, 2].axis('off')
    
    # Row 2: Different enhancement methods
    gamma_result = apply_gamma_correction(low_light, gamma=2.2)
    hist_result = apply_histogram_equalization(low_light)
    clahe_result = apply_clahe(low_light)
    
    axes[1, 0].imshow(cv2.cvtColor(gamma_result, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Gamma Correction')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(hist_result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Histogram Equalization')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(clahe_result, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('CLAHE')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison grid saved to: {output_path}")

def enhance_existing_image(image_path, output_path=None):
    """Enhance an existing image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    print(f"Enhancing image: {image_path}")
    
    # Apply enhancement
    enhanced = enhance_image_comprehensive(image)
    
    # Set output path
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_enhanced.png"
    
    # Save enhanced image
    cv2.imwrite(output_path, enhanced)
    print(f"Enhanced image saved to: {output_path}")
    
    return enhanced

def main():
    parser = argparse.ArgumentParser(description='Quick Demo for Low Light Image Enhancement')
    parser.add_argument('--input', type=str, help='Input image path (optional)')
    parser.add_argument('--output', type=str, help='Output image path (optional)')
    parser.add_argument('--demo', action='store_true', help='Create demo with synthetic image')
    
    args = parser.parse_args()
    
    print("Low Light Image Enhancement - Quick Demo")
    print("=" * 50)
    
    if args.input:
        # Enhance existing image
        enhanced = enhance_existing_image(args.input, args.output)
        if enhanced is not None:
            print("✓ Image enhancement completed!")
    else:
        # Create demo with synthetic image
        print("Creating synthetic demo...")
        
        # Create test image
        original = create_test_image()
        low_light = create_low_light_version(original)
        enhanced = enhance_image_comprehensive(low_light)
        
        # Save individual images
        cv2.imwrite('demo_original.png', original)
        cv2.imwrite('demo_low_light.png', low_light)
        cv2.imwrite('demo_enhanced.png', enhanced)
        
        # Create comparison grid
        create_comparison_grid(original, low_light, enhanced, 'demo_comparison.png')
        
        print("✓ Demo completed!")
        print("\nGenerated files:")
        print("  - demo_original.png")
        print("  - demo_low_light.png")
        print("  - demo_enhanced.png")
        print("  - demo_comparison.png")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nThis demo showcases:")
    print("  - Low light image simulation")
    print("  - Multiple enhancement techniques")
    print("  - Visual comparison of results")
    print("\nFor deep learning enhancement, train a model with:")
    print("  python train.py")

if __name__ == '__main__':
    main()
