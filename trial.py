"""
Trial Version - Low Light Image Enhancement
A simple trial that showcases the project capabilities
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import time

def create_trial_image():
    """Create a sample image for the trial"""
    print("🎨 Creating trial image...")
    
    # Create a colorful test image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some patterns and text
    img[50:100, 50:150] = [255, 100, 100]  # Red rectangle
    img[150:200, 200:350] = [100, 255, 100]  # Green rectangle
    img[100:150, 100:200] = [100, 100, 255]  # Blue rectangle
    
    # Add text
    cv2.putText(img, "TRIAL", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(img, "IMAGE", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "ENHANCEMENT", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    return img

def create_dark_version(image, darkness_level=0.3):
    """Create a dark version of the image"""
    print("🌙 Creating dark version...")
    
    # Apply gamma correction to darken
    gamma = 2.5
    dark_img = np.power(image / 255.0, gamma) * 255.0
    
    # Reduce brightness
    dark_img = dark_img * darkness_level
    
    # Add some noise
    noise = np.random.normal(0, 8, dark_img.shape)
    dark_img = np.clip(dark_img + noise, 0, 255)
    
    return dark_img.astype(np.uint8)

def enhance_gamma(image, gamma=2.2):
    """Apply gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_histogram(image):
    """Apply histogram equalization"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def enhance_clahe(image):
    """Apply CLAHE"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def enhance_brightness_contrast(image, alpha=1.5, beta=30):
    """Apply brightness and contrast"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_comprehensive(image):
    """Apply comprehensive enhancement"""
    # Step 1: Gamma correction
    enhanced = enhance_gamma(image, gamma=2.2)
    
    # Step 2: Brightness and contrast
    enhanced = enhance_brightness_contrast(enhanced, alpha=1.3, beta=20)
    
    # Step 3: CLAHE
    enhanced = enhance_clahe(enhanced)
    
    return enhanced

def run_trial():
    """Run the complete trial"""
    print("🚀 Low Light Image Enhancement - TRIAL VERSION")
    print("=" * 60)
    print("This trial demonstrates the key features of our enhancement system.")
    print()
    
    # Create trial image
    original = create_trial_image()
    dark = create_dark_version(original)
    
    print("✨ Applying enhancement methods...")
    
    # Apply different enhancement methods
    methods = {
        'Gamma Correction': enhance_gamma(dark),
        'Histogram Equalization': enhance_histogram(dark),
        'CLAHE': enhance_clahe(dark),
        'Brightness/Contrast': enhance_brightness_contrast(dark),
        'Comprehensive': enhance_comprehensive(dark)
    }
    
    # Create comparison visualization
    print("📊 Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Low Light Image Enhancement - Trial Results', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Dark image
    axes[0, 1].imshow(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Low Light Input', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Best result (comprehensive)
    axes[0, 2].imshow(cv2.cvtColor(methods['Comprehensive'], cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Enhanced (Comprehensive)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Other methods
    method_names = list(methods.keys())[:3]  # First 3 methods
    for i, method_name in enumerate(method_names):
        row = 1
        col = i
        axes[row, col].imshow(cv2.cvtColor(methods[method_name], cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'{method_name}', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('trial_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save individual images
    print("💾 Saving trial images...")
    cv2.imwrite('trial_original.png', original)
    cv2.imwrite('trial_dark.png', dark)
    cv2.imwrite('trial_enhanced.png', methods['Comprehensive'])
    
    # Display results
    print("\n" + "=" * 60)
    print("🎉 TRIAL COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 Generated files:")
    print("  - trial_original.png (Original image)")
    print("  - trial_dark.png (Low light version)")
    print("  - trial_enhanced.png (Enhanced result)")
    print("  - trial_results.png (Comparison visualization)")
    print()
    print("🔍 What you just saw:")
    print("  ✅ Low light image simulation")
    print("  ✅ Multiple enhancement techniques")
    print("  ✅ Real-time image processing")
    print("  ✅ Visual comparison of results")
    print()
    print("🚀 Ready for more? Try the full version:")
    print("  python quick_demo.py          # Full demo")
    print("  python run_web_app.py         # Web interface")
    print("  python minimal_train.py       # Train your own model")
    print()
    print("💡 This trial used 5 different enhancement methods:")
    for i, method in enumerate(methods.keys(), 1):
        print(f"  {i}. {method}")
    print()
    print("🎯 The 'Comprehensive' method combines multiple techniques")
    print("   for the best results - as you can see in the comparison!")

def trial_with_custom_image(image_path):
    """Run trial with a custom image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🖼️  Running trial with custom image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image. Please check the file format.")
        return
    
    # Resize if too large
    height, width = image.shape[:2]
    if width > 800 or height > 600:
        scale = min(800/width, 600/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
        print(f"📏 Resized image to {new_width}x{new_height}")
    
    # Create dark version
    dark = create_dark_version(image, darkness_level=0.4)
    
    # Apply comprehensive enhancement
    enhanced = enhance_comprehensive(dark)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f'{base_name}_trial_dark.png', dark)
    cv2.imwrite(f'{base_name}_trial_enhanced.png', enhanced)
    
    # Create comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Low Light')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Enhanced')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{base_name}_trial_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Trial completed! Check the generated files:")
    print(f"  - {base_name}_trial_dark.png")
    print(f"  - {base_name}_trial_enhanced.png")
    print(f"  - {base_name}_trial_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Low Light Image Enhancement - Trial Version')
    parser.add_argument('--image', type=str, help='Path to custom image for trial')
    parser.add_argument('--demo', action='store_true', help='Run demo trial (default)')
    
    args = parser.parse_args()
    
    if args.image:
        trial_with_custom_image(args.image)
    else:
        run_trial()

if __name__ == '__main__':
    main()
