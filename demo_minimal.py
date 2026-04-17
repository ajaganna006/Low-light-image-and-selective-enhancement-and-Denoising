#!/usr/bin/env python3
"""
Minimal demo script for Low Light Enhancement
Works with only PyTorch and PIL
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_demo_images():
    """Create demo images for testing"""
    print("Creating demo images...")
    
    # Create a test image with patterns
    size = (256, 256)
    normal = Image.new('RGB', size, (128, 128, 128))
    draw = ImageDraw.Draw(normal)
    
    # Add some patterns
    draw.rectangle([50, 50, 200, 200], fill=(255, 255, 255))
    draw.ellipse([98, 98, 158, 158], fill=(0, 0, 0))
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((80, 100), "DEMO", fill=(0, 0, 0), font=font)
    draw.text((80, 150), "IMAGE", fill=(255, 255, 255), font=font)
    
    # Add some noise
    normal_array = np.array(normal)
    noise = np.random.randint(-30, 30, normal_array.shape, dtype=np.int16)
    normal_array = np.clip(normal_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    normal = Image.fromarray(normal_array)
    
    # Save normal image
    normal_path = "demo_normal.png"
    normal.save(normal_path)
    
    # Create low light version
    low_light = create_low_light_version(normal)
    low_light_path = "demo_low_light.png"
    low_light.save(low_light_path)
    
    print(f"✓ Demo images created:")
    print(f"  Normal: {normal_path}")
    print(f"  Low light: {low_light_path}")
    
    return normal_path, low_light_path

def create_low_light_version(image, gamma=2.5, brightness=0.2):
    """Create low light version of image"""
    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Gamma correction
    low_light = np.power(img_array, gamma)
    
    # Brightness reduction
    low_light = low_light * brightness
    
    # Add noise
    noise = np.random.normal(0, 0.05, low_light.shape)
    low_light = np.clip(low_light + noise, 0, 1)
    
    # Convert back to PIL Image
    low_light = (low_light * 255).astype(np.uint8)
    return Image.fromarray(low_light)

def apply_gamma_correction(image, gamma):
    """Apply gamma correction"""
    img_array = np.array(image, dtype=np.float32) / 255.0
    corrected = np.power(img_array, 1.0 / gamma)
    corrected = (corrected * 255).astype(np.uint8)
    return Image.fromarray(corrected)

def apply_brightness_adjustment(image, factor):
    """Apply brightness adjustment"""
    img_array = np.array(image, dtype=np.float32)
    adjusted = np.clip(img_array * factor, 0, 255)
    return Image.fromarray(adjusted.astype(np.uint8))

def demo_basic_enhancement():
    """Demo basic image processing techniques"""
    print("\nDemo: Basic Image Enhancement Techniques")
    print("=" * 45)
    
    # Create demo images
    normal_path, low_light_path = create_demo_images()
    
    # Load images
    normal = Image.open(normal_path)
    low_light = Image.open(low_light_path)
    
    # Apply different enhancement techniques
    gamma_corrected = apply_gamma_correction(low_light, gamma=2.2)
    brightness_adjusted = apply_brightness_adjustment(low_light, factor=3.0)
    
    # Create comparison
    comparison = Image.new('RGB', (1024, 256))
    comparison.paste(normal, (0, 0))
    comparison.paste(low_light, (256, 0))
    comparison.paste(gamma_corrected, (512, 0))
    comparison.paste(brightness_adjusted, (768, 0))
    
    # Save comparison
    comparison.save('demo_basic_enhancement.png')
    
    print("✓ Basic enhancement demo completed")
    print("  Results saved to: demo_basic_enhancement.png")
    print("  Left to right: Original, Low Light, Gamma Corrected, Brightness Adjusted")

def demo_with_model():
    """Demo with trained model (if available)"""
    print("\nDemo: Deep Learning Enhancement")
    print("=" * 35)
    
    # Check if model exists
    model_path = "best_model_minimal.pth"
    if not os.path.exists(model_path):
        print("⚠ No trained model found.")
        print("To train a model, run: py train_minimal.py")
        return
    
    try:
        import torch
        from train_minimal import MinimalEnhancementNet
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MinimalEnhancementNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print("✓ Model loaded successfully")
        
        # Create demo images
        normal_path, low_light_path = create_demo_images()
        
        # Enhance with model
        from inference_minimal import enhance_image
        enhanced = enhance_image(model, low_light_path, device)
        
        # Load images for comparison
        normal = Image.open(normal_path)
        low_light = Image.open(low_light_path)
        
        # Create comparison
        comparison = Image.new('RGB', (768, 256))
        comparison.paste(low_light, (0, 0))
        comparison.paste(enhanced, (256, 0))
        comparison.paste(normal, (512, 0))
        
        comparison.save('demo_deep_learning_enhancement.png')
        
        print("✓ Deep learning enhancement demo completed")
        print("  Results saved to: demo_deep_learning_enhancement.png")
        print("  Left to right: Low Light, Enhanced, Ground Truth")
        
    except ImportError:
        print("⚠ PyTorch not available. Install with: py -m pip install torch")
    except Exception as e:
        print(f"✗ Error with model demo: {e}")

def main():
    """Main demo function"""
    print("Minimal Low Light Image Enhancement - Demo")
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
    print("  py train_minimal.py")
    print("\nTo enhance your own images:")
    print("  py inference_minimal.py --input your_image.jpg --output enhanced.jpg")

if __name__ == "__main__":
    main()
