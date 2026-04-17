#!/usr/bin/env python3
"""
Simple inference script for Low Light Enhancement
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import argparse

# Import the simple model
from train_simple import SimpleEnhancementNet

def load_model(checkpoint_path, device):
    """Load the trained model"""
    model = SimpleEnhancementNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, size=(256, 256)):
    """Preprocess image for inference"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize
    image = cv2.resize(image, size)
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [-1, 1]
    image = (image / 255.0) * 2 - 1
    
    # Convert to tensor
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    
    return image

def postprocess_image(tensor):
    """Convert tensor back to image"""
    # Convert to numpy
    image = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize from [-1, 1] to [0, 255]
    image = ((image + 1) / 2 * 255).astype(np.uint8)
    
    return image

def enhance_image(model, image_path, device, output_path=None):
    """Enhance a single image"""
    print(f"Enhancing: {image_path}")
    
    # Preprocess
    input_tensor = preprocess_image(image_path).to(device)
    
    # Enhance
    with torch.no_grad():
        enhanced_tensor = model(input_tensor)
    
    # Postprocess
    enhanced_image = postprocess_image(enhanced_tensor)
    
    # Save if output path provided
    if output_path:
        # Convert RGB to BGR for OpenCV
        enhanced_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_bgr)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced_image

def create_comparison(original_path, enhanced_image, output_path):
    """Create side-by-side comparison"""
    # Load original
    original = cv2.imread(original_path)
    original = cv2.resize(original, (256, 256))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Create comparison
    comparison = np.hstack([original_rgb, enhanced_image])
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, "Enhanced", (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, comparison_bgr)
    print(f"Comparison saved to: {output_path}")

def enhance_batch(model, input_dir, device, output_dir=None):
    """Enhance all images in a directory"""
    if output_dir is None:
        output_dir = os.path.join(input_dir, "enhanced")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Found {len(image_files)} images to enhance")
    
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        
        # Create output filename
        name, ext = os.path.splitext(image_file)
        output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
        comparison_path = os.path.join(output_dir, f"{name}_comparison{ext}")
        
        try:
            # Enhance image
            enhanced = enhance_image(model, input_path, device, output_path)
            
            # Create comparison
            create_comparison(input_path, enhanced, comparison_path)
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Simple Low Light Enhancement Inference')
    parser.add_argument('--checkpoint', type=str, default='best_model_simple.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output image or directory')
    parser.add_argument('--batch', action='store_true', help='Process batch of images')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train a model first using: py train_simple.py")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print("✓ Model loaded successfully!")
    
    if args.batch and args.input:
        # Batch processing
        if os.path.isdir(args.input):
            enhance_batch(model, args.input, device, args.output)
        else:
            print("Error: Input must be a directory for batch processing")
    
    elif args.input:
        # Single image
        if os.path.isfile(args.input):
            enhanced = enhance_image(model, args.input, device, args.output)
            
            # Create comparison if no output specified
            if not args.output:
                name, ext = os.path.splitext(args.input)
                comparison_path = f"{name}_comparison{ext}"
                create_comparison(args.input, enhanced, comparison_path)
        else:
            print("Error: Input file not found")
    
    else:
        print("Please provide input path or use --help for options")
        print("\nExamples:")
        print("  py inference_simple.py --input image.jpg --output enhanced.jpg")
        print("  py inference_simple.py --input images/ --output results/ --batch")

if __name__ == "__main__":
    main()
