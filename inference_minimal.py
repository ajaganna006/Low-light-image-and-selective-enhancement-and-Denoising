#!/usr/bin/env python3
"""
Minimal inference script for Low Light Enhancement
Works with only PyTorch and PIL
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse

# Import the minimal model
from train_minimal import MinimalEnhancementNet

def load_model(checkpoint_path, device):
    """Load the trained model"""
    model = MinimalEnhancementNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, size=(256, 256)):
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize(size)
    
    # Convert to numpy and normalize to [-1, 1]
    image = np.array(image, dtype=np.float32) / 255.0 * 2 - 1
    
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
    
    # Convert to PIL Image
    enhanced_pil = Image.fromarray(enhanced_image)
    
    # Save if output path provided
    if output_path:
        enhanced_pil.save(output_path)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced_pil

def create_comparison(original_path, enhanced_image, output_path):
    """Create side-by-side comparison"""
    # Load original
    original = Image.open(original_path).convert('RGB')
    original = original.resize((256, 256))
    
    # Create comparison
    comparison = Image.new('RGB', (512, 256))
    comparison.paste(original, (0, 0))
    comparison.paste(enhanced_image, (256, 0))
    
    # Save
    comparison.save(output_path)
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
    parser = argparse.ArgumentParser(description='Minimal Low Light Enhancement Inference')
    parser.add_argument('--checkpoint', type=str, default='best_model_minimal.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output image or directory')
    parser.add_argument('--batch', action='store_true', help='Process batch of images')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("Please train a model first using: py train_minimal.py")
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
        print("  py inference_minimal.py --input image.jpg --output enhanced.jpg")
        print("  py inference_minimal.py --input images/ --output results/ --batch")

if __name__ == "__main__":
    main()
