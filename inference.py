"""
Inference script for low light image enhancement and denoising
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
import cv2
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from models import create_model
from utils import load_checkpoint, enhance_image, create_comparison_grid, image_to_tensor, tensor_to_image

def enhance_single_image(model, image_path, device, output_path=None):
    """Enhance a single image"""
    print(f"Enhancing image: {image_path}")
    
    # Enhance image
    enhanced = enhance_image(model, image_path, device, output_path)
    
    if output_path:
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced

def enhance_batch_images(model, input_dir, device, output_dir=None):
    """Enhance a batch of images"""
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images to enhance")
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'enhanced')
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhance each image
    for image_path in tqdm(image_paths, desc="Enhancing images"):
        # Get output filename
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_enhanced{ext}")
        
        # Enhance image
        enhance_single_image(model, image_path, device, output_path)
    
    print(f"All images enhanced and saved to: {output_dir}")

def create_comparison_video(model, input_dir, device, output_path, fps=2):
    """Create a comparison video showing original and enhanced images"""
    import cv2
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    # Sort images
    image_paths = sorted(image_paths)
    
    # Get first image to determine video dimensions
    first_img = Image.open(image_paths[0]).convert('RGB')
    width, height = first_img.size
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    print(f"Creating comparison video with {len(image_paths)} images...")
    
    for image_path in tqdm(image_paths, desc="Processing video frames"):
        # Load original image
        original = Image.open(image_path).convert('RGB')
        original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        
        # Enhance image
        enhanced = enhance_image(model, image_path, device)
        enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
        
        # Resize images to match
        original_cv = cv2.resize(original_cv, (width, height))
        enhanced_cv = cv2.resize(enhanced_cv, (width, height))
        
        # Combine images side by side
        combined = np.hstack([original_cv, enhanced_cv])
        
        # Write frame
        video_writer.write(combined)
    
    video_writer.release()
    print(f"Comparison video saved to: {output_path}")

def interactive_enhancement(model, device):
    """Interactive image enhancement"""
    print("Interactive Image Enhancement")
    print("=" * 40)
    
    while True:
        image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not os.path.exists(image_path):
            print("Image not found. Please try again.")
            continue
        
        try:
            # Enhance image
            enhanced = enhance_image(model, image_path, device)
            
            # Show results
            original = Image.open(image_path).convert('RGB')
            
            # Create comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            ax1.imshow(original)
            ax1.set_title('Original')
            ax1.axis('off')
            
            ax2.imshow(enhanced)
            ax2.set_title('Enhanced')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Ask if user wants to save
            save = input("Save enhanced image? (y/n): ").strip().lower()
            if save == 'y':
                output_path = input("Enter output path: ").strip()
                enhanced.save(output_path)
                print(f"Enhanced image saved to: {output_path}")
        
        except Exception as e:
            print(f"Error processing image: {e}")

def benchmark_inference_speed(model, device, num_runs=100):
    """Benchmark inference speed"""
    print("Benchmarking inference speed...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Warm up
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        start_time.record()
    else:
        import time
        start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        total_time = start_time.elapsed_time(end_time)
        avg_time = total_time / num_runs
    else:
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to ms
        avg_time = total_time / num_runs
    
    print(f"Inference Speed Results:")
    print(f"Total time for {num_runs} runs: {total_time:.2f} ms")
    print(f"Average time per image: {avg_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.2f}")

def main():
    parser = argparse.ArgumentParser(description='Inference for Low Light Enhancement Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, help='Input image or directory path')
    parser.add_argument('--output', type=str, help='Output image or directory path')
    parser.add_argument('--batch', action='store_true', help='Process batch of images')
    parser.add_argument('--video', action='store_true', help='Create comparison video')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark inference speed')
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    print("Model loaded successfully!")
    
    # Run inference based on arguments
    if args.benchmark:
        benchmark_inference_speed(model, device)
    
    elif args.interactive:
        interactive_enhancement(model, device)
    
    elif args.batch:
        if not args.input:
            print("Please provide input directory with --input")
            return
        enhance_batch_images(model, args.input, device, args.output)
    
    elif args.video:
        if not args.input:
            print("Please provide input directory with --input")
            return
        if not args.output:
            args.output = "comparison_video.mp4"
        create_comparison_video(model, args.input, device, args.output)
    
    elif args.input:
        if os.path.isfile(args.input):
            # Single image
            enhance_single_image(model, args.input, device, args.output)
        elif os.path.isdir(args.input):
            # Directory of images
            enhance_batch_images(model, args.input, device, args.output)
        else:
            print("Input path not found")
    
    else:
        print("Please provide input path or use --interactive mode")
        print("Use --help for more options")

if __name__ == '__main__':
    main()
