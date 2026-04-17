"""
Simple inference script for low light image enhancement
Works with the existing model file format
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import argparse
import torchvision.transforms as transforms

from config import Config
from simple_model import create_simple_model

def load_simple_model(checkpoint_path, device):
    """Load model from simple state dict format"""
    model = create_simple_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, size=(256, 256)):
    """Preprocess image for inference"""
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

def postprocess_image(tensor):
    """Postprocess tensor to image"""
    # Convert from [-1, 1] to [0, 1] range
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)

def enhance_image_simple(model, image_path, device, output_path=None):
    """Enhance a single image"""
    # Preprocess
    tensor = preprocess_image(image_path).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        enhanced = model(tensor)
    
    # Postprocess
    enhanced_image = postprocess_image(enhanced.squeeze(0))
    
    if output_path:
        enhanced_image.save(output_path)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced_image

def main():
    parser = argparse.ArgumentParser(description='Simple Inference for Low Light Enhancement')
    parser.add_argument('--checkpoint', type=str, default='best_model_minimal.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, help='Output image path')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_simple_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Set output path
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_enhanced.png"
    
    # Enhance image
    print(f"Enhancing image: {args.input}")
    enhanced = enhance_image_simple(model, args.input, device, args.output)
    
    print("Enhancement completed!")

if __name__ == '__main__':
    main()
