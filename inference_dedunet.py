"""
DEDUNet Inference Script
High-quality low-light image enhancement using DEDUNet
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import argparse
import os
import time
from pathlib import Path

from dedunet_simple import create_dedunet_simple
from utils import calculate_psnr, calculate_ssim, save_image


class DEDUNetInference:
    """
    DEDUNet Inference Engine
    """
    def __init__(self, checkpoint_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Load model
        print(f"🔄 Loading DEDUNet model from {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_model(self, checkpoint_path):
        """Load DEDUNet model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        model = create_dedunet_simple()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image_path, target_size=512):
        """Preprocess input image"""
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # Convert to numpy
        image_np = np.array(image)
        
        # Resize if needed
        if max(image_np.shape[:2]) > target_size:
            scale = target_size / max(image_np.shape[:2])
            new_h = int(image_np.shape[0] * scale)
            new_w = int(image_np.shape[1] * scale)
            image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device), image_np
    
    def postprocess_image(self, tensor, original_shape):
        """Postprocess output tensor to image"""
        # Convert to numpy
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tensor = np.clip(tensor, 0, 1)
        
        # Convert to uint8
        image = (tensor * 255).astype(np.uint8)
        
        # Resize to original shape if needed
        if image.shape[:2] != original_shape[:2]:
            image = cv2.resize(image, (original_shape[1], original_shape[0]), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        return image
    
    def enhance_image(self, image_path, output_path=None, save_comparison=True):
        """Enhance a single image"""
        print(f"🖼️  Processing: {image_path}")
        
        # Preprocess
        input_tensor, original_image = self.preprocess_image(image_path)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            enhanced_tensor = self.model(input_tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        enhanced_image = self.postprocess_image(enhanced_tensor, original_image.shape)
        
        print(f"⏱️  Inference time: {inference_time:.3f}s")
        
        # Save results
        if output_path is None:
            base_name = Path(image_path).stem
            output_path = f"{base_name}_dedunet_enhanced.png"
        
        # Save enhanced image
        enhanced_pil = Image.fromarray(enhanced_image)
        enhanced_pil.save(output_path)
        print(f"💾 Enhanced image saved: {output_path}")
        
        # Save comparison
        if save_comparison:
            comparison_path = f"{Path(output_path).stem}_comparison.png"
            self._save_comparison(original_image, enhanced_image, comparison_path)
            print(f"📊 Comparison saved: {comparison_path}")
        
        return enhanced_image, inference_time
    
    def _save_comparison(self, original, enhanced, output_path):
        """Save before/after comparison"""
        h, w = original.shape[:2]
        
        # Create comparison image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = original
        comparison[:, w:] = enhanced
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "DEDUNet Enhanced", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save
        cv2.imwrite(output_path, comparison)
    
    def batch_enhance(self, input_dir, output_dir, file_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """Enhance multiple images in batch"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"🔍 Found {len(image_files)} images to process")
        
        total_time = 0
        for i, image_file in enumerate(image_files):
            print(f"\n📸 Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            output_file = output_path / f"{image_file.stem}_dedunet_enhanced.png"
            
            try:
                _, inference_time = self.enhance_image(
                    str(image_file), 
                    str(output_file),
                    save_comparison=True
                )
                total_time += inference_time
                
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
        
        print(f"\n🎉 Batch processing completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"📊 Average time per image: {total_time/len(image_files):.3f}s")
        print(f"💾 Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DEDUNet Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/dedunet_best.pth', 
                       help='Model checkpoint path')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("💡 Please train the model first using: python train_dedunet.py")
        return
    
    # Create inference engine
    try:
        inference = DEDUNetInference(args.checkpoint, args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Process input
    if args.batch or os.path.isdir(args.input):
        # Batch processing
        output_dir = args.output or f"{args.input}_dedunet_results"
        inference.batch_enhance(args.input, output_dir)
    else:
        # Single image processing
        inference.enhance_image(args.input, args.output)


if __name__ == "__main__":
    main()

