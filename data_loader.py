"""
Data loading and preprocessing utilities for low light image enhancement
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class LowLightDataset(Dataset):
    """
    Dataset class for low light image enhancement
    Expects paired images: low light and corresponding normal light images
    """
    
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        
        # Look for paired images
        self.image_pairs = self._load_image_pairs()
        
    def _load_image_pairs(self):
        """Load pairs of low light and normal light images"""
        pairs = []
        
        # Look for subdirectories or specific naming patterns
        if os.path.exists(os.path.join(self.data_dir, "low")):
            # Separate directories for low and normal light images
            low_dir = os.path.join(self.data_dir, "low")
            normal_dir = os.path.join(self.data_dir, "normal")
            
            low_images = sorted(os.listdir(low_dir))
            normal_images = sorted(os.listdir(normal_dir))
            
            for low_img, normal_img in zip(low_images, normal_images):
                if self._is_image_file(low_img) and self._is_image_file(normal_img):
                    pairs.append({
                        'low': os.path.join(low_dir, low_img),
                        'normal': os.path.join(normal_dir, normal_img)
                    })
        else:
            # Single directory with paired images
            images = sorted(os.listdir(self.data_dir))
            low_images = [img for img in images if 'low' in img.lower() or 'dark' in img.lower()]
            
            for low_img in low_images:
                # Try to find corresponding normal light image
                base_name = self._get_base_name(low_img)
                normal_img = self._find_normal_image(images, base_name)
                
                if normal_img:
                    pairs.append({
                        'low': os.path.join(self.data_dir, low_img),
                        'normal': os.path.join(self.data_dir, normal_img)
                    })
        
        return pairs
    
    def _is_image_file(self, filename):
        """Check if file is an image"""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    
    def _get_base_name(self, filename):
        """Extract base name from filename"""
        name = os.path.splitext(filename)[0]
        # Remove common low light suffixes
        for suffix in ['_low', '_dark', '_lowlight', '_night']:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name
    
    def _find_normal_image(self, images, base_name):
        """Find corresponding normal light image"""
        for img in images:
            if (self._is_image_file(img) and 
                base_name in img and 
                not any(suffix in img.lower() for suffix in ['_low', '_dark', '_lowlight', '_night'])):
                return img
        return None
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load images
        low_img = self._load_image(pair['low'])
        normal_img = self._load_image(pair['normal'])
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=low_img, mask=normal_img)
            low_img = transformed['image']
            normal_img = transformed['mask']
        else:
            # Convert to tensors
            low_img = torch.from_numpy(low_img).permute(2, 0, 1).float() / 255.0
            normal_img = torch.from_numpy(normal_img).permute(2, 0, 1).float() / 255.0
        
        return low_img, normal_img
    
    def _load_image(self, path):
        """Load and preprocess image"""
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def get_transforms(is_training=True, input_size=(256, 256)):
    """Get data augmentation transforms"""
    if is_training:
        transform = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
    
    return transform

def create_data_loaders(config):
    """Create training and validation data loaders"""
    train_transform = get_transforms(is_training=True, input_size=config.INPUT_SIZE)
    val_transform = get_transforms(is_training=False, input_size=config.INPUT_SIZE)
    
    train_dataset = LowLightDataset(
        config.TRAIN_DIR, 
        transform=train_transform, 
        is_training=True
    )
    
    val_dataset = LowLightDataset(
        config.VAL_DIR, 
        transform=val_transform, 
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_synthetic_dataset(config, num_samples=1000):
    """
    Create synthetic low light dataset by applying transformations to normal images
    This is useful when you don't have paired low light images
    """
    import glob
    import random
    
    # Look for any images in the data directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(config.DATA_ROOT, '**', ext), recursive=True))
    
    if not image_paths:
        print("No images found in data directory. Please add some images to create synthetic dataset.")
        return
    
    # Create synthetic low light images
    low_dir = os.path.join(config.TRAIN_DIR, "low")
    normal_dir = os.path.join(config.TRAIN_DIR, "normal")
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    
    for i, img_path in enumerate(image_paths[:num_samples]):
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create low light version
        low_light = create_low_light_image(image)
        
        # Save both versions
        base_name = f"synthetic_{i:04d}.png"
        cv2.imwrite(os.path.join(normal_dir, base_name), 
                   cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(low_dir, base_name), 
                   cv2.cvtColor(low_light, cv2.COLOR_RGB2BGR))
    
    print(f"Created {min(num_samples, len(image_paths))} synthetic image pairs")

def create_low_light_image(image, gamma_range=(1.5, 3.0), brightness_range=(0.1, 0.4)):
    """
    Create synthetic low light image from normal image
    """
    import cv2
    
    # Random gamma correction
    gamma = random.uniform(*gamma_range)
    low_light = np.power(image / 255.0, gamma) * 255.0
    
    # Random brightness reduction
    brightness = random.uniform(*brightness_range)
    low_light = low_light * brightness
    
    # Add some noise
    noise = np.random.normal(0, 5, low_light.shape)
    low_light = np.clip(low_light + noise, 0, 255)
    
    return low_light.astype(np.uint8)
