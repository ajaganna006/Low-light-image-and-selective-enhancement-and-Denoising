#!/usr/bin/env python3
"""
Simple package installer for the Low Light Enhancement project
"""

import subprocess
import sys

def install_package(package):
    """Install a single package"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install packages one by one"""
    print("Installing packages for Low Light Enhancement project...")
    print("=" * 60)
    
    # Core packages
    packages = [
        "torch",
        "torchvision", 
        "torchaudio",
        "numpy",
        "opencv-python",
        "matplotlib",
        "pillow",
        "tqdm",
        "tensorboard"
    ]
    
    # Optional packages (may fail on some systems)
    optional_packages = [
        "scikit-image",
        "albumentations",
        "wandb"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    # Install core packages
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    # Install optional packages
    print("Installing optional packages...")
    for package in optional_packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Installation completed: {success_count}/{total_count} core packages installed")
    
    if success_count >= 6:  # At least torch, numpy, opencv, matplotlib, pillow, tqdm
        print("✓ Core packages installed successfully!")
        print("\nYou can now run:")
        print("  py train.py")
        print("  py demo.py")
    else:
        print("⚠ Some core packages failed to install.")
        print("You may need to install them manually or use conda.")
    
    # Test imports
    print("\nTesting imports...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not available")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not available")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not available")

if __name__ == "__main__":
    main()
