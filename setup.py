"""
Setup script for Low Light Image Enhancement project
"""

import os
import sys
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    directories = [
        "data",
        "data/train",
        "data/train/low",
        "data/train/normal", 
        "data/val",
        "data/val/low",
        "data/val/normal",
        "data/test",
        "checkpoints",
        "logs",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("✓ All directories created successfully!")

def create_sample_config():
    """Create a sample configuration file"""
    print("Creating sample configuration...")
    
    sample_config = """# Sample configuration for Low Light Image Enhancement
# Copy this to config.py and modify as needed

class Config:
    # Data paths
    DATA_ROOT = "data"
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    TEST_DIR = "data/test"
    
    # Model parameters
    INPUT_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""
    
    with open("config_sample.py", "w", encoding="utf-8") as f:
        f.write(sample_config)
    
    print("✓ Sample configuration created: config_sample.py")

def check_gpu():
    """Check GPU availability"""
    print("Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
        else:
            print("⚠ No GPU available. Training will use CPU (slower)")
    except ImportError:
        print("⚠ PyTorch not installed yet. Run setup again after installation.")

def create_sample_script():
    """Create a sample script to test the setup"""
    sample_script = """#!/usr/bin/env python3
\"\"\"
Sample script to test the Low Light Image Enhancement setup
\"\"\"

import torch
from config import Config
from models import create_model, count_parameters
from data_loader import create_synthetic_dataset

def test_setup():
    print("Testing Low Light Image Enhancement Setup")
    print("=" * 50)
    
    # Test configuration
    config = Config()
    print(f"✓ Configuration loaded")
    print(f"  Device: {config.DEVICE}")
    print(f"  Input size: {config.INPUT_SIZE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    
    # Test model creation
    try:
        model = create_model(config)
        num_params = count_parameters(model)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {num_params:,}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False
    
    # Test synthetic dataset creation
    try:
        print("\\nCreating synthetic dataset for testing...")
        create_synthetic_dataset(config, num_samples=10)
        print("✓ Synthetic dataset created")
    except Exception as e:
        print(f"✗ Error creating synthetic dataset: {e}")
        return False
    
    print("\\n✓ Setup test completed successfully!")
    print("\\nNext steps:")
    print("1. Add your own images to the data/train/ directory")
    print("2. Run: python train.py")
    print("3. Or use synthetic data for testing")
    
    return True

if __name__ == "__main__":
    test_setup()
"""
    
    with open("test_setup.py", "w", encoding="utf-8") as f:
        f.write(sample_script)
    
    print("✓ Test script created: test_setup.py")

/*************  ✨ Windsurf Command 🌟  *************/
def main():
    """Main setup function"""
    print("Low Light Image Enhancement - Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation")
        return
    
    # Create directories
    create_directories()
    
    # Create sample files
    create_sample_config()
    create_sample_script()
    
    # Check GPU
    check_gpu()
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nTo test your setup, run:")
    print("  python test_setup.py")
    print("\nTo start training, run:")
    print("  python train.py")
    print("\nFor more information, see README.md")
    print("\nWebsite zoom options:")
    print("  Ctrl + (zoom in)")
    print("  Ctrl - (zoom out)")
    print("  Ctrl 0 (reset zoom)")
/*******  fb9b6944-df78-467a-b858-2f3c95305e9c  *******/

if __name__ == "__main__":
    main()
