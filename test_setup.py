#!/usr/bin/env python3
"""
Sample script to test the Low Light Image Enhancement setup
"""

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
        print("\nCreating synthetic dataset for testing...")
        create_synthetic_dataset(config, num_samples=10)
        print("✓ Synthetic dataset created")
    except Exception as e:
        print(f"✗ Error creating synthetic dataset: {e}")
        return False
    
    print("\n✓ Setup test completed successfully!")
    print("\nNext steps:")
    print("1. Add your own images to the data/train/ directory")
    print("2. Run: python train.py")
    print("3. Or use synthetic data for testing")
    
    return True

if __name__ == "__main__":
    test_setup()
