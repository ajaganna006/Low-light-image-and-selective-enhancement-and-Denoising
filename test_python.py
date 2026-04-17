#!/usr/bin/env python3
"""
Simple test script to verify Python and package installation
"""

def test_python():
    """Test basic Python functionality"""
    print("Testing Python installation...")
    
    try:
        import sys
        print(f"✓ Python version: {sys.version}")
        print(f"✓ Python executable: {sys.executable}")
        return True
    except Exception as e:
        print(f"✗ Python test failed: {e}")
        return False

def test_packages():
    """Test package imports"""
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('skimage', 'scikit-image'),
        ('tqdm', 'tqdm'),
        ('albumentations', 'Albumentations')
    ]
    
    print("\nTesting package imports...")
    results = {}
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name} imported successfully")
            results[package] = True
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            results[package] = False
    
    return results

def test_torch_functionality():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch functionality...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Test tensor creation
        x = torch.randn(2, 3)
        print(f"✓ Tensor creation works: {x.shape}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU only)")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_opencv_functionality():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV functionality...")
    
    try:
        import cv2
        import numpy as np
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        # Test image creation
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        print("✓ Image creation works")
        
        return True
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Python Environment Test")
    print("=" * 30)
    
    # Test Python
    python_ok = test_python()
    
    if not python_ok:
        print("\n❌ Python installation is corrupted!")
        print("Please reinstall Python following the INSTALLATION_GUIDE.md")
        return
    
    # Test packages
    package_results = test_packages()
    
    # Test specific functionality
    torch_ok = test_torch_functionality()
    opencv_ok = test_opencv_functionality()
    
    # Summary
    print("\n" + "=" * 30)
    print("Test Summary:")
    print(f"Python: {'✓' if python_ok else '✗'}")
    print(f"PyTorch: {'✓' if torch_ok else '✗'}")
    print(f"OpenCV: {'✓' if opencv_ok else '✗'}")
    
    all_packages_ok = all(package_results.values())
    print(f"All packages: {'✓' if all_packages_ok else '✗'}")
    
    if python_ok and torch_ok and opencv_ok and all_packages_ok:
        print("\n🎉 All tests passed! You're ready to run the project.")
        print("\nNext steps:")
        print("1. python setup.py")
        print("2. python demo.py")
        print("3. python train.py")
    else:
        print("\n❌ Some tests failed. Please install missing packages:")
        print("pip install -r requirements.txt")
        print("\nOr follow the INSTALLATION_GUIDE.md for detailed instructions.")

if __name__ == "__main__":
    main()
