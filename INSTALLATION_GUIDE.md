# Installation Guide for Low Light Image Enhancement Project

## Issue: Python Installation Problem

Your current Python installation appears to be corrupted. Here's how to fix it:

## Solution 1: Reinstall Python (Recommended)

### Step 1: Download Python
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11** or **Python 3.12** (recommended for better compatibility)
3. Choose the Windows installer (64-bit)

### Step 2: Install Python
1. Run the downloaded installer
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Choose "Install Now" or "Customize installation"
4. If customizing, make sure to include:
   - pip
   - tcl/tk and IDLE
   - Python test suite
   - py launcher

### Step 3: Verify Installation
Open a new Command Prompt or PowerShell and run:
```bash
python --version
pip --version
```

## Solution 2: Use Microsoft Store Python

1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Install the official Python package
4. This automatically adds Python to PATH

## Solution 3: Use Anaconda (Alternative)

1. Download Anaconda from https://www.anaconda.com/download
2. Install Anaconda
3. Open Anaconda Prompt
4. Use conda instead of pip:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install opencv numpy matplotlib pillow scikit-image tqdm
   ```

## After Fixing Python Installation

Once Python is working properly, run these commands in your project directory:

```bash
# Install requirements
pip install -r requirements.txt

# Or if pip doesn't work, try:
python -m pip install -r requirements.txt

# Run setup
python setup.py

# Test the installation
python test_setup.py

# Run demo
python demo.py
```

## Alternative: Manual Package Installation

If you continue having issues, you can install packages manually:

```bash
pip install torch torchvision torchaudio
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pillow
pip install scikit-image
pip install tqdm
pip install albumentations
pip install tensorboard
pip install wandb
```

## Troubleshooting

### If you get "pip is not recognized":
- Make sure Python was installed with "Add to PATH" option
- Restart your command prompt/PowerShell
- Try using `python -m pip` instead of just `pip`

### If you get permission errors:
- Run Command Prompt as Administrator
- Or use `pip install --user package_name`

### If you get SSL errors:
- Try: `pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org package_name`

## Quick Test

After installation, test with this simple script:

```python
# test_python.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("✓ All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
```

Run with: `python test_python.py`

## Need Help?

If you're still having issues:
1. Check your Python installation path
2. Verify environment variables
3. Try a different Python version
4. Consider using a virtual environment
