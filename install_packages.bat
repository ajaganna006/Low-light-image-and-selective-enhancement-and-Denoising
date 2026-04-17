@echo off
echo Low Light Image Enhancement - Package Installation
echo ================================================

echo.
echo Testing Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo.
echo Python found! Testing pip...
python -m pip --version
if %errorlevel% neq 0 (
    echo ERROR: pip is not working
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo This may take several minutes...

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    echo Trying CPU-only version...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

python -m pip install opencv-python
python -m pip install numpy
python -m pip install matplotlib
python -m pip install pillow
python -m pip install scikit-image
python -m pip install tqdm
python -m pip install albumentations
python -m pip install tensorboard
python -m pip install wandb

echo.
echo Testing installation...
python test_python.py

echo.
echo Installation completed!
echo.
echo Next steps:
echo 1. python setup.py
echo 2. python demo.py
echo 3. python train.py
echo.
pause
