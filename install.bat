@echo off
echo Installing packages for Low Light Enhancement...
echo ================================================

echo.
echo Installing PyTorch...
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing OpenCV...
py -m pip install opencv-python

echo.
echo Installing NumPy...
py -m pip install numpy

echo.
echo Installing Matplotlib...
py -m pip install matplotlib

echo.
echo Installing Pillow...
py -m pip install pillow

echo.
echo Installing tqdm...
py -m pip install tqdm

echo.
echo Testing installation...
py -c "import torch; print('PyTorch version:', torch.__version__)"
py -c "import cv2; print('OpenCV version:', cv2.__version__)"
py -c "import numpy; print('NumPy version:', numpy.__version__)"

echo.
echo Installation completed!
echo.
echo You can now run:
echo   py train_simple.py
echo   py demo_simple.py
echo.
pause
