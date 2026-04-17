# Low Light Image Enhancement and Denoising

A comprehensive deep learning project for enhancing low light images and reducing noise using advanced neural network architectures.

## Features

- **Advanced Model Architecture**: U-Net based network with attention mechanisms, dense blocks, and residual connections
- **Multiple Loss Functions**: Combined L1, SSIM, perceptual, gradient, and total variation losses
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Synthetic Dataset Generation**: Create training data from normal images when paired data is unavailable
- **Comprehensive Evaluation**: PSNR, SSIM metrics with baseline comparisons
- **Flexible Inference**: Single image, batch processing, and interactive modes
- **Visualization Tools**: Comparison grids, metric plots, and result analysis

## Project Structure

```
pro-jet/
├── config.py              # Configuration settings
├── data_loader.py         # Data loading and preprocessing
├── models.py              # Neural network architectures
├── losses.py              # Loss functions
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Inference script
├── utils.py               # Utility functions
├── requirements.txt       # Dependencies
├── README.md              # This file
├── data/                  # Data directory
│   ├── train/            # Training data
│   ├── val/              # Validation data
│   └── test/             # Test data
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
└── results/              # Results and visualizations
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Install Python (if not already installed)
1. Download Python from https://www.python.org/downloads/
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
   ```bash
   python --version
   pip --version
   ```

### Step 2: Install Dependencies

**Option A: Using pip (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: If pip doesn't work, try:**
```bash
python -m pip install -r requirements.txt
```

**Option C: Use the batch file (on Windows)**
install_packages.bat
```

**Option D: Manual installation**
/*************  ✨ Windsurf Command 🌟  *************/
```
```bash
/*******  4f13489a-15f5-4f70-b719-0c4a23156b2b  *******/a
pip install torch torchvision torchaudio
pip install opencv-python numpy matplotlib pillow
pip install scikit-image tqdm albumentations tensorboard wandb
```

### Step 3: Test Installation
```bash
python test_python.py
```

### Step 4: Setup Project
```bash
python setup.py
```

### Troubleshooting
If you encounter issues:
- See `INSTALLATION_GUIDE.md` for detailed troubleshooting
- Make sure Python is properly installed and in PATH
- Try using `py` instead of `python` on Windows
- Consider using Anaconda for a complete Python environment

## Data Preparation

### Option 1: Paired Dataset
Organize your data in the following structure:
```
data/
├── train/
│   ├── low/              # Low light images
│   └── normal/           # Corresponding normal light images
├── val/
│   ├── low/
│   └── normal/
└── test/
    ├── low/
    └── normal/
```

### Option 2: Synthetic Dataset
If you don't have paired low light images, the system can create synthetic training data:
```python
from data_loader import create_synthetic_dataset
from config import Config

config = Config()
create_synthetic_dataset(config, num_samples=1000)
```

### Option 3: Single Directory
Place images in a single directory with naming conventions:
- Low light images: `*_low.*`, `*_dark.*`, `*_lowlight.*`, `*_night.*`
- Normal light images: corresponding names without these suffixes

## Usage

### 🚀 Quick Start (Recommended)

### **🎯 Option 1: Trial Version (Start Here!)**
```bash
python start_trial.py
```
- Interactive trial with 4 different options
- Perfect for testing and evaluation
- No setup required - works immediately!

### **🎨 Option 2: Quick Demo**
```bash
python quick_demo.py
```
- Creates demo images and shows enhancement results
- Multiple enhancement methods demonstration
- No setup required - works out of the box!

### **🌐 Option 3: Web Interface (Best Experience)**
```bash
python run_web_app.py
```
- Opens beautiful web interface at http://localhost:5000
- Drag & drop images for enhancement
- Multiple enhancement methods available

### **🖼️ Option 4: Command Line Enhancement**
```bash
python quick_demo.py --input your_image.jpg --output enhanced.jpg
```

### **🧠 Option 5: Train Your Own Model**
```bash
python minimal_train.py --epochs 10 --batch_size 8
```

### Full Version Training

**Basic training**:
```bash
python train.py
```

**Training with custom parameters**:
```bash
python train.py --epochs 200 --batch_size 32 --lr 0.0001
```

**Resume training from checkpoint**:
```bash
python train.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Evaluation

**Evaluate model**:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

**Compare with baseline methods**:
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --compare_baselines
```

### Inference

**Enhance single image**:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --input image.jpg --output enhanced.jpg
```

**Batch processing**:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --input input_dir/ --output output_dir/ --batch
```

**Interactive mode**:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --interactive
```

**Create comparison video**:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --input input_dir/ --output video.mp4 --video
```

**Benchmark inference speed**:
```bash
python inference.py --checkpoint checkpoints/best_model.pth --benchmark
```

## Model Architecture

The model combines several advanced techniques:

- **U-Net Architecture**: Encoder-decoder structure with skip connections
- **Dense Blocks**: Feature reuse and gradient flow improvement
- **Attention Mechanisms**: Channel and spatial attention (CBAM)
- **Residual Connections**: Help with training deep networks
- **Multi-scale Processing**: Different resolution levels for robust features

### Key Components:

1. **Encoder**: Progressive downsampling with dense blocks and attention
2. **Bottleneck**: Deep feature extraction with residual connections
3. **Decoder**: Progressive upsampling with skip connections
4. **Attention Modules**: Focus on important features and spatial regions

## Loss Functions

The model uses a combination of multiple loss functions:

- **L1 Loss**: Basic pixel-wise reconstruction
- **SSIM Loss**: Structural similarity preservation
- **Perceptual Loss**: High-level feature matching using VGG
- **Gradient Loss**: Edge preservation
- **Total Variation Loss**: Smoothness regularization

## Configuration

Modify `config.py` to adjust:

- **Model parameters**: Architecture, filters, blocks
- **Training settings**: Epochs, batch size, learning rate
- **Data paths**: Input/output directories
- **Loss weights**: Balance between different loss components

## Monitoring Training

Training progress can be monitored using:

1. **TensorBoard**:
   ```bash
   tensorboard --logdir logs/
   ```

2. **Console output**: Real-time loss and metric updates

3. **Saved images**: Sample results saved during training

## Results and Evaluation

The system provides comprehensive evaluation:

- **Quantitative Metrics**: PSNR, SSIM with statistical analysis
- **Visual Comparisons**: Side-by-side result comparisons
- **Baseline Comparisons**: Against gamma correction and histogram equalization
- **Performance Benchmarks**: Inference speed and memory usage

## Advanced Features

### Data Augmentation
- Random flips and rotations
- Brightness and contrast adjustments
- Color space modifications
- Gaussian noise injection

### Model Optimization
- Weight initialization strategies
- Learning rate scheduling
- Gradient clipping
- Mixed precision training support

### Visualization Tools
- Metric distribution plots
- Training curve visualization
- Sample result grids
- Histogram analysis

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size in config
2. **No data found**: Check data directory structure
3. **Poor results**: Adjust loss weights or learning rate
4. **Slow training**: Use GPU acceleration and optimize data loading

### Performance Tips:

1. Use SSD storage for faster data loading
2. Enable mixed precision training for faster training
3. Use multiple GPUs for large-scale training
4. Pre-process data for consistent input sizes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{low_light_enhancement,
  title={Low Light Image Enhancement and Denoising},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pro-jet}
}
```

## Acknowledgments

- PyTorch team for the deep learning framework
- OpenCV and PIL for image processing
- scikit-image for evaluation metrics
- The computer vision community for research inspiration

## Future Improvements

- [ ] Support for video enhancement
- [ ] Real-time inference optimization
- [ ] Mobile deployment support
- [ ] Additional loss functions
- [ ] Advanced data augmentation techniques
- [ ] Model compression and quantization
