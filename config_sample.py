# Sample configuration for Low Light Image Enhancement
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
