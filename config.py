"""
Configuration file for Low Light Image Enhancement and Denoising project
"""

import os

class Config:
    # Data paths
    DATA_ROOT = "data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    
    # Model parameters
    INPUT_SIZE = (256, 256)
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Model architecture
    NUM_CHANNELS = 3
    NUM_FILTERS = 64
    NUM_BLOCKS = 8
    
    # Training parameters
    SAVE_INTERVAL = 10
    LOG_INTERVAL = 100
    VAL_INTERVAL = 5
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    RESULTS_DIR = "results"
    
    # Device
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Loss weights
    L1_WEIGHT = 1.0
    SSIM_WEIGHT = 0.1
    PERCEPTUAL_WEIGHT = 0.1
    
    # Data augmentation
    AUGMENTATION_PROB = 0.5
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        dirs = [cls.DATA_ROOT, cls.CHECKPOINT_DIR, cls.LOG_DIR, 
                cls.RESULTS_DIR, cls.TRAIN_DIR, cls.VAL_DIR, cls.TEST_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
