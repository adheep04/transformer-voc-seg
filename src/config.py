from pathlib import Path
import numpy as np

class Config:
    
    # training/validation run num/type
    RUN = 'PASCAL_LION_LOSS2'
    
    # Paths
    CHECKPOINT_DIR = Path("checkpoints/")
    
    LOG_DIR = Path("logs")

    # Model hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 20
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-5
    
    # image crop to save memory during training
    MAX_PIXELS = 700*700
    
   

config = Config()
