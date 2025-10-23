"""
Set random seeds for reproducibility
"""
import random
import numpy as np
import tensorflow as tf
import os


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries
    
    Args:
        seed (int): Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Configure TensorFlow for deterministic behavior
    tf.config.experimental.enable_op_determinism()
    
    print(f"Random seed set to {seed} for reproducibility")


def get_seed() -> int:
    """
    Get the current random seed from TensorFlow
    
    Returns:
        int: Current random seed
    """
    return tf.random.get_global_generator().state.numpy()[0]


if __name__ == "__main__":
    # Test seed setting
    set_seed(42)
    print("Seed setting test completed!")
