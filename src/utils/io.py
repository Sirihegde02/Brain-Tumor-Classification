"""
Input/Output utilities for model saving, loading, and logging
"""
import os
import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import tensorflow as tf
import numpy as np


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save data to YAML file
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from YAML file
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_model(model: tf.keras.Model, 
               filepath: Union[str, Path],
               save_format: str = "tf") -> None:
    """
    Save TensorFlow model
    
    Args:
        model: Keras model to save
        filepath: Output file path
        save_format: Save format ("tf", "h5", "saved_model")
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    if save_format == "tf":
        model.save(str(filepath))
    elif save_format == "h5":
        model.save(str(filepath) + ".h5")
    elif save_format == "saved_model":
        model.save(str(filepath), save_format="tf")
    else:
        raise ValueError(f"Unsupported save format: {save_format}")


def load_model(filepath: Union[str, Path]) -> tf.keras.Model:
    """
    Load TensorFlow model
    
    Args:
        filepath: Model file path
        
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(str(filepath))


def save_weights(model: tf.keras.Model, filepath: Union[str, Path]) -> None:
    """
    Save model weights only
    
    Args:
        model: Keras model
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    model.save_weights(str(filepath))


def load_weights(model: tf.keras.Model, filepath: Union[str, Path]) -> None:
    """
    Load model weights
    
    Args:
        model: Keras model to load weights into
        filepath: Weights file path
    """
    model.load_weights(str(filepath))


def save_history(history: tf.keras.callbacks.History, 
                 filepath: Union[str, Path]) -> None:
    """
    Save training history
    
    Args:
        history: Training history object
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Convert history to serializable format
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]
    
    save_json(history_dict, filepath)


def load_history(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load training history
    
    Args:
        filepath: History file path
        
    Returns:
        History dictionary
    """
    return load_json(filepath)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data using pickle
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data using pickle
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_model_summary(model: tf.keras.Model, 
                      filepath: Union[str, Path]) -> None:
    """
    Save model summary to text file
    
    Args:
        model: Keras model
        filepath: Output file path
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    with open(filepath, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def get_model_info(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Get model information including parameter count
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "num_layers": len(model.layers),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape
    }


def create_checkpoint_callback(filepath: Union[str, Path],
                              monitor: str = "val_loss",
                              mode: str = "min",
                              save_best_only: bool = True,
                              save_weights_only: bool = False) -> tf.keras.callbacks.ModelCheckpoint:
    """
    Create model checkpoint callback
    
    Args:
        filepath: Checkpoint file path
        monitor: Metric to monitor
        mode: "min" or "max"
        save_best_only: Save only best model
        save_weights_only: Save only weights
        
    Returns:
        ModelCheckpoint callback
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=str(filepath),
        monitor=monitor,
        mode=mode,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        verbose=1
    )


def create_tensorboard_callback(log_dir: Union[str, Path]) -> tf.keras.callbacks.TensorBoard:
    """
    Create TensorBoard callback
    
    Args:
        log_dir: Log directory path
        
    Returns:
        TensorBoard callback
    """
    log_dir = Path(log_dir)
    ensure_dir(log_dir)
    
    return tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch'
    )


if __name__ == "__main__":
    # Test utilities
    print("IO utilities test completed!")
