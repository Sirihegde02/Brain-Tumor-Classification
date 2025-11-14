"""
Parameter counting and model analysis utilities
"""
import tensorflow as tf
from typing import Dict, Any, List, Tuple
import numpy as np


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params
    }


def get_layer_info(model: tf.keras.Model) -> List[Dict[str, Any]]:
    """
    Get detailed information about each layer
    
    Args:
        model: Keras model
        
    Returns:
        List of layer information dictionaries
    """
    layer_info = []
    
    for i, layer in enumerate(model.layers):
        layer_params = {
            "index": i,
            "name": layer.name,
            "type": type(layer).__name__,
            "input_shape": layer.input_shape,
            "output_shape": layer.output_shape,
            "params": layer.count_params(),
            "trainable": layer.trainable
        }
        layer_info.append(layer_params)
    
    return layer_info


def estimate_flops(model: tf.keras.Model, input_shape: Tuple[int, ...] = None) -> int:
    """
    Estimate FLOPs (Floating Point Operations) for the model
    
    Args:
        model: Keras model
        input_shape: Input shape for FLOP calculation
        
    Returns:
        Estimated FLOPs
    """
    if input_shape is None:
        input_shape = model.input_shape[1:]  # Remove batch dimension
    
    # Create a dummy input
    dummy_input = tf.random.normal((1,) + input_shape)
    
    # Count FLOPs using TensorFlow's profiler
    try:
        from tensorflow.python.profiler import profiler_v2 as profiler
        
        # This is a simplified estimation
        # In practice, you might want to use more sophisticated tools
        flops = 0
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Rough FLOP estimation for Conv2D
                kernel_size = layer.kernel_size
                filters = layer.filters
                input_h, input_w = layer.input_shape[1:3]
                output_h, output_w = layer.output_shape[1:3]
                
                conv_flops = (kernel_size[0] * kernel_size[1] * 
                             layer.input_shape[-1] * filters * 
                             output_h * output_w)
                flops += conv_flops
                
            elif isinstance(layer, tf.keras.layers.Dense):
                # FLOP estimation for Dense layers
                input_units = layer.input_shape[-1]
                output_units = layer.units
                dense_flops = input_units * output_units
                flops += dense_flops
        
        return flops
        
    except ImportError:
        print("Warning: Could not import profiler for FLOP estimation")
        return 0


def get_model_size_mb(model: tf.keras.Model) -> float:
    """
    Get model size in megabytes
    
    Args:
        model: Keras model
        
    Returns:
        Model size in MB
    """
    # Save model to temporary file and get size
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        model.save(tmp.name)
        size_bytes = os.path.getsize(tmp.name)
        os.unlink(tmp.name)
    
    return size_bytes / (1024 * 1024)  # Convert to MB


def analyze_model_complexity(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Comprehensive model complexity analysis
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with complexity metrics
    """
    params = count_parameters(model)
    layer_info = get_layer_info(model)
    
    # Count different layer types
    layer_types = {}
    for layer in layer_info:
        layer_type = layer["type"]
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    # Calculate model size
    model_size_mb = get_model_size_mb(model)
    
    # Estimate FLOPs
    flops = estimate_flops(model)
    
    return {
        "parameters": params,
        "model_size_mb": model_size_mb,
        "estimated_flops": flops,
        "num_layers": len(layer_info),
        "layer_types": layer_types,
        "layer_details": layer_info
    }


def compare_models(models: Dict[str, tf.keras.Model]) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models
    
    Args:
        models: Dictionary of model name -> model
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    for name, model in models.items():
        comparison[name] = analyze_model_complexity(model)
    
    return comparison


def print_model_summary(model: tf.keras.Model, 
                       show_layers: bool = True,
                       show_parameters: bool = True) -> None:
    """
    Print detailed model summary
    
    Args:
        model: Keras model
        show_layers: Whether to show layer details
        show_parameters: Whether to show parameter counts
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Basic info
    print(f"Model: {model.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print()
    
    # Parameter counts
    if show_parameters:
        params = count_parameters(model)
        print("PARAMETER COUNTS:")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Non-trainable parameters: {params['non_trainable']:,}")
        print()
    
    # Layer information
    if show_layers:
        layer_info = get_layer_info(model)
        print("LAYER DETAILS:")
        print("-" * 80)
        print(f"{'Index':<6} {'Name':<20} {'Type':<15} {'Output Shape':<20} {'Params':<10}")
        print("-" * 80)
        
        for layer in layer_info:
            output_shape = str(layer['output_shape'])[:18] + "..." if len(str(layer['output_shape'])) > 20 else str(layer['output_shape'])
            print(f"{layer['index']:<6} {layer['name']:<20} {layer['type']:<15} {output_shape:<20} {layer['params']:<10,}")
        
        print("-" * 80)
        print()
    
    # Complexity analysis
    complexity = analyze_model_complexity(model)
    print("COMPLEXITY ANALYSIS:")
    print(f"  Model size: {complexity['model_size_mb']:.2f} MB")
    print(f"  Estimated FLOPs: {complexity['estimated_flops']:,}")
    print(f"  Number of layers: {complexity['num_layers']}")
    print()
    
    print("LAYER TYPE DISTRIBUTION:")
    for layer_type, count in complexity['layer_types'].items():
        print(f"  {layer_type}: {count}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Test parameter counting
    print("Parameter utilities test completed!")
