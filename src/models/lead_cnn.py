"""
LEAD-CNN model implementation for brain tumor classification

This module implements a faithful reproduction of the LEAD-CNN architecture
with dimension-reduction blocks and LeakyReLU activations.
"""
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional, Dict, Any
from .blocks import DimensionReductionBlock


class LEADCNN(keras.Model):
    """
    LEAD-CNN model for brain tumor classification
    
    Implements the architecture described in the LEAD-CNN paper with:
    - Dimension reduction blocks
    - LeakyReLU activations
    - Dropout for regularization
    - Global average pooling
    - Dense classification head
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 4,
                 dropout_rate: float = 0.5,
                 name: str = "LEAD_CNN",
                 **kwargs):
        """
        Initialize LEAD-CNN model
        
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Input layer
        self.input_layer = keras.layers.Input(shape=input_shape, name="input")
        
        # Initial convolution
        self.conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=False,
            name="conv1"
        )
        self.bn1 = keras.layers.BatchNormalization(name="bn1")
        self.activation1 = keras.layers.LeakyReLU(alpha=0.1, name="leaky_relu1")
        self.pool1 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name="pool1")
        
        # Dimension reduction blocks
        self.dr_block1 = DimensionReductionBlock(
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            dropout_rate=0.2,
            name="dr_block1"
        )
        
        self.dr_block2 = DimensionReductionBlock(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            dropout_rate=0.3,
            name="dr_block2"
        )
        
        self.dr_block3 = DimensionReductionBlock(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            dropout_rate=0.3,
            name="dr_block3"
        )
        
        self.dr_block4 = DimensionReductionBlock(
            filters=512,
            kernel_size=(3, 3),
            strides=(2, 2),
            dropout_rate=0.4,
            name="dr_block4"
        )
        
        # Global average pooling
        self.global_pool = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        
        # Classification head
        self.dropout = keras.layers.Dropout(dropout_rate, name="dropout")
        
        self.dense1 = keras.layers.Dense(
            units=512,
            activation='leaky_relu',
            name="dense1"
        )
        
        self.dense2 = keras.layers.Dense(
            units=256,
            activation='leaky_relu',
            name="dense2"
        )
        
        self.output_layer = keras.layers.Dense(
            units=num_classes,
            activation='softmax',
            name="output"
        )
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """Build the model architecture"""
        # Forward pass
        x = self.input_layer
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        
        # Dimension reduction blocks
        x = self.dr_block1(x)
        x = self.dr_block2(x)
        x = self.dr_block3(x)
        x = self.dr_block4(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification head
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        
        # Create model
        self.model = keras.Model(inputs=self.input_layer, outputs=x, name=self.name)
    
    def call(self, inputs, training=None):
        """Forward pass"""
        return self.model(inputs, training=training)
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    def summary(self, **kwargs):
        """Print model summary"""
        return self.model.summary(**kwargs)
    
    def save(self, filepath, **kwargs):
        """Save model"""
        return self.model.save(filepath, **kwargs)
    
    def load_weights(self, filepath, **kwargs):
        """Load model weights"""
        return self.model.load_weights(filepath, **kwargs)


def create_lead_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 4,
                   dropout_rate: float = 0.5,
                   **kwargs) -> LEADCNN:
    """
    Create LEAD-CNN model
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
        dropout_rate: Dropout rate
        **kwargs: Additional arguments
        
    Returns:
        LEAD-CNN model instance
    """
    return LEADCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        **kwargs
    )


def get_lead_cnn_architecture() -> Dict[str, Any]:
    """
    Get LEAD-CNN architecture description
    
    Returns:
        Dictionary with architecture details
    """
    return {
        "name": "LEAD-CNN",
        "description": "Lightweight Efficient Attention-based Deep CNN for brain tumor classification",
        "input_shape": (224, 224, 3),
        "num_classes": 4,
        "key_features": [
            "Dimension reduction blocks with LeakyReLU",
            "Global average pooling",
            "Dropout regularization",
            "Multi-scale feature extraction"
        ],
        "blocks": [
            {"name": "conv1", "type": "Conv2D", "filters": 32, "kernel": 7, "stride": 2},
            {"name": "pool1", "type": "MaxPool2D", "pool_size": 3, "stride": 2},
            {"name": "dr_block1", "type": "DimensionReduction", "filters": 64},
            {"name": "dr_block2", "type": "DimensionReduction", "filters": 128, "stride": 2},
            {"name": "dr_block3", "type": "DimensionReduction", "filters": 256, "stride": 2},
            {"name": "dr_block4", "type": "DimensionReduction", "filters": 512, "stride": 2},
            {"name": "global_pool", "type": "GlobalAvgPool2D"},
            {"name": "dense1", "type": "Dense", "units": 512},
            {"name": "dense2", "type": "Dense", "units": 256},
            {"name": "output", "type": "Dense", "units": 4, "activation": "softmax"}
        ]
    }


if __name__ == "__main__":
    # Test LEAD-CNN model
    print("Testing LEAD-CNN model...")
    
    # Create model
    model = create_lead_cnn()
    
    # Print model info
    print(f"Model created: {model.name}")
    print(f"Input shape: {model.input_shape_}")
    print(f"Number of classes: {model.num_classes}")
    
    # Count parameters
    total_params = model.model.count_params()
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    test_input = tf.random.normal((1, 224, 224, 3))
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    
    print("LEAD-CNN model test completed!")
