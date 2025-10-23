"""
Custom blocks for LEAD-CNN and LightNet models
"""
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional


class DimensionReductionBlock(keras.layers.Layer):
    """
    LEAD-CNN style dimension reduction block
    
    This block reduces spatial dimensions while maintaining feature richness
    through a combination of convolutions and pooling operations.
    """
    
    def __init__(self, 
                 filters: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (2, 2),
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True,
                 activation: str = "leaky_relu",
                 name: str = None,
                 **kwargs):
        """
        Initialize dimension reduction block
        
        Args:
            filters: Number of output filters
            kernel_size: Convolution kernel size
            strides: Stride for convolution and pooling
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            activation: Activation function
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # Convolution layer
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=not use_batch_norm,
            name=f"{name}_conv" if name else None
        )
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = keras.layers.BatchNormalization(
                name=f"{name}_bn" if name else None
            )
        else:
            self.batch_norm = None
        
        # Activation
        if activation == "leaky_relu":
            self.activation_layer = keras.layers.LeakyReLU(alpha=0.1)
        elif activation == "relu":
            self.activation_layer = keras.layers.ReLU()
        else:
            self.activation_layer = keras.layers.Activation(activation)
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass"""
        x = self.conv(inputs)
        
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        
        x = self.activation_layer(x)
        
        if self.dropout and training:
            x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'activation': self.activation
        })
        return config


class LiteDRBlock(keras.layers.Layer):
    """
    Lightweight dimension reduction block for LightNet
    
    Uses depthwise-separable convolutions and efficient operations
    to reduce parameters while maintaining performance.
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (2, 2),
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 use_se: bool = True,
                 se_ratio: int = 4,
                 activation: str = "leaky_relu",
                 name: str = None,
                 **kwargs):
        """
        Initialize lightweight dimension reduction block
        
        Args:
            filters: Number of output filters
            kernel_size: Convolution kernel size
            strides: Stride for convolution
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_se: Whether to use squeeze-and-excitation
            se_ratio: SE compression ratio
            activation: Activation function
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_se = use_se
        self.se_ratio = se_ratio
        self.activation = activation
        
        # Depthwise separable convolution
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=not use_batch_norm,
            name=f"{name}_depthwise" if name else None
        )
        
        self.pointwise_conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=not use_batch_norm,
            name=f"{name}_pointwise" if name else None
        )
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = keras.layers.BatchNormalization(
                name=f"{name}_bn" if name else None
            )
        else:
            self.batch_norm = None
        
        # Activation
        if activation == "leaky_relu":
            self.activation_layer = keras.layers.LeakyReLU(alpha=0.1)
        elif activation == "relu":
            self.activation_layer = keras.layers.ReLU()
        else:
            self.activation_layer = keras.layers.Activation(activation)
        
        # Squeeze-and-Excitation
        if use_se:
            self.se = SqueezeExcitation(filters, se_ratio, name=f"{name}_se" if name else None)
        else:
            self.se = None
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass"""
        x = self.depthwise_conv(inputs)
        x = self.pointwise_conv(x)
        
        if self.batch_norm:
            x = self.batch_norm(x, training=training)
        
        x = self.activation_layer(x)
        
        if self.se:
            x = self.se(x)
        
        if self.dropout and training:
            x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'use_se': self.use_se,
            'se_ratio': self.se_ratio,
            'activation': self.activation
        })
        return config


class SqueezeExcitation(keras.layers.Layer):
    """
    Squeeze-and-Excitation block for channel attention
    
    Reduces spatial dimensions to global average pooling,
    then applies two fully connected layers with ReLU and Sigmoid.
    """
    
    def __init__(self, 
                 filters: int,
                 ratio: int = 4,
                 name: str = None,
                 **kwargs):
        """
        Initialize SE block
        
        Args:
            filters: Number of input/output filters
            ratio: Compression ratio for the first FC layer
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.filters = filters
        self.ratio = ratio
        self.squeeze_filters = max(1, filters // ratio)
        
        # Global average pooling
        self.global_pool = keras.layers.GlobalAveragePooling2D()
        
        # Dense layers
        self.fc1 = keras.layers.Dense(
            self.squeeze_filters,
            activation='relu',
            name=f"{name}_fc1" if name else None
        )
        
        self.fc2 = keras.layers.Dense(
            filters,
            activation='sigmoid',
            name=f"{name}_fc2" if name else None
        )
    
    def call(self, inputs):
        """Forward pass"""
        # Squeeze: Global average pooling
        x = self.global_pool(inputs)
        
        # Excitation: Two FC layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Reshape for broadcasting
        x = tf.reshape(x, [-1, 1, 1, self.filters])
        
        # Scale the input
        return inputs * x
    
    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'ratio': self.ratio
        })
        return config


class ResidualBlock(keras.layers.Layer):
    """
    Residual block with optional projection shortcut
    """
    
    def __init__(self,
                 filters: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 strides: Tuple[int, int] = (1, 1),
                 use_projection: bool = False,
                 dropout_rate: float = 0.1,
                 name: str = None,
                 **kwargs):
        """
        Initialize residual block
        
        Args:
            filters: Number of filters
            kernel_size: Convolution kernel size
            strides: Stride for convolution
            use_projection: Whether to use projection shortcut
            dropout_rate: Dropout rate
            name: Layer name
        """
        super().__init__(name=name, **kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_projection = use_projection
        self.dropout_rate = dropout_rate
        
        # Main path
        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=f"{name}_conv1" if name else None
        )
        
        self.bn1 = keras.layers.BatchNormalization(
            name=f"{name}_bn1" if name else None
        )
        
        self.activation1 = keras.layers.ReLU()
        
        self.conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            name=f"{name}_conv2" if name else None
        )
        
        self.bn2 = keras.layers.BatchNormalization(
            name=f"{name}_bn2" if name else None
        )
        
        # Projection shortcut
        if use_projection:
            self.projection = keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=strides,
                padding='same',
                name=f"{name}_projection" if name else None
            )
        else:
            self.projection = None
        
        # Final activation
        self.activation2 = keras.layers.ReLU()
        
        # Dropout
        if dropout_rate > 0:
            self.dropout = keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=None):
        """Forward pass"""
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        if self.projection:
            shortcut = self.projection(inputs)
        else:
            shortcut = inputs
        
        # Add residual connection
        x = x + shortcut
        
        x = self.activation2(x)
        
        if self.dropout and training:
            x = self.dropout(x, training=training)
        
        return x
    
    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'use_projection': self.use_projection,
            'dropout_rate': self.dropout_rate
        })
        return config


if __name__ == "__main__":
    # Test blocks
    print("Testing custom blocks...")
    
    # Test DimensionReductionBlock
    dr_block = DimensionReductionBlock(64, name="test_dr")
    print(f"DR Block created: {dr_block}")
    
    # Test LiteDRBlock
    lite_dr_block = LiteDRBlock(64, use_se=True, name="test_lite_dr")
    print(f"LiteDR Block created: {lite_dr_block}")
    
    # Test SE block
    se_block = SqueezeExcitation(64, name="test_se")
    print(f"SE Block created: {se_block}")
    
    print("All blocks created successfully!")
