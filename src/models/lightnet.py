"""
LightNet - parameter-efficient variant of LEAD-CNN for brain tumor classification.

This module implements a lightweight network with ≤10% of LEAD-CNN's parameters
using depthwise-separable convolutions, squeeze-and-excitation, and efficient blocks.
"""
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional, Dict, Any
from .blocks import LiteDRBlock, SqueezeExcitation


class LightNet(keras.Model):
    """
    LightNet - Lightweight CNN for brain tumor classification
    
    Features:
    - Depthwise-separable convolutions
    - Squeeze-and-excitation blocks
    - Efficient dimension reduction
    - Target: ≤113k parameters (≤10% of LEAD-CNN)
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 4,
                 dropout_rate: float = 0.3,
                 use_se: bool = True,
                 channel_multiplier: float = 0.5,
                 name: str = "LightNet",
                 **kwargs):
        """
        Initialize LightNet model
        
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
            use_se: Whether to use squeeze-and-excitation
            channel_multiplier: Channel width multiplier for scaling
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_se = use_se
        self.channel_multiplier = channel_multiplier
        
        # Calculate channel widths
        base_channels = [32, 64, 128, 256]
        self.channels = [int(c * channel_multiplier) for c in base_channels]
        
        # Input layer
        self.input_layer = keras.layers.Input(shape=input_shape, name="input")
        
        # Initial stem
        self.stem_conv = keras.layers.Conv2D(
            filters=self.channels[0],
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            name="stem_conv"
        )
        self.stem_bn = keras.layers.BatchNormalization(name="stem_bn")
        self.stem_activation = keras.layers.LeakyReLU(alpha=0.1, name="stem_activation")
        
        # Lightweight dimension reduction blocks
        self.lite_dr1 = LiteDRBlock(
            filters=self.channels[1],
            kernel_size=(3, 3),
            strides=(1, 1),
            dropout_rate=0.1,
            use_se=use_se,
            name="lite_dr1"
        )
        
        self.lite_dr2 = LiteDRBlock(
            filters=self.channels[2],
            kernel_size=(3, 3),
            strides=(2, 2),
            dropout_rate=0.2,
            use_se=use_se,
            name="lite_dr2"
        )
        
        self.lite_dr3 = LiteDRBlock(
            filters=self.channels[3],
            kernel_size=(3, 3),
            strides=(2, 2),
            dropout_rate=0.2,
            use_se=use_se,
            name="lite_dr3"
        )
        
        # Additional lightweight block
        self.lite_dr4 = LiteDRBlock(
            filters=self.channels[3],
            kernel_size=(3, 3),
            strides=(2, 2),
            dropout_rate=0.3,
            use_se=use_se,
            name="lite_dr4"
        )
        
        # Global average pooling
        self.global_pool = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        
        # Classification head (lightweight)
        self.dropout = keras.layers.Dropout(dropout_rate, name="dropout")
        
        self.dense1 = keras.layers.Dense(
            units=128,
            activation='leaky_relu',
            name="dense1"
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
        
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_activation(x)
        
        # Lightweight dimension reduction blocks
        x = self.lite_dr1(x)
        x = self.lite_dr2(x)
        x = self.lite_dr3(x)
        x = self.lite_dr4(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification head
        x = self.dropout(x)
        x = self.dense1(x)
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
            'dropout_rate': self.dropout_rate,
            'use_se': self.use_se,
            'channel_multiplier': self.channel_multiplier
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


def build_lightnet(input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 4,
                   dropout_rate: float = 0.3,
                   use_se: bool = True,
                   channel_multiplier: float = 0.5,
                   **kwargs) -> LightNet:
    """
    Build a LightNet model with reduced channel counts and efficient blocks.
    
    Args:
        input_shape: Input tensor shape.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate applied before the classifier.
        use_se: Whether to enable squeeze-and-excitation.
        channel_multiplier: Width multiplier controlling parameter budget.
    
    Returns:
        Configured LightNet instance.
    """
    return LightNet(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_se=use_se,
        channel_multiplier=channel_multiplier,
        **kwargs,
    )


def build_lightnet_v2(input_shape: Tuple[int, int, int] = (224, 224, 3),
                      num_classes: int = 4,
                      dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Build LightNetV2 - a slightly larger yet still efficient network.
    
    Args:
        input_shape: Input tensor shape.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate before the dense classifier.
    
    Returns:
        Configured Keras Model instance named LightNetV2.
    """
    inputs = keras.layers.Input(shape=input_shape, name="input")
    
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.LeakyReLU(alpha=0.1, name="stem_activation")(x)
    
    channels = [32, 64, 128, 192]
    strides = [(1, 1), (2, 2), (2, 2), (2, 2)]
    dropouts = [0.1, 0.15, 0.2, 0.25]
    
    for idx, (filters, stride, dr) in enumerate(zip(channels, strides, dropouts), start=1):
        x = LiteDRBlock(
            filters=filters,
            kernel_size=(3, 3),
            strides=stride,
            dropout_rate=dr,
            use_se=True,
            name=f"lite_dr2_{idx}",
        )(x)
    
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = keras.layers.Dense(256, activation="relu", name="dense1")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    return keras.Model(inputs=inputs, outputs=outputs, name="LightNetV2")

class LightNetV2(keras.Model):
    """
    LightNet V2 - Even more lightweight version
    
    Features:
    - More aggressive parameter reduction
    - MobileNet-style blocks
    - Optimized for edge deployment
    """
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 4,
                 dropout_rate: float = 0.2,
                 name: str = "LightNetV2",
                 **kwargs):
        """
        Initialize LightNet V2 model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Input layer
        self.input_layer = keras.layers.Input(shape=input_shape, name="input")
        
        # Very lightweight stem
        self.stem = keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            name="stem"
        )
        self.stem_bn = keras.layers.BatchNormalization(name="stem_bn")
        self.stem_activation = keras.layers.ReLU(name="stem_activation")
        
        # MobileNet-style blocks
        self.mobile_block1 = self._make_mobile_block(24, 1, name="mobile1")
        self.mobile_block2 = self._make_mobile_block(32, 2, name="mobile2")
        self.mobile_block3 = self._make_mobile_block(48, 2, name="mobile3")
        self.mobile_block4 = self._make_mobile_block(64, 2, name="mobile4")
        
        # Global pooling
        self.global_pool = keras.layers.GlobalAveragePooling2D(name="global_pool")
        
        # Minimal classification head
        self.dropout = keras.layers.Dropout(dropout_rate, name="dropout")
        self.classifier = keras.layers.Dense(num_classes, activation='softmax', name="classifier")
        
        # Build model
        self._build_model()
    
    def _make_mobile_block(self, filters, strides, name):
        """Create MobileNet-style block"""
        return keras.Sequential([
            keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=strides,
                padding='same',
                use_bias=False,
                name=f"{name}_depthwise"
            ),
            keras.layers.BatchNormalization(name=f"{name}_bn1"),
            keras.layers.ReLU(name=f"{name}_relu1"),
            keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                use_bias=False,
                name=f"{name}_pointwise"
            ),
            keras.layers.BatchNormalization(name=f"{name}_bn2"),
            keras.layers.ReLU(name=f"{name}_relu2")
        ], name=name)
    
    def _build_model(self):
        """Build the model architecture"""
        x = self.input_layer
        
        # Stem
        x = self.stem(x)
        x = self.stem_bn(x)
        x = self.stem_activation(x)
        
        # Mobile blocks
        x = self.mobile_block1(x)
        x = self.mobile_block2(x)
        x = self.mobile_block3(x)
        x = self.mobile_block4(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        
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


def create_lightnet(input_shape: Tuple[int, int, int] = (224, 224, 3),
                   num_classes: int = 4,
                   version: str = "v1",
                   **kwargs) -> keras.Model:
    """
    Create LightNet model
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
        version: Model version ("v1" or "v2")
        **kwargs: Additional arguments
        
    Returns:
        LightNet model instance
    """
    if version == "v1":
        return LightNet(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )
    elif version == "v2":
        return LightNetV2(
            input_shape=input_shape,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown version: {version}")


def get_lightnet_architecture(version: str = "v1") -> Dict[str, Any]:
    """
    Get LightNet architecture description
    
    Args:
        version: Model version
        
    Returns:
        Dictionary with architecture details
    """
    if version == "v1":
        return {
            "name": "LightNet V1",
            "description": "Lightweight CNN with depthwise-separable convolutions and SE blocks",
            "input_shape": (224, 224, 3),
            "num_classes": 4,
            "key_features": [
                "Depthwise-separable convolutions",
                "Squeeze-and-excitation blocks",
                "Efficient dimension reduction",
                "Target: ≤113k parameters"
            ],
            "blocks": [
                {"name": "stem", "type": "Conv2D", "filters": 32, "kernel": 3, "stride": 2},
                {"name": "lite_dr1", "type": "LiteDRBlock", "filters": 64},
                {"name": "lite_dr2", "type": "LiteDRBlock", "filters": 128, "stride": 2},
                {"name": "lite_dr3", "type": "LiteDRBlock", "filters": 256, "stride": 2},
                {"name": "lite_dr4", "type": "LiteDRBlock", "filters": 256, "stride": 2},
                {"name": "global_pool", "type": "GlobalAvgPool2D"},
                {"name": "dense1", "type": "Dense", "units": 128},
                {"name": "output", "type": "Dense", "units": 4, "activation": "softmax"}
            ]
        }
    else:
        return {
            "name": "LightNet V2",
            "description": "Ultra-lightweight CNN with MobileNet-style blocks",
            "input_shape": (224, 224, 3),
            "num_classes": 4,
            "key_features": [
                "MobileNet-style depthwise-separable convolutions",
                "Minimal parameter count",
                "Optimized for edge deployment"
            ]
        }


if __name__ == "__main__":
    # Test LightNet models
    print("Testing LightNet models...")
    
    # Test LightNet V1
    model_v1 = create_lightnet(version="v1")
    print(f"LightNet V1 created: {model_v1.name}")
    print(f"Parameters: {model_v1.model.count_params():,}")
    
    # Test LightNet V2
    model_v2 = create_lightnet(version="v2")
    print(f"LightNet V2 created: {model_v2.name}")
    print(f"Parameters: {model_v2.model.count_params():,}")
    
    # Test forward pass
    test_input = tf.random.normal((1, 224, 224, 3))
    output_v1 = model_v1(test_input)
    output_v2 = model_v2(test_input)
    
    print(f"V1 output shape: {output_v1.shape}")
    print(f"V2 output shape: {output_v2.shape}")
    
    print("LightNet model tests completed!")
