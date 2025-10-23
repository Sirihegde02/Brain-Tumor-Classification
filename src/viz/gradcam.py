"""
GradCAM visualization for brain tumor classification

Generates GradCAM heatmaps to visualize which regions of the input
images the model focuses on for making predictions.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
import random


class GradCAM:
    """
    GradCAM implementation for visualizing model attention
    """
    
    def __init__(self, model: keras.Model, layer_name: str = None):
        """
        Initialize GradCAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of the layer to visualize (last conv layer if None)
        """
        self.model = model
        
        # Find the last convolutional layer
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        self.grad_model = self._create_grad_model()
    
    def _create_grad_model(self) -> keras.Model:
        """Create model for computing gradients"""
        # Get the layer output
        layer = self.model.get_layer(self.layer_name)
        
        # Create model that outputs both predictions and layer activations
        grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.output, layer.output]
        )
        
        return grad_model
    
    def compute_gradcam(self, image: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Compute GradCAM heatmap
        
        Args:
            image: Input image (batch of 1)
            class_idx: Class index to visualize (None for predicted class)
            
        Returns:
            GradCAM heatmap
        """
        # Get predictions and layer outputs
        with tf.GradientTape() as tape:
            tape.watch(self.grad_model.input)
            predictions, conv_outputs = self.grad_model(image)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_score = predictions[0, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_score, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray,
                       alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image
            heatmap: GradCAM heatmap
            alpha: Transparency of heatmap
            
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to 3-channel
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid


def generate_gradcam_for_image(model: keras.Model, image: np.ndarray,
                              class_names: List[str], layer_name: str = None) -> dict:
    """
    Generate GradCAM visualization for a single image
    
    Args:
        model: Trained model
        image: Input image
        class_names: List of class names
        layer_name: Layer to visualize
        
    Returns:
        Dictionary with visualization results
    """
    # Create GradCAM instance
    gradcam = GradCAM(model, layer_name)
    
    # Get prediction
    pred = model.predict(image[np.newaxis, ...])
    pred_class = np.argmax(pred[0])
    pred_confidence = pred[0][pred_class]
    
    # Generate heatmap
    heatmap = gradcam.compute_gradcam(image[np.newaxis, ...], pred_class)
    
    # Overlay on image
    overlaid = gradcam.overlay_heatmap(image, heatmap)
    
    return {
        'image': image,
        'heatmap': heatmap,
        'overlaid': overlaid,
        'predicted_class': pred_class,
        'predicted_name': class_names[pred_class],
        'confidence': pred_confidence
    }


def generate_gradcam_visualizations(model: keras.Model, test_data: tf.data.Dataset,
                                  class_names: List[str], output_dir: str,
                                  num_samples_per_class: int = 2,
                                  layer_name: str = None) -> None:
    """
    Generate GradCAM visualizations for test samples
    
    Args:
        model: Trained model
        test_data: Test dataset
        class_names: List of class names
        output_dir: Output directory for visualizations
        num_samples_per_class: Number of samples per class
        layer_name: Layer to visualize
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect samples from each class
    class_samples = {i: [] for i in range(len(class_names))}
    
    for batch_x, batch_y in test_data:
        for i in range(len(batch_x)):
            if len(class_samples) >= num_samples_per_class * len(class_names):
                break
            
            # Get true class
            true_class = np.argmax(batch_y[i])
            
            if len(class_samples[true_class]) < num_samples_per_class:
                class_samples[true_class].append({
                    'image': batch_x[i],
                    'true_class': true_class
                })
        
        if len(class_samples) >= num_samples_per_class * len(class_names):
            break
    
    # Generate visualizations for each sample
    for class_idx, samples in class_samples.items():
        for sample_idx, sample in enumerate(samples):
            # Generate GradCAM
            result = generate_gradcam_for_image(
                model, sample['image'], class_names, layer_name
            )
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(result['image'])
            axes[0].set_title(f"Original\nTrue: {class_names[result['predicted_class']]}")
            axes[0].axis('off')
            
            # Heatmap
            im = axes[1].imshow(result['heatmap'], cmap='jet')
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Overlaid
            axes[2].imshow(result['overlaid'])
            axes[2].set_title(f"Overlaid\nPred: {result['predicted_name']} ({result['confidence']:.3f})")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            save_path = output_path / f"gradcam_{class_names[class_idx]}_{sample_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"GradCAM visualizations saved to: {output_path}")


def create_gradcam_comparison(models: dict, test_data: tf.data.Dataset,
                             class_names: List[str], output_dir: str,
                             num_samples: int = 4) -> None:
    """
    Create GradCAM comparison for multiple models
    
    Args:
        models: Dictionary of model name -> model
        test_data: Test dataset
        class_names: List of class names
        output_dir: Output directory
        num_samples: Number of samples to visualize
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect random samples
    samples = []
    for batch_x, batch_y in test_data:
        for i in range(len(batch_x)):
            if len(samples) >= num_samples:
                break
            samples.append({
                'image': batch_x[i],
                'true_class': np.argmax(batch_y[i])
            })
        if len(samples) >= num_samples:
            break
    
    # Generate comparison for each sample
    for sample_idx, sample in enumerate(samples):
        fig, axes = plt.subplots(2, len(models) + 1, figsize=(5 * (len(models) + 1), 10))
        
        # Original image
        axes[0, 0].imshow(sample['image'])
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(sample['image'])
        axes[1, 0].set_title(f"True: {class_names[sample['true_class']]}")
        axes[1, 0].axis('off')
        
        # Model predictions
        for model_idx, (model_name, model) in enumerate(models.items()):
            result = generate_gradcam_for_image(model, sample['image'], class_names)
            
            # Heatmap
            axes[0, model_idx + 1].imshow(result['heatmap'], cmap='jet')
            axes[0, model_idx + 1].set_title(f"{model_name}\nHeatmap")
            axes[0, model_idx + 1].axis('off')
            
            # Overlaid
            axes[1, model_idx + 1].imshow(result['overlaid'])
            axes[1, model_idx + 1].set_title(f"Pred: {result['predicted_name']}\n({result['confidence']:.3f})")
            axes[1, model_idx + 1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        save_path = output_path / f"gradcam_comparison_{sample_idx}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"GradCAM comparison saved to: {output_path}")


if __name__ == "__main__":
    # Test GradCAM implementation
    print("Testing GradCAM implementation...")
    
    # Create dummy model
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(4, activation='softmax')
    ])
    
    # Test GradCAM
    gradcam = GradCAM(model)
    print(f"GradCAM initialized with layer: {gradcam.layer_name}")
    
    print("GradCAM test completed!")
