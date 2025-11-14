"""
Quick start example for Brain Tumor Lightweight Classifier

This script demonstrates how to use the project for training and evaluation.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import tensorflow as tf
from models.lead_cnn import create_lead_cnn
from models.lightnet import create_lightnet
from data.transforms import BrainTumorTransforms
from eval.metrics import ClassificationMetrics
from utils.seed import set_seed


def main():
    """Quick start example"""
    print("Brain Tumor Lightweight Classifier - Quick Start")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # 1. Create models
    print("\n1. Creating models...")
    
    # LEAD-CNN baseline
    lead_cnn = create_lead_cnn()
    print(f"LEAD-CNN created with {lead_cnn.model.count_params():,} parameters")
    
    # LightNet
    lightnet = create_lightnet(version="v1")
    print(f"LightNet created with {lightnet.model.count_params():,} parameters")
    
    # Check parameter reduction
    reduction = (1 - lightnet.model.count_params() / lead_cnn.model.count_params()) * 100
    print(f"Parameter reduction: {reduction:.1f}%")
    
    # 2. Test data loading (with dummy data)
    print("\n2. Testing data loading...")
    
    # Create dummy data
    batch_size = 4
    image_size = (224, 224, 3)
    num_classes = 4
    
    # Dummy images and labels
    dummy_images = tf.random.normal((batch_size,) + image_size)
    dummy_labels = tf.one_hot(tf.random.uniform((batch_size,), 0, num_classes, dtype=tf.int32), num_classes)
    
    print(f"Created dummy dataset: {dummy_images.shape} images, {dummy_labels.shape} labels")
    
    # 3. Test model forward pass
    print("\n3. Testing model forward pass...")
    
    # LEAD-CNN forward pass
    lead_cnn_pred = lead_cnn(dummy_images)
    print(f"LEAD-CNN output shape: {lead_cnn_pred.shape}")
    
    # LightNet forward pass
    lightnet_pred = lightnet(dummy_images)
    print(f"LightNet output shape: {lightnet_pred.shape}")
    
    # 4. Test data transforms
    print("\n4. Testing data transforms...")
    
    # Create transforms
    transforms = BrainTumorTransforms(
        image_size=(224, 224),
        normalize=True,
        augment=True
    )
    
    # Test preprocessing with tensor helper
    from data.transforms import preprocess_tensor
    dummy = tf.random.uniform((224, 224, 3), 0, 1, dtype=tf.float32)
    processed_image, _ = preprocess_tensor(dummy, augment=True)
    print(f"Transform smoke test OK: {processed_image.shape}, {processed_image.dtype}")
    
    # 5. Test evaluation metrics
    print("\n5. Testing evaluation metrics...")
    
    # Create dummy predictions
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 3])  # One wrong prediction
    y_pred_proba = np.random.rand(4, 4)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    metrics_calc = ClassificationMetrics()
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1-score: {metrics['f1_macro']:.3f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
    
    # 6. Model comparison
    print("\n6. Model comparison...")
    
    models = {
        "LEAD-CNN": lead_cnn.model,
        "LightNet": lightnet.model
    }
    
    print("Model Parameter Comparison:")
    print("-" * 30)
    for name, model in models.items():
        params = model.count_params()
        print(f"{name:12s}: {params:8,} parameters")
    
    # 7. Architecture visualization
    print("\n7. Testing architecture visualization...")
    
    try:
        from viz.plot_arch import plot_lead_cnn_architecture, plot_lightnet_architecture
        
        # Create output directory
        output_dir = Path("outputs/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot architectures
        plot_lead_cnn_architecture(str(output_dir / "lead_cnn_architecture.png"))
        plot_lightnet_architecture(str(output_dir / "lightnet_architecture.png"))
        
        print("Architecture diagrams saved to outputs/figures/")
        
    except Exception as e:
        print(f"Architecture visualization failed: {e}")
    
    print("\n" + "=" * 50)
    print("Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Download the dataset: python src/data/download_kaggle.py")
    print("2. Prepare data splits: python src/data/prepare_splits.py")
    print("3. Train models: make train-baseline")
    print("4. Evaluate models: make evaluate")
    print("5. Generate report: make report")


if __name__ == "__main__":
    main()
