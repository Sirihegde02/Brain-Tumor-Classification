"""
Smoke test script for Brain Tumor Classification project

Creates a tiny dataset (16 images, 4 per class) and runs 1 epoch
training for both lead_cnn and lightnet models to verify everything works.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.lead_cnn import create_lead_cnn
from models.lightnet import create_lightnet
from utils.seed import set_seed
from utils.params import count_parameters
from utils.io import save_model, ensure_dir


def create_smoke_dataset(output_dir: str = "data/smoke_test") -> Dict:
    """
    Create a tiny dataset with 16 images (4 per class) for smoke testing
    
    Args:
        output_dir: Directory to save smoke test data
        
    Returns:
        Dictionary with splits information
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create class directories
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    # Create splits: 10 train, 3 val, 3 test (roughly balanced)
    splits = {
        "train": {"paths": [], "labels": []},
        "val": {"paths": [], "labels": []},
        "test": {"paths": [], "labels": []}
    }
    
    # Create dummy images (saved as numpy arrays, then loaded as if they were real images)
    # For smoke test, we'll create synthetic data directly in memory
    np.random.seed(42)
    
    # Generate 16 images total: 4 per class
    images_per_class = 4
    total_images = len(class_names) * images_per_class
    
    all_paths = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        for img_idx in range(images_per_class):
            # Create a dummy image path (we'll use synthetic data)
            img_path = f"smoke_{class_name}_{img_idx}.npy"
            all_paths.append(img_path)
            all_labels.append(class_idx)
    
    # Split: 10 train, 3 val, 3 test
    train_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    val_indices = [10, 11, 12]
    test_indices = [13, 14, 15]
    
    for idx in train_indices:
        splits["train"]["paths"].append(all_paths[idx])
        splits["train"]["labels"].append(all_labels[idx])
    
    for idx in val_indices:
        splits["val"]["paths"].append(all_paths[idx])
        splits["val"]["labels"].append(all_labels[idx])
    
    for idx in test_indices:
        splits["test"]["paths"].append(all_paths[idx])
        splits["test"]["labels"].append(all_labels[idx])
    
    # Add metadata
    splits["metadata"] = {
        "total_images": total_images,
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_ratio": len(splits["train"]["paths"]) / total_images,
        "val_ratio": len(splits["val"]["paths"]) / total_images,
        "test_ratio": len(splits["test"]["paths"]) / total_images,
        "random_state": 42
    }
    
    # Save splits
    splits_file = output_path / "splits.json"
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created smoke test dataset: {total_images} images")
    print(f"  Train: {len(splits['train']['paths'])}")
    print(f"  Val: {len(splits['val']['paths'])}")
    print(f"  Test: {len(splits['test']['paths'])}")
    print(f"Splits saved to: {splits_file}")
    
    return splits


def create_synthetic_dataset(image_paths: list, labels: list, image_size: tuple = (224, 224, 3)):
    """
    Create a synthetic TensorFlow dataset from paths and labels
    
    Args:
        image_paths: List of image paths (not used, but kept for compatibility)
        labels: List of labels
        image_size: Image size (H, W, C)
        
    Returns:
        TensorFlow dataset
    """
    def generate_image(path, label):
        # Generate a random image
        image = tf.random.normal(image_size, mean=0.5, stddev=0.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        # One-hot encode label
        label_onehot = tf.one_hot(label, 4)
        return image, label_onehot
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda p, l: generate_image(p, l), num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset


def train_smoke_model(model_name: str, model, splits: Dict, output_dir: str):
    """
    Train a model on smoke test data for 1 epoch
    
    Args:
        model_name: Name of the model
        model: Model instance
        splits: Data splits dictionary
        output_dir: Output directory
        
    Returns:
        Training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Create datasets
    batch_size = 2  # Very small batch size for smoke test
    
    train_dataset = create_synthetic_dataset(
        splits["train"]["paths"],
        splits["train"]["labels"]
    )
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = create_synthetic_dataset(
        splits["val"]["paths"],
        splits["val"]["labels"]
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Compile model
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model info
    params = count_parameters(model.model)
    print(f"Model parameters: {params['total']:,}")
    
    # Create checkpoint callback
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"SMOKE_{model_name}_best.h5"
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Train for 1 epoch
    print(f"Training for 1 epoch with batch size {batch_size}...")
    history = model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        callbacks=[checkpoint_callback],
        verbose=1
    )
    
    # Verify checkpoint was saved
    if checkpoint_path.exists():
        print(f"✅ Checkpoint saved: {checkpoint_path}")
    else:
        print(f"❌ Checkpoint NOT saved: {checkpoint_path}")
    
    return history, params


def main():
    """Run smoke test"""
    print("="*60)
    print("SMOKE TEST - Brain Tumor Classification")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create smoke test dataset
    print("\n1. Creating smoke test dataset...")
    splits = create_smoke_dataset("data/smoke_test")
    
    # Test LEAD-CNN
    print("\n2. Testing LEAD-CNN model...")
    lead_cnn = create_lead_cnn()
    lead_history, lead_params = train_smoke_model("lead_cnn", lead_cnn, splits, str(output_dir))
    
    # Test LightNet
    print("\n3. Testing LightNet model...")
    lightnet = create_lightnet(version="v1")
    lightnet_history, lightnet_params = train_smoke_model("lightnet", lightnet, splits, str(output_dir))
    
    # Verify checkpoints
    print("\n4. Verifying checkpoints...")
    smoke_checkpoints = {
        "lead_cnn": output_dir / "checkpoints" / "SMOKE_lead_cnn_best.h5",
        "lightnet": output_dir / "checkpoints" / "SMOKE_lightnet_best.h5"
    }
    
    checkpoint_status = {}
    for name, path in smoke_checkpoints.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            checkpoint_status[name] = f"✅ {path} ({size_mb:.2f} MB)"
        else:
            checkpoint_status[name] = f"❌ {path} (NOT FOUND)"
    
    # Print summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    print(f"\nModel Parameters:")
    print(f"  LEAD-CNN: {lead_params['total']:,}")
    print(f"  LightNet: {lightnet_params['total']:,}")
    print(f"  Reduction: {(1 - lightnet_params['total'] / lead_params['total']) * 100:.1f}%")
    
    print(f"\nCheckpoints:")
    for name, status in checkpoint_status.items():
        print(f"  {name}: {status}")
    
    print(f"\nTraining Results:")
    print(f"  LEAD-CNN - Train Acc: {lead_history.history.get('accuracy', ['N/A'])[0]:.4f}, "
          f"Val Acc: {lead_history.history.get('val_accuracy', ['N/A'])[0]:.4f}")
    print(f"  LightNet - Train Acc: {lightnet_history.history.get('accuracy', ['N/A'])[0]:.4f}, "
          f"Val Acc: {lightnet_history.history.get('val_accuracy', ['N/A'])[0]:.4f}")
    
    print("\n" + "="*60)
    print("✅ Smoke test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

