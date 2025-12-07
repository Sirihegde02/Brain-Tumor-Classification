"""
Train LightNet with Knowledge Distillation

This script trains the LightNet model using knowledge distillation
from a pre-trained teacher model (LEAD-CNN or DenseNet121).
"""
import os
import sys
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lightnet import create_lightnet, build_lightnet_v2
from models.lead_cnn import create_lead_cnn
from models.kd_losses import DistillationModel, create_distillation_loss
from data.transforms import create_data_generators, get_class_weights
from utils.seed import set_seed
from utils.io import (
    save_model,
    save_history,
    save_model_summary,
    create_checkpoint_callback,
    create_tensorboard_callback,
)
from utils.params import count_parameters, print_model_summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train LightNet with Knowledge Distillation")
    parser.add_argument("--config", type=str, default="experiments/kd.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                       help="Path to dataset directory")
    parser.add_argument("--splits_file", type=str, default="data/splits.json",
                       help="Path to data splits file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for models and logs")
    parser.add_argument("--teacher_path", type=str, required=True,
                       help="Path to pre-trained teacher model")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_teacher_model(teacher_path, config):
    """Load pre-trained teacher model"""
    print(f"Loading teacher model from: {teacher_path}")
    
    # Try to load as LEAD-CNN first
    try:
        teacher = keras.models.load_model(teacher_path)
        print("Loaded teacher model as Keras model")
    except:
        # Try to load as custom LEAD-CNN
        try:
            teacher = create_lead_cnn(
                input_shape=tuple(config['model']['input_shape']),
                num_classes=config['model']['num_classes']
            )
            teacher.load_weights(teacher_path)
            teacher = teacher.model
            print("Loaded teacher model as LEAD-CNN")
        except Exception as e:
            print(f"Error loading teacher model: {e}")
            raise
    
    # Freeze teacher model
    teacher.trainable = False
    
    return teacher


def create_student_model(config):
    """Create student model (LightNet)"""
    model_config = config['model']
    
    # For KD final config, use the full LightNetV2 builder (≈120k params) with config-driven knobs.
    if model_config.get('version', 'v2') == 'v2':
        student = build_lightnet_v2(
            input_shape=tuple(model_config['input_shape']),
            num_classes=model_config['num_classes'],
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_se=model_config.get('use_se', True),
            channel_multiplier=model_config.get('channel_multiplier', 1.0),
        )
    else:
        student = create_lightnet(
            input_shape=tuple(model_config['input_shape']),
            num_classes=model_config['num_classes'],
            version=model_config.get('version', 'v1'),
            dropout_rate=model_config.get('dropout_rate', 0.3),
            use_se=model_config.get('use_se', True),
            channel_multiplier=model_config.get('channel_multiplier', 1.0)
        )
    
    return student


def create_distillation_model(teacher, student, config):
    """Create distillation model"""
    # Get feature layers for distillation
    feature_layers = config['distillation'].get('feature_layers', [])
    
    # Use underlying Keras Models if wrappers are provided
    teacher_model = getattr(teacher, "model", teacher)
    student_model = getattr(student, "model", student)
    
    # Create distillation model
    distillation_model = DistillationModel(
        teacher_model=teacher_model,
        student_model=student_model,
        feature_layers=feature_layers,
        name="distillation_model"
    )
    
    return distillation_model


def create_distillation_loss_fn(config):
    """Create distillation loss function"""
    distillation_config = config['distillation']
    
    loss_fn = create_distillation_loss(
        temperature=distillation_config.get('temperature', 3.0),
        alpha=distillation_config.get('alpha', 0.5),
        beta=distillation_config.get('beta', 0.3),
        gamma=distillation_config.get('gamma', 0.2),
        feature_layers=distillation_config.get('feature_layers', [])
    )
    
    return loss_fn


def compile_distillation_model(distillation_model, config):
    """Compile distillation model"""
    # Create loss function
    loss_fn = create_distillation_loss_fn(config)
    
    # Optimizer
    if config['training']['optimizer']['type'] == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=config['training']['optimizer']['learning_rate'],
            beta_1=config['training']['optimizer'].get('beta_1', 0.9),
            beta_2=config['training']['optimizer'].get('beta_2', 0.999)
        )
    elif config['training']['optimizer']['type'] == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=config['training']['optimizer']['learning_rate'],
            weight_decay=config['training']['optimizer'].get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']['type']}")
    
    # Metrics
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2'),
    ]
    
    # Compile model
    distillation_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=metrics
    )
    
    return distillation_model


def create_callbacks(config, output_dir):
    """Create training callbacks"""
    callbacks = []
    
    # Resolve monitor/mode from training or early_stopping sections
    monitor = config['training'].get('monitor') or config['training'].get('early_stopping', {}).get('monitor', 'val_accuracy')
    monitor_mode = config['training'].get('monitor_mode') or config['training'].get('early_stopping', {}).get('mode', 'max')
    
    # Model checkpoint
    checkpoint_path = Path(output_dir) / "checkpoints" / "lightnet_kd_best.h5"
    checkpoint_callback = create_checkpoint_callback(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=monitor_mode,
        save_best_only=True,
        save_weights_only=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config['training'].get('early_stopping', {}).get('enabled', True):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=monitor_mode,
            patience=config['training']['early_stopping']['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Learning rate scheduler
    if config['training'].get('lr_scheduler', {}).get('enabled', False):
        if config['training']['lr_scheduler']['type'] == 'reduce_on_plateau':
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=monitor_mode,
                factor=config['training']['lr_scheduler']['factor'],
                patience=config['training']['lr_scheduler']['patience'],
                min_lr=config['training']['lr_scheduler']['min_lr'],
                verbose=1
            )
            callbacks.append(lr_scheduler)
    
    # TensorBoard
    if config['training'].get('tensorboard', {}).get('enabled', True):
        tensorboard_dir = Path(output_dir) / "logs" / "lightnet_kd"
        tensorboard_callback = create_tensorboard_callback(tensorboard_dir)
        callbacks.append(tensorboard_callback)
    
    return callbacks


def train_distillation_model(distillation_model, train_data, val_data, config, callbacks):
    """Train the distillation model"""
    print("Starting knowledge distillation training...")
    print(f"Training samples: {len(train_data) * config['data']['batch_size']}")
    print(f"Validation samples: {len(val_data) * config['data']['batch_size']}")
    
    # Training parameters
    epochs = config['training']['epochs']
    batch_size = config['data']['batch_size']
    
    # Train model
    history = distillation_model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def save_results(distillation_model, history, config, output_dir):
    """Save training results"""
    output_path = Path(output_dir)
    
    # Save distillation model
    model_path = output_path / "checkpoints" / "lightnet_kd_final.h5"
    save_model(distillation_model, model_path)
    
    # Save student model separately
    student_path = output_path / "checkpoints" / "lightnet_kd_student.h5"
    save_model(distillation_model.student_model, student_path)
    
    # Save history
    history_path = output_path / "logs" / "lightnet_kd_history.json"
    save_history(history, history_path)
    
    # Save model summary
    summary_path = output_path / "reports" / "lightnet_kd_summary.txt"
    save_model_summary(distillation_model, summary_path)
    
    # Save training config
    config_path = output_path / "logs" / "lightnet_kd_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Results saved to: {output_path}")


def compare_models(teacher, student, test_data):
    """Compare teacher and student performance"""
    print("\nComparing teacher and student models...")

    # Ensure models are compiled for evaluation
    eval_metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
    ]
    teacher.compile(loss="sparse_categorical_crossentropy", metrics=eval_metrics)
    student.compile(loss="sparse_categorical_crossentropy", metrics=eval_metrics)
    
    # Evaluate teacher
    teacher_results = teacher.evaluate(test_data, verbose=0)
    print(f"Teacher Results:")
    print(f"  Loss: {teacher_results[0]:.4f}")
    print(f"  Accuracy: {teacher_results[1]:.4f}")
    print(f"  Top-2 Accuracy: {teacher_results[2]:.4f}")
    
    # Evaluate student
    student_results = student.evaluate(test_data, verbose=0)
    print(f"Student Results:")
    print(f"  Loss: {student_results[0]:.4f}")
    print(f"  Accuracy: {student_results[1]:.4f}")
    print(f"  Top-2 Accuracy: {student_results[2]:.4f}")
    
    # Calculate performance retention
    accuracy_retention = student_results[1] / teacher_results[1] * 100
    print(f"Accuracy retention: {accuracy_retention:.1f}%")
    
    return {
        'teacher': teacher_results,
        'student': student_results,
        'accuracy_retention': accuracy_retention
    }


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data generators
    print("Creating data generators...")
    datasets = create_data_generators(
        splits_file=args.splits_file,
        batch_size=config['data']['batch_size'],
        image_size=tuple(config['data']['image_size']),
        augmentation_config=config['data'].get('augmentation', {})
    )
    
    train_data = datasets['train']
    val_data = datasets['val']
    test_data = datasets['test']
    
    # Load teacher model
    teacher = load_teacher_model(args.teacher_path, config)
    
    # Create student model
    print("Creating student model...")
    student = create_student_model(config)
    
    # Print model info
    print("\nTeacher model info:")
    teacher_params = count_parameters(teacher)
    print(f"Teacher parameters: {teacher_params['total']:,}")
    
    print("\nStudent model info:")
    student_params = count_parameters(student)
    print(f"Student parameters: {student_params['total']:,}")
    print(f"Parameter reduction: {(1 - student_params['total'] / teacher_params['total']) * 100:.1f}%")
    
    # Create distillation model
    print("Creating distillation model...")
    distillation_model = create_distillation_model(teacher, student, config)
    
    # Compile distillation model
    distillation_model = compile_distillation_model(distillation_model, config)
    
    # Create callbacks
    callbacks = create_callbacks(config, args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        distillation_model.load_weights(args.resume)
    
    # Train distillation model
    history = train_distillation_model(distillation_model, train_data, val_data, config, callbacks)
    
    # Save results
    save_results(distillation_model, history, config, args.output_dir)
    
    # Compare models
    comparison_results = compare_models(teacher, student, test_data)
    
    # Save comparison results
    comparison_path = Path(args.output_dir) / "reports" / "kd_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print("Knowledge distillation training completed successfully!")


if __name__ == "__main__":
    main()
