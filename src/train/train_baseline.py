"""
Train LEAD-CNN baseline model

This script trains the LEAD-CNN model on the brain tumor dataset
with proper data splits and evaluation metrics.
"""
import os
import sys
import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lead_cnn import create_lead_cnn
from models.lightnet import build_lightnet, build_lightnet_v2
from models.blocks import DimensionReductionBlock, LiteDRBlock, SqueezeExcitation
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
from eval.metrics import ClassificationMetrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train LEAD-CNN baseline model")
    parser.add_argument("--config", type=str, default="experiments/baseline_leadcnn.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                       help="Path to dataset directory")
    parser.add_argument("--splits_file", type=str, default="data/splits.json",
                       help="Path to data splits file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for models and logs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--sanity_steps", type=int, default=0,
                       help="Limit each dataset split to this many batches for quick sanity checks (0 disables)")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create model (LEAD-CNN or LightNet) based on configuration."""
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "lead_cnn").lower()
    input_shape = tuple(model_cfg.get("input_shape", (224, 224, 3)))
    num_classes = model_cfg.get("num_classes", 4)
    dropout_rate = model_cfg.get("dropout_rate", 0.5)
    
    if model_type == "lead_cnn":
        model = create_lead_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )
    elif model_type == "lightnet":
        model = build_lightnet(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_se=model_cfg.get("use_se", True),
            channel_multiplier=model_cfg.get("channel_multiplier", 0.5),
        )
    elif model_type == "lightnet_v2":
        model = build_lightnet_v2(
            input_shape=input_shape,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    # Ensure consistent interface with .model attribute
    if not hasattr(model, "model"):
        model.model = model
    
    return model
    
    return model


def compile_model(model, config):
    """Compile model with optimizer and loss"""
    compile_cfg = config.get("compile", {})
    training_cfg = config.get("training", {})
    
    optimizer_cfg = training_cfg.get("optimizer") or compile_cfg.get("optimizer")
    
    learning_rate = compile_cfg.get("learning_rate")
    optimizer_type = "adam"
    
    if optimizer_cfg is None:
        optimizer_cfg = {"type": "adam"}
    
    if isinstance(optimizer_cfg, str):
        optimizer_type = optimizer_cfg.lower()
        optimizer_params = {}
    else:
        optimizer_type = optimizer_cfg.get("type", "adam").lower()
        optimizer_params = optimizer_cfg
    
    if learning_rate is None:
        learning_rate = optimizer_params.get("learning_rate", 1e-3)
    optimizer_params.setdefault("learning_rate", learning_rate)
    
    if optimizer_type == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=optimizer_params.get("learning_rate", 1e-3),
            beta_1=optimizer_params.get("beta_1", 0.9),
            beta_2=optimizer_params.get("beta_2", 0.999)
        )
    elif optimizer_type == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=optimizer_params.get("learning_rate", 1e-3),
            weight_decay=optimizer_params.get("weight_decay", 0.0),
            beta_1=optimizer_params.get("beta_1", 0.9),
            beta_2=optimizer_params.get("beta_2", 0.999)
        )
    elif optimizer_type == "sgd":
        optimizer = keras.optimizers.SGD(
            learning_rate=optimizer_params.get("learning_rate", 1e-3),
            momentum=optimizer_params.get("momentum", 0.9),
            nesterov=optimizer_params.get("nesterov", True)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Loss function
    loss_name = (
        compile_cfg.get("loss")
        or training_cfg.get("loss")
        or "sparse_categorical_crossentropy"
    )
    label_smoothing = training_cfg.get("label_smoothing", 0.0)
    
    if loss_name == "categorical_crossentropy":
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    elif loss_name == "sparse_categorical_crossentropy":
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        loss = loss_name
    
    # Metrics (default to sparse categorical accuracy for integer labels)
    metrics_cfg = compile_cfg.get("metrics", ["accuracy"])
    metrics = []
    for metric in metrics_cfg:
        if isinstance(metric, str):
            metric_name = metric.lower()
            if metric_name in ("accuracy", "acc", "sparse_categorical_accuracy"):
                metrics.append(keras.metrics.SparseCategoricalAccuracy(name="accuracy"))
            elif metric_name == "precision":
                metrics.append(keras.metrics.Precision(name="precision"))
            elif metric_name == "recall":
                metrics.append(keras.metrics.Recall(name="recall"))
            elif metric_name in ("auc", "roc_auc"):
                metrics.append(keras.metrics.AUC(name="auc"))
            else:
                metrics.append(metric)
        else:
            metrics.append(metric)
    
    if not metrics:
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    
    return model


def create_callbacks(config, output_dir):
    """Create training callbacks"""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = Path(output_dir) / "checkpoints" / "lead_cnn_best.h5"
    checkpoint_callback = create_checkpoint_callback(
        filepath=checkpoint_path,
        monitor=config['training']['monitor'],
        mode=config['training']['monitor_mode'],
        save_best_only=True,
        save_weights_only=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config['training'].get('early_stopping', {}).get('enabled', True):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=config['training']['monitor'],
            mode=config['training']['monitor_mode'],
            patience=config['training']['early_stopping']['patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Learning rate scheduler
    lr_cfg = config['training'].get('lr_scheduler', {})
    if lr_cfg.get('enabled', False):
        if lr_cfg.get('type') == 'reduce_on_plateau':
            lr_scheduler = keras.callbacks.ReduceLROnPlateau(
                monitor=config['training']['monitor'],
                mode=config['training']['monitor_mode'],
                factor=float(lr_cfg.get('factor', 0.5)),
                patience=int(lr_cfg.get('patience', 5)),
                min_lr=float(lr_cfg.get('min_lr', 1e-6)),
                verbose=1
            )
            callbacks.append(lr_scheduler)
    
    # TensorBoard
    if config['training'].get('tensorboard', {}).get('enabled', True):
        tensorboard_dir = Path(output_dir) / "logs" / "lead_cnn"
        tensorboard_callback = create_tensorboard_callback(tensorboard_dir)
        callbacks.append(tensorboard_callback)
    
    return callbacks


def train_model(model, train_data, val_data, config, callbacks, output_dir, class_weights=None):
    """Train the model"""
    print("Starting training...")
    print(f"Training samples: {len(train_data) * config['data']['batch_size']}")
    print(f"Validation samples: {len(val_data) * config['data']['batch_size']}")
    
    # Training parameters
    epochs = config.get("training", {}).get("epochs", 3)  # default 3 if missing
    
    # Train model
    history = model.model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    
    # Persist training curves for downstream analysis
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_csv_path = os.path.join(output_dir, "history.csv")
    Path(history_csv_path).parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Saved training history to {history_csv_path}")


    return history


def save_results(model, history, config, output_dir):
    """Save training results"""
    output_path = Path(output_dir)
    
    # Save model
    model_path = output_path / "checkpoints" / "lead_cnn_final.h5"
    save_model(model, model_path)
    
    # Save history
    history_path = output_path / "logs" / "lead_cnn_history.json"
    save_history(history, history_path)
    
    # Save model summary
    summary_path = output_path / "reports" / "leadcnn_summary.txt"
    save_model_summary(model, summary_path)
    
    # Save training config
    config_path = output_path / "logs" / "lead_cnn_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Results saved to: {output_path}")


def evaluate_and_log_metrics(test_data, output_dir, splits_file):
    """
    Load the best checkpoint, run it on the test set, and log detailed metrics.
    """
    output_path = Path(output_dir)
    best_model_path = output_path / "checkpoints" / "lead_cnn_best.h5"
    
    if not best_model_path.exists():
        print(f"Best checkpoint not found at {best_model_path}, skipping detailed evaluation.")
        return
    
    print("\nRunning detailed evaluation with ClassificationMetrics...")
    custom_objects = {
        "DimensionReductionBlock": DimensionReductionBlock,
        "LiteDRBlock": LiteDRBlock,
        "SqueezeExcitation": SqueezeExcitation,
    }
    best_model = tf.keras.models.load_model(str(best_model_path), custom_objects=custom_objects)
    
    y_true = []
    y_pred_proba = []
    
    for batch_x, batch_y in test_data:
        preds = best_model.predict(batch_x, verbose=0)
        y_pred_proba.append(preds)
        y_true.append(batch_y.numpy())
    
    if not y_true:
        print("Test dataset is empty; skipping detailed metrics.")
        return
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    class_names = None
    try:
        with open(splits_file, "r") as f:
            splits = json.load(f)
            class_names = splits.get("metadata", {}).get("class_names")
    except FileNotFoundError:
        class_names = None
    
    metrics_calc = ClassificationMetrics(class_names=class_names)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba=y_pred_proba)
    metrics_calc.print_metrics(metrics)
    
    metrics_path = output_path / "test_metrics.json"
    metrics_calc.save_metrics(metrics, metrics_path)
    print(f"Detailed metrics saved to: {metrics_path}")


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
    
    # Inspect one batch to verify label shapes/dtypes
    for images, labels in train_data.take(1):
        print(f"Train batch debug -> x: {images.shape}, y: {labels.shape}, dtype: {labels.dtype}")
        break
    
    # Optional sanity-check mode to cap number of batches per split
    if args.sanity_steps > 0:
        print(f"Sanity-check mode enabled: limiting each split to {args.sanity_steps} batches.")
        train_data = train_data.take(args.sanity_steps)
        val_data = val_data.take(max(1, args.sanity_steps))
        test_data = test_data.take(max(1, args.sanity_steps))
    
    # Get class weights
    class_weights = None
    if config['training'].get('use_class_weights', False):
        class_weights = get_class_weights(args.splits_file)
    
    # Create model
    model_type = config.get("model", {}).get("type", "lead_cnn").upper()
    print(f"Creating {model_type} model...")
    model = create_model(config)
    
    # Print model info
    model.model.summary()
    
    # Compile model
    model = compile_model(model, config)
    
    # Create callbacks
    callbacks = create_callbacks(config, args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_weights(args.resume)
    
    # Train model
    history = train_model(
        model,
        train_data,
        val_data,
        config,
        callbacks,
        str(output_dir),
        class_weights,
    )
    
    # Save results
    save_results(model, history, config, args.output_dir)
    
    # Final evaluation
    print("\nEvaluating on test set...")
    test_results = model.model.evaluate(test_data, verbose=1)
    
    print("Test Results:")
    for name, value in zip(model.model.metrics_names, test_results):
        print(f"  {name}: {value:.4f}")
    
    evaluate_and_log_metrics(test_data, output_dir, args.splits_file)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
