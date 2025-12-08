"""
Train LightNet lightweight model

This script trains the LightNet model with various configurations
and ablation studies for parameter efficiency.
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

from models.lightnet import create_lightnet
from data.transforms import create_data_generators, get_class_weights
from utils.seed import set_seed
from utils.io import (
    save_model,
    save_history,
    create_checkpoint_callback,
    create_tensorboard_callback,
    save_model_summary,
)
from utils.params import count_parameters, print_model_summary, analyze_model_complexity


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train LightNet model")
    parser.add_argument("--config", type=str, default="experiments/lightnet_ablation.yaml",
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
    parser.add_argument("--version", type=str, default="v1",
                       choices=["v1", "v2"],
                       help="LightNet version")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, version="v1"):
    """Create LightNet model"""
    model_config = config['model']
    
    model = create_lightnet(
        input_shape=tuple(model_config['input_shape']),
        num_classes=model_config['num_classes'],
        version=version,
        dropout_rate=model_config.get('dropout_rate', 0.3),
        use_se=model_config.get('use_se', True),
        channel_multiplier=model_config.get('channel_multiplier', 1.0)
    )
    
    return model

def compile_model(model, config):
    """Compile model with optimizer and loss"""
    # Optimizer
    opt_cfg = config['training']['optimizer']
    opt_type = opt_cfg['type']

    if opt_type == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=opt_cfg['learning_rate'],
            beta_1=opt_cfg.get('beta_1', 0.9),
            beta_2=opt_cfg.get('beta_2', 0.999),
        )
    elif opt_type == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=opt_cfg['learning_rate'],
            weight_decay=opt_cfg.get('weight_decay', 0.01),
        )
    elif opt_type == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=opt_cfg['learning_rate'],
            momentum=opt_cfg.get('momentum', 0.9),
            nesterov=opt_cfg.get('nesterov', True),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    # LOSS: generators emit integer class indices → use sparse CE
    loss = keras.losses.SparseCategoricalCrossentropy()

    # METRICS: use sparse accuracy so it matches integer labels
    metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]

    model.model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


def create_callbacks(config, output_dir, model_name):
    callbacks = []

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.h5")

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config["training"].get("early_stopping_patience", 10),
        restore_best_weights=True,
        verbose=1,
    )

    # Optional but nice: standard ReduceLROnPlateau instead of fancy cosine scheduler
    lr_plateau_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    callbacks.extend([checkpoint_cb, early_stopping_cb, lr_plateau_cb])
    return callbacks


def train_model(model, train_data, val_data, config, callbacks, class_weights=None):
    """Train the model"""
    print("Starting training...")
    print(f"Training samples: {len(train_data) * config['data']['batch_size']}")
    print(f"Validation samples: {len(val_data) * config['data']['batch_size']}")
    
    # Training parameters
    epochs = config['training']['epochs']
    batch_size = config['data']['batch_size']
    
    # Train model
    history = model.model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


def run_ablation_study(config, datasets, output_dir):
    """Run ablation study with different configurations"""
    ablation_results = {}
    
    # Base configuration
    base_config = config.copy()
    
    # Ablation parameters
    ablation_params = {
        'use_se': [True, False],
        'channel_multiplier': [0.5, 0.75, 1.0, 1.25],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4]
    }
    
    for param_name, param_values in ablation_params.items():
        print(f"\nRunning ablation study for {param_name}...")
        
        param_results = {}
        
        for value in param_values:
            print(f"  Testing {param_name}={value}")
            
            # Update config
            test_config = base_config.copy()
            test_config['model'][param_name] = value
            
            # Create model
            model = create_model(test_config, version="v1")
            
            # Analyze complexity
            complexity = analyze_model_complexity(model.model)
            
            # Store results
            param_results[value] = {
                'parameters': complexity['parameters']['total'],
                'model_size_mb': complexity['model_size_mb'],
                'estimated_flops': complexity['estimated_flops']
            }
            
            print(f"    Parameters: {complexity['parameters']['total']:,}")
            print(f"    Model size: {complexity['model_size_mb']:.2f} MB")
        
        ablation_results[param_name] = param_results
    
    # Save ablation results
    ablation_path = Path(output_dir) / "reports" / "ablation_study.json"
    with open(ablation_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"Ablation study results saved to: {ablation_path}")
    
    return ablation_results


def save_results(model, history, config, output_dir, model_name):
    """Save training results"""
    output_path = Path(output_dir)
    
    # Save model
    model_path = output_path / "checkpoints" / f"{model_name}_final.h5"
    save_model(model, model_path)
    
    # Save history
    history_path = output_path / "logs" / f"{model_name}_history.json"
    save_history(history, history_path)
    
    # Save model summary
    summary_path = output_path / "reports" / f"{model_name}_summary.txt"
    save_model_summary(model, summary_path)
    
    # Save training config
    config_path = output_path / "logs" / f"{model_name}_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Results saved to: {output_path}")


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
    
    # Get class weights
    class_weights = None
    if config['training'].get('use_class_weights', False):
        class_weights = get_class_weights(args.splits_file)
    
    # Run ablation study if enabled
    if config.get('ablation_study', {}).get('enabled', False):
        print("Running ablation study...")
        ablation_results = run_ablation_study(config, datasets, args.output_dir)
    
    # Create model
    print(f"Creating LightNet {args.version} model...")
    model = create_model(config, version=args.version)
    
    # Print model info
    print_model_summary(model.model, show_layers=False, show_parameters=True)
    
    # Check parameter count
    params = count_parameters(model.model)
    print(f"Parameter count: {params['total']:,}")
    
    if params['total'] > 113000:  # Target: ≤113k parameters
        print(f"WARNING: Model has {params['total']:,} parameters, exceeding target of 113k")
    
    # Compile model
    model = compile_model(model, config)
    
    # Create callbacks
    model_name = f"lightnet_{args.version}"
    callbacks = create_callbacks(config, args.output_dir, model_name)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_weights(args.resume)
    
    # Train model
    history = train_model(model, train_data, val_data, config, callbacks, class_weights)
    
    # Save results
    save_results(model, history, config, args.output_dir, model_name)
    
    # Final evaluation
    print("\nEvaluating on test set...")
    test_results = model.model.evaluate(test_data, verbose=1)

    print("Test Results:")
    loss, acc = test_results  # [loss, accuracy]
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
