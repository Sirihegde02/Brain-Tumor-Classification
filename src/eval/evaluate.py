"""
Comprehensive model evaluation script

Evaluates trained models on test data and generates detailed reports
including metrics, confusion matrices, and GradCAM visualizations.
"""
import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lead_cnn import create_lead_cnn
from models.lightnet import create_lightnet, build_lightnet_v2
from data.transforms import create_data_generators
from eval.metrics import ClassificationMetrics, evaluate_model, compare_models, create_comparison_table
from eval.confusion import plot_confusion_matrix, analyze_confusion_matrix, save_confusion_analysis, plot_multiple_confusion_matrices
from viz.gradcam import generate_gradcam_visualizations
from eval.utils_eval import evaluate_student_on_test


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", type=str, default="experiments/baseline_leadcnn.yaml",
                       help="Path to configuration file")
    parser.add_argument("--splits_file", type=str, default="data/splits.json",
                       help="Path to data splits file")
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                       help="Paths to trained models")
    parser.add_argument("--model_names", type=str, nargs='+', required=True,
                       help="Names for the models")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for results")
    parser.add_argument("--generate_gradcam", action="store_true",
                       help="Generate GradCAM visualizations")
    parser.add_argument("--compare", action="store_true",
                       help="Generate comparison table")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def load_model(model_path: str, model_type: str = "auto", config: dict = None) -> tf.keras.Model:
    """
    Load a trained model
    
    Args:
        model_path: Path to model file
        model_type: Type of model ("lead_cnn", "lightnet", "auto")
        config: Optional config dict for model rebuilds
        
    Returns:
        Loaded Keras model
    """
    print(f"Loading model from: {model_path}")
    
    try:
        # Try to load as Keras model first
        model = keras.models.load_model(model_path)
        print("Loaded as Keras model")
        return model
    except Exception:
        # Try to load as custom model
        try:
            if model_type == "lead_cnn" or "lead" in model_path.lower():
                model = create_lead_cnn()
                model.load_weights(model_path)
                return model.model
            elif model_type == "lightnet" or "light" in model_path.lower():
                # Prefer full LightNetV2 for KD/V2 checkpoints; rebuild from config if available.
                if "kd_student_best" in os.path.basename(model_path) or "kd" in model_path.lower() or "v2" in model_path.lower():
                    cfg = config
                    if cfg is None:
                        try:
                            with open("experiments/lightnet_v2_kd_final.yaml", "r") as f:
                                cfg = yaml.safe_load(f)
                        except Exception:
                            cfg = {"model": {"input_shape": [224, 224, 3], "num_classes": 4, "dropout_rate": 0.3, "use_se": True, "channel_multiplier": 1.0}}
                    model = build_lightnet_v2(
                        input_shape=tuple(cfg["model"]["input_shape"]),
                        num_classes=cfg["model"]["num_classes"],
                        dropout_rate=cfg["model"].get("dropout_rate", 0.3),
                        use_se=cfg["model"].get("use_se", True),
                        channel_multiplier=cfg["model"].get("channel_multiplier", 1.0),
                    )
                    model.load_weights(model_path)
                    return model
                model = create_lightnet()
                model.load_weights(model_path)
                return model.model
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


def evaluate_single_model(model: tf.keras.Model, test_data: tf.data.Dataset,
                         model_name: str, output_dir: str, class_names: List[str]) -> dict:
    """
    Evaluate a single model and save results
    
    Args:
        model: Trained Keras model
        test_data: Test dataset
        model_name: Name of the model
        output_dir: Output directory
        class_names: List of class names
        
    Returns:
        Evaluation results
    """
    print(f"Evaluating {model_name}...")
    
    # Get predictions
    y_true = []
    y_pred_proba = []
    
    for batch_x, batch_y in test_data:
        y_true.extend(batch_y.numpy())
        y_pred_proba.extend(model.predict(batch_x, verbose=0))
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics_calc = ClassificationMetrics(class_names)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print metrics
    metrics_calc.print_metrics(metrics)
    
    # Save metrics
    metrics_path = Path(output_dir) / "reports" / f"{model_name}_metrics.json"
    metrics_calc.save_metrics(metrics, metrics_path)
    
    # Generate confusion matrix
    cm_path = Path(output_dir) / "figures" / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title=f"{model_name} Confusion Matrix",
        save_path=cm_path
    )
    
    # Analyze confusion matrix
    cm_analysis = analyze_confusion_matrix(y_true, y_pred, class_names)
    cm_analysis_path = Path(output_dir) / "reports" / f"{model_name}_confusion_analysis.json"
    save_confusion_analysis(cm_analysis, cm_analysis_path)
    
    # Generate GradCAM if requested
    if args.generate_gradcam:
        print(f"Generating GradCAM visualizations for {model_name}...")
        gradcam_path = Path(output_dir) / "figures" / f"{model_name}_gradcam"
        generate_gradcam_visualizations(
            model, test_data, class_names,
            output_dir=gradcam_path,
            num_samples_per_class=2
        )
    
    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def evaluate_multiple_models(model_paths: List[str], model_names: List[str],
                           test_data: tf.data.Dataset, output_dir: str,
                           class_names: List[str]) -> dict:
    """
    Evaluate multiple models and compare results
    
    Args:
        model_paths: List of model file paths
        model_names: List of model names
        test_data: Test dataset
        output_dir: Output directory
        class_names: List of class names
        
    Returns:
        Dictionary with all evaluation results
    """
    results = {}
    
    # Evaluate each model
    for model_path, model_name in zip(model_paths, model_names):
        try:
            # Load model
            model = load_model(model_path)
            
            # Evaluate model
            if model_name.lower() == "lightnetv2_kd":
                # Use shared helper for student to mirror train_kd evaluation
                metrics = evaluate_student_on_test(model, {"test": test_data}, model_name=model_name)
                results[model_name] = {"metrics": metrics}
            else:
                result = evaluate_single_model(model, test_data, model_name, output_dir, class_names)
                results[model_name] = result
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
    
    return results


def generate_comparison_report(results: dict, output_dir: str, class_names: List[str]) -> None:
    """
    Generate comprehensive comparison report
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Output directory
        class_names: List of class names
    """
    print("\nGenerating comparison report...")
    
    # Create comparison table
    metrics_only = {name: result['metrics'] for name, result in results.items()}
    comparison_table = create_comparison_table(metrics_only)
    
    print(comparison_table)
    
    # Save comparison table
    table_path = Path(output_dir) / "reports" / "comparison_table.txt"
    with open(table_path, 'w') as f:
        f.write(comparison_table)
    
    # Generate comparison confusion matrices
    if len(results) > 1:
        cm_data = {}
        for name, result in results.items():
            cm_data[name] = {
                'y_true': result['y_true'],
                'y_pred': result['y_pred']
            }
        
        cm_path = Path(output_dir) / "figures" / "comparison_confusion_matrices.png"
        plot_multiple_confusion_matrices(cm_data, class_names, save_path=cm_path)
    
    # Save detailed comparison results
    comparison_results = {
        'models': list(results.keys()),
        'metrics': metrics_only,
        'class_names': class_names,
        'timestamp': str(np.datetime64('now'))
    }
    
    comparison_path = Path(output_dir) / "reports" / "comparison_results.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"Comparison report saved to: {output_dir}")


def main():
    """Main evaluation function"""
    global args
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data generators
    print("Creating data generators...")
    datasets = create_data_generators(
        splits_file=args.splits_file,
        batch_size=config['data']['batch_size'],
        image_size=tuple(config['data']['image_size']),
        augmentation_config={}  # No augmentation for evaluation
    )
    
    test_data = datasets['test']
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    
    print(f"Test dataset size: {len(test_data) * config['data']['batch_size']} samples")
    
    # Evaluate models
    results = evaluate_multiple_models(
        args.model_paths, args.model_names,
        test_data, args.output_dir, class_names
    )
    
    if not results:
        print("No models were successfully evaluated!")
        return
    
    # Generate comparison report if requested
    if args.compare and len(results) > 1:
        generate_comparison_report(results, args.output_dir, class_names)
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
