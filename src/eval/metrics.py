"""
Comprehensive evaluation metrics for brain tumor classification

Includes Cohen's kappa, precision, recall, F1, and other metrics
for model evaluation and comparison.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path


class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or ["glioma", "meningioma", "pituitary", "no_tumor"]
        self.num_classes = len(self.class_names)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels (one-hot encoded or integer)
            y_pred: Predicted labels (one-hot encoded or integer)
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        # Convert to integer labels if needed
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Cohen's kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            except:
                roc_auc = None
        
        # Compile results
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision),
            'recall_macro': float(recall),
            'f1_macro': float(f1),
            'cohen_kappa': float(kappa),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'per_class': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist()
            }
        }
        
        return metrics
    
    def calculate_detailed_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate detailed metrics including ROC curves
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with detailed metrics
        """
        # Convert to integer labels
        if y_true.ndim > 1:
            y_true_int = np.argmax(y_true, axis=1)
        else:
            y_true_int = y_true
        
        # ROC curves for each class
        roc_curves = {}
        for i, class_name in enumerate(self.class_names):
            try:
                fpr, tpr, _ = roc_curve(y_true_int == i, y_pred_proba[:, i])
                roc_auc = roc_auc_score(y_true_int == i, y_pred_proba[:, i])
                roc_curves[class_name] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(roc_auc)
                }
            except:
                roc_curves[class_name] = None
        
        # Precision-Recall curves
        pr_curves = {}
        for i, class_name in enumerate(self.class_names):
            try:
                precision, recall, _ = precision_recall_curve(y_true_int == i, y_pred_proba[:, i])
                pr_curves[class_name] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
            except:
                pr_curves[class_name] = None
        
        return {
            'roc_curves': roc_curves,
            'pr_curves': pr_curves
        }
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Print formatted metrics
        
        Args:
            metrics: Metrics dictionary
        """
        print("=" * 60)
        print("CLASSIFICATION METRICS")
        print("=" * 60)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC (macro): {metrics['roc_auc']:.4f}")
        
        print("\nPer-class metrics:")
        print("-" * 40)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:12s}: "
                  f"P={metrics['per_class']['precision'][i]:.3f}, "
                  f"R={metrics['per_class']['recall'][i]:.3f}, "
                  f"F1={metrics['per_class']['f1'][i]:.3f}")
        
        print("\nConfusion Matrix:")
        print("-" * 40)
        cm = np.array(metrics['confusion_matrix'])
        print("Predicted ->")
        print("Actual")
        print("     ", end="")
        for name in self.class_names:
            print(f"{name:8s}", end="")
        print()
        
        for i, name in enumerate(self.class_names):
            print(f"{name:8s}", end="")
            for j in range(len(self.class_names)):
                print(f"{cm[i, j]:8d}", end="")
            print()
    
    def save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """
        Save metrics to JSON file
        
        Args:
            metrics: Metrics dictionary
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to: {filepath}")


def evaluate_model(model: tf.keras.Model, test_data: tf.data.Dataset, 
                  class_names: List[str] = None) -> Dict[str, Any]:
    """
    Evaluate a model on test data
    
    Args:
        model: Trained Keras model
        test_data: Test dataset
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
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
    
    return metrics


def compare_models(models: Dict[str, tf.keras.Model], test_data: tf.data.Dataset,
                  class_names: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models
    
    Args:
        models: Dictionary of model name -> model
        test_data: Test dataset
        class_names: List of class names
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model, test_data, class_names)
        results[name] = metrics
    
    return results


def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Create a comparison table for model results
    
    Args:
        results: Dictionary with model results
        
    Returns:
        Formatted comparison table
    """
    table = "\n" + "=" * 80 + "\n"
    table += "MODEL COMPARISON\n"
    table += "=" * 80 + "\n"
    
    # Header
    table += f"{'Model':<20s} {'Acc':<8s} {'Prec':<8s} {'Rec':<8s} {'F1':<8s} {'Kappa':<8s} {'AUC':<8s}\n"
    table += "-" * 80 + "\n"
    
    # Results
    for name, metrics in results.items():
        table += f"{name:<20s} "
        table += f"{metrics['accuracy']:<8.3f} "
        table += f"{metrics['precision_macro']:<8.3f} "
        table += f"{metrics['recall_macro']:<8.3f} "
        table += f"{metrics['f1_macro']:<8.3f} "
        table += f"{metrics['cohen_kappa']:<8.3f} "
        if metrics['roc_auc'] is not None:
            table += f"{metrics['roc_auc']:<8.3f}\n"
        else:
            table += f"{'N/A':<8s}\n"
    
    table += "=" * 80 + "\n"
    
    return table


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing classification metrics...")
    
    # Create dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 2])
    y_pred_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.6, 0.3],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    
    # Calculate metrics
    metrics_calc = ClassificationMetrics()
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print results
    metrics_calc.print_metrics(metrics)
    
    print("Metrics calculation test completed!")
