"""
Confusion matrix visualization and analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Tuple
import json
from pathlib import Path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str] = None,
                         normalize: bool = False,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix with optional normalization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += " (Normalized)"
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    # Customize plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def plot_multiple_confusion_matrices(results: dict, class_names: List[str],
                                   figsize: Tuple[int, int] = (15, 5),
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple confusion matrices in a grid
    
    Args:
        results: Dictionary with model names and their predictions
        class_names: List of class names
        figsize: Figure size
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, data) in enumerate(results.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i], cbar_kws={'label': 'Count'})
        
        axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontsize=10)
        axes[i].set_ylabel('True Label', fontsize=10)
        
        # Rotate tick labels
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multiple confusion matrices saved to: {save_path}")
    
    return fig


def analyze_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: List[str] = None) -> dict:
    """
    Analyze confusion matrix and provide insights
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary with analysis results
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Per-class metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Most confused pairs
    confusion_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    'true_class': class_names[i] if class_names else f'Class {i}',
                    'predicted_class': class_names[j] if class_names else f'Class {j}',
                    'count': int(cm[i, j]),
                    'percentage': float(cm[i, j] / np.sum(cm[i, :]) * 100)
                })
    
    # Sort by count
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
    
    # Analysis results
    analysis = {
        'overall_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist()
        },
        'most_confused_pairs': confusion_pairs[:5],  # Top 5
        'class_distribution': {
            'true': np.sum(cm, axis=1).tolist(),
            'predicted': np.sum(cm, axis=0).tolist()
        }
    }
    
    return analysis


def save_confusion_analysis(analysis: dict, filepath: str) -> None:
    """
    Save confusion matrix analysis to JSON file
    
    Args:
        analysis: Analysis results
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Confusion analysis saved to: {filepath}")


def print_confusion_analysis(analysis: dict, class_names: List[str] = None) -> None:
    """
    Print confusion matrix analysis
    
    Args:
        analysis: Analysis results
        class_names: List of class names
    """
    print("=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    print(f"Overall Accuracy: {analysis['overall_accuracy']:.4f}")
    
    print("\nPer-class Metrics:")
    print("-" * 40)
    if class_names:
        for i, name in enumerate(class_names):
            print(f"{name:12s}: "
                  f"Precision={analysis['per_class_metrics']['precision'][i]:.3f}, "
                  f"Recall={analysis['per_class_metrics']['recall'][i]:.3f}, "
                  f"F1={analysis['per_class_metrics']['f1'][i]:.3f}")
    else:
        for i in range(len(analysis['per_class_metrics']['precision'])):
            print(f"Class {i:2d}: "
                  f"Precision={analysis['per_class_metrics']['precision'][i]:.3f}, "
                  f"Recall={analysis['per_class_metrics']['recall'][i]:.3f}, "
                  f"F1={analysis['per_class_metrics']['f1'][i]:.3f}")
    
    print("\nMost Confused Pairs:")
    print("-" * 40)
    for pair in analysis['most_confused_pairs']:
        print(f"{pair['true_class']} -> {pair['predicted_class']}: "
              f"{pair['count']} ({pair['percentage']:.1f}%)")


if __name__ == "__main__":
    # Test confusion matrix functions
    print("Testing confusion matrix functions...")
    
    # Create dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    
    # Test analysis
    analysis = analyze_confusion_matrix(y_true, y_pred, class_names)
    print_confusion_analysis(analysis, class_names)
    
    print("Confusion matrix functions test completed!")
