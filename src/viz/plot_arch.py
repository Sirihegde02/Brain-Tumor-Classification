"""
Model architecture visualization tools

Creates diagrams of model architectures including overall model graphs
and detailed block diagrams for LEAD-CNN and LightNet models.
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json


def plot_model_architecture(model: keras.Model, output_path: str,
                           title: str = "Model Architecture",
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot model architecture using TensorFlow's plot_model
    
    Args:
        model: Keras model
        output_path: Output file path
        title: Plot title
        figsize: Figure size
    """
    try:
        # Use TensorFlow's built-in plot_model
        tf.keras.utils.plot_model(
            model,
            to_file=output_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',  # Top to bottom
            expand_nested=True,
            dpi=300
        )
        print(f"Model architecture saved to: {output_path}")
    except Exception as e:
        print(f"Error plotting model: {e}")
        # Fallback to manual plotting
        plot_model_manual(model, output_path, title, figsize)


def plot_model_manual(model: keras.Model, output_path: str,
                     title: str = "Model Architecture",
                     figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Manually plot model architecture
    
    Args:
        model: Keras model
        output_path: Output file path
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get model layers
    layers = model.layers
    
    # Calculate positions
    num_layers = len(layers)
    y_positions = np.linspace(0.9, 0.1, num_layers)
    
    # Plot layers
    for i, (layer, y_pos) in enumerate(zip(layers, y_positions)):
        # Layer box
        box = FancyBboxPatch(
            (0.1, y_pos - 0.03), 0.8, 0.06,
            boxstyle="round,pad=0.01",
            facecolor='lightblue',
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(box)
        
        # Layer name
        ax.text(0.5, y_pos, layer.name, ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        # Layer type
        layer_type = type(layer).__name__
        ax.text(0.5, y_pos - 0.02, layer_type, ha='center', va='center',
                fontsize=8, style='italic')
        
        # Connect to next layer
        if i < num_layers - 1:
            arrow = ConnectionPatch(
                (0.5, y_pos - 0.03), (0.5, y_positions[i + 1] + 0.03),
                "data", "data",
                arrowstyle="->", shrinkA=5, shrinkB=5,
                mutation_scale=20, fc="black"
            )
            ax.add_patch(arrow)
    
    # Set plot properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Manual model architecture saved to: {output_path}")


def plot_lead_cnn_architecture(output_path: str) -> None:
    """
    Plot LEAD-CNN architecture diagram
    
    Args:
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define blocks and their positions
    blocks = [
        {"name": "Input", "pos": (1, 9), "size": (2, 0.8), "color": "lightgray"},
        {"name": "Conv1\n(32 filters)", "pos": (1, 8), "size": (2, 0.8), "color": "lightblue"},
        {"name": "MaxPool1", "pos": (1, 7), "size": (2, 0.8), "color": "lightgreen"},
        {"name": "DR Block 1\n(64 filters)", "pos": (1, 6), "size": (2, 0.8), "color": "orange"},
        {"name": "DR Block 2\n(128 filters)", "pos": (1, 5), "size": (2, 0.8), "color": "orange"},
        {"name": "DR Block 3\n(256 filters)", "pos": (1, 4), "size": (2, 0.8), "color": "orange"},
        {"name": "DR Block 4\n(512 filters)", "pos": (1, 3), "size": (2, 0.8), "color": "orange"},
        {"name": "Global\nAvg Pool", "pos": (1, 2), "size": (2, 0.8), "color": "lightcoral"},
        {"name": "Dense\n(512 units)", "pos": (1, 1), "size": (2, 0.8), "color": "lightyellow"},
        {"name": "Output\n(4 classes)", "pos": (1, 0), "size": (2, 0.8), "color": "lightpink"}
    ]
    
    # Draw blocks
    for block in blocks:
        rect = FancyBboxPatch(
            block["pos"], block["size"],
            boxstyle="round,pad=0.1",
            facecolor=block["color"],
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(block["pos"][0] + block["size"][0]/2, 
                block["pos"][1] + block["size"][1]/2,
                block["name"], ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # Add arrows between blocks
    for i in range(len(blocks) - 1):
        start_y = blocks[i]["pos"][1] + blocks[i]["size"][1]
        end_y = blocks[i + 1]["pos"][1] + blocks[i + 1]["size"][1]
        
        arrow = ConnectionPatch(
            (blocks[i]["pos"][0] + blocks[i]["size"][0]/2, start_y),
            (blocks[i + 1]["pos"][0] + blocks[i + 1]["size"][0]/2, end_y),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc="black"
        )
        ax.add_patch(arrow)
    
    # Add dimension reduction block detail
    dr_detail_x = 5
    dr_detail_y = 6
    
    # DR Block components
    dr_components = [
        {"name": "Conv2D", "pos": (dr_detail_x, dr_detail_y + 1), "size": (1.5, 0.6), "color": "lightblue"},
        {"name": "BatchNorm", "pos": (dr_detail_x, dr_detail_y), "size": (1.5, 0.6), "color": "lightgreen"},
        {"name": "LeakyReLU", "pos": (dr_detail_x, dr_detail_y - 1), "size": (1.5, 0.6), "color": "orange"},
        {"name": "Dropout", "pos": (dr_detail_x, dr_detail_y - 2), "size": (1.5, 0.6), "color": "lightcoral"}
    ]
    
    # Draw DR block components
    for comp in dr_components:
        rect = FancyBboxPatch(
            comp["pos"], comp["size"],
            boxstyle="round,pad=0.05",
            facecolor=comp["color"],
            edgecolor="black",
            linewidth=1
        )
        ax.add_patch(rect)
        
        ax.text(comp["pos"][0] + comp["size"][0]/2, 
                comp["pos"][1] + comp["size"][1]/2,
                comp["name"], ha='center', va='center',
                fontsize=8, fontweight='bold')
    
    # Connect DR block to main architecture
    dr_arrow = ConnectionPatch(
        (3, 6.4), (dr_detail_x, dr_detail_y + 1.3),
        "data", "data",
        arrowstyle="->", shrinkA=5, shrinkB=5,
        mutation_scale=15, fc="red", linestyle="--"
    )
    ax.add_patch(dr_arrow)
    
    # Add title and labels
    ax.set_title("LEAD-CNN Architecture", fontsize=16, fontweight='bold', pad=20)
    ax.text(2, 9.5, "Input: 224×224×3", ha='center', fontsize=12)
    ax.text(2, -0.5, "Output: 4 classes", ha='center', fontsize=12)
    ax.text(dr_detail_x + 0.75, dr_detail_y - 2.5, "Dimension Reduction Block", 
            ha='center', fontsize=10, fontweight='bold', color='red')
    
    # Set plot properties
    ax.set_xlim(0, 8)
    ax.set_ylim(-1, 10)
    ax.axis('off')
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"LEAD-CNN architecture saved to: {output_path}")


def plot_lightnet_architecture(output_path: str) -> None:
    """
    Plot LightNet architecture diagram
    
    Args:
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define blocks and their positions
    blocks = [
        {"name": "Input", "pos": (1, 9), "size": (2, 0.8), "color": "lightgray"},
        {"name": "Stem\n(32 filters)", "pos": (1, 8), "size": (2, 0.8), "color": "lightblue"},
        {"name": "LiteDR 1\n(64 filters)", "pos": (1, 7), "size": (2, 0.8), "color": "orange"},
        {"name": "LiteDR 2\n(128 filters)", "pos": (1, 6), "size": (2, 0.8), "color": "orange"},
        {"name": "LiteDR 3\n(256 filters)", "pos": (1, 5), "size": (2, 0.8), "color": "orange"},
        {"name": "LiteDR 4\n(256 filters)", "pos": (1, 4), "size": (2, 0.8), "color": "orange"},
        {"name": "Global\nAvg Pool", "pos": (1, 3), "size": (2, 0.8), "color": "lightcoral"},
        {"name": "Dense\n(128 units)", "pos": (1, 2), "size": (2, 0.8), "color": "lightyellow"},
        {"name": "Output\n(4 classes)", "pos": (1, 1), "size": (2, 0.8), "color": "lightpink"}
    ]
    
    # Draw blocks
    for block in blocks:
        rect = FancyBboxPatch(
            block["pos"], block["size"],
            boxstyle="round,pad=0.1",
            facecolor=block["color"],
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(block["pos"][0] + block["size"][0]/2, 
                block["pos"][1] + block["size"][1]/2,
                block["name"], ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # Add arrows between blocks
    for i in range(len(blocks) - 1):
        start_y = blocks[i]["pos"][1] + blocks[i]["size"][1]
        end_y = blocks[i + 1]["pos"][1] + blocks[i + 1]["size"][1]
        
        arrow = ConnectionPatch(
            (blocks[i]["pos"][0] + blocks[i]["size"][0]/2, start_y),
            (blocks[i + 1]["pos"][0] + blocks[i + 1]["size"][0]/2, end_y),
            "data", "data",
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=20, fc="black"
        )
        ax.add_patch(arrow)
    
    # Add LiteDR block detail
    lite_dr_x = 5
    lite_dr_y = 5
    
    # LiteDR Block components
    lite_dr_components = [
        {"name": "Depthwise\nConv2D", "pos": (lite_dr_x, lite_dr_y + 1.5), "size": (1.5, 0.6), "color": "lightblue"},
        {"name": "Pointwise\nConv2D", "pos": (lite_dr_x, lite_dr_y + 0.5), "size": (1.5, 0.6), "color": "lightgreen"},
        {"name": "BatchNorm", "pos": (lite_dr_x, lite_dr_y - 0.5), "size": (1.5, 0.6), "color": "orange"},
        {"name": "LeakyReLU", "pos": (lite_dr_x, lite_dr_y - 1.5), "size": (1.5, 0.6), "color": "lightcoral"},
        {"name": "SE Block", "pos": (lite_dr_x, lite_dr_y - 2.5), "size": (1.5, 0.6), "color": "lightyellow"}
    ]
    
    # Draw LiteDR block components
    for comp in lite_dr_components:
        rect = FancyBboxPatch(
            comp["pos"], comp["size"],
            boxstyle="round,pad=0.05",
            facecolor=comp["color"],
            edgecolor="black",
            linewidth=1
        )
        ax.add_patch(rect)
        
        ax.text(comp["pos"][0] + comp["size"][0]/2, 
                comp["pos"][1] + comp["size"][1]/2,
                comp["name"], ha='center', va='center',
                fontsize=8, fontweight='bold')
    
    # Connect LiteDR block to main architecture
    lite_dr_arrow = ConnectionPatch(
        (3, 5.4), (lite_dr_x, lite_dr_y + 2.1),
        "data", "data",
        arrowstyle="->", shrinkA=5, shrinkB=5,
        mutation_scale=15, fc="red", linestyle="--"
    )
    ax.add_patch(lite_dr_arrow)
    
    # Add SE block detail
    se_x = 7
    se_y = 2
    
    se_components = [
        {"name": "Global\nAvg Pool", "pos": (se_x, se_y + 1), "size": (1.2, 0.5), "color": "lightblue"},
        {"name": "Dense\n(ReLU)", "pos": (se_x, se_y), "size": (1.2, 0.5), "color": "lightgreen"},
        {"name": "Dense\n(Sigmoid)", "pos": (se_x, se_y - 1), "size": (1.2, 0.5), "color": "orange"}
    ]
    
    for comp in se_components:
        rect = FancyBboxPatch(
            comp["pos"], comp["size"],
            boxstyle="round,pad=0.05",
            facecolor=comp["color"],
            edgecolor="black",
            linewidth=1
        )
        ax.add_patch(rect)
        
        ax.text(comp["pos"][0] + comp["size"][0]/2, 
                comp["pos"][1] + comp["size"][1]/2,
                comp["name"], ha='center', va='center',
                fontsize=7, fontweight='bold')
    
    # Connect SE block
    se_arrow = ConnectionPatch(
        (lite_dr_x + 1.5, lite_dr_y - 2.2), (se_x, se_y + 1.2),
        "data", "data",
        arrowstyle="->", shrinkA=5, shrinkB=5,
        mutation_scale=15, fc="blue", linestyle="--"
    )
    ax.add_patch(se_arrow)
    
    # Add title and labels
    ax.set_title("LightNet Architecture", fontsize=16, fontweight='bold', pad=20)
    ax.text(2, 9.5, "Input: 224×224×3", ha='center', fontsize=12)
    ax.text(2, 0.5, "Output: 4 classes", ha='center', fontsize=12)
    ax.text(lite_dr_x + 0.75, lite_dr_y - 3.2, "LiteDR Block", 
            ha='center', fontsize=10, fontweight='bold', color='red')
    ax.text(se_x + 0.6, se_y - 1.8, "SE Block", 
            ha='center', fontsize=10, fontweight='bold', color='blue')
    
    # Set plot properties
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"LightNet architecture saved to: {output_path}")


def create_architecture_comparison(output_dir: str) -> None:
    """
    Create comparison diagram of all architectures
    
    Args:
        output_dir: Output directory
    """
    output_path = Path(output_dir) / "figures" / "architecture_comparison.png"
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # LEAD-CNN
    ax1 = axes[0]
    ax1.set_title("LEAD-CNN", fontsize=14, fontweight='bold')
    ax1.text(0.5, 0.5, "LEAD-CNN\nArchitecture", ha='center', va='center',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # LightNet
    ax2 = axes[1]
    ax2.set_title("LightNet", fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, "LightNet\nArchitecture", ha='center', va='center',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Knowledge Distillation
    ax3 = axes[2]
    ax3.set_title("Knowledge Distillation", fontsize=14, fontweight='bold')
    ax3.text(0.5, 0.5, "Teacher → Student\nDistillation", ha='center', va='center',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Architecture comparison saved to: {output_path}")


if __name__ == "__main__":
    # Test architecture plotting
    print("Testing architecture plotting...")
    
    # Create dummy model
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(4, activation='softmax')
    ])
    
    # Test plotting
    plot_model_architecture(model, "test_architecture.png")
    plot_lead_cnn_architecture("test_lead_cnn.png")
    plot_lightnet_architecture("test_lightnet.png")
    
    print("Architecture plotting test completed!")
