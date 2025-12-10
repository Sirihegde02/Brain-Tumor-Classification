"""
Model architecture visualization tools

Creates diagrams of model architectures including overall model graphs
and detailed block diagrams for LEAD-CNN and LightNet models.
"""
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    TF_AVAILABLE = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import json
import argparse

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore


# Shared palette to keep diagrams visually consistent
PALETTE = {
    "ink": "#111827",
    "ink_light": "#4b5563",
    "canvas": "#f8fafc",
    "grid": "#e5e7eb",
    "input": "#d1d5db",
    "stem": "#9ac7d8",
    "pool": "#9ee4c2",
    "dr": "#f59e0b",
    "dr_detail": "#fb923c",
    "gap": "#fca5a5",
    "dense": "#fef3c7",
    "output": "#fbcfe8",
    "callout": "#f3f4f6",
    "lite_block": "#9ac7d8",
    "litedr": "#f59e0b",
    "se": "#c7f9cc",
}


def _add_block(ax, x: float, y: float, width: float, height: float,
               label: str, color: str, text_color: str = PALETTE["ink"],
               lw: float = 1.6) -> Tuple[float, float]:
    """Draw a rounded block and return its center."""
    rect = FancyBboxPatch(
        xy=(x, y),
        width=width,
        height=height,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=color,
        edgecolor=PALETTE["ink"],
        linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=text_color,
    )
    return x + width / 2, y + height / 2


def _add_arrow(ax, start: Tuple[float, float], end: Tuple[float, float],
               color: str = PALETTE["ink"], style: str = "simple",
               lw: float = 1.4, mutation: float = 14.0,
               linestyle: str = "-") -> None:
    """Add a clean arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def _style_axes(fig, ax) -> None:
    """Apply a subtle background and remove axis clutter."""
    fig.patch.set_facecolor(PALETTE["canvas"])
    ax.set_facecolor(PALETTE["canvas"])
    ax.axis("off")


def plot_model_architecture(model: "keras.Model", output_path: str,
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
    if not TF_AVAILABLE:
        print("TensorFlow not available; using manual matplotlib diagram.")
        plot_model_manual(model, output_path, title, figsize)
        return

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
        print(f"Error plotting model with tf.keras.utils.plot_model: {e}")
        # Fallback to simple matplotlib layer-bar diagram so the pipeline doesn't break
        try:
            safe_plot_layers_fallback(model, output_path, title)
            print(f"Graphviz unavailable; saved fallback diagram to {output_path}")
        except Exception as e2:
            print(f"Fallback plotting also failed: {e2}. Using manual plot.")
            plot_model_manual(model, output_path, title, figsize)


def plot_model_manual(model: "keras.Model", output_path: str,
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
            xy=(0.1, y_pos - 0.03), width=0.8, height=0.06,
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

def safe_plot_layers_fallback(model: "keras.Model", output_path: str,
                              title: str = "Model Architecture (Fallback)") -> None:
    """
    Simpler, robust fallback diagram: horizontal bars with layer names on y-axis.
    Saves to the same output path to keep pipeline unbroken.
    """
    # Always use a safe figsize tuple
    plt.figure(figsize=(10, 6))
    layers = [layer.name for layer in model.layers]
    lengths = list(range(1, len(layers) + 1))
    y_pos = list(range(len(layers)))
    plt.barh(y_pos, lengths, color="lightblue", edgecolor="black")
    plt.yticks(y_pos, layers, fontsize=8)
    plt.xlabel("Layer order")
    plt.title(title)
    plt.gca().invert_yaxis()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_lead_cnn_architecture(output_path: str) -> None:
    """
    Plot LEAD-CNN architecture diagram
    
    Args:
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    _style_axes(fig, ax)

    stack_x = 1.2
    stack_width = 2.6
    stack_height = 0.9
    y_start = 8.6
    y_gap = 1.05

    lead_blocks = [
        ("Input\n224×224×3", PALETTE["input"]),
        ("Conv1\n32 filters", PALETTE["stem"]),
        ("MaxPool1\nstride 2", PALETTE["pool"]),
        ("DR Block 1\n64 filters", PALETTE["dr"]),
        ("DR Block 2\n128 filters", PALETTE["dr"]),
        ("DR Block 3\n256 filters", PALETTE["dr"]),
        ("DR Block 4\n512 filters", PALETTE["dr"]),
        ("Global Avg Pool", PALETTE["gap"]),
        ("Dense\n512 units", PALETTE["dense"]),
        ("Output\n4 classes", PALETTE["output"]),
    ]

    centers = []
    y_cursor = y_start
    for label, color in lead_blocks:
        centers.append(_add_block(ax, stack_x, y_cursor, stack_width, stack_height, label, color))
        y_cursor -= y_gap

    for i in range(len(centers) - 1):
        _add_arrow(
            ax,
            (centers[i][0], centers[i][1] - stack_height / 2 - 0.05),
            (centers[i + 1][0], centers[i + 1][1] + stack_height / 2 + 0.05),
            color=PALETTE["ink"],
            mutation=18,
        )

    # Dimension reduction block (callout)
    dr_x = 5.4
    dr_y = 5.8
    dr_width = 1.8
    dr_height = 0.62
    dr_components = [
        "Conv2D\n3×3, stride 1",
        "BatchNorm",
        "LeakyReLU\nα=0.1",
        "Dropout\nrate 0.2",
    ]

    dr_centers = []
    for idx, label in enumerate(dr_components):
        y_pos = dr_y - idx * 0.9
        dr_centers.append(
            _add_block(
                ax,
                dr_x,
                y_pos,
                dr_width,
                dr_height,
                label,
                PALETTE["dr_detail"],
                lw=1.2,
            )
        )
        if idx:
            _add_arrow(
                ax,
                (dr_centers[idx - 1][0], dr_centers[idx - 1][1] - dr_height / 2 - 0.04),
                (dr_centers[idx][0], dr_centers[idx][1] + dr_height / 2 + 0.04),
                color=PALETTE["ink_light"],
                mutation=12,
            )

    # Connector from main stack to DR detail
    _add_arrow(
        ax,
        (stack_x + stack_width, centers[3][1]),
        (dr_x - 0.15, dr_centers[0][1] + 0.25),
        color=PALETTE["dr_detail"],
        linestyle="--",
        mutation=14,
    )

    # Text callouts
    ax.text(
        centers[0][0],
        y_start + 0.85,
        "Input: 224×224×3",
        ha="center",
        fontsize=11,
        color=PALETTE["ink"],
    )
    ax.text(
        centers[-1][0],
        centers[-1][1] - 1.1,
        "Output: 4 classes",
        ha="center",
        fontsize=11,
        color=PALETTE["ink_light"],
    )
    ax.text(
        dr_x + dr_width / 2,
        dr_y - len(dr_components) * 0.9 - 0.3,
        "Dimension Reduction Block",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["dr_detail"],
    )

    ax.set_title("LEAD-CNN Architecture", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    ax.set_xlim(0, 9)
    ax.set_ylim(-1, 10)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close()

    print(f"LEAD-CNN architecture saved to: {output_path}")


def plot_lightnet_architecture(output_path: str) -> None:
    """
    Plot LightNet architecture diagram
    
    Args:
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    _style_axes(fig, ax)

    stack_x = 1.2
    stack_width = 2.6
    stack_height = 0.9
    y_start = 8.6
    y_gap = 1.05

    light_blocks = [
        ("Input\n224×224×3", PALETTE["input"]),
        ("Stem\n32 filters", PALETTE["stem"]),
        ("LiteDR 1\n64 filters", PALETTE["dr"]),
        ("LiteDR 2\n128 filters", PALETTE["dr"]),
        ("LiteDR 3\n256 filters", PALETTE["dr"]),
        ("LiteDR 4\n256 filters", PALETTE["dr"]),
        ("Global Avg Pool", PALETTE["gap"]),
        ("Dense\n128 units", PALETTE["dense"]),
        ("Output\n4 classes", PALETTE["output"]),
    ]

    centers = []
    y_cursor = y_start
    for label, color in light_blocks:
        centers.append(_add_block(ax, stack_x, y_cursor, stack_width, stack_height, label, color))
        y_cursor -= y_gap

    for i in range(len(centers) - 1):
        _add_arrow(
            ax,
            (centers[i][0], centers[i][1] - stack_height / 2 - 0.05),
            (centers[i + 1][0], centers[i + 1][1] + stack_height / 2 + 0.05),
            color=PALETTE["ink"],
            mutation=18,
        )

    # LiteDR block detail
    litedr_x = 5.4
    litedr_y = 6.4
    litedr_width = 1.9
    litedr_height = 0.62
    litedr_components = [
        ("Depthwise Conv2D\n3×3", PALETTE["lite_block"]),
        ("Pointwise Conv2D\n1×1", PALETTE["se"]),
        ("BatchNorm", PALETTE["dr_detail"]),
        ("LeakyReLU", PALETTE["gap"]),
        ("SE Block", PALETTE["dense"]),
    ]

    litedr_centers = []
    for idx, (label, color) in enumerate(litedr_components):
        y_pos = litedr_y - idx * 0.9
        litedr_centers.append(_add_block(ax, litedr_x, y_pos, litedr_width, litedr_height, label, color, lw=1.2))
        if idx:
            _add_arrow(
                ax,
                (litedr_centers[idx - 1][0], litedr_centers[idx - 1][1] - litedr_height / 2 - 0.04),
                (litedr_centers[idx][0], litedr_centers[idx][1] + litedr_height / 2 + 0.04),
                color=PALETTE["ink_light"],
                mutation=12,
            )

    # SE block detail
    se_x = 7.5
    se_y = 3.4
    se_width = 1.5
    se_height = 0.52
    se_components = [
        ("Global Avg Pool", PALETTE["lite_block"]),
        ("Dense (ReLU)", PALETTE["se"]),
        ("Dense (Sigmoid)", PALETTE["dr_detail"]),
    ]

    se_centers = []
    for idx, (label, color) in enumerate(se_components):
        y_pos = se_y - idx * 0.8
        se_centers.append(_add_block(ax, se_x, y_pos, se_width, se_height, label, color, lw=1.1))
        if idx:
            _add_arrow(
                ax,
                (se_centers[idx - 1][0], se_centers[idx - 1][1] - se_height / 2 - 0.03),
                (se_centers[idx][0], se_centers[idx][1] + se_height / 2 + 0.03),
                color=PALETTE["ink_light"],
                mutation=12,
            )

    # Connect callouts
    _add_arrow(
        ax,
        (stack_x + stack_width, centers[3][1]),
        (litedr_x - 0.15, litedr_centers[0][1] + 0.25),
        color=PALETTE["dr_detail"],
        linestyle="--",
        mutation=14,
    )
    _add_arrow(
        ax,
        (litedr_x + litedr_width, litedr_centers[-1][1]),
        (se_x - 0.1, se_centers[0][1] + 0.15),
        color=PALETTE["lite_block"],
        linestyle="--",
        mutation=14,
    )

    # Text callouts
    ax.text(
        centers[0][0],
        y_start + 0.85,
        "Input: 224×224×3",
        ha="center",
        fontsize=11,
        color=PALETTE["ink"],
    )
    ax.text(
        centers[-1][0],
        centers[-1][1] - 1.1,
        "Output: 4 classes",
        ha="center",
        fontsize=11,
        color=PALETTE["ink_light"],
    )
    ax.text(
        litedr_x + litedr_width / 2,
        litedr_y - len(litedr_components) * 0.9 - 0.25,
        "LiteDR Block",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["dr_detail"],
    )
    ax.text(
        se_x + se_width / 2,
        se_y - len(se_components) * 0.8 - 0.25,
        "SE Block",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=PALETTE["ink_light"],
    )

    ax.set_title("LightNet Architecture", fontsize=18, fontweight="bold", color=PALETTE["ink"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close()

    print(f"LightNet architecture saved to: {output_path}")


def create_architecture_comparison(output_dir: str) -> None:
    """
    Create comparison diagram of all architectures
    
    Args:
        output_dir: Output directory
    """
    output_path = Path(output_dir) / "figures" / "architecture_comparison.png"
    
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax in axes:
        _style_axes(fig, ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # LEAD-CNN
    axes[0].set_title("LEAD-CNN", fontsize=13, fontweight="bold", color=PALETTE["ink"])
    _add_block(axes[0], 0.24, 0.55, 0.52, 0.16, "DR Blocks ×4\n2M params", PALETTE["dr"])
    _add_block(axes[0], 0.35, 0.25, 0.30, 0.12, "Teacher\nAccuracy ≈0.94", PALETTE["gap"])
    _add_arrow(axes[0], (0.5, 0.55), (0.5, 0.42), mutation=16)

    # LightNet
    axes[1].set_title("LightNet", fontsize=13, fontweight="bold", color=PALETTE["ink"])
    _add_block(axes[1], 0.24, 0.55, 0.52, 0.16, "LiteDR Blocks ×4\n≈121k params", PALETTE["dr"])
    _add_block(axes[1], 0.35, 0.25, 0.30, 0.12, "Student\nLightweight", PALETTE["dense"])
    _add_arrow(axes[1], (0.5, 0.55), (0.5, 0.42), mutation=16)

    # Knowledge Distillation
    axes[2].set_title("Knowledge Distillation", fontsize=13, fontweight="bold", color=PALETTE["ink"])
    _add_block(axes[2], 0.18, 0.55, 0.28, 0.14, "Teacher\nLEAD-CNN", PALETTE["gap"])
    _add_block(axes[2], 0.54, 0.55, 0.28, 0.14, "Student\nLightNetV2", PALETTE["dense"])
    _add_block(axes[2], 0.36, 0.24, 0.30, 0.12, "Hard + Soft Labels", PALETTE["se"])
    _add_arrow(axes[2], (0.46, 0.62), (0.46, 0.48), mutation=14, color=PALETTE["ink"])
    _add_arrow(axes[2], (0.54, 0.62), (0.54, 0.48), mutation=14, color=PALETTE["ink"])
    _add_arrow(axes[2], (0.32, 0.55), (0.48, 0.55), mutation=12, color=PALETTE["dr_detail"], linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close()

    print(f"Architecture comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot model architectures and blocks.")
    parser.add_argument("--model", choices=["leadcnn", "lightnet"], help="Model to plot")
    parser.add_argument("--block", choices=["litedr"], help="Specific block diagram to plot")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"], help="LightNet version")
    args = parser.parse_args()
    
    out_path = args.out
    
    # If a specific block is requested
    if args.block:
        if args.block == "litedr":
            plot_lightnet_architecture(out_path)
            return
    
    if args.model:
        if args.model == "leadcnn":
            try:
                from src.models.lead_cnn import create_lead_cnn
            except ImportError:
                from models.lead_cnn import create_lead_cnn
            model = create_lead_cnn().model
            plot_model_architecture(model, out_path, title="LEAD-CNN Architecture")
        elif args.model == "lightnet":
            try:
                from src.models.lightnet import create_lightnet
            except ImportError:
                from models.lightnet import create_lightnet
            model = create_lightnet(version=args.version).model
            plot_model_architecture(model, out_path, title=f"LightNet {args.version.upper()} Architecture")
        return
    
    # Default behavior: generate LEAD-CNN overview
    plot_lead_cnn_architecture(out_path)


if __name__ == "__main__":
    main()
