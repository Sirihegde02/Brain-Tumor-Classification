"""
Compare two trained models using saved metrics and parameter summaries.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml

# Ensure repo root/src are on the path when executed as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    sys.path.insert(0, str(path))

from utils.params import analyze_model_complexity
from train.train_baseline import create_model as create_baseline_model


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics JSON from disk."""
    with open(path, "r") as f:
        return json.load(f)


def format_markdown_table(name_a: str, metrics_a: Dict[str, Any],
                          name_b: str, metrics_b: Dict[str, Any]) -> str:
    """Create a markdown comparison table for selected metrics."""
    keys = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision (macro)"),
        ("recall_macro", "Recall (macro)"),
        ("f1_macro", "F1 (macro)"),
        ("cohen_kappa", "Cohen's Kappa"),
        ("roc_auc", "ROC AUC"),
    ]
    
    lines = ["| Metric | {a} | {b} |".format(a=name_a, b=name_b),
             "|---|---|---|"]
    for key, label in keys:
        a_val = metrics_a.get(key)
        b_val = metrics_b.get(key)
        a_str = f"{a_val:.4f}" if isinstance(a_val, (int, float)) else str(a_val)
        b_str = f"{b_val:.4f}" if isinstance(b_val, (int, float)) else str(b_val)
        lines.append(f"| {label} | {a_str} | {b_str} |")
    return "\n".join(lines)


def find_summary_file(run_dir: Path, summary_filename: str) -> Optional[Path]:
    """Try to find an existing summary file."""
    candidate = run_dir / summary_filename
    if candidate.exists():
        return candidate
    
    reports_dir = run_dir / "reports"
    if reports_dir.exists():
        txt_files = list(reports_dir.glob("*.txt"))
        if txt_files:
            return txt_files[0]
    return None


def find_config_path(run_dir: Path) -> Optional[Path]:
    """Find the saved config file within the run directory."""
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return None
    
    preferred = logs_dir / "lead_cnn_config.yaml"
    if preferred.exists():
        return preferred
    
    yaml_files = list(logs_dir.glob("*.yaml"))
    if yaml_files:
        return yaml_files[0]
    return None


def analyze_from_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Rebuild the model from config and analyze its complexity."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_wrapper = create_baseline_model(config)
    base_model = getattr(model_wrapper, "model", model_wrapper)
    complexity = analyze_model_complexity(base_model)
    return complexity


def report_model_info(name: str, run_dir: Path, summary_filename: str) -> None:
    """Print summary text or reconstructed complexity information."""
    summary_path = find_summary_file(run_dir, summary_filename)
    if summary_path is not None:
        print(f"\n{name} summary ({summary_path}):")
        print(summary_path.read_text())
        return
    
    config_path = find_config_path(run_dir)
    if config_path is None:
        print(f"\nNo summary or config found for {name} in {run_dir}.")
        return
    
    complexity = analyze_from_config(config_path)
    if complexity is None:
        print(f"\nFailed to analyze model complexity for {name} using {config_path}.")
        return
    
    total_params = complexity["parameters"]["total"]
    model_size_mb = complexity["model_size_mb"]
    print(f"\n{name} complexity (reconstructed from {config_path}):")
    print(f"  Total params: {total_params:,}")
    print(f"  Model size (MB): {model_size_mb:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Compare two trained models using saved metrics.")
    parser.add_argument("--name_a", required=True, help="Display name for model A.")
    parser.add_argument("--results_a", required=True, help="Path to test_metrics.json for model A.")
    parser.add_argument("--name_b", required=True, help="Display name for model B.")
    parser.add_argument("--results_b", required=True, help="Path to test_metrics.json for model B.")
    parser.add_argument("--summary_filename", default="model_summary.txt",
                        help="Filename for saved model summaries (default: model_summary.txt).")
    args = parser.parse_args()
    
    path_a = Path(args.results_a)
    path_b = Path(args.results_b)
    
    metrics_a = load_metrics(path_a)
    metrics_b = load_metrics(path_b)
    
    print("Loaded metrics for:")
    print(f"  {args.name_a}: {path_a}")
    print(f"  {args.name_b}: {path_b}\n")
    
    table = format_markdown_table(args.name_a, metrics_a, args.name_b, metrics_b)
    print("### Metric Comparison")
    print(table)
    
    report_model_info(args.name_a, path_a.parent, args.summary_filename)
    report_model_info(args.name_b, path_b.parent, args.summary_filename)


if __name__ == "__main__":
    main()
