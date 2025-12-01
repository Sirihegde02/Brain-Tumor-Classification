"""
Compare two trained models using saved metrics and parameter summaries.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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


def format_markdown_table(models: List[Tuple[str, Dict[str, Any]]]) -> str:
    """Create a markdown comparison table for selected metrics."""
    keys = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision (macro)"),
        ("recall_macro", "Recall (macro)"),
        ("f1_macro", "F1 (macro)"),
        ("cohen_kappa", "Cohen's Kappa"),
        ("roc_auc", "ROC AUC"),
    ]

    header = "| Metric | " + " | ".join(name for name, _ in models) + " |"
    divider = "|---|" + "|".join(["---"] * len(models)) + "|"
    lines = [header, divider]

    for key, label in keys:
        row = [label]
        for _, metrics in models:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

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


def build_model_list(args) -> List[Tuple[str, Path]]:
    models = []
    if args.models:
        for entry in args.models:
            if "=" not in entry:
                raise ValueError(f"Invalid --models entry '{entry}', expected Name=path")
            name, path = entry.split("=", 1)
            models.append((name.strip(), Path(path.strip())))
    else:
        if args.name_a and args.results_a and args.name_b and args.results_b:
            models.append((args.name_a, Path(args.results_a)))
            models.append((args.name_b, Path(args.results_b)))
        else:
            raise ValueError("Provide either --models or both --name_a/--results_a and --name_b/--results_b")
    return models


def main():
    parser = argparse.ArgumentParser(description="Compare trained models using saved metrics.")
    parser.add_argument("--name_a", help="Display name for model A.")
    parser.add_argument("--results_a", help="Path to test_metrics.json for model A.")
    parser.add_argument("--name_b", help="Display name for model B.")
    parser.add_argument("--results_b", help="Path to test_metrics.json for model B.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of NAME=path entries (e.g., LEAD=outputs/baseline/test_metrics.json).",
    )
    parser.add_argument(
        "--summary_filename",
        default="model_summary.txt",
        help="Filename for saved model summaries (default: model_summary.txt).",
    )
    args = parser.parse_args()

    model_specs = build_model_list(args)
    metrics_entries = []

    print("Loaded metrics for:")
    for name, path in model_specs:
        metrics = load_metrics(path)
        metrics_entries.append((name, metrics))
        print(f"  {name}: {path}")
    print()

    table = format_markdown_table(metrics_entries)
    print("### Metric Comparison")
    print(table)

    for name, path in model_specs:
        report_model_info(name, path.parent, args.summary_filename)


if __name__ == "__main__":
    main()
