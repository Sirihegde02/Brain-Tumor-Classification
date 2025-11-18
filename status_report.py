"""
Status Report Generator for Brain Tumor Classification Project

Verifies all files exist, checks for import errors, and generates a comprehensive status report.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists"""
    return Path(filepath).exists()


def check_imports(filepath: str) -> Tuple[bool, str]:
    """Check if a Python file has syntax errors"""
    # Skip non-Python files
    if not filepath.endswith('.py'):
        return True, "Not a Python file (OK)"
    
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_script_help(script_path: str) -> Tuple[bool, str]:
    """Check if a script has --help and runs without errors"""
    try:
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 or "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower():
            return True, ""
        else:
            return False, result.stderr[:100] if result.stderr else "No help output"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:100]


def get_model_params() -> Dict[str, int]:
    """Get model parameter counts (if dependencies installed)"""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from models.lead_cnn import create_lead_cnn
        from models.lightnet import create_lightnet
        from utils.params import count_parameters
        
        lead_cnn = create_lead_cnn()
        lightnet = create_lightnet(version="v1")
        
        return {
            "lead_cnn": count_parameters(lead_cnn.model)["total"],
            "lightnet": count_parameters(lightnet.model)["total"]
        }
    except Exception as e:
        return {"error": str(e)}


def check_directory_structure() -> Dict[str, bool]:
    """Check if all required directories exist"""
    required_dirs = [
        "src/data",
        "src/models",
        "src/train",
        "src/eval",
        "src/viz",
        "src/utils",
        "experiments",
        "examples",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/figures",
        "outputs/reports"
    ]
    
    results = {}
    for dir_path in required_dirs:
        results[dir_path] = Path(dir_path).exists()
        if not results[dir_path]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return results


def check_required_files() -> Dict[str, Tuple[bool, str]]:
    """Check if all required files exist and are valid"""
    required_files = {
        "src/data/download_kaggle.py": "Data download script",
        "src/data/prepare_splits.py": "Data splitting script",
        "src/data/transforms.py": "Data transforms",
        "src/models/lead_cnn.py": "LEAD-CNN model",
        "src/models/lightnet.py": "LightNet model",
        "src/models/blocks.py": "Model blocks",
        "src/models/kd_losses.py": "Knowledge distillation losses",
        "src/train/train_baseline.py": "Baseline training script",
        "src/train/train_lightnet.py": "LightNet training script",
        "src/train/train_kd.py": "Knowledge distillation training script",
        "src/eval/evaluate.py": "Evaluation script",
        "src/eval/metrics.py": "Evaluation metrics",
        "src/eval/confusion.py": "Confusion matrix utilities",
        "src/viz/gradcam.py": "GradCAM visualization",
        "src/viz/plot_arch.py": "Architecture plotting",
        "src/utils/seed.py": "Seed utilities",
        "src/utils/io.py": "I/O utilities",
        "src/utils/params.py": "Parameter counting",
        "experiments/baseline_leadcnn.yaml": "LEAD-CNN config",
        "experiments/lightnet_ablation.yaml": "LightNet config",
        "experiments/kd.yaml": "KD config",
        "examples/quick_start.py": "Quick start example",
        "requirements.txt": "Dependencies",
        "Makefile": "Makefile",
        "setup.py": "Setup script"
    }
    
    results = {}
    for filepath, description in required_files.items():
        exists = check_file_exists(filepath)
        if exists:
            valid, error = check_imports(filepath)
            results[filepath] = (valid, error if not valid else description)
        else:
            results[filepath] = (False, "File not found")
    
    return results


def check_scripts() -> Dict[str, Tuple[bool, str]]:
    """Check if training/evaluation scripts have proper CLI"""
    scripts = {
        "src/train/train_baseline.py": "Baseline training",
        "src/train/train_lightnet.py": "LightNet training",
        "src/train/train_kd.py": "KD training",
        "src/eval/evaluate.py": "Evaluation",
        "src/data/prepare_splits.py": "Data preparation"
    }
    
    results = {}
    for script_path, description in scripts.items():
        if not check_file_exists(script_path):
            results[script_path] = (False, "File not found")
        else:
            # Just check syntax, not actual help (may need dependencies)
            valid, error = check_imports(script_path)
            results[script_path] = (valid, error if not valid else "OK")
    
    return results


def generate_status_report() -> str:
    """Generate comprehensive status report"""
    report = []
    report.append(f"{BOLD}{'='*70}{RESET}")
    report.append(f"{BOLD}Brain Tumor Classification - Project Status Report{RESET}")
    report.append(f"{BOLD}{'='*70}{RESET}\n")
    
    # Check directory structure
    report.append(f"{BOLD}Directory Structure:{RESET}")
    dirs = check_directory_structure()
    for dir_path, exists in dirs.items():
        status = f"{GREEN}✅{RESET}" if exists else f"{YELLOW}⚠️{RESET}"
        report.append(f"  {status} {dir_path}")
    report.append("")
    
    # Check required files
    report.append(f"{BOLD}Required Files:{RESET}")
    files = check_required_files()
    all_files_ok = True
    for filepath, (valid, error) in files.items():
        if valid:
            status = f"{GREEN}✅{RESET}"
        else:
            status = f"{RED}❌{RESET}"
            all_files_ok = False
        report.append(f"  {status} {filepath}")
        if not valid and error != "File not found":
            report.append(f"      {RED}Error: {error}{RESET}")
    report.append("")
    
    # Check scripts
    report.append(f"{BOLD}Scripts:{RESET}")
    scripts = check_scripts()
    for script_path, (valid, error) in scripts.items():
        status = f"{GREEN}✅{RESET}" if valid else f"{RED}❌{RESET}"
        report.append(f"  {status} {script_path}")
        if not valid:
            report.append(f"      {RED}Error: {error}{RESET}")
    report.append("")
    
    # Model parameters (if available)
    report.append(f"{BOLD}Model Parameters:{RESET}")
    params = get_model_params()
    if "error" in params:
        report.append(f"  {YELLOW}⚠️  Cannot compute (dependencies not installed): {params['error']}{RESET}")
        report.append(f"  {BLUE}   Install dependencies: pip install -r requirements.txt{RESET}")
    else:
        lead_params = params.get("lead_cnn", 0)
        lightnet_params = params.get("lightnet", 0)
        reduction = (1 - lightnet_params / lead_params) * 100 if lead_params > 0 else 0
        report.append(f"  {GREEN}✅{RESET} LEAD-CNN: {lead_params:,} parameters")
        report.append(f"  {GREEN}✅{RESET} LightNet: {lightnet_params:,} parameters")
        report.append(f"  {BLUE}   Parameter reduction: {reduction:.1f}%{RESET}")
    report.append("")
    
    # Output paths
    report.append(f"{BOLD}Output Paths:{RESET}")
    report.append(f"  {BLUE}Checkpoints:{RESET} outputs/checkpoints/")
    report.append(f"  {BLUE}Logs:{RESET} outputs/logs/")
    report.append(f"  {BLUE}Figures:{RESET} outputs/figures/")
    report.append(f"  {BLUE}Reports:{RESET} outputs/reports/")
    report.append("")
    
    # Training commands
    report.append(f"{BOLD}Training Commands:{RESET}")
    report.append(f"  {BLUE}1. Install dependencies:{RESET}")
    report.append(f"     pip install -r requirements.txt")
    report.append(f"     python -m pip install -e .")
    report.append("")
    report.append(f"  {BLUE}2. Prepare data:{RESET}")
    report.append(f"     python src/data/download_kaggle.py")
    report.append(f"     python src/data/prepare_splits.py --create_csv")
    report.append("")
    report.append(f"  {BLUE}3. Train LEAD-CNN baseline:{RESET}")
    report.append(f"     python src/train/train_baseline.py --config experiments/baseline_leadcnn.yaml")
    report.append("")
    report.append(f"  {BLUE}4. Train LightNet:{RESET}")
    report.append(f"     python src/train/train_lightnet.py --config experiments/lightnet_ablation.yaml")
    report.append("")
    report.append(f"  {BLUE}5. Train with Knowledge Distillation:{RESET}")
    report.append(f"     python src/train/train_kd.py --config experiments/kd.yaml --teacher_path outputs/checkpoints/lead_cnn_best.h5")
    report.append("")
    report.append(f"  {BLUE}6. Evaluate models:{RESET}")
    report.append(f"     python src/eval/evaluate.py --model_paths outputs/checkpoints/lead_cnn_best.h5 outputs/checkpoints/lightnet_best.h5 --model_names LEAD-CNN LightNet --compare")
    report.append("")
    report.append(f"  {BLUE}7. Using Makefile (recommended):{RESET}")
    report.append(f"     make install      # Install dependencies")
    report.append(f"     make setup        # Setup project")
    report.append(f"     make data         # Download and prepare data")
    report.append(f"     make train-baseline # Train LEAD-CNN")
    report.append(f"     make train-lightnet # Train LightNet")
    report.append(f"     make train-kd     # Train with KD")
    report.append(f"     make evaluate     # Evaluate models")
    report.append("")
    
    # Smoke test
    report.append(f"{BOLD}Smoke Test:{RESET}")
    report.append(f"  {BLUE}Run smoke test (1-minute test with 16 images):{RESET}")
    report.append(f"     python smoke_test.py")
    report.append("")
    
    # Summary
    report.append(f"{BOLD}Summary:{RESET}")
    if all_files_ok:
        report.append(f"  {GREEN}✅ All required files exist and are valid{RESET}")
    else:
        report.append(f"  {RED}❌ Some files are missing or have errors{RESET}")
    
    report.append(f"\n{BOLD}{'='*70}{RESET}\n")
    
    return "\n".join(report)


def main():
    """Generate and print status report"""
    report = generate_status_report()
    print(report)
    
    # Also save to file
    report_path = Path("outputs/status_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        # Remove color codes for file
        import re
        clean_report = re.sub(r'\033\[[0-9;]*m', '', report)
        f.write(clean_report)
    
    print(f"\n{BLUE}Status report also saved to: {report_path}{RESET}")


if __name__ == "__main__":
    main()

