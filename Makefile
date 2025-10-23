# Brain Tumor Lightweight Classifier Makefile

.PHONY: help install setup data train-baseline train-lightnet train-kd evaluate clean

# Default target
help:
	@echo "Brain Tumor Lightweight Classifier"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  setup         - Setup project structure"
	@echo "  data          - Download and prepare data"
	@echo "  train-baseline- Train LEAD-CNN baseline"
	@echo "  train-lightnet- Train LightNet model"
	@echo "  train-kd      - Train with knowledge distillation"
	@echo "  evaluate      - Evaluate all models"
	@echo "  report        - Generate final report"
	@echo "  clean         - Clean output files"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Setup project
setup:
	python -m pip install -e .

# Download and prepare data
data:
	python src/data/download_kaggle.py
	python src/data/prepare_splits.py --create_csv

# Train LEAD-CNN baseline
train-baseline:
	python src/train/train_baseline.py --config experiments/baseline_leadcnn.yaml

# Train LightNet
train-lightnet:
	python src/train/train_lightnet.py --config experiments/lightnet_ablation.yaml

# Train with knowledge distillation
train-kd:
	python src/train/train_kd.py --config experiments/kd.yaml --teacher_path outputs/checkpoints/lead_cnn_best.h5

# Evaluate models
evaluate:
	python src/eval/evaluate.py \
		--model_paths outputs/checkpoints/lead_cnn_best.h5 outputs/checkpoints/lightnet_best.h5 outputs/checkpoints/lightnet_kd_best.h5 \
		--model_names LEAD-CNN LightNet KD-LightNet \
		--compare --generate_gradcam

# Generate final report
report:
	@echo "Generating final report..."
	@echo "=========================="
	@echo ""
	@echo "Model Performance Summary:"
	@echo "========================="
	@if [ -f outputs/reports/comparison_results.json ]; then \
		python -c "import json; data=json.load(open('outputs/reports/comparison_results.json')); \
		print('\\n'.join([f'{name}: Acc={metrics[\"accuracy\"]:.3f}, F1={metrics[\"f1_macro\"]:.3f}, Kappa={metrics[\"cohen_kappa\"]:.3f}' \
		for name, metrics in data['metrics'].items()]))"; \
	else \
		echo "No comparison results found. Run 'make evaluate' first."; \
	fi
	@echo ""
	@echo "Architecture diagrams saved to: outputs/figures/"
	@echo "Model summaries saved to: outputs/reports/"
	@echo "Training logs saved to: outputs/logs/"

# Clean output files
clean:
	rm -rf outputs/checkpoints/*.h5
	rm -rf outputs/logs/*
	rm -rf outputs/figures/*
	rm -rf outputs/reports/*.json
	rm -rf outputs/reports/*.txt
	@echo "Cleaned output files"

# Full pipeline
pipeline: setup data train-baseline train-lightnet train-kd evaluate report

# Development setup
dev-setup: install setup
	@echo "Development environment ready!"
	@echo "Run 'make data' to download dataset"
	@echo "Run 'make pipeline' for full training pipeline"
