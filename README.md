# Brain Tumor Lightweight Classifier (LEAD-CNN-inspired)

A comprehensive brain tumor classification system that reproduces the LEAD-CNN baseline and creates an over 10× lighter student model using knowledge distillation.

## 🎯 Project Overview

This project implements a brain tumor classification system with the following goals:
- Reproduce the LEAD-CNN baseline on the Kaggle Brain Tumor MRI Dataset (4 classes: glioma, meningioma, pituitary, normal)
- Design a lightweight student model with around 10 percent of LEAD-CNN's parameters (target about 113k parameters)
- Use knowledge distillation to retain accuracy while reducing model size
- Provide comprehensive evaluation with Cohen's kappa and other metrics

## 📊 Dataset

- **Source**: [Kaggle "Brain Tumor MRI Dataset"](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- **Classes**: 4 (glioma, meningioma, pituitary, normal)
- **Images**: about 7023 total
- **Input size**: 224 × 224 RGB
- **Splits**: Train 65 percent, validation 15 percent, test 20 percent (stratified)
- **Preprocessing**: Resize to 224×224, scale to [0,1], normalize with ImageNet mean/std
- **Augmentation (train only)**: Horizontal flip + coarse rotation via `tf.image.rot90` (from a ±10° setting); zoom/brightness disabled in code; no augmentation on val/test

## 🏗️ Project Structure

```text
Brain-Tumor-Classification/
├── README.md
├── requirements.txt
├── setup.py
├── Makefile
├── src/
│   ├── data/
│   │   ├── download_kaggle.py      # Dataset download
│   │   ├── prepare_splits.py       # Stratified data splitting
│   │   └── transforms.py           # Data augmentation and preprocessing
│   ├── models/
│   │   ├── lead_cnn.py             # LEAD-CNN implementation
│   │   ├── lightnet.py             # LightNet and LightNetV2
│   │   ├── blocks.py               # Custom blocks
│   │   └── kd_losses.py            # Knowledge distillation losses
│   ├── train/
│   │   ├── train_baseline.py       # LEAD-CNN training
│   │   ├── train_lightnet.py       # LightNet training and ablations
│   │   └── train_kd.py             # Knowledge distillation
│   ├── eval/
│   │   ├── evaluate.py             # Model evaluation and comparison
│   │   ├── metrics.py              # Metrics helper class
│   │   └── confusion.py            # Confusion matrix utilities
│   ├── viz/
│   │   ├── plot_arch.py            # Architecture diagrams
│   │   └── gradcam.py              # GradCAM visualization
│   └── utils/
│       ├── seed.py                 # Reproducibility
│       ├── io.py                   # I/O utilities
│       └── params.py               # Parameter counting and summaries
├── experiments/
│   ├── baseline_leadcnn.yaml       # LEAD-CNN config
│   ├── lightnet_ablation.yaml      # LightNet v1 config + ablations
│   └── lightnet_v2_kd_final.yaml   # Final KD setup for LightNetV2
├── examples/
│   └── quick_start.py              # Quick start example
└── outputs/
    ├── baseline_leadcnn/
    │   ├── checkpoints/            # Teacher checkpoints
    │   ├── logs/                   # Training logs
    │   └── reports/                # Baseline reports
    ├── lightnet_ablation/
    │   ├── checkpoints/            # LightNet v1 checkpoints
    │   ├── logs/                   # LightNet v1 logs
    │   └── reports/                # Ablation study results
    └── lightnet_v2_kd_final/
        ├── checkpoints/            # KD wrapper and student checkpoints
        ├── logs/                   # KD logs
        └── reports/                # KD comparison reports
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Brain-Tumor-Classification

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
python -m pip install -e .
```

### 2. Data Preparation

```bash
# Download dataset (requires Kaggle API setup)
python src/data/download_kaggle.py

# Prepare stratified splits (writes src/data/splits.json)
python src/data/prepare_splits.py --create_csv
```

### 3. Training Models

```bash
# Train LEAD-CNN baseline
python src/train/train_baseline.py \
  --config experiments/baseline_leadcnn.yaml \
  --splits_file src/data/splits.json \
  --output_dir outputs/baseline_leadcnn

# Train LightNet v1 + run ablation study
python src/train/train_lightnet.py \
  --config experiments/lightnet_ablation.yaml \
  --splits_file src/data/splits.json \
  --output_dir outputs/lightnet_ablation

# Train LightNetV2 with knowledge distillation (final setup)
python src/train/train_kd.py \
  --config experiments/lightnet_v2_kd_final.yaml \
  --teacher_path outputs/baseline_leadcnn/checkpoints/lead_cnn_best.h5 \
  --splits_file src/data/splits.json \
  --output_dir outputs/lightnet_v2_kd_final

```

### 4. Evaluation

```bash
python src/eval/evaluate.py \
  --model_paths outputs/baseline_leadcnn/checkpoints/lead_cnn_best.h5 \
  --model_names LEAD-CNN \
  --splits_file src/data/splits.json \
  --output_dir outputs/final_eval \
  --compare
```

#### This produces:
- Metrics JSON: outputs/final_eval/reports/LEAD-CNN_metrics.json
- Confusion matrix: outputs/final_eval/figures/LEAD-CNN_confusion_matrix.png
- Confusion analysis: outputs/final_eval/reports/LEAD-CNN_confusion_analysis.json

Multi-model comparison and KD student evaluation are available once all compatible checkpoints are trained under the same architecture. See src/eval/evaluate.py --help for more options.

### 5. Using Makefile (Recommended)

```bash
# Full pipeline
make pipeline

# Individual steps
make data          # Download and prepare data
make train-baseline # Train LEAD-CNN
make train-lightnet # Train LightNet
make train-kd      # Train with KD
make evaluate      # Evaluate models
make report        # Generate final report
```

## 📈 Model Performance

| Model         | Params    | Test Accuracy | F1 (macro) | Cohen kappa | ROC AUC (macro) |
| ------------- | --------- | ------------- | ---------- | ----------- | --------------- |
| LEAD-CNN      | 1,970,404 | 0.9388        | 0.9368     | 0.9182      | 0.9931          |
| LightNet v1   | 221,364   | 0.3295        | 0.6907     | 0.6063      | 0.9007          |
| LightNetV2 KD | 120,940   | 0.8043        | 0.6487     | 0.5575      | 0.8927          |

LightNetV2 KD achieves about 93.9 percent parameter reduction compared to LEAD-CNN (1.97M to about 121k) while retaining roughly 85.7 percent of the baseline accuracy.

## 🔧 Key Features

### Models
- **LEAD-CNN**: Teacher model based on dimension-reduction blocks and LeakyReLU activation with about 2M parameters.
- **LightNet**: Lightweight convolutional model using depthwise separable blocks and channel multipliers. Used to explore parameter and accuracy tradeoffs.
- **LightNetV2 KD (student)**: Distilled student model trained from LEAD-CNN using a combination of hard labels and soft teacher logits.

### Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, Cohen's kappa, ROC-AUC (KD loss uses α=0.6, γ=0.4, T=4; no feature term)
- **Visualizations**: Confusion matrices, GradCAM attention maps
- **Architecture Diagrams**: Model structure visualization

### Reproducibility
- **Fixed Seeds**: Deterministic training across runs
- **Stratified Splits**: Balanced train/val/test splits
- **Configuration Files**: YAML-based experiment management

## 🧪 Ablation Studies

LightNet v1 ablation studies are implemented in train_lightnet.py and controlled by experiments/lightnet_ablation.yaml.

Current ablations include:
- **Squeeze and Excitation (SE) usage**: `use_se` in `{True, False}`
- **Channel multiplier**: `channel_multiplier` in `{0.5, 0.75, 1.0, 1.25}`
- **Dropout rate**: `dropout_rate` in `{0.1, 0.2, 0.3, 0.4}`

The script writes parameter counts and model sizes for each configuration to:

```bash
outputs/lightnet_ablation/reports/ablation_study.json
```

## 📊 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged
- **Cohen's Kappa**: Inter-rater agreement measure
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification analysis
- **GradCAM**: Visual attention analysis

## 🎨 Visualizations

- **Architecture Diagrams**: Model structure visualization
- **Confusion Matrices**: Classification performance analysis
- **GradCAM Heatmaps**: Model attention visualization
- **Training Curves**: Loss and accuracy progression
- **Comparison Tables**: Model performance comparison

## ⚠️ Limitations & Future Work

### Current Limitations
- **Data Diversity**: Limited to single dataset; multi-site validation needed
- **External Validation**: Cross-dataset evaluation on GI endoscopy data (optional)
- **Deployment**: No full on-device deployment pipeline in this repository
- **Leakage Prevention**: Improved split protocols for clinical deployment

### Future Work
- **Multi-site Validation**: Cross-dataset and multi-site validation on other brain MRI datasets
- **Real-time Inference**: Model optimization for clinical deployment
- **Uncertainty Quantification**: Bayesian approaches for confidence estimation
- **Federated Learning**: Privacy-preserving multi-site training

## 🛠️ Requirements

- **Python**: 3.10–3.12
- **TensorFlow**: 2.12+ (Apple Metal supported)
- **Keras**: 2.12+
- **scikit-learn**: 1.3+
- **pandas**: 2.0+
- **matplotlib**: 3.7+
- **seaborn**: 0.12+
- **tensorflow-model-optimization**: 0.7+

Install with:
```bash
pip install -r requirements.txt
```

## 📝 Usage Examples

### Quick Start Example
```python
# Run the quick start example
python examples/quick_start.py
```

### Custom Training
```python
from src.models.lead_cnn import create_lead_cnn
from src.data.transforms import create_data_generators

# Create model
model = create_lead_cnn()

# Load data
datasets = create_data_generators("src/data/splits.json")

# Train model
model.model.fit(datasets["train"], validation_data=datasets["val"])
```

### Custom Metrics
```python
from src.eval.metrics import ClassificationMetrics

metrics_calc = ClassificationMetrics()
metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)

print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
print(f"Macro F1: {metrics['f1_macro']:.3f}")
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{brain_tumor_lightnet,
  title={Brain Tumor Lightweight Classifier},
  author={Siri Hegde},
  year={2025},
  url={https://github.com/Sirihegde02/Brain-Tumor-Classification}
}
```

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples in the `examples/` directory
