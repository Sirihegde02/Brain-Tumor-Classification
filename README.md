# Brain Tumor Lightweight Classifier (LEAD-CNN-inspired)

A comprehensive brain tumor classification system that reproduces the LEAD-CNN baseline and creates a 10× lighter model using knowledge distillation techniques.

## 🎯 Project Overview

This project implements a brain tumor classification system with the following goals:
- **Reproduce LEAD-CNN baseline** on Kaggle Brain Tumor MRI Dataset (4 classes: glioma, meningioma, pituitary, normal)
- **Create lightweight model** with ≤10% of LEAD-CNN's parameters (target: ≤113k params)
- **Implement knowledge distillation** for improved performance
- **Provide comprehensive evaluation** with Cohen's kappa and other metrics

## 📊 Dataset

- **Source**: [Kaggle "Brain Tumor MRI Dataset"](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
- **Classes**: 4 (glioma, meningioma, pituitary, normal)
- **Images**: ~7,023 total
- **Input size**: 224×224 RGB
- **Splits**: Train 65% / Val 15% / Test 20% (stratified)

## 🏗️ Project Structure

```
brain-tumor-lightnet/
├── README.md
├── requirements.txt
├── setup.py
├── Makefile
├── src/
│   ├── data/
│   │   ├── download_kaggle.py      # Dataset download
│   │   ├── prepare_splits.py       # Stratified data splitting
│   │   └── transforms.py           # Data augmentation
│   ├── models/
│   │   ├── lead_cnn.py             # LEAD-CNN implementation
│   │   ├── lightnet.py             # LightNet implementation
│   │   ├── blocks.py               # Custom blocks
│   │   └── kd_losses.py            # Knowledge distillation
│   ├── train/
│   │   ├── train_baseline.py       # LEAD-CNN training
│   │   ├── train_lightnet.py       # LightNet training
│   │   └── train_kd.py             # Knowledge distillation
│   ├── eval/
│   │   ├── evaluate.py             # Model evaluation
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── confusion.py            # Confusion matrix
│   ├── viz/
│   │   ├── plot_arch.py            # Architecture diagrams
│   │   └── gradcam.py              # GradCAM visualization
│   └── utils/
│       ├── seed.py                 # Reproducibility
│       ├── io.py                   # I/O utilities
│       └── params.py               # Parameter counting
├── experiments/
│   ├── baseline_leadcnn.yaml      # LEAD-CNN config
│   ├── lightnet_ablation.yaml     # LightNet config
│   └── kd.yaml                     # Knowledge distillation config
├── examples/
│   └── quick_start.py              # Quick start example
└── outputs/
    ├── checkpoints/                 # Model checkpoints
    ├── logs/                        # Training logs
    ├── figures/                    # Visualizations
    └── reports/                     # Evaluation reports
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd brain-tumor-lightnet

# Install dependencies
pip install -r requirements.txt

# Setup project
python -m pip install -e .
```

### 2. Data Preparation

```bash
# Download dataset (requires Kaggle API setup)
python src/data/download_kaggle.py

# Prepare stratified splits
python src/data/prepare_splits.py --create_csv
```

### 3. Training Models

```bash
# Train LEAD-CNN baseline
python src/train/train_baseline.py --config experiments/baseline_leadcnn.yaml

# Train LightNet
python src/train/train_lightnet.py --config experiments/lightnet_ablation.yaml

# Train with knowledge distillation
python src/train/train_kd.py --config experiments/kd.yaml --teacher_path outputs/checkpoints/lead_cnn_best.h5
```

### 4. Evaluation

```bash
# Evaluate all models
python src/eval/evaluate.py \
    --model_paths outputs/checkpoints/lead_cnn_best.h5 outputs/checkpoints/lightnet_best.h5 outputs/checkpoints/lightnet_kd_best.h5 \
    --model_names LEAD-CNN LightNet KD-LightNet \
    --compare --generate_gradcam
```

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

| Model | Params | FLOPs | Test Acc | F1 | Cohen κ |
|-------|--------|-------|----------|----|---------| 
| LEAD-CNN | ~1.13M | - | - | - | - |
| LightNet | ≤113k | - | - | - | - |
| KD-LightNet | ≤113k | - | - | - | - |

*Results will be populated after training*

## 🔧 Key Features

### Models
- **LEAD-CNN**: Faithful reproduction with dimension-reduction blocks and LeakyReLU
- **LightNet**: Lightweight architecture using depthwise-separable convolutions
- **Knowledge Distillation**: Soft target and feature distillation losses

### Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, Cohen's kappa, ROC-AUC
- **Visualizations**: Confusion matrices, GradCAM attention maps
- **Architecture Diagrams**: Model structure visualization

### Reproducibility
- **Fixed Seeds**: Deterministic training across runs
- **Stratified Splits**: Balanced train/val/test splits
- **Configuration Files**: YAML-based experiment management

## 🧪 Ablation Studies

The project includes comprehensive ablation studies for LightNet:
- **Squeeze-and-Excitation**: With/without SE blocks
- **Channel Multipliers**: Different channel width scaling
- **Dropout Rates**: Regularization analysis
- **Architecture Variants**: V1 vs V2 comparisons

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
- **Deployment**: Model optimization for on-device inference
- **Leakage Prevention**: Improved split protocols for clinical deployment

### Future Work
- **Multi-site Validation**: Cross-institutional dataset evaluation
- **Real-time Inference**: Model optimization for clinical deployment
- **Uncertainty Quantification**: Bayesian approaches for confidence estimation
- **Federated Learning**: Privacy-preserving multi-site training

## 🛠️ Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.12+
- **Keras**: 2.12+
- **scikit-learn**: 1.3+
- **pandas**: 2.0+
- **matplotlib**: 3.7+
- **seaborn**: 0.12+
- **tensorflow-model-optimization**: 0.7+

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
datasets = create_data_generators("data/splits.json")

# Train model
model.model.fit(datasets['train'], validation_data=datasets['val'])
```

### Model Evaluation
```python
from src.eval.metrics import ClassificationMetrics

# Calculate metrics
metrics_calc = ClassificationMetrics()
metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
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
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/brain-tumor-lightnet}
}
```

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the examples in the `examples/` directory
