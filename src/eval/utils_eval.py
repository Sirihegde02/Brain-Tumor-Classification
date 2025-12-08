"""
Shared evaluation utilities for student/teacher models.
"""
from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras import metrics as keras_metrics, losses as keras_losses, optimizers as keras_optimizers


def evaluate_student_on_test(model: tf.keras.Model, datagens: Dict[str, Any], model_name: str = "LightNetV2_KD") -> Dict[str, float]:
    """
    Evaluate a student model on the test split using Keras metrics attached to the model.
    
    Args:
        model: Keras model to evaluate.
        datagens: Dict with "test" dataset (from create_data_generators).
        model_name: Label for printing.
    
    Returns:
        Dict of metric name -> value.
    """
    test_dataset = datagens["test"]

    # Ensure model is compiled before evaluating.
    # Some callers build a fresh LightNetV2 and just load weights (no compile).
    if not hasattr(model, "optimizer") or model.optimizer is None:
        model.compile(
            optimizer=keras_optimizers.Adam(),
            loss=keras_losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[
                keras_metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras_metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
            ],
        )

    results = model.evaluate(test_dataset, verbose=0)

    metrics = {}
    for name, value in zip(model.metrics_names, results):
        metrics[name] = float(value)

    print("============================================================")
    print("CLASSIFICATION METRICS")
    print("============================================================")
    print(f"{model_name} raw Keras metrics: {metrics}")

    return metrics
