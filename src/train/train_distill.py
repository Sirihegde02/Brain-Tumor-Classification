"""
Knowledge distillation training script.

Trains LightNetV2 as a student using a frozen LEAD-CNN teacher with a KD loss
that blends hard-label cross-entropy and soft-teacher KL divergence.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import yaml

# Ensure src is importable
sys.path.append(str(Path(__file__).parent.parent))

from data.transforms import create_data_generators, get_class_weights
from eval.metrics import ClassificationMetrics
from models.blocks import DimensionReductionBlock, LiteDRBlock, SqueezeExcitation
from models.lead_cnn import create_lead_cnn
from models.lightnet import build_lightnet_v2
from utils.io import (
    create_checkpoint_callback,
    create_tensorboard_callback,
    save_history,
    save_model,
    save_model_summary,
)
from utils.params import count_parameters
from utils.seed import set_seed


class Distiller(keras.Model):
    """Simple KD wrapper that trains a student under a frozen teacher."""

    def __init__(self, student, teacher, temperature=3.0, alpha=0.5):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy()
        self.distill_loss_fn = keras.losses.KLDivergence()
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distill_loss_tracker = keras.metrics.Mean(name="distill_loss")
        self.acc_metric = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.student_loss_tracker,
            self.distill_loss_tracker,
            self.acc_metric,
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "alpha": self.alpha,
        })
        return config

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)

    def train_step(self, data):
        sample_weight = None
        if isinstance(data, (list, tuple)) and len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
        teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            student_losses = tf.keras.losses.sparse_categorical_crossentropy(y, student_pred)
            if sample_weight is not None:
                student_loss = tf.reduce_sum(student_losses * sample_weight) / tf.reduce_sum(sample_weight)
            else:
                student_loss = tf.reduce_mean(student_losses)

            teacher_soft = tf.nn.softmax(teacher_pred / self.temperature)
            student_soft = tf.nn.softmax(student_pred / self.temperature)
            distill_losses = tf.keras.losses.kullback_leibler_divergence(teacher_soft, student_soft)
            distill_losses *= self.temperature ** 2
            if sample_weight is not None:
                distill_loss = tf.reduce_sum(distill_losses * sample_weight) / tf.reduce_sum(sample_weight)
            else:
                distill_loss = tf.reduce_mean(distill_losses)

            loss = (1 - self.alpha) * student_loss + self.alpha * distill_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.total_loss_tracker.update_state(loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distill_loss_tracker.update_state(distill_loss)
        self.acc_metric.update_state(y, student_pred, sample_weight=sample_weight)

        return {
            "loss": self.total_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distill_loss": self.distill_loss_tracker.result(),
            "accuracy": self.acc_metric.result(),
        }

    def test_step(self, data):
        sample_weight = None
        if isinstance(data, (list, tuple)) and len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
        teacher_pred = self.teacher(x, training=False)
        student_pred = self.student(x, training=False)

        student_losses = tf.keras.losses.sparse_categorical_crossentropy(y, student_pred)
        if sample_weight is not None:
            student_loss = tf.reduce_sum(student_losses * sample_weight) / tf.reduce_sum(sample_weight)
        else:
            student_loss = tf.reduce_mean(student_losses)

        teacher_soft = tf.nn.softmax(teacher_pred / self.temperature)
        student_soft = tf.nn.softmax(student_pred / self.temperature)
        distill_losses = tf.keras.losses.kullback_leibler_divergence(teacher_soft, student_soft)
        distill_losses *= self.temperature ** 2
        if sample_weight is not None:
            distill_loss = tf.reduce_sum(distill_losses * sample_weight) / tf.reduce_sum(sample_weight)
        else:
            distill_loss = tf.reduce_mean(distill_losses)

        loss = (1 - self.alpha) * student_loss + self.alpha * distill_loss

        self.total_loss_tracker.update_state(loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distill_loss_tracker.update_state(distill_loss)
        self.acc_metric.update_state(y, student_pred, sample_weight=sample_weight)

        return {
            "loss": self.total_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distill_loss": self.distill_loss_tracker.result(),
            "accuracy": self.acc_metric.result(),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Distill LEAD-CNN -> LightNetV2")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/lightnet_kd.yaml",
        help="KD configuration file",
    )
    parser.add_argument(
        "--splits_file", type=str, default="data/splits.json", help="Dataset splits"
    )
    parser.add_argument(
        "--teacher_path", type=str, required=True, help="Path to frozen teacher .h5"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lightnet_v2_kd",
        help="Where to store logs/checkpoints",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sanity_steps",
        type=int,
        default=0,
        help="Limit tf.data steps per split for debugging",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_teacher(teacher_path):
    custom_objects = {
        "DimensionReductionBlock": DimensionReductionBlock,
        "LiteDRBlock": LiteDRBlock,
        "SqueezeExcitation": SqueezeExcitation,
    }
    teacher = tf.keras.models.load_model(teacher_path, custom_objects=custom_objects)
    teacher.trainable = False
    return teacher


def build_optimizer(config):
    compile_cfg = config.get("compile", {})
    training_cfg = config.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer") or compile_cfg.get("optimizer") or {
        "type": "adam"
    }

    if isinstance(optimizer_cfg, str):
        optimizer_type = optimizer_cfg.lower()
        optimizer_params = {}
    else:
        optimizer_type = optimizer_cfg.get("type", "adam").lower()
        optimizer_params = dict(optimizer_cfg)

    lr = optimizer_params.get("learning_rate", compile_cfg.get("learning_rate", 5e-4))

    if optimizer_type == "adam":
        return keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=optimizer_params.get("beta_1", 0.9),
            beta_2=optimizer_params.get("beta_2", 0.999),
        )
    elif optimizer_type == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=optimizer_params.get("weight_decay", 0.0),
            beta_1=optimizer_params.get("beta_1", 0.9),
            beta_2=optimizer_params.get("beta_2", 0.999),
        )
    elif optimizer_type == "sgd":
        return keras.optimizers.SGD(
            learning_rate=lr,
            momentum=optimizer_params.get("momentum", 0.9),
            nesterov=optimizer_params.get("nesterov", True),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_callbacks(config, output_dir):
    callbacks = []
    training_cfg = config.get("training", {})

    checkpoint_path = Path(output_dir) / "checkpoints" / "lightnet_v2_kd_best.h5"
    callbacks.append(
        create_checkpoint_callback(
            filepath=checkpoint_path,
            monitor=training_cfg["monitor"],
            mode=training_cfg["monitor_mode"],
            save_best_only=True,
            save_weights_only=False,
        )
    )

    early_cfg = training_cfg.get("early_stopping", {})
    if early_cfg.get("enabled", True):
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=training_cfg["monitor"],
                mode=training_cfg["monitor_mode"],
                patience=early_cfg.get("patience", 10),
                restore_best_weights=True,
                verbose=1,
            )
        )

    lr_cfg = training_cfg.get("lr_scheduler", {})
    if lr_cfg.get("enabled", False) and lr_cfg.get("type") == "reduce_on_plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=training_cfg["monitor"],
                mode=training_cfg["monitor_mode"],
                factor=float(lr_cfg.get("factor", 0.5)),
                patience=int(lr_cfg.get("patience", 5)),
                min_lr=float(lr_cfg.get("min_lr", 1e-6)),
                verbose=1,
            )
        )

    if training_cfg.get("tensorboard", {}).get("enabled", True):
        tb_dir = Path(output_dir) / "logs" / "lightnet_v2_kd"
        callbacks.append(create_tensorboard_callback(tb_dir))

    return callbacks


def limit_dataset(ds, steps):
    if steps <= 0:
        return ds
    return ds.take(steps)


def save_history_csv(history, output_dir):
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    csv_path = Path(output_dir) / "history.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(csv_path, index=False)
    print(f"Saved KD history to {csv_path}")


def evaluate_student(student_model, test_data, output_dir, splits_file):
    print("\nEvaluating student on test data...")
    student_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    results = student_model.evaluate(test_data, verbose=1)
    print(f"Student test results: {dict(zip(student_model.metrics_names, results))}")

    # Collect predictions for richer metrics
    y_true = []
    y_pred_proba = []
    for bx, by in test_data:
        y_true.append(by.numpy())
        y_pred_proba.append(student_model.predict(bx, verbose=0))
    y_true = np.concatenate(y_true, axis=0)
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    with open(splits_file, "r") as f:
        splits = json.load(f)
    class_names = splits.get("metadata", {}).get("class_names")

    metrics_calc = ClassificationMetrics(class_names=class_names)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_pred_proba)
    metrics_calc.print_metrics(metrics)

    metrics_path = Path(output_dir) / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved KD metrics to {metrics_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = create_data_generators(
        splits_file=args.splits_file,
        batch_size=config["data"]["batch_size"],
        image_size=tuple(config["data"]["image_size"]),
        augmentation_config=config["data"].get("augmentation", {}),
    )
    train_data = limit_dataset(datasets["train"], args.sanity_steps)
    val_data = limit_dataset(datasets["val"], args.sanity_steps)
    test_data = limit_dataset(datasets["test"], args.sanity_steps)

    class_weights = None
    if config["training"].get("use_class_weights", False):
        class_weights = get_class_weights(args.splits_file)

    print("Loading teacher model...")
    teacher = load_teacher(args.teacher_path)
    print("Building student (LightNetV2)...")
    student = build_lightnet_v2(
        input_shape=tuple(config["model"]["input_shape"]),
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"].get("dropout_rate", 0.3),
    )

    print("Teacher parameters:", count_parameters(teacher)["total"])
    print("Student parameters:", count_parameters(student)["total"])

    distill_cfg = config.get("distillation", {})
    distiller = Distiller(
        student=student,
        teacher=teacher,
        temperature=distill_cfg.get("temperature", 3.0),
        alpha=distill_cfg.get("alpha", 0.5),
    )

    optimizer = build_optimizer(config)
    distiller.compile(optimizer=optimizer)

    callbacks = create_callbacks(config, output_dir)

    print("Starting KD training...")
    history = distiller.fit(
        train_data,
        validation_data=val_data,
        epochs=config["training"]["epochs"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_history(history, output_dir / "logs" / "lightnet_v2_kd_history.json")
    save_history_csv(history, output_dir)

    # Save student and distiller checkpoints
    student_ckpt = output_dir / "checkpoints" / "lightnet_v2_kd_student.h5"
    save_model(student, student_ckpt)
    distiller_ckpt = output_dir / "checkpoints" / "lightnet_v2_kd_distiller.h5"
    save_model(distiller, distiller_ckpt)
    summary_path = output_dir / "reports" / "lightnet_v2_kd_summary.txt"
    save_model_summary(student, summary_path)
    config_path = output_dir / "logs" / "lightnet_v2_kd_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    evaluate_student(student, test_data, output_dir, args.splits_file)
    print("Knowledge distillation completed.")


if __name__ == "__main__":
    main()
