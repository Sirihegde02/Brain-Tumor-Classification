"""
Knowledge Distillation loss functions

Implements soft target distillation and feature distillation losses
for training lightweight models with teacher guidance.
"""
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, List, Optional


def safe_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Graph-safe categorical crossentropy that handles dynamic shapes."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)


def safe_kl_divergence(p: tf.Tensor, q: tf.Tensor) -> tf.Tensor:
    """Graph-safe KL divergence KL(p || q) for batched distributions."""
    p = tf.cast(p, tf.float32)
    q = tf.cast(q, tf.float32)
    p = tf.clip_by_value(p, 1e-7, 1.0)
    q = tf.clip_by_value(q, 1e-7, 1.0)
    return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)


class SoftTargetDistillationLoss(keras.losses.Loss):
    """
    Soft target distillation loss using KL divergence
    
    Computes KL divergence between teacher and student softened predictions.
    """
    
    def __init__(self,
                 temperature: float = 3.0,
                 alpha: float = 0.7,
                 name: str = "soft_target_distillation",
                 **kwargs):
        """
        Initialize soft target distillation loss
        
        Args:
            temperature: Temperature for softening predictions
            alpha: Weight for soft target loss (1-alpha for hard target)
            name: Loss name
        """
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.num_classes = None
    
    def call(self, y_true, y_pred):
        """
        Compute soft target distillation loss
        
        Args:
            y_true: Unused (kept for API compatibility)
            y_pred: Dictionary with 'student' and 'teacher' predictions
            
        Returns:
            KL divergence loss
        """
        student_pred = y_pred['student']
        teacher_pred = y_pred['teacher']
        
        # Soft target loss (KL divergence)
        # Soften teacher predictions
        teacher_soft = tf.nn.softmax(teacher_pred / self.temperature)
        student_soft = tf.nn.softmax(student_pred / self.temperature)
        
        # KL divergence: KL(teacher_soft || student_soft)
        kl_loss = safe_kl_divergence(teacher_soft, student_soft)
        
        # Scale KL loss by temperature squared
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return tf.reduce_mean(kl_loss)
    
    def get_config(self):
        """Get loss configuration"""
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'alpha': self.alpha
        })
        return config


class FeatureDistillationLoss(keras.losses.Loss):
    """
    Feature distillation loss using L2 distance between feature maps
    
    Distills knowledge from intermediate teacher features to student features.
    """
    
    def __init__(self,
                 feature_layers: List[str],
                 weights: Optional[List[float]] = None,
                 name: str = "feature_distillation",
                 **kwargs):
        """
        Initialize feature distillation loss
        
        Args:
            feature_layers: List of layer names to distill
            weights: Weights for each feature layer
            name: Loss name
        """
        super().__init__(name=name, **kwargs)
        self.feature_layers = feature_layers
        self.weights = weights or [1.0] * len(feature_layers)
        
        if len(self.weights) != len(self.feature_layers):
            raise ValueError("Number of weights must match number of feature layers")
    
    def call(self, y_true, y_pred):
        """
        Compute feature distillation loss
        
        Args:
            y_true: True labels (not used in feature distillation)
            y_pred: Dictionary with teacher and student features
            
        Returns:
            Feature distillation loss
        """
        teacher_features = y_pred['teacher_features']
        student_features = y_pred['student_features']
        
        total_loss = 0.0
        
        for i, layer_name in enumerate(self.feature_layers):
            if layer_name in teacher_features and layer_name in student_features:
                teacher_feat = teacher_features[layer_name]
                student_feat = student_features[layer_name]
                
                # Ensure same spatial dimensions
                if teacher_feat.shape[1:3] != student_feat.shape[1:3]:
                    # Resize student features to match teacher
                    student_feat = tf.image.resize(
                        student_feat, 
                        teacher_feat.shape[1:3],
                        method='bilinear'
                    )
                
                # L2 loss between feature maps
                l2_loss = tf.reduce_mean(tf.square(teacher_feat - student_feat))
                total_loss += self.weights[i] * l2_loss
        
        return total_loss
    
    def get_config(self):
        """Get loss configuration"""
        config = super().get_config()
        config.update({
            'feature_layers': self.feature_layers,
            'weights': self.weights
        })
        return config


class CombinedDistillationLoss(keras.losses.Loss):
    """
    Combined knowledge distillation loss
    
    Combines soft target distillation and feature distillation.
    """
    
    def __init__(self,
                 temperature: float = 3.0,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2,
                 feature_layers: Optional[List[str]] = None,
                 feature_weights: Optional[List[float]] = None,
                 name: str = "combined_distillation",
                 **kwargs):
        """
        Initialize combined distillation loss
        
        Args:
            temperature: Temperature for soft targets
            alpha: Weight for soft target loss
            beta: Weight for feature distillation loss
            gamma: Weight for hard target loss
            feature_layers: Layers for feature distillation
            feature_weights: Weights for feature layers
            name: Loss name
        """
        super().__init__(name=name, **kwargs)
        
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize component losses
        self.soft_target_loss = SoftTargetDistillationLoss(
            temperature=temperature,
            alpha=1.0,  # Will be weighted by alpha
            name="soft_target"
        )
        
        if feature_layers:
            self.feature_loss = FeatureDistillationLoss(
                feature_layers=feature_layers,
                weights=feature_weights,
                name="feature_distillation"
            )
        else:
            self.feature_loss = None
    
    def call(self, y_true, y_pred):
        """
        Compute combined distillation loss
        
        Args:
            y_true: True labels
            y_pred: Dictionary with predictions and features
            
        Returns:
            Combined loss value
        """
        student_logits = y_pred['student']
        teacher_logits = y_pred['teacher']
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(student_logits)[-1]
        labels_onehot = tf.one_hot(labels, depth=num_classes)

        # Soft KD term
        t = self.temperature
        teacher_soft = tf.nn.softmax(teacher_logits / t)
        student_soft = tf.nn.softmax(student_logits / t)
        soft_loss = safe_kl_divergence(teacher_soft, student_soft) * (t ** 2)

        # Hard CE term (on student probabilities)
        hard_loss = tf.keras.losses.categorical_crossentropy(labels_onehot, student_soft)

        # No feature loss (beta typically 0 in configs)
        total_loss = self.alpha * soft_loss + self.gamma * hard_loss

        return tf.reduce_mean(total_loss)
    
    def get_config(self):
        """Get loss configuration"""
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        })
        return config


class DistillationModel(keras.Model):
    """
    Wrapper model for knowledge distillation
    
    Handles teacher and student models with feature extraction.
    """
    
    def __init__(self,
                 teacher_model: keras.Model,
                 student_model: keras.Model,
                 feature_layers: Optional[List[str]] = None,
                 name: str = "distillation_model",
                 **kwargs):
        """
        Initialize distillation model
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            feature_layers: Layers to extract features from
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.feature_layers = feature_layers or []
        
        # Freeze teacher model
        self.teacher_model.trainable = False
        
        # Create feature extraction models
        self._create_feature_extractors()
        # Placeholder for loss set at compile time
        self.distillation_loss = None
    
    def _create_feature_extractors(self):
        """Create feature extraction models for teacher and student"""
        self.teacher_feature_extractor = None
        self.student_feature_extractor = None
        
        if self.feature_layers:
            # Teacher feature extractor
            teacher_outputs = []
            for layer_name in self.feature_layers:
                layer = self.teacher_model.get_layer(layer_name)
                teacher_outputs.append(layer.output)
            
            self.teacher_feature_extractor = keras.Model(
                inputs=self.teacher_model.input,
                outputs=teacher_outputs,
                name="teacher_feature_extractor"
            )
            
            # Student feature extractor
            student_outputs = []
            for layer_name in self.feature_layers:
                layer = self.student_model.get_layer(layer_name)
                student_outputs.append(layer.output)
            
            self.student_feature_extractor = keras.Model(
                inputs=self.student_model.input,
                outputs=student_outputs,
                name="student_feature_extractor"
            )
    
    def compile(self, optimizer, loss, metrics=None, **kwargs):
        """Custom compile to store distillation loss and metrics."""
        super().compile(optimizer=optimizer, metrics=metrics or [], **kwargs)
        self.distillation_loss = loss
    
    def _forward(self, inputs, training):
        """Compute teacher/student preds and optional features."""
        teacher_pred = self.teacher_model(inputs, training=False)
        student_pred = self.student_model(inputs, training=training)
        
        teacher_features = {}
        student_features = {}
        
        if self.teacher_feature_extractor and self.student_feature_extractor:
            teacher_feat_outputs = self.teacher_feature_extractor(inputs, training=False)
            student_feat_outputs = self.student_feature_extractor(inputs, training=training)
            
            for i, layer_name in enumerate(self.feature_layers):
                teacher_features[layer_name] = teacher_feat_outputs[i]
                student_features[layer_name] = student_feat_outputs[i]
        
        return {
            'teacher': teacher_pred,
            'student': student_pred,
            'teacher_features': teacher_features,
            'student_features': student_features
        }
    
    def train_step(self, data):
        """Custom training step for distillation."""
        inputs, y_true = data
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        with tf.GradientTape() as tape:
            preds = self._forward(inputs, training=True)
            loss = self.distillation_loss(labels, preds)
            # Add potential regularization losses
            if self.losses:
                loss += tf.add_n(self.losses)
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        # Update metrics on student predictions
        self.compiled_metrics.update_state(labels, preds['student'])
        
        results = {'loss': loss}
        for m in self.metrics:
            results[m.name] = m.result()
        return results
    
    def test_step(self, data):
        """Custom evaluation step for distillation."""
        inputs, y_true = data
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        preds = self._forward(inputs, training=False)
        loss = self.distillation_loss(labels, preds)
        if self.losses:
            loss += tf.add_n(self.losses)
        
        self.compiled_metrics.update_state(labels, preds['student'])
        
        results = {'loss': loss}
        for m in self.metrics:
            results[m.name] = m.result()
        return results
    
    def call(self, inputs, training=None):
        """Forward pass used for inference; returns student predictions."""
        return self.student_model(inputs, training=training)
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'feature_layers': self.feature_layers
        })
        return config


def create_distillation_loss(temperature: float = 3.0,
                           alpha: float = 0.5,
                           beta: float = 0.3,
                           gamma: float = 0.2,
                           feature_layers: Optional[List[str]] = None) -> CombinedDistillationLoss:
    """
    Create combined distillation loss
    
    Args:
        temperature: Temperature for soft targets
        alpha: Weight for soft target loss
        beta: Weight for feature distillation loss
        gamma: Weight for hard target loss
        feature_layers: Layers for feature distillation
        
    Returns:
        Combined distillation loss
    """
    return CombinedDistillationLoss(
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        feature_layers=feature_layers
    )


if __name__ == "__main__":
    # Test distillation losses
    print("Testing knowledge distillation losses...")
    
    # Create dummy models
    teacher = keras.Sequential([
        keras.layers.Dense(10, activation='relu', name='teacher_dense1'),
        keras.layers.Dense(4, activation='softmax', name='teacher_output')
    ])
    
    student = keras.Sequential([
        keras.layers.Dense(5, activation='relu', name='student_dense1'),
        keras.layers.Dense(4, activation='softmax', name='student_output')
    ])
    
    # Test soft target loss
    soft_loss = SoftTargetDistillationLoss(temperature=3.0, alpha=0.7)
    print(f"Soft target loss created: {soft_loss.name}")
    
    # Test combined loss
    combined_loss = create_distillation_loss(
        temperature=3.0,
        alpha=0.5,
        beta=0.3,
        gamma=0.2
    )
    print(f"Combined loss created: {combined_loss.name}")
    
    print("Knowledge distillation loss tests completed!")
