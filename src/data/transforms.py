"""
Data augmentation and preprocessing transforms for Brain Tumor MRI dataset
"""
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any


def preprocess_tensor(x, augment=False, target_size=(224, 224)):
    """
    Takes a float32 tensor in [0,1], resizes, optional augment, returns (img, label_dummy).
    
    Args:
        x: Input tensor in [0, 1] range
        augment: Whether to apply augmentation
        target_size: Target image size (height, width)
        
    Returns:
        Tuple of (processed_image, dummy_label)
    """
    x = tf.image.resize(x, target_size)
    
    if augment:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
    
    return x, -1


class BrainTumorTransforms:
    """
    Data augmentation and preprocessing transforms for brain tumor MRI images
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 augment: bool = False,
                 augmentation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize transforms
        
        Args:
            image_size: Target image size (height, width)
            normalize: Whether to normalize images
            augment: Whether to apply augmentation
            augmentation_config: Configuration for augmentation parameters
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Default augmentation config
        self.aug_config = {
            "horizontal_flip": True,
            "vertical_flip": False,  # Usually not good for medical images
            "rotation_range": 15.0,  # degrees
            "zoom_range": 0.1,
            "brightness_range": 0.1,
            "contrast_range": 0.1,
            "noise_std": 0.01
        }
        
        if augmentation_config:
            self.aug_config.update(augmentation_config)
    
    def preprocess_image(self, image_path: str, label: int, allow_dummy: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess a single image
        
        Args:
            image_path: Path to image file
            label: Image label
            allow_dummy: If True and file doesn't exist, create dummy image
            
        Returns:
            Tuple of (processed_image, label)
        """
        # Check if file exists
        file_exists = tf.io.gfile.exists(image_path)
        
        # If file doesn't exist and allow_dummy is True, create dummy image
        if not file_exists and allow_dummy:
            image = tf.random.uniform((224, 224, 3), 0, 1, dtype=tf.float32)
            # Resize to target size (in case self.image_size is different)
            image = tf.image.resize(image, self.image_size)
        else:
            # Load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.cast(image, tf.float32)
            
            # Resize
            image = tf.image.resize(image, self.image_size)
            
            # Normalize to [0, 1]
            image = image / 255.0
        
        # Apply augmentation if enabled
        if self.augment:
            image = self._apply_augmentation(image)
        
        # Final normalization
        if self.normalize:
            image = self._normalize_image(image)
        
        return image, tf.cast(label, tf.int32)
    
    def _apply_augmentation(self, image: tf.Tensor) -> tf.Tensor:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image tensor
            
        Returns:
            Augmented image tensor
        """
        # Random horizontal flip
        if self.aug_config["horizontal_flip"]:
            image = tf.image.random_flip_left_right(image)
        
        # Random rotation
        if self.aug_config["rotation_range"] > 0:
            angle = tf.random.uniform([], 
                                    -self.aug_config["rotation_range"], 
                                    self.aug_config["rotation_range"])
            image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))
        
        # Random zoom
        if self.aug_config["zoom_range"] > 0:
            zoom_factor = tf.random.uniform([], 
                                         1 - self.aug_config["zoom_range"],
                                         1 + self.aug_config["zoom_range"])
            image = tf.image.resize_with_crop_or_pad(image, 
                                                   tf.cast(tf.shape(image)[0] * zoom_factor, tf.int32),
                                                   tf.cast(tf.shape(image)[1] * zoom_factor, tf.int32))
            image = tf.image.resize(image, self.image_size)
        
        # Random brightness
        if self.aug_config["brightness_range"] > 0:
            delta = tf.random.uniform([], 
                                    -self.aug_config["brightness_range"],
                                    self.aug_config["brightness_range"])
            image = tf.image.adjust_brightness(image, delta)
        
        # Random contrast
        if self.aug_config["contrast_range"] > 0:
            contrast_factor = tf.random.uniform([], 
                                              1 - self.aug_config["contrast_range"],
                                              1 + self.aug_config["contrast_range"])
            image = tf.image.adjust_contrast(image, contrast_factor)
        
        # Add noise
        if self.aug_config["noise_std"] > 0:
            noise = tf.random.normal(tf.shape(image), 0, self.aug_config["noise_std"])
            image = image + noise
        
        # Ensure values are in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    def _normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        """
        Normalize image using ImageNet statistics
        
        Args:
            image: Input image tensor
            
        Returns:
            Normalized image tensor
        """
        # ImageNet normalization
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        
        image = (image - mean) / std
        
        return image
    
    def create_dataset(self, 
                      image_paths: list, 
                      labels: list, 
                      batch_size: int = 32,
                      shuffle: bool = True,
                      buffer_size: Optional[int] = None) -> tf.data.Dataset:
        """
        Create a TensorFlow dataset from image paths and labels
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            batch_size: Batch size for dataset
            shuffle: Whether to shuffle the dataset
            buffer_size: Buffer size for shuffling
            
        Returns:
            TensorFlow dataset
        """
        if buffer_size is None:
            buffer_size = len(image_paths)
        
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        
        # Map preprocessing function
        dataset = dataset.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


def create_data_generators(splits_file: str = "data/splits.json",
                          batch_size: int = 32,
                          image_size: Tuple[int, int] = (224, 224),
                          augmentation_config: Optional[Dict[str, Any]] = None) -> Dict[str, tf.data.Dataset]:
    """
    Create data generators for train, validation, and test sets
    
    Args:
        splits_file: Path to splits JSON file
        batch_size: Batch size for datasets
        image_size: Target image size
        augmentation_config: Augmentation configuration
        
    Returns:
        Dictionary containing train, val, and test datasets
    """
    import json
    
    # Load splits
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    datasets = {}
    
    for split_name in ["train", "val", "test"]:
        if split_name == "train":
            # Use augmentation for training
            transforms = BrainTumorTransforms(
                image_size=image_size,
                normalize=True,
                augment=True,
                augmentation_config=augmentation_config
            )
            shuffle = True
        else:
            # No augmentation for validation/test
            transforms = BrainTumorTransforms(
                image_size=image_size,
                normalize=True,
                augment=False
            )
            shuffle = False
        
        # Create dataset
        dataset = transforms.create_dataset(
            image_paths=splits[split_name]["paths"],
            labels=splits[split_name]["labels"],
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        datasets[split_name] = dataset
        
        print(f"Created {split_name} dataset: {len(splits[split_name]['paths'])} samples")
    
    return datasets


def get_class_weights(splits_file: str = "data/splits.json") -> Dict[int, float]:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        splits_file: Path to splits JSON file
        
    Returns:
        Dictionary mapping class indices to weights
    """
    import json
    from collections import Counter
    
    # Load splits
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Get training labels
    train_labels = splits["train"]["labels"]
    
    # Count class frequencies
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    num_classes = len(class_counts)
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for class_id, count in class_counts.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_id] = weight
    
    print("Class weights:")
    for class_id, weight in class_weights.items():
        class_name = splits["metadata"]["class_names"][class_id]
        print(f"  {class_name} (class {class_id}): {weight:.3f}")
    
    return class_weights


if __name__ == "__main__":
    # Test the transforms
    import json
    
    # Create test data
    test_paths = ["test1.jpg", "test2.jpg"]
    test_labels = [0, 1]
    
    # Test without augmentation
    transforms = BrainTumorTransforms(augment=False)
    dataset = transforms.create_dataset(test_paths, test_labels, batch_size=2)
    
    print("Test dataset created successfully!")
    print(f"Dataset element spec: {dataset.element_spec}")
