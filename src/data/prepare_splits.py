"""
Prepare stratified train/validation/test splits for Brain Tumor MRI dataset
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
from collections import Counter


def collect_image_paths(data_dir="data/raw"):
    """
    Collect all image paths and their corresponding labels
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        tuple: (image_paths, labels, class_names)
    """
    data_path = Path(data_dir)
    
    # Define class mapping (accept both spellings and normalize to no_tumor)
    normalized_classes = ["glioma", "meningioma", "pituitary", "no_tumor"]
    alias_to_normalized = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "pituitary": "pituitary",
        "no_tumor": "no_tumor",
        "notumor": "no_tumor"
    }
    class_mapping = {name: idx for idx, name in enumerate(normalized_classes)}
    
    image_paths = []
    labels = []
    class_names = normalized_classes[:]
    
    print("Collecting image paths...")

    # Detect Kaggle nested layout (Training/ and Testing/ under data_dir)
    nested_training = data_path / "Training"
    nested_testing = data_path / "Testing"
    use_nested = nested_training.exists() and nested_testing.exists()

    if use_nested:
        print("Detected Kaggle nested layout (Training/ and Testing/). Combining all images before stratification.")
        per_class_counts = {}
        for alias, norm_name in alias_to_normalized.items():
            # Aggregate from both Training and Testing for each alias spellings
            for split_dir in [nested_training, nested_testing]:
                class_dir = split_dir / alias
                if not class_dir.exists():
                    continue
                image_files = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    image_files.extend(class_dir.glob(ext))
                for img_path in image_files:
                    image_paths.append(str(img_path.relative_to(Path.cwd())))
                    labels.append(class_mapping[norm_name])
                    per_class_counts[norm_name] = per_class_counts.get(norm_name, 0) + 1
        # Print summary per normalized class
        for norm_name in class_names:
            print(f"Found {per_class_counts.get(norm_name, 0)} images in {norm_name}")
    else:
        # Flat layout: data_dir/<class>/*
        for alias, norm_name in alias_to_normalized.items():
            class_dir = data_path / alias
            if not class_dir.exists():
                continue
            # Get all image files
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(class_dir.glob(ext))
            
            print(f"Found {len(image_files)} images in {norm_name} (from '{alias}')")
            
            for img_path in image_files:
                image_paths.append(str(img_path.relative_to(Path.cwd())))
                labels.append(class_mapping[norm_name])
    
    print(f"Total images collected: {len(image_paths)}")
    print(f"Class distribution: {Counter(labels)}")
    if len(image_paths) == 0:
        print("No images found! Please verify your --data_dir path and layout.")
    
    return image_paths, labels, class_names


def create_stratified_splits(image_paths, labels, train_ratio=0.65, val_ratio=0.15, test_ratio=0.20, random_state=42):
    """
    Create stratified train/validation/test splits
    
    Args:
        image_paths (list): List of image file paths
        labels (list): List of corresponding labels
        train_ratio (float): Proportion for training set
        val_ratio (float): Proportion for validation set  
        test_ratio (float): Proportion for test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Split indices and metadata
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, 
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: separate train and validation from remaining data
    val_size = val_ratio / (train_ratio + val_ratio)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Create split indices
    splits = {
        "train": {
            "indices": np.where(np.isin(np.arange(len(image_paths)), 
                                      np.where(np.isin(image_paths, X_train))[0]))[0].tolist(),
            "paths": X_train.tolist(),
            "labels": y_train.tolist()
        },
        "val": {
            "indices": np.where(np.isin(np.arange(len(image_paths)), 
                                      np.where(np.isin(image_paths, X_val))[0]))[0].tolist(),
            "paths": X_val.tolist(), 
            "labels": y_val.tolist()
        },
        "test": {
            "indices": np.where(np.isin(np.arange(len(image_paths)), 
                                      np.where(np.isin(image_paths, X_test))[0]))[0].tolist(),
            "paths": X_test.tolist(),
            "labels": y_test.tolist()
        }
    }
    
    # Add metadata
    splits["metadata"] = {
        "total_images": len(image_paths),
        "num_classes": len(np.unique(labels)),
        "class_names": ["glioma", "meningioma", "pituitary", "no_tumor"],
        "train_ratio": train_ratio,
        "val_ratio": val_ratio, 
        "test_ratio": test_ratio,
        "random_state": random_state
    }
    
    # Print statistics
    print(f"\nSplit Statistics:")
    print(f"Total images: {len(image_paths)}")
    print(f"Train: {len(X_train)} ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"Validation: {len(X_val)} ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"Test: {len(X_test)} ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    for split_name, split_data in splits.items():
        if split_name == "metadata":
            continue
        print(f"\n{split_name.capitalize()} class distribution:")
        unique, counts = np.unique(split_data["labels"], return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = splits["metadata"]["class_names"][class_id]
            print(f"  {class_name}: {count} ({count/len(split_data['labels'])*100:.1f}%)")
    
    return splits


def save_splits(splits, output_file="data/splits.json"):
    """
    Save split information to JSON file
    
    Args:
        splits (dict): Split data and metadata
        output_file (str): Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to: {output_path}")


def create_csv_splits(splits, output_dir="data"):
    """
    Create CSV files for each split
    
    Args:
        splits (dict): Split data and metadata
        output_dir (str): Output directory for CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        if split_name == "metadata":
            continue
            
        df = pd.DataFrame({
            "image_path": split_data["paths"],
            "label": split_data["labels"],
            "class_name": [splits["metadata"]["class_names"][label] for label in split_data["labels"]]
        })
        
        csv_file = output_path / f"{split_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Saved {split_name} split to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare stratified data splits")
    parser.add_argument("--data_dir", default="data/raw", help="Path to dataset directory")
    parser.add_argument("--output", default="data/splits.json", help="Output file for splits")
    parser.add_argument("--train_ratio", type=float, default=0.65, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.20, help="Test set ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--create_csv", action="store_true", help="Also create CSV files")
    
    args = parser.parse_args()
    
    # Collect image paths and labels
    image_paths, labels, class_names = collect_image_paths(args.data_dir)
    
    if len(image_paths) == 0:
        print("No images found! Please check your data directory.")
        return
    
    # Create stratified splits
    splits = create_stratified_splits(
        image_paths, labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    # Save splits
    save_splits(splits, args.output)
    
    if args.create_csv:
        create_csv_splits(splits, Path(args.output).parent)


if __name__ == "__main__":
    main()
