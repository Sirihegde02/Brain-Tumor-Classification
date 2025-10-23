"""
Download and setup Kaggle Brain Tumor MRI Dataset
"""
import os
import zipfile
import kaggle
from pathlib import Path
import argparse


def download_dataset(dataset_name="sartajbhuvaji/brain-tumor-classification-mri", 
                    output_dir="data/raw"):
    """
    Download the Brain Tumor MRI dataset from Kaggle
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        output_dir (str): Directory to save the dataset
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=output_dir, 
            unzip=True
        )
        print("Dataset downloaded successfully!")
        
        # List downloaded files
        data_path = Path(output_dir)
        print(f"\nDownloaded files in {data_path}:")
        for item in data_path.rglob("*"):
            if item.is_file():
                print(f"  {item.relative_to(data_path)}")
                
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have:")
        print("1. Kaggle API credentials in ~/.kaggle/kaggle.json")
        print("2. Accepted the dataset terms on Kaggle")
        raise


def setup_data_structure(data_dir="data/raw"):
    """
    Organize the downloaded data into a clean structure
    
    Args:
        data_dir (str): Path to raw data directory
    """
    data_path = Path(data_dir)
    
    # Expected class directories
    classes = ["glioma", "meningioma", "pituitary", "no_tumor"]
    
    print("Organizing data structure...")
    
    # Check if data is already organized
    if all((data_path / cls).exists() for cls in classes):
        print("Data already organized!")
        return
    
    # Look for the actual data structure in subdirectories
    for subdir in data_path.iterdir():
        if subdir.is_dir():
            print(f"Found subdirectory: {subdir.name}")
            # Check if this contains our class folders
            if any((subdir / cls).exists() for cls in classes):
                print(f"Found organized data in: {subdir}")
                # Move files to main directory
                for cls in classes:
                    src = subdir / cls
                    dst = data_path / cls
                    if src.exists():
                        if not dst.exists():
                            dst.mkdir(parents=True)
                        # Move files
                        for file in src.iterdir():
                            if file.is_file():
                                file.rename(dst / file.name)
                break
    
    # Verify final structure
    print("\nFinal data structure:")
    for cls in classes:
        cls_path = data_path / cls
        if cls_path.exists():
            count = len(list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")))
            print(f"  {cls}: {count} images")
        else:
            print(f"  {cls}: NOT FOUND")


def main():
    parser = argparse.ArgumentParser(description="Download Brain Tumor MRI Dataset")
    parser.add_argument("--dataset", default="sartajbhuvaji/brain-tumor-classification-mri",
                       help="Kaggle dataset identifier")
    parser.add_argument("--output", default="data/raw",
                       help="Output directory for dataset")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup data structure, don't download")
    
    args = parser.parse_args()
    
    if not args.setup_only:
        download_dataset(args.dataset, args.output)
    
    setup_data_structure(args.output)


if __name__ == "__main__":
    main()
