"""
Setup script for Brain Tumor Lightweight Classifier
"""
from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="brain-tumor-lightnet",
    version="1.0.0",
    author="Brain Tumor Classification Team",
    author_email="team@example.com",
    description="Lightweight CNN for brain tumor classification with knowledge distillation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/brain-tumor-lightnet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-tumor-train=src.train.train_baseline:main",
            "brain-tumor-lightnet=src.train.train_lightnet:main",
            "brain-tumor-kd=src.train.train_kd:main",
            "brain-tumor-eval=src.eval.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
