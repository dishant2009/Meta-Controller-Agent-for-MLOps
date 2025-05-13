"""
Setup script for the Meta-Controller Agent for MLOps package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meta-controller-mlops",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-driven Meta-Controller Agent for MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/meta-controller-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "requests>=2.26.0",
        "python-dateutil>=2.8.2",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "jsonschema>=4.4.0",
        "openai>=0.27.0",
        "psutil>=5.9.0",
        "pyarrow>=7.0.0",
        "fastapi>=0.75.0",
        "uvicorn>=0.17.0",
        "prometheus-client>=0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=2.12.0",
            "black>=22.1.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meta-controller=meta_controller.main:main",
        ],
    },
)
