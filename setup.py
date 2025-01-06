import os

from setuptools import find_packages, setup

# Package metadata
NAME = "Cloaking-Daniel"
VERSION = "1.0"
DESCRIPTION = "Cloaking images"
URL = "https://github.com/vickybudhiraja/cloaking-daniel"
AUTHOR = "Daniel"
AUTHOR_EMAIL = "helpnet.in.vicky@gmail.com"
LICENSE = "MIT"

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Required dependencies
REQUIRED_PACKAGES = [
        "numpy>=1.19.5",
        #"intel-tensorflow",
        "keras",
        "mtcnn",
        "pillow>=7.0.0",
        "bleach>=2.1.0",
        "matplotlib",
        "onnxruntime",
        "onnx",
        # "onnx",
        #"onnx2keras",
        "mediapipe",
        "tqdm",
        "scikit-learn",
        # "tensorflow==2.15.1",
        "tensorflow",
        "tblib",
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.0",
)