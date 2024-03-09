import os

from setuptools import find_packages, setup

with open(os.path.join("flycraft", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="flycraft",
    description="A fixed-wing UAV environment based on gymnasium.",
    author="Xudong Gong",
    author_email="gongxudong_cs@aliyun.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gongxudong/fly-craft",
    packages=find_packages(),
    include_package_data=True,
    # package_data={"flycraft": ["version.txt"]},
    version=__version__,
    install_requires=["gymnasium>=0.26", "numpy", "pandas"],
    extras_require={
        "develop": ["pytest-cov", "black", "isort", "pytype", "sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
