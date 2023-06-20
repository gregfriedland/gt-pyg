from setuptools import setup, find_packages

setup(
    name="gt_pyg",
    description="Implementation of the Graph Transformer architecture in Pytorch-geometric",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=1.13.0",
        "torch_geometric",
        "numpy",
        "rdkit",
    ],
    include_package_data=True,
)
