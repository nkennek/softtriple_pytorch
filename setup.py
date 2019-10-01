from setuptools import find_packages, setup

setup(
    name="softtriple_pytorch",
    version="0.1.0",
    description="Unofficial implementation of \
        `SoftTriple Loss: Deep Metric Learning Without Triplet Sampling`",
    author="Kenichi Nakahara",
    packages=find_packages(),
    install_requires=[],  # note: you need torch but pip-install is not necessary
)
