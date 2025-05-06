from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="jax-ml-template",
    version="0.1",
    packages=find_packages(),
    install_requires=required_packages,
)