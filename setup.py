from setuptools import setup, find_packages

setup(
    name="diffusion_backbone",
    version="0.1.0",
    packages=find_packages('..'),   # or specify subfolders
    package_dir={"diffusion_backbone": "."},               # interpret current dir as top-level
)