from setuptools import setup, find_packages

setup(
    name="uni3d",
    version="0.1.0",
    packages=find_packages(include=["uni3d", "uni3d.*"]),
    install_requires=[
        "timm>=0.9.7",
        "einops",
        "plyfile",
        "trimesh",
    ],
)