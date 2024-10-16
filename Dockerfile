# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04
# FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Set the working directory to /app
WORKDIR /ws

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install git python3.8 python3.8-dev python3.8-venv python3-pip python3-tk python-is-python3 -y && rm -rf /var/lib/apt/lists/*


RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip3 install --upgrade pip

RUN pip3 install \
    numpy==1.23.5 \
    pillow \
    einops \
    typed-argument-parser \
    tqdm \
    transformers \
    absl-py \
    matplotlib \
    scipy

RUN pip3 install \
    tensorboard \
    opencv-python \
    blosc \
    setuptools==57.5.0 \
    beautifulsoup4 \
    bleach>=6.0.0 \
    defusedxml \
    jinja2>=3.0 \
    jupyter-core>=4.7 \
    jupyterlab-pygments \
    mistune==2.0.5 \
    nbclient>=0.5.0 \
    nbformat>=5.7 \
    pandocfilters>=1.4.1 \
    tinycss2 \
    traitlets>=5.1

RUN pip3 install diffusers["torch"]
RUN pip3 install dgl==1.1.3+cu116 -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp38-cp38-manylinux1_x86_64.whl
RUN pip3 install packaging
RUN pip3 install ninja==1.11.1.1
#RUN pip3 install flash-attn==2.6.1 --no-build-isolation
RUN pip3 install git+https://github.com/openai/CLIP.git@a9b1bf5
RUN pip3 install open3d

RUN apt update
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0
