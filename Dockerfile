# Use an official Python runtime as a parent image
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM nvcr.io/nvidia/dgl:23.07-py3

# Set the working directory to /app
WORKDIR /ws

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --no-install-recommends \ 
    && apt-get install -y apt-utils 
# RUN apt install git python3.8 python3.8-dev python3.8-venv python3-pip python3-tk python-is-python3 -y && rm -rf /var/lib/apt/lists/*
RUN apt install git python3-pip python3-tk python-is-python3 -y && rm -rf /var/lib/apt/lists/*


# RUN pip3 install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

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
# RUN pip3 install dgl==1.1.3+cu116 -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp38-cp38-manylinux1_x86_64.whl

# RUN pip3 install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html

RUN pip3 install packaging
RUN pip3 install ninja
RUN pip3 install flash-attn --no-build-isolation
RUN pip3 install git+https://github.com/openai/CLIP.git@a9b1bf5
RUN pip3 install open3d

RUN apt update
RUN apt-get install -y libgl1-mesa-dev libglib2.0-0
RUN pip install opencv-python==4.8.0.74
# RUN apt install libcurl4


#ros2
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y \
   tzdata \
 && ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && apt-get clean 

RUN apt update && apt install locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

RUN apt-get install software-properties-common -y
RUN add-apt-repository universe
RUN apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN apt upgrade -y
RUN apt install ros-humble-desktop -y
RUN apt install ros-dev-tools -y
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN ldconfig