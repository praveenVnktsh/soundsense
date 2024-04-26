# This is an auto generated Dockerfile for ros:robot
# generated from docker_images/create_ros_image.Dockerfile.em
FROM ros:noetic-ros-base-focal

RUN apt-get update && apt-get install -y curl
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

# ARG distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
RUN curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
RUN sudo apt-get update
RUN sudo apt-get install -y nvidia-docker2
RUN sudo apt-get install -y nvidia-container-toolkit
RUN sudo apt-get install -y ros-noetic-cv-bridge 

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-robot=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip


RUN pip install pytorch-lightning
RUN pip install torchvision

RUN apt-get --yes install libsndfile1
RUN pip install soundfile

RUN pip install transformers
RUN pip install torchaudio
RUN pip install opencv-python
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install matplotlib
RUN pip install albumentations
RUN pip install configargparse

RUN apt-get install -y ros-noetic-audio-common