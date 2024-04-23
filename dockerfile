# This is an auto generated Dockerfile for ros:robot
# generated from docker_images/create_ros_image.Dockerfile.em
FROM ros:noetic-ros-base-focal

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



