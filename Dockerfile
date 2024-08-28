ARG TAG=0.0.1
ARG DEBIAN_FRONTEND=noninteractive

# This is an auto generated Dockerfile for ros:ros-base
# generated from docker_images/create_ros_image.Dockerfile.em
FROM ros:noetic-ros-core-focal

###################################################################################################
# Set up guest user  and tools for development
ARG DEV_GROUP_ID=8888
ARG ID_NAME=guest
ENV ID_NAME=${ID_NAME}
# Create default users
RUN groupadd -g ${DEV_GROUP_ID} developers \
    && useradd -ms /bin/bash $ID_NAME \
    && usermod -aG sudo $ID_NAME \
    && usermod -aG developers $ID_NAME \
    && echo "${ID_NAME}:${ID_NAME}" | chpasswd
USER $ID_NAME
WORKDIR /home/${ID_NAME}
RUN mkdir /home/${ID_NAME}/.ssh

###################################################################################################
# Build the image as root from the / folder
USER root
WORKDIR /

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    python3-vcstool \
    python3-catkin-tools \
    git \
    openssh-client \
    python3-pip \
    libeigen3-dev \
    libzmqpp-dev \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    ros-noetic-tf2-eigen \
    ros-noetic-interactive-markers \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-tf2-ros \
    ros-noetic-image-proc \
    ros-noetic-depth-image-proc \
    ros-noetic-rviz \
    qtbase5-dev \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    ros-noetic-gtsam \
    libopencv-dev \
    libpcl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Update PATH environment variable to include Conda
ENV PATH /opt/conda/bin:$PATH
