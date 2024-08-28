#!/bin/bash

# Name of the Docker container
CONTAINER_NAME="ros-noetic-habitat-hydra-bridge"

# Docker image to use
#DOCKER_IMAGE="blakerbuchanan/ros-noetic-habitat-hydra-bridge:0.0.1"
DOCKER_IMAGE="ros-noetic-habitat-hydra-bridge-with-environment:latest"

# Path to the workspace directory
WORKSPACE_DIR="$(pwd)"

# Environment variables
DISPLAY_VAR=$DISPLAY
SSH_AUTH_SOCK_VAR=$SSH_AUTH_SOCK

# Run the Docker container with the appropriate arguments
docker run -it \
  --name $CONTAINER_NAME \
  --privileged \
  --gpus all \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
  --env="XAUTHORITY:$XAUTHORITY" \
  --env="QT_X11_NO_MITSHM=1" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e QT_X11_NO_MITSHM=1 \
  -v $SSH_AUTH_SOCK_VAR:/run/ssh-agent \
  -e SSH_AUTH_SOCK=/run/ssh-agent \
  -v $WORKSPACE_DIR:/workspace:cached \
  --user guest \
  --runtime nvidia \
  $DOCKER_IMAGE \
  /bin/bash
