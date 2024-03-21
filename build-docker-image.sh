#!/bin/bash
clear
echo "Removing docker image..."
docker image rm -f tps:ruby-fann
echo "Building docker image..."
docker build --platform linux/arm64 -t tps:ruby-fann .
docker image ls
