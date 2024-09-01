#!/bin/bash
set -e

sudo apt-get update
sudo apt-get --assume-yes install \
    python3-opencv \
    ffmpeg \
    cifs-utils

echo 'export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}' >> ~/.profile
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.profile
