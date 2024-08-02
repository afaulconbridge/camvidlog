FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

#RUN wget 'https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb' \
#    && sudo dpkg -i cuda-keyring_1.1-1_all.deb \
#    && rm cuda-keyring_1.1-1_all.deb
RUN sudo apt update \
  && sudo apt -y install \
  python3-opencv \
  ffmpeg

COPY .devcontainer/install_nvidia_key.sh /
RUN /install_nvidia_key.sh

RUN sudo apt update \
  && sudo apt -y install software-properties-common \
  && sudo add-apt-repository -y contrib
RUN sudo apt update \
  && sudo apt -y install \
  cuda-toolkit-12-2

RUN echo 'export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}' >> ~/.profile
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.profile
