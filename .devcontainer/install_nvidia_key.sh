#!/bin/bash
set -e

# Add NVIDIA's package repository to apt so that we can download packages
# Always use the ubuntu2004 repo because the other repos (e.g., debian11) are missing packages
NVIDIA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64"
KEYRING_PACKAGE="cuda-keyring_1.0-1_all.deb"
KEYRING_PACKAGE_URL="$NVIDIA_REPO_URL/$KEYRING_PACKAGE"
KEYRING_PACKAGE_PATH="$(mktemp -d)"
KEYRING_PACKAGE_FILE="$KEYRING_PACKAGE_PATH/$KEYRING_PACKAGE"
wget -O "$KEYRING_PACKAGE_FILE" "$KEYRING_PACKAGE_URL"
apt-get install -yq "$KEYRING_PACKAGE_FILE"
apt-get update -yq
