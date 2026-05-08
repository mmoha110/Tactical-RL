#!/bin/bash
set -e

echo "=== Installing system dependencies for gfootball ==="

sudo apt-get update
sudo apt-get install -y \
  cmake \
  build-essential \
  libgl1-mesa-dev \
  libsdl2-dev \
  libsdl2-image-dev \
  libsdl2-ttf-dev \
  libsdl2-gfx-dev \
  zlib1g-dev \
  python3-dev \
  libboost-all-dev \
  libgflags-dev \
  libgoogle-glog-dev

echo "=== System deps installed ==="
