FROM python:3.9-bullseye

# System dependencies required by gfootball
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    libgoogle-glog-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# STEP 1: Install gym BEFORE any pip upgrade.
# gym==0.21.0 has `opencv-python>=3.` (malformed spec) in its setup.py.
# packaging>=24 rejects it; the base image's pip 23.0.1 is lenient.
# --no-build-isolation uses the current env's setuptools instead of a fresh isolated one.
RUN pip install --timeout=600 --no-build-isolation "gym==0.21.0"

# STEP 2: Torch gets its own layer — 199 MB download needs retries on slow connections.
# Once this layer is cached by Docker, rebuilds never re-download it.
RUN pip install --timeout=3600 --retries=10 \
    "torch==1.13.1+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# STEP 3: Install remaining deps — DO NOT upgrade pip.
# pip>=24.1 refuses to operate when any installed package has invalid METADATA
# (gym's METADATA contains `opencv-python>=3.`). pip 23.0.1 is not affected.
RUN pip install --timeout=600 wheel "setuptools<67" && \
    pip install --timeout=600 -r requirements.txt && \
    pip install --timeout=600 --no-build-isolation gfootball==2.10.2 && \
    # Force reinstall numpy+pandas LAST to ensure binary ABI compatibility.
    # gym's install may have left an older numpy whose dtype struct size (88 bytes)
    # differs from what pandas expects (96 bytes), causing import errors.
    pip install --timeout=600 --force-reinstall "numpy==1.23.5" "pandas==1.5.3"

# Copy project (volume mount in docker-compose.yml makes this optional for dev)
COPY . .

# Headless rendering — gfootball needs these in Docker (no display server)
ENV DISPLAY=:0
ENV SDL_VIDEODRIVER=offscreen
ENV SDL_AUDIODRIVER=dummy

CMD ["bash"]
