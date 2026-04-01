FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set default python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

# Copy project source (excluding submodules — cloned fresh below)
COPY pyproject.toml ./
COPY src/ ./src/
COPY train.py evaluate.py render_video.py ./

# PyTorch must be installed before compiling the CUDA submodules
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Clone and build CUDA submodules (self-contained — no local checkout needed)
RUN git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization submodules/diff-gaussian-rasterization && \
    pip3 install --no-cache-dir --break-system-packages ./submodules/diff-gaussian-rasterization

RUN git clone --recursive https://github.com/camenduru/simple-knn submodules/simple-knn && \
    pip3 install --no-cache-dir --break-system-packages ./submodules/simple-knn

# Install remaining Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy Pillow plyfile pytorch-msssim tqdm opencv-python lpips

# Install the package itself
RUN pip3 install --no-cache-dir --break-system-packages -e .

CMD ["python", "train.py", "--help"]
