FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Copy the pyproject first for caching
COPY pyproject.toml ./

# Copy everything else (including submodules)
COPY . .

# PyTorch must be installed before compiling the CUDA submodules
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision numpy

# Install the CUDA-dependent submodules manually
RUN pip3 install --no-cache-dir --break-system-packages ./submodules/diff-gaussian-rasterization ./submodules/simple-knn

# Install the package itself in editable mode
RUN pip3 install --no-cache-dir --break-system-packages -e .

CMD ["python", "train.py", "--help"]
