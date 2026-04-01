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
COPY train.py evaluate.py ./

# PyTorch must be installed before compiling the CUDA submodules
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Tell nvcc which GPU architectures to compile for.
# Without this, setup.py calls torch.cuda.get_device_capability() which fails
# on CI runners (no GPU) and raises IndexError: list index out of range.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Clone and build CUDA submodules (self-contained — no local checkout needed)
RUN git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization submodules/diff-gaussian-rasterization && \
    sed -i '1s/^/#include <cstdint>\n/' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h && \
    pip3 install --no-cache-dir --break-system-packages ./submodules/diff-gaussian-rasterization

RUN git clone --recursive https://github.com/camenduru/simple-knn submodules/simple-knn && \
    pip3 install --no-cache-dir --break-system-packages ./submodules/simple-knn

# Install remaining Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy Pillow plyfile pytorch-msssim tqdm opencv-python lpips

# Install the package itself
RUN pip3 install --no-cache-dir --break-system-packages -e .

CMD ["python", "train.py", "--help"]
