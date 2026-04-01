FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY train.py evaluate.py ./

# Install PyTorch (CUDA 12.4 wheels are compatible with CUDA 12.6 runtime)
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Required for building CUDA extensions on machines without a physical GPU
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Build diff-gaussian-rasterization (cloned inside Docker to keep image self-contained)
RUN git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization \
        submodules/diff-gaussian-rasterization && \
    # Fix missing <cstdint> include for GCC 13+ (Ubuntu 24.04)
    sed -i '1s/^/#include <cstdint>\n/' \
        submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h && \
    pip3 install --no-cache-dir --break-system-packages \
        ./submodules/diff-gaussian-rasterization

# Build simple-knn
RUN git clone --recursive https://github.com/camenduru/simple-knn \
        submodules/simple-knn && \
    pip3 install --no-cache-dir --break-system-packages \
        ./submodules/simple-knn

# Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    numpy Pillow plyfile pytorch-msssim tqdm opencv-python lpips

# Install project in editable mode
RUN pip3 install --no-cache-dir --break-system-packages -e .

CMD ["python", "train.py", "--help"]
