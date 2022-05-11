ARG CUDA_VERSION=10.2
ARG PYTHON_VERSION=3.10.3

# Image based on CUDA 10.2 runtime (maybe should be devel?)
FROM NVIDIA_RUNTIME_IMAGE=nvidia/cuda:${CUDA_VERSION}-runtime

RUN echo "Building with: $CUDA_VERSION"

# Update and install neccesary software
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends && \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    wget \
    libbz2-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Get python source
RUN wget https://www.python.org/ftp/python/3.10.3/Python-3.10.3.tgz