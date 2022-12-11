FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Use the below version for running on RTX2060
# FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# Use the below version when not using a GPU, and trying to use Intel MKL
# FROM ubuntu:jammy

# Disable prompts from apt.
ENV DEBIAN_FRONTEND=noninteractive

# Install python3, pip, and other requisites
RUN apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y --no-install-recommends \
  curl \
  git \
#  intel-mkl \
  python3.10 \
  python3.10-dev \
  python3.10-distutils \
  python3-pip \
  python3-venv \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*
  
# Symbolic link so python3 resolves to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

# Fixes odd issue where pip3 cannot upgrade/install libs
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Now upgrade, and add some packages for testing
COPY requirements.txt .
RUN pip3 --no-cache-dir install --upgrade setuptools pip \
  && pip3 install -r requirements.txt

RUN mkdir /supy_res
WORKDIR /supy_res

# Update LD_LIBRARY_PATH to locate OpenMP/IntelMKL (bit of a hack)
# ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH

# Run jupyter lab as a service, ip=0.0.0.0 allows external container access
# CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=$JUPY_PORT", "--no-browser", "--allow-root"]
 CMD jupyter-lab --ip=0.0.0.0 --port=$JUPY_PORT --no-browser --allow-root
