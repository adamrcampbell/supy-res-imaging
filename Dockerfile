FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# Disable prompts from apt.
ENV DEBIAN_FRONTEND=noninteractive

# Install python3, pip, and other requisites
RUN apt-get update && apt-get install -y --no-install-recommends \
  software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update && apt-get install -y --no-install-recommends \
  curl \
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
COPY notebooks /supy_res/notebooks
COPY data /supy_res/data
WORKDIR /supy_res

# Run jupyter lab as a service, ip=0.0.0.0 allows external container access
# CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=$JUPY_PORT", "--no-browser", "--allow-root"]
CMD jupyter-lab --ip=0.0.0.0 --port=$JUPY_PORT --no-browser --allow-root
