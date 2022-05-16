FROM ubuntu:22.04

# Disable prompts from apt.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3.10-venv \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install --upgrade pip \
  && pip3 install numpy \
  && pip3 install jupyterlab

# Run jupyter lab as a service, ip=0.0.0.0 allows external container access
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# To build:
# sudo docker build -t supy_res:latest -f Dockerfile .
# To run (opens jupyter notebook as service)
# sudo docker run -p 8888:8888 supy_res
