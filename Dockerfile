FROM ubuntu:22.04

# Disable prompts from apt.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y python3-pip python3.10-venv \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv venv \
  && . venv/bin/activate \
  && pip3 --no-cache-dir install --upgrade pip \
  && pip3 install numpy

# All of this needs a rework...
# See: https://stackoverflow.com/a/48562835

# Note: look for a way to reduce this, Ubuntu img is heavy...

# To build:
# docker build -t supy:latest -f Dockerfile .
# To run (opens bash terminal):
# docker run supy