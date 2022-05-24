# Set dir of Makefile to a variable to use later
MAKEPATH := $(abspath $(lastword $(MAKEFILE_LIST)))
BASEDIR := $(patsubst %/,%,$(dir $(MAKEPATH)))

# Cross platform make declarations
# \
!ifndef 0 # \
# nmake code here \
MKDIR=mkdir # \
RMRF=del /f /s /q # \
!else
# make code here
MKDIR=mkdir -p
RMRF=rm -rf
# \
!endif


# Args for environment (i.e., Docker, port forwarding, virtual environments)
NAME := supy_res_imaging
TAG := latest
IMAGE := $(NAME):$(TAG)
HOST_TO_JUPYTER_PORT := 8888
JUPYTER_PORT := 8888
VENV := venv

DATA_DIR := $(BASEDIR)/data

.PHONY: vars help test k8s show lint deploy delete logs describe namespace default all clean
.DEFAULT_GOAL := help

env: ## Creates and activates local virtual env, and installs requirements.txt for local execution
	python3 -m venv $(BASEDIR)/$(VENV)
	. $(BASEDIR)/$(VENV)/bin/activate
	pip3 install -r $(BASEDIR)/requirements.txt

build: ## Build Docker image for GPU-supported supy_res_imaging jupter labs environment
	docker build -t $(IMAGE) -f Dockerfile .

lab: ## Runs jupyter labs locally; assumes the virtual environment is set up

run: ## Run supy_res_imaging via Docker hosted Jupyter Labs (with NVIDIA GPU support)
	docker run --rm --it --gpus all -p $(HOST_TO_JUPYTER_PORT):$(JUPYTER_PORT) \
		--env JUPY_PORT=$(JUPYTER_PORT) $(IMAGE)

clean: ## Deactivates and deletes the local virtual env
	deactivate
	$(RMRF) $(BASEDIR)/$(VENV)

help:  ## Show this help menu
	@echo "$(MAKE) targets:"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ": .*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""; echo "make vars (+defaults):"
	@grep -E '^[0-9a-zA-Z_-]+ \?=.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = " \\?\\= | ## "}; {printf "\033[36m%-30s\033[0m %-20s %-30s\n", $$1, $$2, $$3}'

