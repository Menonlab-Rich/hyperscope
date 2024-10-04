#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = hyper-scope
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
VENV_MANAGER = venv # options: venv, conda, pipenv, poetry, mamba
VENV_PATH = ./.venv

# Detect OS: Linux/Unix or Windows
UNAME_S := $(shell uname -s)
# Setup venv bin path depending on OS and venv manager
# Setup venv bin path depending on OS and venv manager
ifeq ($(UNAME_S),Linux)
    BIN_DIR = bin
else ifeq ($(UNAME_S),Darwin)
    BIN_DIR = bin
else ifeq ($(OS),Windows_NT)
    BIN_DIR = Scripts
else
    BIN_DIR = bin
endif

VENV_BIN = $(VENV_PATH)/$(BIN_DIR)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Setup IPyKernel
.PHONY: kernel
kernel:
	@echo ">>> Setting up IPyKernel"
	./scripts/setup_kernel.sh $(VENV_BIN)



## Install Python Dependencies
.PHONY: requirements
requirements:
## check which venv manager is in use
ifeq ($(VENV_MANAGER), venv)
	$(VENV_BIN)/pip install -r requirements.txt
endif
ifeq ($(VENV_MANAGER), conda)
	conda env update --name $(PROJECT_NAME) --file requirements.txt
endif
ifeq ($(VENV_MANAGER), pipenv)
	pipenv install
endif
ifeq ($(VENV_MANAGER), poetry)
	poetry install
endif
ifeq ($(VENV_MANAGER), mamba)
	mamba env update --name $(PROJECT_NAME) --file requirements.txt
endif

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 hypserscope
	isort --check --diff --profile black hypserscope
	black --check --config pyproject.toml hypserscope

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml hypserscope

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
ifeq ($(VENV_MANAGER), venv)
	if [ -d $(VENV_PATH) ]; then exit 0; fi
	$(PYTHON_INTERPRETER) -m venv $(VENV_PATH)
	$(VENV_BIN)/pip install --upgrade pip setuptools
endif
ifeq ($(VENV_MANAGER), conda)
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
endif
ifeq ($(VENV_MANAGER), pipenv)
	pipenv --python $(PYTHON_VERSION)
endif
ifeq ($(VENV_MANAGER), poetry)
	poetry env use $(PYTHON_VERSION)
endif
ifeq ($(VENV_MANAGER), mamba)
	mamba env create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
endif

## Install package in editable mode
.PHONY: install
install:
	$(VENV_BIN)/pip install -e . 

.PHONY: setup
setup: create_environment requirements install kernel
	@echo ">>> Setup complete. Activate with:\nsource $(VENV_BIN)/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) hypserscope/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
