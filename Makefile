# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Makefile for Healthcare Knowledge Base Project

# Python settings
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip

# Project structure
SRC_DIR := geniusrise_healthcare
TEST_DIR := tests
DATA_DIR := data
DOCS_DIR := docs

# Data files
UMLS_DIR := $(DATA_DIR)/umls
SNOMED_DIR := $(DATA_DIR)/snomed
RXNORM_DIR := $(DATA_DIR)/rxnorm
MESH_DIR := $(DATA_DIR)/mesh
GO_DIR := $(DATA_DIR)/gene_ontology
DO_DIR := $(DATA_DIR)/disease_ontology

# Docker settings
DOCKER_IMAGE := healthcare-kb
DOCKER_TAG := latest

.PHONY: all setup clean test lint format docs build run deploy help

all: setup test lint format docs build

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

test:
	$(VENV)/bin/pytest $(TEST_DIR)

lint:
	$(VENV)/bin/flake8 $(SRC_DIR)
	$(VENV)/bin/mypy $(SRC_DIR)

format:
	$(VENV)/bin/black $(SRC_DIR) $(TEST_DIR)
	$(VENV)/bin/isort $(SRC_DIR) $(TEST_DIR)

docs:
	$(VENV)/bin/sphinx-build -b html $(DOCS_DIR)/source $(DOCS_DIR)/build

build:
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

run:
	docker run -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

# Development helpers
dev-setup: setup
	$(PIP) install -r requirements-dev.txt

jupyter:
	$(VENV)/bin/jupyter notebook

help:
	@echo "Available targets:"
	@echo "  setup       - Set up virtual environment and install dependencies"
	@echo "  clean       - Remove virtual environment and cache files"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linters (flake8, mypy)"
	@echo "  format      - Format code (black, isort)"
	@echo "  docs        - Build documentation"
	@echo "  build       - Build Docker image"
	@echo "  run         - Run Docker container"
	@echo "  dev-setup   - Set up development environment"
	@echo "  jupyter     - Start Jupyter notebook"
