# Build helpers; correctness and performance runs belong to Weft Bench.
PYTHON ?= python3
ROOT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(shell PYTHONPATH="$(ROOT_DIR)/python" $(PYTHON) -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')

.PHONY: all dev-install wheel triton-opt docs
all:
	cmake --build $(BUILD_DIR)

dev-install:
	$(PYTHON) -m pip install --no-build-isolation -e .

wheel:
	$(PYTHON) -m pip wheel --no-build-isolation --no-deps . -w dist

triton-opt:
	cmake --build $(BUILD_DIR) --target triton-opt

docs:
	$(PYTHON) -m mkdocs build --strict -f third_party/weft/mkdocs.yml
