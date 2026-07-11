#!/usr/bin/env bash
set -euo pipefail

# Reproducible, isolated setup for the audited stable gprMax 3.1.7 release.
# Prerequisites: Miniconda/Miniforge, git, and a C compiler with OpenMP.
# GPU execution additionally requires an NVIDIA driver/CUDA runtime supported
# by gprMax. This script never modifies the MyNet training environment.

WORK_ROOT="${1:-$PWD/gprmax_runtime_3_1_7}"
REPO_DIR="$WORK_ROOT/gprMax"
ENV_NAME="gprmax317"

mkdir -p "$WORK_ROOT"
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --branch v3.1.7 --depth 1 https://github.com/gprMax/gprMax.git "$REPO_DIR"
fi

cd "$REPO_DIR"
# The official repository's conda_env.yml is the dependency source of truth.
conda env create --name "$ENV_NAME" --file conda_env.yml

# `conda activate` is not reliable in non-interactive scripts; conda run keeps
# the build explicitly inside the isolated environment.
conda run -n "$ENV_NAME" python setup.py build
conda run -n "$ENV_NAME" python setup.py install
conda run -n "$ENV_NAME" python -m gprMax --help

echo "gprMax 3.1.7 environment ready: $ENV_NAME"
