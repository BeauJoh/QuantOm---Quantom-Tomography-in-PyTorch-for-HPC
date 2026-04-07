#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/setup-backends.sh"

cd "$SCRIPT_DIR"

./setup-pixi.sh install

if [[ "${BACKENDS:-}" == *"cuda"* ]]; then
  echo "Installing CUDA PyTorch wheels into Pixi environment..."
  ./setup-pixi.sh run install-pytorch-cuda
elif [[ "${BACKENDS:-}" == *"hip"* ]]; then
  echo "Installing HIP PyTorch wheels into Pixi environment..."
  ./setup-pixi.sh run install-pytorch-hip
else
  echo "No CUDA/HIP backend detected in BACKENDS=${BACKENDS:-}" >&2
  exit 1
fi

./setup-pixi.sh run python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
