#!/usr/bin/env bash
# Create an isolated Python env with CPU torch for the grokking sweeps.
# Usage: bash experiments/grokking/setup_env.sh [env_dir]
set -euo pipefail
ENV_DIR="${1:-.venv}"
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install --quiet --upgrade pip
# CPU wheels are much smaller than the default CUDA build on Linux
"$ENV_DIR/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
  || "$ENV_DIR/bin/pip" install --quiet torch
"$ENV_DIR/bin/pip" install --quiet numpy matplotlib pytest
"$ENV_DIR/bin/python" -c "import torch, os; print('torch', torch.__version__, 'cpus', os.cpu_count())"
