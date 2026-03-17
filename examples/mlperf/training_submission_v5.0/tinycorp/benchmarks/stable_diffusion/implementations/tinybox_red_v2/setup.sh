#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -d .venv-sd-mlperf ]]; then
  . .venv-sd-mlperf/bin/activate
else
  python3 -m venv .venv-sd-mlperf
  . .venv-sd-mlperf/bin/activate
fi

python3 -m pip install --upgrade pip
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install tqdm numpy ftfy regex pillow scipy wandb webdataset

python3 -m pip list
if command -v rocm-smi >/dev/null 2>&1; then rocm-smi --version; fi

