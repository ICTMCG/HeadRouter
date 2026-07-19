#!/usr/bin/env bash
# One-shot setup for HeadRouter (conda env + patched diffusers).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx 'diffusers0.18.0'; then
  echo "Env diffusers0.18.0 already exists; skipping create."
else
  conda env create -f environment.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffusers0.18.0
pip install -r requirements.txt
pip install -e ./diffusers

echo
echo "Setup done. Next:"
echo "  conda activate diffusers0.18.0"
echo "  cd ${ROOT}"
echo "  bash scripts/infer_tree2.sh"
