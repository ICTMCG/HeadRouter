#!/usr/bin/env bash
# Example: edit tree2.png with HeadRouter.
# Usage (from repo root OR from scripts/):
#   conda activate diffusers0.18.0
#   bash scripts/infer_tree2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

python main.py \
  --image "${ROOT}/examples/images/tree2.png" \
  --eta 0.9 \
  --gamma 0.5 \
  --start_timestep 0 \
  --stop_timestep 8 \
  --prompt_list \
  "A tree" "A tree with lightning and the Moon"
