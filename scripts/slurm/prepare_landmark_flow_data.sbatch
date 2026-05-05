#!/usr/bin/env bash
#SBATCH --job-name=ad-flow-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs data/landmark_flow

# Defaults target roughly 10k train clips from the bundled CMU mirror.
# Set MAX_FILES=all to use every BVH under BVH_ROOT.
BVH_ROOT="${BVH_ROOT:-cmu-mocap/data}"
STRIDE="${STRIDE:-30}"
MAX_FILES="${MAX_FILES:-400}"
CLIP_LEN="${CLIP_LEN:-31}"
MIN_SPAN="${MIN_SPAN:-3}"
MAX_SPAN="${MAX_SPAN:-30}"
WHOLE_LIMB_PROB="${WHOLE_LIMB_PROB:-0.35}"
SWAP_PROB="${SWAP_PROB:-0.20}"
HIGH_CONF_WRONG_PROB="${HIGH_CONF_WRONG_PROB:-0.0}"

ARGS=(
  --bvh-root "$BVH_ROOT"
  --output-dir data/landmark_flow
  --clip-len "$CLIP_LEN"
  --stride "$STRIDE"
  --seed 1337
  --min-span "$MIN_SPAN"
  --max-span "$MAX_SPAN"
  --whole-limb-prob "$WHOLE_LIMB_PROB"
  --swap-prob "$SWAP_PROB"
  --high-conf-wrong-prob "$HIGH_CONF_WRONG_PROB"
)

if [[ "$MAX_FILES" != "all" ]]; then
  ARGS+=(--max-files "$MAX_FILES")
fi

uv run --with-requirements requirements-landmark-flow-data.txt python -m landmark_flow.bvh_dataset \
  "${ARGS[@]}"
