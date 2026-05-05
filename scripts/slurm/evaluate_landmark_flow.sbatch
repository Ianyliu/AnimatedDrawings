#!/usr/bin/env bash
#SBATCH --job-name=ad-flow-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source ~/.local/bin/env
nvidia-smi || true

uv run --with-requirements requirements-landmark-flow.txt python -m landmark_flow.evaluate \
  --checkpoint outputs/landmark_flow/landmark_flow_corrector.pt \
  --data-dir data/landmark_flow \
  --split test \
  --threshold "${THRESHOLD:-0.5}"
