#!/usr/bin/env bash
#SBATCH --job-name=ad-flow-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs outputs/landmark_flow

source ~/.local/bin/env
nvidia-smi || true

uv run --with-requirements requirements-landmark-flow.txt python -m landmark_flow.train \
  --data-dir data/landmark_flow \
  --output-dir outputs/landmark_flow \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --lr "${LR:-1e-3}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --dropout "${DROPOUT:-0.1}" \
  --smooth-weight "${SMOOTH_WEIGHT:-0.01}" \
  --inference-steps "${INFERENCE_STEPS:-8}" \
  --threshold "${THRESHOLD:-0.5}" \
  --num-workers "${SLURM_NTASKS:-4}"
