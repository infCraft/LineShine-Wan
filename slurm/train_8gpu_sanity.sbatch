#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=760G
#SBATCH --output=../reports/slurm/train_8gpu_%j.out
#SBATCH --error=../reports/slurm/train_8gpu_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${LINESHINE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
CODE=${LINESHINE_CODE_ROOT:-$ROOT/code}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/train}
RUN_DIR=${RUN_DIR:-$ROOT/runs/wan_t2v_1_3b_openvid_sanity}
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lineshine-wan}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE"
export PYTHONPATH=.

torchrun --nproc_per_node=8 src/train/train.py \
  --cache-dir "$CACHE_DIR" \
  --cache-pattern "train-*.tar" \
  --run-dir "$RUN_DIR" \
  --empty-context "$ROOT/cache/prompts/empty.safetensors" \
  --from-scratch \
  --max-steps "${MAX_STEPS:-2000}" \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --warmup-steps 1000 \
  --weight-decay 1e-3 \
  --clip-grad-norm 1.0 \
  --cfg-dropout 0.10 \
  --save-every 500 \
  --val-every 0 \
  --val-batches 512 \
  --no-tiny-model
