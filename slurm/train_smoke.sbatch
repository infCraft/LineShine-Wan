#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --output=/mnt/beegfs/home/huang_z/lineshine/reports/slurm/train_smoke_%j.out
#SBATCH --error=/mnt/beegfs/home/huang_z/lineshine/reports/slurm/train_smoke_%j.err

set -euo pipefail

ROOT=${LINESHINE_ROOT:-/mnt/beegfs/home/huang_z/lineshine}
CODE=$ROOT/code
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/smoke_real64}
RUN_DIR=${RUN_DIR:-$ROOT/runs/smoke_train_real64}

source /mnt/beegfs/home/huang_z/miniforge3/etc/profile.d/conda.sh
conda activate lineshine-wan
cd "$CODE"
export PYTHONPATH=.

python src/train/train.py \
  --cache-dir "$CACHE_DIR" \
  --cache-pattern "smoke_real-*.tar" \
  --run-dir "$RUN_DIR" \
  --empty-context "$ROOT/cache/prompts/empty.safetensors" \
  --max-steps "${MAX_STEPS:-20}" \
  --batch-size 1 \
  --save-every "${SAVE_EVERY:-10}" \
  --tiny-model \
  --tiny-dim 128 \
  --tiny-ffn-dim 512 \
  --tiny-heads 4 \
  --tiny-layers 2

