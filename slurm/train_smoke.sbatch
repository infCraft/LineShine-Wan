#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --output=../reports/slurm/train_smoke_%j.out
#SBATCH --error=../reports/slurm/train_smoke_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${LINESHINE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
CODE=${LINESHINE_CODE_ROOT:-$ROOT/code}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/smoke_real64}
RUN_DIR=${RUN_DIR:-$ROOT/runs/smoke_train_synthetic}
SMOKE_MODE=${SMOKE_MODE:-synthetic}
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lineshine-wan}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE"
export PYTHONPATH=.

if [ "$SMOKE_MODE" = "real" ]; then
  python src/train/train.py \
    --cache-dir "$CACHE_DIR" \
    --cache-pattern "smoke_real-*.tar" \
    --run-dir "$RUN_DIR" \
    --empty-context "$ROOT/cache/prompts/empty.safetensors" \
    --from-scratch \
    --max-steps "${MAX_STEPS:-20}" \
    --batch-size 1 \
    --grad-accum-steps 1 \
    --num-workers 0 \
    --shuffle \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --warmup-steps 0 \
    --weight-decay 1e-3 \
    --clip-grad-norm 1.0 \
    --cfg-dropout 0.10 \
    --save-every "${SAVE_EVERY:-10}" \
    --val-every 0 \
    --tiny-model \
    --tiny-dim 128 \
    --tiny-ffn-dim 512 \
    --tiny-heads 4 \
    --tiny-layers 2
else
  python src/train/train.py \
    --synthetic \
    --synthetic-count "${SYNTHETIC_COUNT:-16}" \
    --synthetic-text-len "${SYNTHETIC_TEXT_LEN:-8}" \
    --run-dir "$RUN_DIR" \
    --from-scratch \
    --max-steps "${MAX_STEPS:-20}" \
    --batch-size 1 \
    --grad-accum-steps 1 \
    --num-workers 0 \
    --shuffle \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --warmup-steps 0 \
    --weight-decay 1e-3 \
    --clip-grad-norm 1.0 \
    --cfg-dropout 0.10 \
    --save-every "${SAVE_EVERY:-10}" \
    --val-every 0 \
    --tensorboard \
    --tiny-model \
    --tiny-dim 128 \
    --tiny-ffn-dim 512 \
    --tiny-heads 4 \
    --tiny-layers 2
fi
