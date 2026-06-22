#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --output=../reports/slurm/cache_%A_%a.out
#SBATCH --error=../reports/slurm/cache_%A_%a.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${LINESHINE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
CODE=${LINESHINE_CODE_ROOT:-$ROOT/code}
MANIFEST=${MANIFEST:-$ROOT/data/openvid/meta/openvid_shared_train_98k_extracted.jsonl}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/train}
PREFIX=${PREFIX:-train}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
FRAMES=${FRAMES:-49}
TARGET_FPS=${TARGET_FPS:-16.0}
IMAGE_SIZE=${IMAGE_SIZE:-256}
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lineshine-wan}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE"
export PYTHONPATH=.

python src/data/preprocess_cache.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --prefix "$PREFIX" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --frames "$FRAMES" \
  --target-fps "$TARGET_FPS" \
  --image-size "$IMAGE_SIZE" \
  --device cuda
