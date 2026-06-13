#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --output=/mnt/beegfs/home/huang_z/lineshine/reports/slurm/cache_%A_%a.out
#SBATCH --error=/mnt/beegfs/home/huang_z/lineshine/reports/slurm/cache_%A_%a.err

set -euo pipefail

ROOT=${LINESHINE_ROOT:-/mnt/beegfs/home/huang_z/lineshine}
CODE=$ROOT/code
MANIFEST=${MANIFEST:-$ROOT/data/openvid/meta/openvid_shared_train_98k.jsonl}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/train}
PREFIX=${PREFIX:-train}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

source /mnt/beegfs/home/huang_z/miniforge3/etc/profile.d/conda.sh
conda activate lineshine-wan
cd "$CODE"
export PYTHONPATH=.

python src/data/preprocess_cache.py \
  --manifest "$MANIFEST" \
  --cache-dir "$CACHE_DIR" \
  --prefix "$PREFIX" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --device cuda

