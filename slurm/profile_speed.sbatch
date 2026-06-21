#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --output=../reports/profile/profile_%j.out
#SBATCH --error=../reports/profile/profile_%j.err

# Single-GPU throughput profiler for the Wan DiT training loop.
# Usage: sbatch slurm/profile_speed.sbatch
set -euo pipefail

ROOT=${LINESHINE_ROOT:-/mnt/beegfs/home/huang_z/lineshine}
CODE=${LINESHINE_CODE_ROOT:-$ROOT/code}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/train}
CACHE_PATTERN=${CACHE_PATTERN:-train-*.tar}
MAX_SHARDS=${MAX_SHARDS:-8}
REPORT=${REPORT:-$ROOT/reports/profile/w5_profile.json}
COMPILE_FLAG=${COMPILE_FLAG:-}
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lineshine-wan}

mkdir -p "$ROOT/reports/profile"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE"
export PYTHONPATH=. LINESHINE_ROOT="$ROOT" LINESHINE_CODE_ROOT="$CODE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/train/profile_speed.py \
  --cache-dir "$CACHE_DIR" \
  --cache-pattern "$CACHE_PATTERN" \
  --max-shards "$MAX_SHARDS" \
  --warmup 5 --iters 24 \
  $COMPILE_FLAG \
  --report "$REPORT"
