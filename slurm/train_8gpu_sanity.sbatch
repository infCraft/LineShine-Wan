#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=760G
#SBATCH --output=/mnt/beegfs/home/huang_z/lineshine/reports/slurm/train_8gpu_%j.out
#SBATCH --error=/mnt/beegfs/home/huang_z/lineshine/reports/slurm/train_8gpu_%j.err

set -euo pipefail

ROOT=${LINESHINE_ROOT:-/mnt/beegfs/home/huang_z/lineshine}
CODE=$ROOT/code
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/train}
RUN_DIR=${RUN_DIR:-$ROOT/runs/wan_t2v_1_3b_openvid_sanity}

source /mnt/beegfs/home/huang_z/miniforge3/etc/profile.d/conda.sh
conda activate lineshine-wan
cd "$CODE"
export PYTHONPATH=.

torchrun --nproc_per_node=8 src/train/train.py \
  --cache-dir "$CACHE_DIR" \
  --cache-pattern "train-*.tar" \
  --run-dir "$RUN_DIR" \
  --empty-context "$ROOT/cache/prompts/empty.safetensors" \
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
  --val-every 500 \
  --val-batches 512 \
  --no-tiny-model

