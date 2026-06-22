#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=760G
#SBATCH --output=../reports/slurm/w5_train_%j.out
#SBATCH --error=../reports/slurm/w5_train_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
if [ -n "${LINESHINE_ROOT:-}" ]; then
  ROOT=$LINESHINE_ROOT
elif [ -d "$SUBMIT_DIR/src" ] && [ -d "$SUBMIT_DIR/slurm" ]; then
  ROOT=$(cd "$SUBMIT_DIR/.." && pwd)
else
  ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
fi
CODE=${LINESHINE_CODE_ROOT:-$ROOT/code}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/train}
CACHE_PATTERN=${CACHE_PATTERN:-train-*.tar}
RUN_DIR=${RUN_DIR:?RUN_DIR must be set, normally by slurm/submit_w5_chain.sh}
EMPTY_CONTEXT=${EMPTY_CONTEXT:-$ROOT/cache/prompts/empty.safetensors}
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lineshine-wan}

SEGMENT_INDEX=${SEGMENT_INDEX:-0}
RESUME_MODE=${RESUME_MODE:-resume}
MAX_STEPS=${MAX_STEPS:-30000}
LR_TOTAL_STEPS=${LR_TOTAL_STEPS:-30000}
SAVE_EVERY=${SAVE_EVERY:-2500}
LOG_EVERY=${LOG_EVERY:-10}
OPTIMIZER=${OPTIMIZER:-adamw}
MUON_LR=${MUON_LR:-0.02}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.95}
MUON_WEIGHT_DECAY=${MUON_WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
LR=${LR:-1e-4}
# Throughput knobs (A-class optimizations; see docs/throughput.md):
#   eff batch = BATCH_SIZE * GRAD_ACCUM_STEPS * 8 GPUs. Defaults keep eff batch = 64.
NUM_WORKERS=${NUM_WORKERS:-6}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
COMPILE=${COMPILE:-1}

mkdir -p "$RUN_DIR/logs" "$ROOT/reports/slurm"
RUN_LOG="$RUN_DIR/logs/w5_segment_${SEGMENT_INDEX}_job_${SLURM_JOB_ID:-manual}.log"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "W5 segment index: $SEGMENT_INDEX"
echo "Slurm job id: ${SLURM_JOB_ID:-manual}"
echo "Run dir: $RUN_DIR"
echo "Run log: $RUN_LOG"
echo "Cache: $CACHE_DIR/$CACHE_PATTERN"
echo "Target max steps: $MAX_STEPS"
echo "LR total steps: $LR_TOTAL_STEPS"
echo "Resume mode: $RESUME_MODE"

if [ ! -d "$CACHE_DIR" ]; then
  echo "Missing cache dir: $CACHE_DIR" >&2
  exit 1
fi

if ! compgen -G "$CACHE_DIR/$CACHE_PATTERN" >/dev/null; then
  echo "No cache shards matched: $CACHE_DIR/$CACHE_PATTERN" >&2
  exit 1
fi

if [ ! -f "$EMPTY_CONTEXT" ]; then
  echo "Missing empty prompt cache: $EMPTY_CONTEXT" >&2
  exit 1
fi

if [ "$RESUME_MODE" = "scratch" ]; then
  if compgen -G "$RUN_DIR/checkpoints/step_*.pt" >/dev/null; then
    echo "Refusing scratch run with existing checkpoints in $RUN_DIR/checkpoints" >&2
    exit 1
  fi
  RESUME_ARGS=(--from-scratch)
elif [ "$RESUME_MODE" = "resume" ]; then
  if ! compgen -G "$RUN_DIR/checkpoints/step_*.pt" >/dev/null; then
    echo "Resume segment requires an existing checkpoint in $RUN_DIR/checkpoints" >&2
    exit 1
  fi
  RESUME_ARGS=(--auto-resume --no-from-scratch)
else
  echo "RESUME_MODE must be scratch or resume, got: $RESUME_MODE" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$CODE"
export PYTHONPATH=.
# Reduce allocator fragmentation; lets larger per-GPU batch fit under 80GB.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ "$COMPILE" = "1" ]; then COMPILE_ARG=--compile; else COMPILE_ARG=--no-compile; fi

torchrun --nproc_per_node=8 src/train/train.py \
  --cache-dir "$CACHE_DIR" \
  --cache-pattern "$CACHE_PATTERN" \
  --run-dir "$RUN_DIR" \
  --empty-context "$EMPTY_CONTEXT" \
  "${RESUME_ARGS[@]}" \
  --max-steps "$MAX_STEPS" \
  --lr-total-steps "$LR_TOTAL_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  $COMPILE_ARG \
  --lr "$LR" \
  --min-lr 1e-5 \
  --warmup-steps "$WARMUP_STEPS" \
  --weight-decay 1e-3 \
  --optimizer "$OPTIMIZER" \
  --muon-lr "$MUON_LR" \
  --muon-momentum "$MUON_MOMENTUM" \
  --muon-weight-decay "$MUON_WEIGHT_DECAY" \
  --clip-grad-norm 1.0 \
  --cfg-dropout 0.10 \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY" \
  --val-every 0 \
  --val-batches 512 \
  --no-tiny-model
