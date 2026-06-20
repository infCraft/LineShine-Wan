#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${LINESHINE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
CODE=${LINESHINE_CODE_ROOT:-$ROOT/code}
REPORT_DIR=${REPORT_DIR:-$ROOT/reports/W5.1_train_chain}
RUN_DIR=${RUN_DIR:-$ROOT/runs/wan_t2v_1_3b_openvid_w5_30k_$(date +%Y%m%d_%H%M)}
CHAIN_STEPS=${CHAIN_STEPS:-5000 10000 15000 20000 25000 30000}
LR_TOTAL_STEPS=${LR_TOTAL_STEPS:-30000}
SAVE_EVERY=${SAVE_EVERY:-2500}
LOG_EVERY=${LOG_EVERY:-10}
NUM_WORKERS=${NUM_WORKERS:-0}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/train}
CACHE_PATTERN=${CACHE_PATTERN:-train-*.tar}
EMPTY_CONTEXT=${EMPTY_CONTEXT:-$ROOT/cache/prompts/empty.safetensors}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-$SCRIPT_DIR/train_8gpu_w5.sbatch}

mkdir -p "$REPORT_DIR" "$ROOT/reports/slurm"

if [ ! -f "$SBATCH_SCRIPT" ]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi

if [ -e "$RUN_DIR" ] && compgen -G "$RUN_DIR/checkpoints/step_*.pt" >/dev/null; then
  echo "Refusing to submit W5 into an existing checkpoint directory: $RUN_DIR" >&2
  exit 1
fi

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

cd "$CODE"
touch "$REPORT_DIR/submitted_jobs.jsonl"

echo "Submitting W5 chain"
echo "Run dir: $RUN_DIR"
echo "Report dir: $REPORT_DIR"
echo "Cache: $CACHE_DIR/$CACHE_PATTERN"
echo "Segment target steps: $CHAIN_STEPS"
echo "LR total steps: $LR_TOTAL_STEPS"
echo "Per-segment tee logs: $RUN_DIR/logs/w5_segment_<index>_job_<jobid>.log"
echo "Slurm logs: $ROOT/reports/slurm/w5_train_<jobid>.{out,err}"

prev_job=""
idx=0
for step in $CHAIN_STEPS; do
  if [ "$idx" -eq 0 ]; then
    resume_mode=scratch
    dependency_args=()
  else
    resume_mode=resume
    dependency_args=(--dependency=afterany:"$prev_job")
  fi

  export LINESHINE_ROOT="$ROOT"
  export LINESHINE_CODE_ROOT="$CODE"
  export RUN_DIR
  export CACHE_DIR
  export CACHE_PATTERN
  export EMPTY_CONTEXT
  export SEGMENT_INDEX="$idx"
  export RESUME_MODE="$resume_mode"
  export MAX_STEPS="$step"
  export LR_TOTAL_STEPS
  export SAVE_EVERY
  export LOG_EVERY
  export NUM_WORKERS

  sbatch_out=$(sbatch --parsable \
    -p compute \
    --gres=gpu:8 \
    --cpus-per-gpu=8 \
    --mem=760G \
    --output=../reports/slurm/w5_train_%j.out \
    --error=../reports/slurm/w5_train_%j.err \
    "${dependency_args[@]}" \
    "$SBATCH_SCRIPT")
  job_id=${sbatch_out%%;*}
  prev_job="$job_id"

  printf '{"segment_index":%s,"job_id":"%s","dependency":"%s","resume_mode":"%s","max_steps":%s,"lr_total_steps":%s,"run_dir":"%s"}\n' \
    "$idx" "$job_id" "${dependency_args[*]:-}" "$resume_mode" "$step" "$LR_TOTAL_STEPS" "$RUN_DIR" \
    >> "$REPORT_DIR/submitted_jobs.jsonl"
  echo "segment=$idx job=$job_id mode=$resume_mode max_steps=$step dependency=${dependency_args[*]:-none}"
  idx=$((idx + 1))
done

cat > "$REPORT_DIR/latest_submission.env" <<EOF
RUN_DIR=$RUN_DIR
REPORT_DIR=$REPORT_DIR
CHAIN_STEPS="$CHAIN_STEPS"
LR_TOTAL_STEPS=$LR_TOTAL_STEPS
FINAL_JOB_ID=$prev_job
EOF

echo "Final chained job id: $prev_job"
echo "Submission record: $REPORT_DIR/submitted_jobs.jsonl"
