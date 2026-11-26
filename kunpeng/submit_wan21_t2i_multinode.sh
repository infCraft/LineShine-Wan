#!/usr/bin/env bash
# Default Wan BF16: raw media -> frozen VAE/UMT5 -> shared-state training.
set -euo pipefail
(($# >= 2 && $# <= 4)) || { echo "Usage: $0 RAW_INDEX_OR_MANIFEST RUN_DIR [MAX_STEPS=40] [scratch|resume]" >&2; exit 2; }
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_INDEX="$1"; RUN_DIR="$2"; MAX_STEPS="${3:-40}"; MODE="${4:-scratch}"
[[ -f "$DATA_INDEX" || -r "$DATA_INDEX/catalog.json" ]] || { echo "Raw media manifest/catalog is missing: $DATA_INDEX" >&2; exit 1; }
NNODES="${WAN21_NNODES:-4}"; RPN="${WAN21_PACK_RPN:-16}"; MEM_GIB="${WAN21_MEM_PER_RANK_GIB:-27}"
for value in "$NNODES" "$RPN" "$MEM_GIB" "$MAX_STEPS"; do [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2; done
(( RPN >= 4 && RPN <= 16 && RPN % 4 == 0 )) || { echo "RPN must be 4, 8, 12, or 16" >&2; exit 2; }
case "$MODE" in
 scratch|--from-scratch) MODE=scratch; [[ ! -e "$RUN_DIR" ]] || { echo "Scratch run already exists" >&2; exit 1; } ;;
 resume|--auto-resume) MODE=resume; compgen -G "$RUN_DIR/checkpoints/step_*/COMPLETED" >/dev/null || { echo "No completed checkpoint" >&2; exit 1; } ;;
 *) echo "Start mode must be scratch or resume" >&2; exit 2 ;;
esac
ENV_ACTIVATE="${WAN21_ENV_ACTIVATE:?set the absolute path to the site runtime activation script}"
: "${WAN21_QUEUE:?set the Donau queue}" "${WAN21_ENCODER_WEIGHTS:?set the encoder weights directory}"
: "${WAN21_BLAS_CHECK:?set the executable link checker for the selected runtime}"
[[ "$ENV_ACTIVATE" == /* && -f "$ENV_ACTIVATE" && -r "$ENV_ACTIVATE" ]] || { echo 'WAN21_ENV_ACTIVATE must be a readable absolute file' >&2; exit 2; }
[[ "$WAN21_ENCODER_WEIGHTS" == /* && -d "$WAN21_ENCODER_WEIGHTS" ]] || { echo 'WAN21_ENCODER_WEIGHTS must be an absolute directory' >&2; exit 2; }
[[ "$WAN21_BLAS_CHECK" == /* && -f "$WAN21_BLAS_CHECK" && -x "$WAN21_BLAS_CHECK" ]] || { echo 'WAN21_BLAS_CHECK must be an absolute executable file' >&2; exit 2; }
# The runner changes directory before launching ranks. Resolve CLI paths on the submit host.
DATA_INDEX="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$DATA_INDEX")"
RUN_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RUN_DIR")"
DSUB="${WAN21_DSUB:-dsub}"
command -v "$DSUB" >/dev/null || { echo 'Donau dsub not found; set WAN21_DSUB' >&2; exit 2; }
TASK="${WAN21_TRAIN_TASK:-t2i}"
[[ "$TASK" == t2i || "$TASK" == t2v ]] || { echo "Invalid training task" >&2; exit 2; }
python3 - "$TASK" "${WAN21_IMAGE_SIZE:-256}" "${WAN21_VIDEO_FRAMES:-49}" "${WAN21_VIDEO_FPS:-16}" <<'PY'
import math,sys
task,size,frames,fps=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4])
if size <= 0 or size % 16 or frames < (5 if task == "t2v" else 1) or (frames-1) % 4 or not math.isfinite(fps) or fps <= 0:
    raise SystemExit("Invalid media shape: positive size divisible by 16, 4k+1 frames (T2V >=5), positive finite fps required")
PY
VERSION="${WAN21_EXPECT_TORCH_VERSION:-2.9.1+shine.hmpi.kblas2521.beta0fix1}"
[[ -z "${WAN21_CACHE_ROOT:-}" ]] || { echo "This entry reads raw media online; WAN21_CACHE_ROOT is unsupported" >&2; exit 2; }
STORED_PRECISION="${WAN21_STORED_PRECISION:-fp32}"
[[ "$STORED_PRECISION" == fp32 || "$STORED_PRECISION" == bf16 ]] || exit 2
[[ "${WAN21_TEXT_PRECISION:-fp32}" == fp32 || "${WAN21_TEXT_PRECISION:-fp32}" == bf16 ]] || exit 2
if [[ "$STORED_PRECISION" == bf16 && "${WAN21_TRAIN_PRECISION:-bf16}" != bf16 ]]; then
 echo "BF16 stored state requires BF16 computation" >&2; exit 2
fi
HIERARCHICAL_COMM="${WAN21_HIERARCHICAL_COMM:-0}"
if [[ -z "${WAN21_HIERARCHICAL_COMM:-}" && "$STORED_PRECISION" == bf16 && "$RPN" == 16 ]]; then HIERARCHICAL_COMM=1; fi
if [[ "$HIERARCHICAL_COMM" == 1 && "$RPN" != 16 ]]; then
 echo "Hierarchical communication requires 16 ranks per node" >&2; exit 2
fi
# A NUMA affinity allocation exposes DDR nodes only; FFN HBM needs a full-node cpuset.
HBM_ALLOC=0
if [[ "$TASK" == t2v && "$STORED_PRECISION" == bf16 && "${WAN21_NATIVE_OPS:-1}" == 1 &&
      "${WAN21_FFN_HBM:-1}" == 1 && "${WAN21_VIDEO_FRAMES:-49}" == 49 &&
      "${WAN21_IMAGE_SIZE:-256}" == 256 && "${WAN21_TRAIN_BATCH_SIZE:-1}" == 1 ]]; then
 HBM_ALLOC=1
fi
FULL_NODE_ALLOC="${WAN21_FULL_NODE_ALLOC:-$HBM_ALLOC}"
TIME_ARGS=(); NODE_ARGS=(); DEP_ARGS=()
if [[ -n "${WAN21_TIME_LIMIT_SECONDS:-}" ]]; then
 [[ "$WAN21_TIME_LIMIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || exit 2
 TIME_ARGS=(-T "$WAN21_TIME_LIMIT_SECONDS")
fi
[[ -z "${WAN21_NODE_LIST:-}" ]] || NODE_ARGS=(-nl "$WAN21_NODE_LIST")
[[ -z "${WAN21_JOB_DEPENDENCY:-}" ]] || DEP_ARGS=(-D "$WAN21_JOB_DEPENDENCY")
LOG_ROOT="${WAN21_LOG_ROOT:-$(dirname "$RUN_DIR")/donau_logs}"; mkdir -p "$LOG_ROOT"
if [[ "$FULL_NODE_ALLOC" == 1 ]]; then
  RESOURCE_ARGS=(-N "$NNODES" -nn "$NNODES" -R "cpu=608;mem=${WAN21_FULL_NODE_MEM_GIB:-400}G" -x job)
else
  RESOURCE_ARGS=(-N "$((NNODES * RPN))" -rpn "$RPN" -nn "$NNODES" -R "cpu=38;mem=${MEM_GIB}G" -x job -a 'numa[count=1,distribution=pack]')
fi
"$DSUB" -n "wan21_${TASK}_${NNODES}n${RPN}r" --mpi hmpi \
 -q "$WAN21_QUEUE" "${RESOURCE_ARGS[@]}" \
 ${TIME_ARGS[@]+"${TIME_ARGS[@]}"} ${NODE_ARGS[@]+"${NODE_ARGS[@]}"} ${DEP_ARGS[@]+"${DEP_ARGS[@]}"} -oo "$LOG_ROOT/${TASK}_%J.out" -eo "$LOG_ROOT/${TASK}_%J.err" \
 /usr/bin/env "WAN21_ENV_ACTIVATE=$ENV_ACTIVATE" "WAN21_BLAS_CHECK=$WAN21_BLAS_CHECK" "WAN21_EXPECT_TORCH_VERSION=$VERSION" \
 "WAN21_EXPECT_BLAS=${WAN21_EXPECT_BLAS:-}" "WAN21_TRAIN_PRECISION=${WAN21_TRAIN_PRECISION:-bf16}" \
  "WAN21_STORED_PRECISION=${WAN21_STORED_PRECISION:-fp32}" \
 "FINAL_GEMM_THREADS=${FINAL_GEMM_THREADS:-36}" "WAN21_BF16_DIRECT_REDUCE=${WAN21_BF16_DIRECT_REDUCE:-1}" "WAN21_BF16_DIRECT_GATHER=${WAN21_BF16_DIRECT_GATHER:-0}" "WAN21_BF16_COMM_CHUNK_MB=${WAN21_BF16_COMM_CHUNK_MB:-64}" "WAN21_EARLY_COMM_CHUNK_MB=${WAN21_EARLY_COMM_CHUNK_MB:-0}" "WAN21_OWNER3_FIRST_COMM_DELAY_MS=${WAN21_OWNER3_FIRST_COMM_DELAY_MS:-0}" "WAN21_OWNER0_BUCKET_MB=${WAN21_OWNER0_BUCKET_MB:-0}" "WAN21_STRIPED_SHARED_WEIGHTS=${WAN21_STRIPED_SHARED_WEIGHTS:-1}" "WAN21_WEIGHT_STRIPE_MB=${WAN21_WEIGHT_STRIPE_MB:-2}" "WAN21_PIPELINE_LOCAL_REDUCE=${WAN21_PIPELINE_LOCAL_REDUCE:-1}" "WAN21_LOCAL_REDUCE_LANES=${WAN21_LOCAL_REDUCE_LANES:-1}" "WAN21_HIERARCHICAL_COMM=$HIERARCHICAL_COMM" \
 "WAN21_UCX_RNDV_RAILS=${WAN21_UCX_RNDV_RAILS:-8}" "WAN21_UCX_DIE_LOCAL=${WAN21_UCX_DIE_LOCAL:-0}" "WAN21_MPI_ALLTOALL_ALG=${WAN21_MPI_ALLTOALL_ALG:-0}" "WAN21_MPI_ALLTOALL_MAX_REQUESTS=${WAN21_MPI_ALLTOALL_MAX_REQUESTS:-4}" "WAN21_COMM_SCRATCH_HBM=${WAN21_COMM_SCRATCH_HBM:-0}" "WAN21_HBM_PROBE_ONLY=${WAN21_HBM_PROBE_ONLY:-0}" "WAN21_SHARED_HBM_PROBE=${WAN21_SHARED_HBM_PROBE:-0}" "WAN21_SHARED_WEIGHTS_HBM=${WAN21_SHARED_WEIGHTS_HBM:-0}" "WAN21_CODE_ROOT=$CODE_ROOT" "WAN21_TRAIN_DATA=$DATA_INDEX" "WAN21_TRAIN_RUN_DIR=$RUN_DIR" "WAN21_NNODES=$NNODES" "WAN21_PACK_RPN=$RPN" "WAN21_COMM_OWNER_MODE=${WAN21_COMM_OWNER_MODE:-contiguous}" "WAN21_COMM_SKIP_CLIP_BARRIER=${WAN21_COMM_SKIP_CLIP_BARRIER:-1}" \
 "WAN21_TRAIN_MAX_STEPS=$MAX_STEPS" "WAN21_TRAIN_START_MODE=$MODE" \
 "WAN21_TRAIN_LR_TOTAL_STEPS=${WAN21_TRAIN_LR_TOTAL_STEPS:-2000}" \
 "WAN21_TRAIN_GRAD_ACCUM=${WAN21_TRAIN_GRAD_ACCUM:-1}" "WAN21_TRAIN_BATCH_SIZE=${WAN21_TRAIN_BATCH_SIZE:-1}" \
 "WAN21_TRAIN_SAVE_EVERY=${WAN21_TRAIN_SAVE_EVERY:-200}" "WAN21_TRAIN_SAVE_FINAL=${WAN21_TRAIN_SAVE_FINAL:-1}" \
 "WAN21_TRAIN_KEEP_LAST_CHECKPOINTS=${WAN21_TRAIN_KEEP_LAST_CHECKPOINTS:-0}" \
 "WAN21_TRAIN_KEEP_CHECKPOINT_STEPS=${WAN21_TRAIN_KEEP_CHECKPOINT_STEPS:-}" \
 "WAN21_TRAIN_WARMUP_STEPS=${WAN21_TRAIN_WARMUP_STEPS:-2}" "WAN21_TRAIN_SEED=${WAN21_TRAIN_SEED:-2026}" \
 "WAN21_TRAIN_BUCKET_MB=${WAN21_TRAIN_BUCKET_MB:-128}" "WAN21_TRAIN_BENCHMARK=${WAN21_TRAIN_BENCHMARK:-0}" \
 "WAN21_VERIFY_PARAMETER_UPDATES=${WAN21_VERIFY_PARAMETER_UPDATES:-0}" \
 "WAN21_ENCODER_WEIGHTS=$WAN21_ENCODER_WEIGHTS" \
 "WAN21_IMAGE_SIZE=${WAN21_IMAGE_SIZE:-256}" "WAN21_IMAGE_BUCKET=${WAN21_IMAGE_BUCKET:-256/256x256}" \
 "WAN21_VIDEO_FRAMES=${WAN21_VIDEO_FRAMES:-49}" "WAN21_VIDEO_FPS=${WAN21_VIDEO_FPS:-16}" \
 "WAN21_TRAIN_TASK=$TASK" "WAN21_VIDEO_ROOT=${WAN21_VIDEO_ROOT:-}" \
 "WAN21_TEXT_PRECISION=${WAN21_TEXT_PRECISION:-fp32}" "WAN21_NATIVE_OPS=${WAN21_NATIVE_OPS:-1}" \
 "WAN21_MONITOR_NODE_MEMORY=${WAN21_MONITOR_NODE_MEMORY:-0}" "WAN21_RECOMPUTE_BLOCKS=${WAN21_RECOMPUTE_BLOCKS:-0}" \
  "WAN21_TRAIN_PROFILE_START_STEP=${WAN21_TRAIN_PROFILE_START_STEP:-0}" \
  "WAN21_TRAIN_PROFILE_STEPS=${WAN21_TRAIN_PROFILE_STEPS:-0}" \
  "WAN21_TRAIN_PROFILE_RANK=${WAN21_TRAIN_PROFILE_RANK:-0}" \
  "WAN21_TRAIN_PROFILE_DETAILED=${WAN21_TRAIN_PROFILE_DETAILED:-0}" \
  "WAN21_TRAIN_PROFILE_MEMORY=${WAN21_TRAIN_PROFILE_MEMORY:-0}" \
  "WAN21_TRAIN_PROFILE_ALL_RANKS=${WAN21_TRAIN_PROFILE_ALL_RANKS:-0}" \
 "WAN21_INIT_CHECKPOINT=${WAN21_INIT_CHECKPOINT:-}" "WAN21_SHARED_ROOT=${WAN21_SHARED_ROOT:-/tmp}" \
 "WAN21_TRAIN_GEMM_BRIDGE=${WAN21_TRAIN_GEMM_BRIDGE:-}" "WAN21_TRAIN_GEMM_LIB=${WAN21_TRAIN_GEMM_LIB:-$CODE_ROOT/kunpeng/wan21_gemm_library/libsme_gemm.so}" \
 "WAN21_GEMM_STATS=${WAN21_GEMM_STATS:-}" \
 "WAN21_XH_LINEAR_OP=${WAN21_XH_LINEAR_OP:-$CODE_ROOT/kunpeng/wan21_gemm_library/xhpops_biasfix.so}" "WAN21_XH_STATS=${WAN21_XH_STATS:-}" "WAN21_FFN_HBM=${WAN21_FFN_HBM:-1}" "WAN21_FFN_HBM_POOL=${WAN21_FFN_HBM_POOL:-1}" \
 "$SCRIPT_DIR/run_wan21_t2i_multinode.sh"
