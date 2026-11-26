#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$CODE_ROOT/src/train"
: "${WAN21_TRAIN_DATA:?}" "${WAN21_TRAIN_RUN_DIR:?}" "${WAN21_NNODES:?}"
: "${WAN21_ENV_ACTIVATE:?}" "${WAN21_EXPECT_TORCH_VERSION:?}" "${CCS_ALLOC_FILE:?}"
RPN="${WAN21_PACK_RPN:-16}"; NNODES="$WAN21_NNODES"; WORLD=$((RPN * NNODES))
export WAN21_SHARED_ROOT="${WAN21_SHARED_ROOT:-/dev/shm}"
[[ "$WAN21_SHARED_ROOT" == /* && -d "$WAN21_SHARED_ROOT" ]] || { echo "WAN21_SHARED_ROOT must be an existing absolute directory" >&2; exit 2; }
(( RPN >= 4 && RPN <= 16 && RPN % 4 == 0 )) || exit 2
JOB_TOKEN="${CCS_JOB_ID:?}"; [[ "$JOB_TOKEN" =~ ^[0-9]+$ ]] || exit 2
RUN_DIR="$WAN21_TRAIN_RUN_DIR"; mkdir -p "$RUN_DIR/logs"
TEE_LOG="$RUN_DIR/logs/train_${JOB_TOKEN}.tee.log"
exec > >(tee -a "$TEE_LOG") 2>&1
printf 'WAN21_ONLINE_LOG=%s\n' "$TEE_LOG"
source "$WAN21_ENV_ACTIVATE"
: "${WAN21_PYTHON:?activation must export the runtime Python executable}" "${WAN21_BLAS_CHECK:?set the site BLAS checker}"
export PYTHONPATH="$SCRIPT_DIR:$CODE_ROOT:$CODE_ROOT/third_party/Wan2.1${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WAN21_PIPELINE_LOCAL_REDUCE="${WAN21_PIPELINE_LOCAL_REDUCE:-1}" WAN21_STRIPED_SHARED_WEIGHTS="${WAN21_STRIPED_SHARED_WEIGHTS:-1}" WAN21_WEIGHT_STRIPE_MB="${WAN21_WEIGHT_STRIPE_MB:-2}" FINAL_GEMM_THREADS="${FINAL_GEMM_THREADS:-36}" WAN21_BF16_DIRECT_REDUCE="${WAN21_BF16_DIRECT_REDUCE:-1}" WAN21_BF16_DIRECT_GATHER="${WAN21_BF16_DIRECT_GATHER:-0}" WAN21_BF16_COMM_CHUNK_MB="${WAN21_BF16_COMM_CHUNK_MB:-64}" WAN21_OWNER0_BUCKET_MB="${WAN21_OWNER0_BUCKET_MB:-0}"
printf "WAN21_FINAL_GEMM_THREADS=%s\n" "$FINAL_GEMM_THREADS"
RUNTIME_ROOT="$RUN_DIR/.runtime.$JOB_TOKEN"; mkdir -p "$RUNTIME_ROOT/tmp" "$RUNTIME_ROOT/huggingface"
export TMPDIR="$RUNTIME_ROOT/tmp" XDG_CACHE_HOME="$RUNTIME_ROOT" HF_HOME="$RUNTIME_ROOT/huggingface"
export DIFFUSERS_CACHE="$RUNTIME_ROOT/diffusers"
[[ "${WAN21_UCX_RNDV_RAILS:-8}" =~ ^(1|2|4|8)$ ]] || { echo "invalid UCX rendezvous rail count" >&2; exit 2; }
export UCX_TLS=self,rc,sm UCX_NET_DEVICES=all UCX_MAX_RNDV_RAILS="${WAN21_UCX_RNDV_RAILS:-8}" UCX_MAX_EAGER_RAILS=1
export UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y OMPI_MCA_coll='^han'
export WAN21_CODE_ROOT="$CODE_ROOT"
export WAN21_NATIVE_OPS="${WAN21_NATIVE_OPS:-1}"
export WAN21_FFN_HBM="${WAN21_FFN_HBM:-1}" WAN21_FFN_HBM_POOL="${WAN21_FFN_HBM_POOL:-1}"
# Fixed native FFN scratch is valid only for the source model geometry and stored BF16.
if [[ "$WAN21_NATIVE_OPS" != 1 || "${WAN21_STORED_PRECISION:-fp32}" != bf16 ||
      "${WAN21_TRAIN_TASK:-t2i}" != t2v || "${WAN21_VIDEO_FRAMES:-49}" != 49 ||
      "${WAN21_IMAGE_SIZE:-256}" != 256 || "${WAN21_TRAIN_BATCH_SIZE:-1}" != 1 ]]; then
 export WAN21_FFN_HBM=0 WAN21_FFN_HBM_POOL=0
fi
[[ "${WAN21_RECOMPUTE_BLOCKS:-0}" != 1 ]] || export WAN21_FFN_HBM_POOL=0
if [[ "$WAN21_NATIVE_OPS" != 1 ]]; then
 unset WAN21_XH_LINEAR_OP WAN21_TRAIN_GEMM_BRIDGE
 export WAN21_FFN_HBM=0 WAN21_FFN_HBM_POOL=0
fi
MPI_ENV=(-x WAN21_NATIVE_OPS -x WAN21_CODE_ROOT -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x PYTHONDONTWRITEBYTECODE -x PYTHONNOUSERSITE
 -x TOKENIZERS_PARALLELISM -x HF_HUB_OFFLINE -x TRANSFORMERS_OFFLINE -x TMPDIR
 -x XDG_CACHE_HOME -x HF_HOME -x DIFFUSERS_CACHE -x WAN21_SHARED_ROOT -x UCX_TLS -x UCX_NET_DEVICES
 -x UCX_MAX_RNDV_RAILS -x UCX_MAX_EAGER_RAILS -x UCX_RC_VERBS_ROCE_LOCAL_SUBNET
 -x UCX_UD_VERBS_ROCE_LOCAL_SUBNET -x OMPI_MCA_coll -x WAN21_RECOMPUTE_BLOCKS -x WAN21_STORED_PRECISION -x WAN21_BF16_DIRECT_REDUCE -x WAN21_BF16_DIRECT_GATHER -x WAN21_BF16_COMM_CHUNK_MB -x WAN21_EARLY_COMM_CHUNK_MB -x WAN21_OWNER3_FIRST_COMM_DELAY_MS -x WAN21_OWNER0_BUCKET_MB -x WAN21_STRIPED_SHARED_WEIGHTS -x WAN21_WEIGHT_STRIPE_MB -x WAN21_PIPELINE_LOCAL_REDUCE -x WAN21_LOCAL_REDUCE_LANES -x WAN21_HIERARCHICAL_COMM -x WAN21_UCX_DIE_LOCAL -x WAN21_COMM_SCRATCH_HBM -x WAN21_SHARED_HBM_PROBE -x WAN21_SHARED_WEIGHTS_HBM -x FINAL_GEMM_THREADS -x WAN21_TRAIN_PROFILE_ALL_RANKS -x WAN21_COMM_OWNER_MODE -x WAN21_COMM_SKIP_CLIP_BARRIER)
if [[ "${WAN21_MPI_ALLTOALL_ALG:-0}" != 0 ]]; then
  [[ "${WAN21_MPI_ALLTOALL_ALG}" =~ ^(1|2|3|4)$ ]] || exit 2
  [[ "${WAN21_MPI_ALLTOALL_MAX_REQUESTS:-4}" =~ ^(0|2|4|8)$ ]] || exit 2
  export OMPI_MCA_coll_tuned_use_dynamic_rules=1
  export OMPI_MCA_coll_tuned_alltoall_algorithm="${WAN21_MPI_ALLTOALL_ALG}"
  export OMPI_MCA_coll_tuned_alltoall_algorithm_max_requests="${WAN21_MPI_ALLTOALL_MAX_REQUESTS:-4}"
  MPI_ENV+=(-x OMPI_MCA_coll_tuned_use_dynamic_rules
            -x OMPI_MCA_coll_tuned_alltoall_algorithm
            -x OMPI_MCA_coll_tuned_alltoall_algorithm_max_requests)
fi
if [[ -n "${WAN21_TRAIN_GEMM_BRIDGE:-}" ]]; then
  [[ -r "$WAN21_TRAIN_GEMM_BRIDGE" && -r "${WAN21_TRAIN_GEMM_LIB:-}" ]] || {
    echo "GEMM bridge or optimized library is missing" >&2; exit 2;
  }
  export WAN21_TRAIN_GEMM_BRIDGE WAN21_TRAIN_GEMM_LIB WAN21_GEMM_STATS
  MPI_ENV+=(-x WAN21_TRAIN_GEMM_BRIDGE -x WAN21_TRAIN_GEMM_LIB -x WAN21_GEMM_STATS)
  sha256sum "$WAN21_TRAIN_GEMM_BRIDGE" "$WAN21_TRAIN_GEMM_LIB"
fi
if [[ -n "${WAN21_XH_LINEAR_OP:-}" ]]; then
  [[ -r "$WAN21_XH_LINEAR_OP" && -r "${WAN21_TRAIN_GEMM_LIB:-}" ]] || {
    echo "xh_linear operator or SME GEMM library is missing" >&2; exit 2;
  }
  export WAN21_XH_LINEAR_OP WAN21_XH_STATS WAN21_TRAIN_GEMM_LIB
  MPI_ENV+=(-x WAN21_XH_LINEAR_OP -x WAN21_XH_STATS -x WAN21_TRAIN_GEMM_LIB -x WAN21_FFN_HBM -x WAN21_FFN_HBM_POOL)
  sha256sum "$WAN21_XH_LINEAR_OP"
fi
if [[ "$WAN21_NATIVE_OPS" == 1 ]]; then
gcc -O3 -fPIC -shared -fopenmp -march=armv8.2-a+sve2 \
 "$NATIVE_DIR/sve_gelu.c" -lm -o "$RUNTIME_ROOT/sve_gelu.so"
export WAN21_SVE_GELU_LIB="$RUNTIME_ROOT/sve_gelu.so"
MPI_ENV+=(-x WAN21_SVE_GELU_LIB)
gcc -O3 -fPIC -shared -fopenmp -march=armv8.2-a+sve2 \
 "$NATIVE_DIR/sve_layernorm.c" -lm -o "$RUNTIME_ROOT/sve_layernorm.so"
export WAN21_SVE_LAYERNORM_LIB="$RUNTIME_ROOT/sve_layernorm.so"
MPI_ENV+=(-x WAN21_SVE_LAYERNORM_LIB)
gcc -O3 -fPIC -shared -fopenmp -march=armv8.2-a+sve2 \
 "$NATIVE_DIR/sve_residual.c" -o "$RUNTIME_ROOT/sve_residual.so"
export WAN21_SVE_RESIDUAL_LIB="$RUNTIME_ROOT/sve_residual.so"
gcc -O3 -fPIC -shared -fopenmp -march=armv8.2-a+sve2 \
 "$NATIVE_DIR/sve_modulation.c" -o "$RUNTIME_ROOT/sve_modulation.so"
export WAN21_SVE_MODULATION_LIB="$RUNTIME_ROOT/sve_modulation.so"
MPI_ENV+=(-x WAN21_SVE_MODULATION_LIB)
MPI_ENV+=(-x WAN21_SVE_RESIDUAL_LIB)
fi
mapfile -t NODES < <(awk 'NF {print $1}' "$CCS_ALLOC_FILE" | sort -u)
(( ${#NODES[@]} == NNODES )) || { echo "Unexpected allocation" >&2; exit 1; }
RANKFILE="$RUNTIME_ROOT/rankfile"; LEADERS="$RUNTIME_ROOT/leaders"
: > "$RANKFILE"; : > "$LEADERS"; rank=0
for i in "${!NODES[@]}"; do
 printf 'rank %d=%s slot=0\n' "$i" "${NODES[$i]}" >> "$LEADERS"
 for ((local=0; local<RPN; local++)); do
  first=$((local * 38))
  printf 'rank %d=%s slot=%d-%d\n' "$rank" "${NODES[$i]}" "$first" "$((first + 36))" >> "$RANKFILE"
  rank=$((rank + 1))
 done
done
cleanup() {
 result=$?; trap - EXIT
 # Always clean this node first, even when MPI teardown cannot start after OOM.
 /usr/bin/python3 -c '
import pathlib,os,sys
prefix="wan21-online.%s.%s.die" % (os.getuid(),sys.argv[1])
for die in range(4):
 for suffix in ("weights","gradients","gradients.ready"):
  path=pathlib.Path(os.environ["WAN21_SHARED_ROOT"])/(prefix+str(die)+"."+suffix)
  if path.exists(): path.unlink()
' "$JOB_TOKEN" || true
 if (( NNODES > 1 )); then
 mpirun -n "$NNODES" --rankfile "$LEADERS" --mca rmaps_rank_file_physical true "${MPI_ENV[@]}" \
 /usr/bin/python3 -c '
import pathlib,os,sys
prefix="wan21-online.%s.%s.die" % (os.getuid(),sys.argv[1])
for die in range(4):
 for suffix in ("weights","gradients","gradients.ready"):
  path=pathlib.Path(os.environ["WAN21_SHARED_ROOT"])/(prefix+str(die)+"."+suffix)
  if path.exists(): path.unlink()
' "$JOB_TOKEN" || true
 fi
 rm -rf -- "$RUNTIME_ROOT"
 exit "$result"
}
trap cleanup EXIT
# Each four-rank group maps one model and four gradient buffers in the stored dtype.
# Reject exhausted node-local tmpfs before mmap writes can raise SIGBUS.
mpirun -n "$NNODES" --rankfile "$LEADERS" --mca rmaps_rank_file_physical true "${MPI_ENV[@]}" \
 /usr/bin/python3 -c '
import json,os,socket,sys
rpn=int(sys.argv[1])
precision=os.environ.get("WAN21_STORED_PRECISION", "fp32")
if precision not in ("fp32", "bf16"):
    raise SystemExit(f"Invalid WAN21_STORED_PRECISION={precision}")
required=1418996800*(2 if precision=="bf16" else 4)*(rpn+rpn//4)+(1<<30)
root=os.environ["WAN21_SHARED_ROOT"];fs=os.statvfs(root)
free=fs.f_bavail*fs.f_frsize
mem={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in open("/proc/meminfo")}
record={"host":socket.gethostname(),"shared_root":root,"shared_total_bytes":fs.f_blocks*fs.f_frsize,"shared_free_bytes":free,"shared_required_bytes":required,
        "mem_available":mem["MemAvailable"],"ranks_per_node":rpn,"stored_precision":precision}
print("WAN21_NODE_MEMORY "+json.dumps(record),flush=True)
if free < required:
    raise SystemExit(f"Insufficient {root} capacity before shared model initialization")
' "$RPN"
"$WAN21_PYTHON" -c 'import os,torch; assert torch.__version__==os.environ["WAN21_EXPECT_TORCH_VERSION"]; assert torch.distributed.is_mpi_available()'
case "${WAN21_EXPECT_BLAS:-$WAN21_EXPECT_TORCH_VERSION}" in
 *openblas*) "$WAN21_BLAS_CHECK" "${VIRTUAL_ENV:?}" ;;
 *kblas*) "$WAN21_BLAS_CHECK" ;;
 *) echo "Unsupported BLAS expectation" >&2; exit 2 ;;
esac
sha256sum "$CODE_ROOT/src/train/shared_train.py" "$CODE_ROOT/src/train/shared_parallel.py" \
 "$CODE_ROOT/src/train/selective_ffn.py" "$CODE_ROOT/src/train/rank_hbm.py" \
 "$CODE_ROOT/src/train/xh_linear.py" "$CODE_ROOT/src/train/shared_checkpoint.py" \
 "$CODE_ROOT/src/data/online_training.py" "$CODE_ROOT/third_party/Wan2.1/wan/modules/model.py"
if [[ "$WAN21_FFN_HBM" == 1 ]]; then
printf "WAN21_HBM_BIND_PREFLIGHT_BEGIN\n"
mpirun -n "$WORLD" --rankfile "$RANKFILE" --mca rmaps_rank_file_physical true "${MPI_ENV[@]}" \
 "$CODE_ROOT/kunpeng/run_wan21_online_rank.sh" /usr/bin/python3 -u \
 "$SCRIPT_DIR/probe_hbm.py"
printf "WAN21_HBM_BIND_PREFLIGHT_OK\n"
fi
if [[ "${WAN21_HBM_PROBE_ONLY:-0}" == 1 ]]; then
 printf "WAN21_HBM_PROBE_ONLY_OK nodes=%s ranks=%s\n" "$NNODES" "$WORLD"
 exit 0
fi
ARGS=(--data-index "$WAN21_TRAIN_DATA" --run-dir "$RUN_DIR" --nnodes "$NNODES" --dies-per-node "$((RPN / 4))"
 --job-token "$JOB_TOKEN" --precision "${WAN21_TRAIN_PRECISION:-bf16}"
 --stored-precision "${WAN21_STORED_PRECISION:-fp32}" --text-precision "${WAN21_TEXT_PRECISION:-fp32}"
 --max-steps "${WAN21_TRAIN_MAX_STEPS:-40}" --lr-total-steps "${WAN21_TRAIN_LR_TOTAL_STEPS:-2000}"
 --warmup-steps "${WAN21_TRAIN_WARMUP_STEPS:-2}" --seed "${WAN21_TRAIN_SEED:-2026}"
 --batch-size "${WAN21_TRAIN_BATCH_SIZE:-1}" --grad-accum "${WAN21_TRAIN_GRAD_ACCUM:-1}"
 --bucket-mb "${WAN21_TRAIN_BUCKET_MB:-128}" --save-every "${WAN21_TRAIN_SAVE_EVERY:-200}"
 --keep-last-checkpoints "${WAN21_TRAIN_KEEP_LAST_CHECKPOINTS:-0}"
 --keep-checkpoint-steps "${WAN21_TRAIN_KEEP_CHECKPOINT_STEPS:-}"
 --encoder-weights "${WAN21_ENCODER_WEIGHTS:?set the encoder weights directory}"
 --image-size "${WAN21_IMAGE_SIZE:-256}" --image-bucket "${WAN21_IMAGE_BUCKET:-256/256x256}"
 --video-frames "${WAN21_VIDEO_FRAMES:-49}" --video-fps "${WAN21_VIDEO_FPS:-16}"
 --training-task "${WAN21_TRAIN_TASK:-t2i}")
[[ -z "${WAN21_VIDEO_ROOT:-}" ]] || ARGS+=(--video-root "$WAN21_VIDEO_ROOT")
[[ "${WAN21_MONITOR_NODE_MEMORY:-0}" != 1 ]] || ARGS+=(--monitor-node-memory)
[[ "${WAN21_TRAIN_START_MODE:-scratch}" != resume ]] || ARGS+=(--auto-resume)
[[ -z "${WAN21_INIT_CHECKPOINT:-}" ]] || ARGS+=(--init-checkpoint "$WAN21_INIT_CHECKPOINT")
[[ "${WAN21_VERIFY_PARAMETER_UPDATES:-0}" != 1 ]] || ARGS+=(--verify-updates)
if [[ "${WAN21_TRAIN_PROFILE_STEPS:-0}" != 0 ]]; then
  ARGS+=(--profile-start-step "${WAN21_TRAIN_PROFILE_START_STEP:-0}"
    --profile-steps "$WAN21_TRAIN_PROFILE_STEPS" --profile-rank "${WAN21_TRAIN_PROFILE_RANK:-0}")
fi
[[ "${WAN21_TRAIN_PROFILE_DETAILED:-0}" != 1 ]] || ARGS+=(--profile-detailed)
[[ "${WAN21_TRAIN_PROFILE_MEMORY:-0}" != 1 ]] || ARGS+=(--profile-memory)
if [[ "${WAN21_TRAIN_BENCHMARK:-0}" == 1 ]]; then
 [[ "${WAN21_TRAIN_SAVE_EVERY:-200}" == 0 ]] || { echo "Benchmark requires SAVE_EVERY=0" >&2; exit 2; }
 ARGS+=(--no-save-final)
elif [[ "${WAN21_TRAIN_SAVE_FINAL:-1}" == 0 ]]; then
 ARGS+=(--no-save-final)
fi
cd "$CODE_ROOT"
RANK_LAUNCHER="$CODE_ROOT/kunpeng/run_wan21_online_rank.sh"
if [[ "${WAN21_UCX_DIE_LOCAL:-0}" == 1 ]]; then
 RANK_LAUNCHER="$SCRIPT_DIR/run_wan21_local_rails.sh"
fi
mpirun -n "$WORLD" --rankfile "$RANKFILE" --mca rmaps_rank_file_physical true \
 --mca grpcomm_direct_priority 100 --mca coll_tuned_use_dynamic_rules true "${MPI_ENV[@]}" \
 "$RANK_LAUNCHER" "$WAN21_PYTHON" -u "$CODE_ROOT/src/train/shared_train.py" "${ARGS[@]}"
TASK="${WAN21_TRAIN_TASK:-t2i}"
INPUT_MODE=raw_online
printf 'WAN21_%s_MULTINODE_OK nnodes=%s world=%s input=%s run_dir=%s\n' "${TASK^^}" "$NNODES" "$WORLD" "$INPUT_MODE" "$RUN_DIR"
