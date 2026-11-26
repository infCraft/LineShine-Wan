#!/usr/bin/env bash
# Reusable core (two-node) and raw-media (one-node) training acceptance checks.
set -euo pipefail
ulimit -c 0
CODE="$(cd "$(dirname "$0")/.." && pwd -P)"
PORT_MODE="${WAN21_CHECK_MODE:-core}"
[[ "$PORT_MODE" == core || "$PORT_MODE" == media || "$PORT_MODE" == media49 ]] || exit 2
[[ "${CCS_JOB_ID:?}" =~ ^[0-9]+$ ]] || exit 2
PORT_WORK=/tmp/wan21-shared-checks.$CCS_JOB_ID
mkdir "$PORT_WORK"
exec > >(tee -a "$PORT_WORK/run.log") 2>&1
finish() {
 result=$?; trap - EXIT
 printf 'PORT_GATE_DONE rc=%s\n' "$result"
 for ((i=0;i<${WAN21_CHECK_CAPTURE_SECONDS:-0};i++)); do [[ ! -e "$PORT_WORK/ack" ]] || break; sleep 1; done
 if [[ -n "${WAN21_PYTHON:-}" && -f "$PORT_WORK/leaders" ]]; then
  mpirun --stdin none -n "$PORT_NNODES" --rankfile "$PORT_WORK/leaders" --mca rmaps_rank_file_physical true \
   /bin/bash -c 'if [[ $(hostname) != "$2" ]]; then rm -rf -- "$1"; fi' bash "$PORT_WORK" "$(hostname)" || true
 fi
 rm -rf -- "$PORT_WORK"
 printf 'PORT_GATE_CLEANUP_OK rc=%s\n' "$result"
 sleep 2
 exit "$result"
}
trap finish EXIT
source "${WAN21_ENV_ACTIVATE:?set the site runtime activation script}"
cd "$CODE"
export PYTHONPATH="$CODE:$CODE/third_party/Wan2.1" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMPI_MCA_orte_tmpdir_base=/tmp
export UCX_TLS=self,rc,sm UCX_NET_DEVICES=all UCX_MAX_RNDV_RAILS=8 UCX_MAX_EAGER_RAILS=1
export UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y OMPI_MCA_coll='^han'
export WAN21_NATIVE_OPS=1 WAN21_FFN_HBM=0 WAN21_FFN_HBM_POOL=0 WAN21_RECOMPUTE_BLOCKS=0
export WAN21_BF16_DIRECT_REDUCE=1 WAN21_BF16_DIRECT_GATHER=0 WAN21_BF16_COMM_CHUNK_MB=64
export WAN21_PIPELINE_LOCAL_REDUCE=1 WAN21_LOCAL_REDUCE_LANES=1 WAN21_HIERARCHICAL_COMM=1
export WAN21_STRIPED_SHARED_WEIGHTS=1 WAN21_WEIGHT_STRIPE_MB=2 WAN21_SHARED_ROOT="$PORT_WORK"
export WAN21_COMM_OWNER_MODE=contiguous WAN21_COMM_SKIP_CLIP_BARRIER=1 FINAL_GEMM_THREADS=36
export WAN21_TRAIN_GEMM_LIB="$CODE/kunpeng/wan21_gemm_library/libsme_gemm.so"
export WAN21_XH_LINEAR_OP="$CODE/kunpeng/wan21_gemm_library/xhpops_biasfix.so"
export WAN21_SVE_GELU_LIB="$PORT_WORK/sve_gelu.so" WAN21_SVE_LAYERNORM_LIB="$PORT_WORK/sve_layernorm.so"
export WAN21_SVE_RESIDUAL_LIB="$PORT_WORK/sve_residual.so" WAN21_SVE_MODULATION_LIB="$PORT_WORK/sve_modulation.so"
export TMPDIR="$PORT_WORK/tmp" XDG_CACHE_HOME="$PORT_WORK/cache" HF_HOME="$PORT_WORK/huggingface"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"
mapfile -t NODES < <(awk 'NF {print $1}' "$CCS_ALLOC_FILE" | sort -u)
PORT_NNODES=${#NODES[@]}
if [[ "$PORT_MODE" == core ]]; then [[ "$PORT_NNODES" == 2 ]]; else [[ "$PORT_NNODES" == 1 ]]; fi
: > "$PORT_WORK/ranks"; : > "$PORT_WORK/leaders"; : > "$PORT_WORK/four"
for ((node=0;node<PORT_NNODES;node++)); do
 printf 'rank %d=%s slot=0-36\n' "$node" "${NODES[$node]}" >> "$PORT_WORK/leaders"
 for ((local=0;local<16;local++)); do
  printf 'rank %d=%s slot=%d-%d\n' "$((node*16+local))" "${NODES[$node]}" "$((local*38))" "$((local*38+36))" >> "$PORT_WORK/ranks"
 done
done
printf 'rank 0=%s slot=0-36\n' "${NODES[0]}" > "$PORT_WORK/one"
for ((local=0;local<4;local++)); do
 printf 'rank %d=%s slot=%d-%d\n' "$local" "${NODES[0]}" "$((local*38))" "$((local*38+36))" >> "$PORT_WORK/four"
done
MPI_ENV=()
for name in PATH LD_LIBRARY_PATH PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONNOUSERSITE \
 TOKENIZERS_PARALLELISM HF_HUB_OFFLINE TRANSFORMERS_OFFLINE UCX_TLS UCX_NET_DEVICES \
 UCX_MAX_RNDV_RAILS UCX_MAX_EAGER_RAILS UCX_RC_VERBS_ROCE_LOCAL_SUBNET UCX_UD_VERBS_ROCE_LOCAL_SUBNET \
 OMPI_MCA_coll WAN21_NATIVE_OPS WAN21_FFN_HBM WAN21_FFN_HBM_POOL WAN21_RECOMPUTE_BLOCKS \
 WAN21_BF16_DIRECT_REDUCE WAN21_BF16_DIRECT_GATHER WAN21_BF16_COMM_CHUNK_MB \
 WAN21_PIPELINE_LOCAL_REDUCE WAN21_LOCAL_REDUCE_LANES WAN21_HIERARCHICAL_COMM \
 WAN21_STRIPED_SHARED_WEIGHTS WAN21_WEIGHT_STRIPE_MB WAN21_SHARED_ROOT WAN21_COMM_OWNER_MODE \
 WAN21_COMM_SKIP_CLIP_BARRIER FINAL_GEMM_THREADS WAN21_TRAIN_GEMM_LIB WAN21_XH_LINEAR_OP \
 WAN21_SVE_GELU_LIB WAN21_SVE_LAYERNORM_LIB WAN21_SVE_RESIDUAL_LIB WAN21_SVE_MODULATION_LIB \
 TMPDIR XDG_CACHE_HOME HF_HOME; do MPI_ENV+=(-x "$name"); done
printf 'PORT_GATE_START mode=%s nodes=%s log=%s/run.log\n' "$PORT_MODE" "${NODES[*]}" "$PORT_WORK"
mpirun --stdin none -n "$PORT_NNODES" --rankfile "$PORT_WORK/leaders" --mca rmaps_rank_file_physical true "${MPI_ENV[@]}" \
 /bin/bash -c 'set -eu; mkdir -p "$1/tmp" "$1/cache" "$1/huggingface"; hostname; lscpu | head -n 18; 
 for name in gelu layernorm residual modulation; do gcc -O3 -fPIC -shared -fopenmp -march=armv8.2-a+sve2 "$2/src/train/sve_${name}.c" -lm -o "$1/sve_${name}.so"; done
 gcc -O3 -fPIC -shared -fopenmp -march=armv8.2-a+sve2+bf16 "$2/src/train/shared_reduce_sve.c" -o "$1/shared_reduce_sve.so"' bash "$PORT_WORK" "$CODE"
cd "$CODE"
if [[ "$PORT_MODE" == core ]]; then
for pattern in test_online_training.py test_node_memory.py test_native_training.py test_t5_loading.py; do
 echo "PORT_UNIT_START $pattern"
 mpirun --stdin none -n 1 --rankfile "$PORT_WORK/one" --mca rmaps_rank_file_physical true "${MPI_ENV[@]}" \
  /usr/bin/env WAN21_TEST_NATIVE_KERNELS=1 "$CODE/kunpeng/run_wan21_online_rank.sh" \
  "$WAN21_PYTHON" -m unittest discover -s tests -p "$pattern" -v
done
echo PORT_MPI_START
mpirun --stdin none -n 32 --rankfile "$PORT_WORK/ranks" --mca rmaps_rank_file_physical true "${MPI_ENV[@]}" \
 "$CODE/kunpeng/run_wan21_online_rank.sh" "$WAN21_PYTHON" -u tests/mpi_shared_gradients.py \
 --root "$PORT_WORK/gradients" --reducer "$PORT_WORK/shared_reduce_sve.so"
export WAN21_HIERARCHICAL_COMM=0
for precision in fp32 bf16; do
 for stage in continuous split resume; do
  steps=4; output="$PORT_WORK/$precision/continuous"; extra=()
  if [[ "$stage" != continuous ]]; then output="$PORT_WORK/$precision/split"; fi
  if [[ "$stage" == split ]]; then steps=2; fi
  if [[ "$stage" == resume ]]; then extra=(--auto-resume); fi
  echo "PORT_RESTORE_START $precision $stage"
  mpirun --stdin none -n 4 --rankfile "$PORT_WORK/four" --mca rmaps_rank_file_physical true "${MPI_ENV[@]}" \
   "$CODE/kunpeng/run_wan21_online_rank.sh" "$WAN21_PYTHON" -u src/train/shared_train.py \
   --toy --nnodes 1 --dies-per-node 1 --run-dir "$output" --stored-precision "$precision" \
   --max-steps "$steps" --grad-accum 2 --lr-total-steps 4 --warmup-steps 1 \
   --save-every 2 --verify-updates --job-token "${CCS_JOB_ID}.${precision}.${stage}" "${extra[@]}"
 done
done
"$WAN21_PYTHON" - "$PORT_WORK" <<'PY'
import sys,torch
from pathlib import Path
from safetensors.torch import load_file
root=Path(sys.argv[1])
def compare(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: compare(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y)
    else: assert a==b,(a,b)
for precision in ('fp32','bf16'):
    a=root/precision/'continuous/checkpoints/step_00000004'
    b=root/precision/'split/checkpoints/step_00000004'
    compare(load_file(str(a/'model.safetensors')),load_file(str(b/'model.safetensors')))
    for owner in range(4):
        name=f'owner_{owner:02d}.pt'
        compare(torch.load(a/name,weights_only=False),torch.load(b/name,weights_only=False))
    print('PORT_RESTORE_EXACT_OK',precision,flush=True)
PY
else
 export WAN21_HIERARCHICAL_COMM=0
 : "${WAN21_CHECK_VIDEO_DATA:?set a video manifest of at least four clips}" "${WAN21_ENCODER_WEIGHTS:?}" "${WAN21_BLAS_CHECK:?}"
 cases=(image video5 video49)
 [[ "$PORT_MODE" != media49 ]] || cases=(video49)
 for case in "${cases[@]}"; do
  input="${WAN21_CHECK_IMAGE_DATA:-}"
  task=t2i; frames=49; stored=fp32; text=fp32; steps=2
  export WAN21_FFN_HBM=0 WAN21_FFN_HBM_POOL=0
  if [[ "$case" != image ]]; then
   input="$WAN21_CHECK_VIDEO_DATA"; task=t2v; stored=bf16; text=bf16; steps=1
   if [[ "$case" == video5 ]]; then frames=5; else export WAN21_FFN_HBM=1 WAN21_FFN_HBM_POOL=1; fi
  else
   : "${WAN21_CHECK_IMAGE_DATA:?set the image manifest or catalog}"
  fi
  echo "PORT_MEDIA_START $case"
  env WAN21_TRAIN_DATA="$input" WAN21_TRAIN_TASK="$task" WAN21_VIDEO_FRAMES="$frames" \
   WAN21_VIDEO_ROOT="${WAN21_VIDEO_ROOT:-}" \
   WAN21_NNODES=1 WAN21_PACK_RPN=4 WAN21_TRAIN_RUN_DIR="$PORT_WORK/media_$case" \
   WAN21_STORED_PRECISION="$stored" WAN21_TEXT_PRECISION="$text" WAN21_TRAIN_PRECISION=bf16 \
   WAN21_ENV_ACTIVATE="$WAN21_ENV_ACTIVATE" \
   WAN21_EXPECT_TORCH_VERSION=2.9.1+shine.hmpi.kblas2521.beta0fix1 \
   WAN21_TRAIN_MAX_STEPS="$steps" WAN21_TRAIN_SAVE_EVERY=0 WAN21_TRAIN_BENCHMARK=1 \
   WAN21_VERIFY_PARAMETER_UPDATES=1 WAN21_TRAIN_START_MODE=scratch \
   "$CODE/kunpeng/run_wan21_t2i_multinode.sh"
 done
 "$WAN21_PYTHON" - "$PORT_WORK" "${cases[@]}" <<'PY'
from pathlib import Path
import json,sys,math
root=Path(sys.argv[1])
for case in sys.argv[2:]:
    steps=2 if case=='image' else 1
    run=root/('media_'+case)
    config=json.loads((run/'train_config.json').read_text())
    records=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert len(records)==steps and (run/'TRAINING_COMPLETED').is_file()
    assert config['input_mode']=='raw_online'
    assert all(r['encoded_samples']==4 and math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in records)
    print('PORT_MEDIA_OK '+json.dumps({'case':case,'steps':steps,'seconds':[r['step_seconds'] for r in records],
          'loss':[r['loss'] for r in records],'stored':config['parameter_dtype'],
          'encoder':config['encoder_dtype']}),flush=True)
PY
fi
echo PORT_ALL_CHECKS_PASSED
