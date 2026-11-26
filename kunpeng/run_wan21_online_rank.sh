#!/usr/bin/env bash
set -euo pipefail
: "${OMPI_COMM_WORLD_LOCAL_RANK:?}"
LOCAL_RANK="$OMPI_COMM_WORLD_LOCAL_RANK"
[[ "$LOCAL_RANK" =~ ^[0-9]+$ ]] && (( LOCAL_RANK < 16 )) || exit 2
FIRST=$((LOCAL_RANK * 38)); COMM_CPU=$((FIRST + 36)); LAST=$((FIRST + 35))
python3 -c '
import pathlib,sys,os
n=int(sys.argv[1]); raw=pathlib.Path(f"/sys/devices/system/node/node{n}/cpulist").read_text().strip()
cpus=set()
for item in raw.split(","):
 a,sep,b=item.partition("-");cpus.update(range(int(a),int(b)+1) if sep else [int(a)])
assert cpus==set(range(n*38,(n+1)*38)),(n,raw)
assert set(range(n*38,n*38+37))<=os.sched_getaffinity(0),os.sched_getaffinity(0)
assert "0xd22" in pathlib.Path("/proc/cpuinfo").read_text()
distances=[int(value) for value in pathlib.Path(f"/sys/devices/system/node/node{n}/distance").read_text().split()]
die_first=(n//4)*4
assert {i for i,d in enumerate(distances[:16]) if d<=20}==set(range(die_first,die_first+4)),(n,distances[:16])
' "$LOCAL_RANK"
PLACES=""
for ((cpu=FIRST; cpu<=LAST; cpu++)); do
 [[ -z "$PLACES" ]] || PLACES+=","; PLACES+="{$cpu}"
done
export OMP_NUM_THREADS=36 OPENBLAS_NUM_THREADS=36 OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=spread OMP_PLACES="$PLACES" OMP_WAIT_POLICY=PASSIVE
export WAN21_COMMUNICATION_CPU="$COMM_CPU" TOKENIZERS_PARALLELISM=false
printf 'ONLINE_RANK_BIND rank=%s compute=%s-%s communication=%s\n' "$OMPI_COMM_WORLD_RANK" "$FIRST" "$LAST" "$COMM_CPU"
DIE_FIRST=$(( (LOCAL_RANK / 4) * 4 ))
DIE_LAST=$(( DIE_FIRST + 3 ))
printf 'ONLINE_DIE_MEMBIND rank=%s mems=%s-%s\n' "$OMPI_COMM_WORLD_RANK" "$DIE_FIRST" "$DIE_LAST"
GEMM_ENV=()
if [[ -n "${WAN21_TRAIN_GEMM_BRIDGE:-}" ]]; then
  GEMM_ENV=(env "LD_PRELOAD=$WAN21_TRAIN_GEMM_BRIDGE${LD_PRELOAD:+:$LD_PRELOAD}"
    "FINAL_GEMM_THREADS=36" "WAN21_GEMM_ENABLED=1")
fi
exec numactl --physcpubind="$FIRST-$COMM_CPU" --membind="$DIE_FIRST-$DIE_LAST" "${GEMM_ENV[@]}" "$@"
