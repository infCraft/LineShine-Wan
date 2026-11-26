#!/usr/bin/env bash
# Bind each rank to the two RDMA ports local to its CPU die before MPI_Init.
set -euo pipefail
: "${OMPI_COMM_WORLD_LOCAL_RANK:?}" "${OMPI_COMM_WORLD_RANK:?}"
[[ "$OMPI_COMM_WORLD_LOCAL_RANK" =~ ^([0-9]|1[0-5])$ ]] || exit 2
die=$((OMPI_COMM_WORLD_LOCAL_RANK / 4))
first=$((die * 4))
last=$((first + 3))
ports=()
for device in /sys/class/infiniband/*; do
  [[ -d "$device" ]] || continue
  [[ -r "$device/device/numa_node" ]] || continue
  read -r numa_node < "$device/device/numa_node"
  [[ "$numa_node" =~ ^[0-9]+$ ]] || continue
  (( numa_node >= first && numa_node <= last )) || continue
  for port in "$device"/ports/*; do
    [[ -r "$port/state" ]] || continue
    read -r state < "$port/state"
    [[ "$state" == '4: ACTIVE' ]] || continue
    ports+=("${device##*/}:${port##*/}")
  done
done
# All tested nodes expose precisely two ports per die; fail closed if topology differs.
(( ${#ports[@]} == 2 )) || { echo "WAN21_LOCAL_UCX_ERROR rank=$OMPI_COMM_WORLD_RANK die=$die ports=${#ports[@]}" >&2; exit 86; }
export UCX_NET_DEVICES="$(IFS=,; echo "${ports[*]}")"
export UCX_MAX_RNDV_RAILS=2
printf 'WAN21_LOCAL_UCX_BIND rank=%s die=%s devices=%s rails=%s\n' \
  "$OMPI_COMM_WORLD_RANK" "$die" "$UCX_NET_DEVICES" "$UCX_MAX_RNDV_RAILS"
exec "$(cd "$(dirname "$0")" && pwd -P)/run_wan21_online_rank.sh" "$@"
