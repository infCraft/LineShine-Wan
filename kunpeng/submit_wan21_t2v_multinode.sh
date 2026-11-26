#!/usr/bin/env bash
# Raw OpenVid -> frozen VAE/UMT5 -> shared-state BF16 T2V training.
set -euo pipefail
if (($# < 2 || $# > 4)); then
 echo "Usage: $0 VIDEO_JSONL_OR_default RUN_DIR [MAX_STEPS=40] [scratch|resume]" >&2
 exit 2
fi
CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA_INDEX="$1"
if [[ "$DATA_INDEX" == default ]]; then
 DATA_INDEX="${WAN21_VIDEO_MANIFEST:?pass a video JSONL or set WAN21_VIDEO_MANIFEST for default}"
fi
[[ -f "$DATA_INDEX" ]] || { echo "T2V requires a video JSONL manifest: $DATA_INDEX" >&2; exit 2; }
export WAN21_TRAIN_TASK=t2v
export WAN21_PACK_RPN="${WAN21_PACK_RPN:-4}"
[[ "$WAN21_PACK_RPN" =~ ^(4|8|12|16)$ ]] || { echo "RPN must be 4, 8, 12, or 16" >&2; exit 2; }
export WAN21_MEM_PER_RANK_GIB="${WAN21_MEM_PER_RANK_GIB:-$((432 / WAN21_PACK_RPN))}"
export WAN21_VIDEO_ROOT="${WAN21_VIDEO_ROOT:-}"
export WAN21_VIDEO_FRAMES="${WAN21_VIDEO_FRAMES:-49}" WAN21_VIDEO_FPS="${WAN21_VIDEO_FPS:-16}"
exec "$CODE_ROOT/kunpeng/submit_wan21_t2i_multinode.sh" "$DATA_INDEX" "$2" "${3:-40}" "${4:-scratch}"
