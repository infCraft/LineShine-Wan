#!/usr/bin/env bash
set -euo pipefail

ROOT=${LINESHINE_ROOT:-/mnt/beegfs/home/huang_z/lineshine}
CODE=${LINESHINE_CODE_ROOT:-$ROOT/code}
CONDA_SH=${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-lineshine-wan}
POLL_SECONDS=${POLL_SECONDS:-1800}

OLD_TRAIN_MANIFEST=${OLD_TRAIN_MANIFEST:-$ROOT/data/openvid/meta/openvid_shared_train_98k_extracted.jsonl}
EXTRA_EXTRACT_SUMMARY=${EXTRA_EXTRACT_SUMMARY:-$ROOT/reports/stage1_expand/extract_extra_summary.json}
EXTRA_EXTRACT_STATE=${EXTRA_EXTRACT_STATE:-$ROOT/data/openvid/meta/stage1_extra_extract_state.json}
EXTRA_EXTRACTED_MANIFEST=${EXTRA_EXTRACTED_MANIFEST:-$ROOT/data/openvid/meta/stage1_extra_current_downloads_extracted.jsonl}
SEGMENT_MANIFEST=${SEGMENT_MANIFEST:-$ROOT/data/openvid/meta/stage1_train_segments_3s_cap8.jsonl}
SEGMENT_SKIPPED=${SEGMENT_SKIPPED:-$ROOT/data/openvid/meta/stage1_train_segments_3s_cap8_skipped.jsonl}
SEGMENT_REPORT=${SEGMENT_REPORT:-$ROOT/reports/stage1_segments/segments_3s_cap8.json}
CACHE_DIR=${CACHE_DIR:-$ROOT/cache/stage1_256x256_49f16fps/train_segments_3s_cap8}
CACHE_PREFIX=${CACHE_PREFIX:-train_segments}
CACHE_ARRAY_TASKS=${CACHE_ARRAY_TASKS:-16}
VERIFY_REPORT=${VERIFY_REPORT:-$ROOT/reports/stage1_segments/verify_train_segments_cache.json}
VERIFY_LOG=${VERIFY_LOG:-$ROOT/reports/stage1_segments/verify_train_segments_cache.log}
SUBMISSION_JSON=${SUBMISSION_JSON:-$ROOT/reports/stage1_segments/cache_array_submission.json}
RUNNER_STATE=${RUNNER_STATE:-$ROOT/reports/stage1_segments/runner_state.json}

mkdir -p "$ROOT/reports/stage1_segments" "$CACHE_DIR"
cd "$CODE"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH=.

write_state() {
  local phase=$1
  local extra=${2:-"{}"}
  python - "$RUNNER_STATE" "$phase" "$extra" <<'PY'
import json, sys
from datetime import datetime, timezone
path, phase, extra = sys.argv[1], sys.argv[2], sys.argv[3]
data = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "phase": phase}
try:
    data.update(json.loads(extra))
except json.JSONDecodeError:
    data["extra"] = extra
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write("\n")
PY
}

echo "[stage1] runner started at $(date -Is)"
write_state "waiting_for_extract"

while [[ ! -f "$EXTRA_EXTRACT_SUMMARY" ]]; do
  if [[ -f "$EXTRA_EXTRACT_STATE" ]]; then
    python - "$EXTRA_EXTRACT_STATE" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
print("[stage1] extract state", {"last_part": r.get("last_part"), "parts_done": len(r.get("parts_done", {})), "extracted": r.get("total_extracted_so_far"), "failed": r.get("total_failed_so_far"), "updated_at": r.get("updated_at")})
PY
  else
    echo "[stage1] waiting for extract state at $(date -Is)"
  fi
  sleep "$POLL_SECONDS"
done

python - "$EXTRA_EXTRACT_SUMMARY" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
print("[stage1] extract summary", {k: s.get(k) for k in ["requested", "extracted", "failed", "parts_tmp_empty", "output_manifest"]})
if s.get("failed") not in (0, None):
    raise SystemExit(f"extract has failures: {s.get('failed')}")
if not s.get("parts_tmp_empty", False):
    raise SystemExit("parts_tmp is not empty after extract")
PY
write_state "extract_done"

echo "[stage1] building 3s cap8 segment manifest at $(date -Is)"
python src/data/make_clip_segments.py \
  --manifest "$OLD_TRAIN_MANIFEST" \
  --manifest "$EXTRA_EXTRACTED_MANIFEST" \
  --output "$SEGMENT_MANIFEST" \
  --skipped "$SEGMENT_SKIPPED" \
  --report "$SEGMENT_REPORT" \
  --clip-duration 3.0 \
  --max-segments 8 \
  --safety-margin 0.0625
write_state "segments_built" "{\"segment_manifest\":\"$SEGMENT_MANIFEST\"}"

echo "[stage1] submitting cache array at $(date -Is)"
JOB_ID=$(MANIFEST="$SEGMENT_MANIFEST" CACHE_DIR="$CACHE_DIR" PREFIX="$CACHE_PREFIX" \
  LINESHINE_ROOT="$ROOT" LINESHINE_CODE_ROOT="$CODE" \
  sbatch --parsable --array=0-$((CACHE_ARRAY_TASKS - 1)) "$CODE/slurm/cache_array.sbatch")
python - "$SUBMISSION_JSON" "$JOB_ID" "$SEGMENT_MANIFEST" "$CACHE_DIR" "$CACHE_ARRAY_TASKS" <<'PY'
import json, sys
from datetime import datetime, timezone
path, job_id, manifest, cache_dir, tasks = sys.argv[1:]
data = {
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "job_id": job_id,
    "manifest": manifest,
    "cache_dir": cache_dir,
    "array_tasks": int(tasks),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY
write_state "cache_submitted" "{\"job_id\":\"$JOB_ID\"}"
echo "[stage1] submitted cache array job $JOB_ID"

while squeue -h -j "$JOB_ID" | grep -q .; do
  echo "[stage1] cache job $JOB_ID still active at $(date -Is)"
  squeue -j "$JOB_ID" || true
  sleep "$POLL_SECONDS"
done

FAILED_COUNT=$(find "$CACHE_DIR" -name "${CACHE_PREFIX}_*_failed.jsonl" -type f -exec sh -c 'cat "$@"' _ {} + | wc -l | tr -d ' ')
echo "[stage1] cache array finished; failure rows=$FAILED_COUNT"
if [[ "$FAILED_COUNT" != "0" ]]; then
  write_state "cache_finished_with_failures" "{\"job_id\":\"$JOB_ID\",\"failed_rows\":$FAILED_COUNT}"
  exit 1
fi

echo "[stage1] verifying cache at $(date -Is)"
python src/data/verify_cache.py \
  --cache-dir "$CACHE_DIR" \
  --pattern "${CACHE_PREFIX}-*.tar" \
  --report "$VERIFY_REPORT" \
  --latent-shape 16 13 32 32 \
  --text-dim 4096 \
  --text-len 512 \
  2>&1 | tee "$VERIFY_LOG"

write_state "complete" "{\"job_id\":\"$JOB_ID\",\"verify_report\":\"$VERIFY_REPORT\"}"
echo "[stage1] complete at $(date -Is)"
