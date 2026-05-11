#!/usr/bin/env bash
# Run pipeline in a loop until target hours reached.
# Uses flock to ensure only one instance runs at a time.
set -euo pipefail

TARGET_HOURS="${1:-1000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG="/tmp/pipeline-loop.log"
LOCKFILE="/tmp/pipeline-loop.lock"
VALIDATE_INTERVAL=120
HF_UPLOAD_INTERVAL=1800
PIPELINE_TIMEOUT=7200
OUTPUT_DIR="$PROJECT_DIR/output/synthetic-dialogue"

# Ensure only one instance runs at a time
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') Another run_loop.sh is already running (lock held). Exiting." >> "$LOG"
    exit 0
fi
echo "Lock acquired: PID $$" >&200

get_hours() {
    $VENV_PYTHON -c "
import json, glob
total_s = 0
for m in glob.glob('$OUTPUT_DIR/stage4_audio/*/manifest.json'):
    d = json.load(open(m))
    total_s += d.get('total_duration_s', 0)
print(f'{total_s/3600:.2f}')
" 2>/dev/null
}

get_dialogue_count() {
    ls "$OUTPUT_DIR/stage4_audio/" 2>/dev/null | wc -l
}

current=$(get_hours)
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting loop. Current: ${current}h / Target: ${TARGET_HOURS}h" | tee -a "$LOG"

# Start ASR validation as a long-running background loop
ASR_LOG="/tmp/pipeline-asr.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting ASR validator in background..." | tee -a "$LOG"
while true; do
    export CUDA_VISIBLE_DEVICES=0
    $VENV_PYTHON "$SCRIPT_DIR/05_validate_asr.py" 2>&1 | tee -a "$ASR_LOG" || true
    sleep "$VALIDATE_INTERVAL"
done &
ASR_PID=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') ASR validator PID: $ASR_PID" | tee -a "$LOG"

# Start HF incremental upload as a long-running background loop
HF_LOG="/tmp/pipeline-hf-upload.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting HF incremental uploader in background..." | tee -a "$LOG"
while true; do
    $VENV_PYTHON "$SCRIPT_DIR/upload_to_hf.py" --incremental 2>&1 | tee -a "$HF_LOG" || true
    sleep "$HF_UPLOAD_INTERVAL"
done &
HF_PID=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') HF uploader PID: $HF_PID (every ${HF_UPLOAD_INTERVAL}s)" | tee -a "$LOG"

# Start periodic final rebuild as a long-running background loop
REBUILD_INTERVAL=1800
REBUILD_LOG="/tmp/pipeline-rebuild.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting final rebuild checkpoint (every ${REBUILD_INTERVAL}s)..." | tee -a "$LOG"
while true; do
    $VENV_PYTHON "$SCRIPT_DIR/05_validate_asr.py" --rebuild-final 2>&1 | tee -a "$REBUILD_LOG" || true
    sleep "$REBUILD_INTERVAL"
done &
REBUILD_PID=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') Rebuild checkpoint PID: $REBUILD_PID (every ${REBUILD_INTERVAL}s)" | tee -a "$LOG"

while true; do
    current=$(get_hours)
    count=$(get_dialogue_count)
    echo "$(date '+%Y-%m-%d %H:%M:%S') Current: ${current}h (${count} dialogues) / Target: ${TARGET_HOURS}h" | tee -a "$LOG"

    if (( $(echo "$current >= $TARGET_HOURS" | bc -l) )); then
        echo "$(date '+%Y-%m-%d %H:%M:%S') Target reached! ${current}h >= ${TARGET_HOURS}h" | tee -a "$LOG"
        break
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') Running pipeline..." | tee -a "$LOG"
    cd "$PROJECT_DIR"

    timeout "$PIPELINE_TIMEOUT" $VENV_PYTHON "$SCRIPT_DIR/run_pipeline_async.py" --max-concurrent 32 --continuous >> "$LOG" 2>&1 || \
        echo "$(date '+%Y-%m-%d %H:%M:%S') Pipeline timed out or failed (exit $?), continuing..." >> "$LOG"

    new_hours=$(get_hours)
    echo "$(date '+%Y-%m-%d %H:%M:%S') Pipeline done. ${current}h -> ${new_hours}h" | tee -a "$LOG"
done

# Final ASR validation pass before exiting
echo "$(date '+%Y-%m-%d %H:%M:%S') Stopping background processes..." | tee -a "$LOG"
kill "$ASR_PID" 2>/dev/null || true
kill "$HF_PID" 2>/dev/null || true
kill "$REBUILD_PID" 2>/dev/null || true
wait "$ASR_PID" 2>/dev/null || true
wait "$HF_PID" 2>/dev/null || true
wait "$REBUILD_PID" 2>/dev/null || true

echo "$(date '+%Y-%m-%d %H:%M:%S') Running final ASR validation..." | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES=0
$VENV_PYTHON "$SCRIPT_DIR/05_validate_asr.py" 2>&1 | tee -a "$LOG" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') Final HF upload..." | tee -a "$LOG"
$VENV_PYTHON "$SCRIPT_DIR/upload_to_hf.py" --incremental 2>&1 | tee -a "$LOG" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') All done. Final: $(get_hours)h ($(get_dialogue_count) dialogues)" | tee -a "$LOG"
