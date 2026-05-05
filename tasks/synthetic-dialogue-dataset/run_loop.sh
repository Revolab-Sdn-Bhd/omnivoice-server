#!/usr/bin/env bash
# Run pipeline in a loop until target hours reached.
# Pipeline and ASR validation run as independent background processes.
# Pipeline never waits for ASR — both GPUs stay busy.
# Usage: ./run_loop.sh [target_hours]
set -euo pipefail

TARGET_HOURS="${1:-1000}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG="/tmp/pipeline-loop.log"
VALIDATE_INTERVAL=120
OUTPUT_DIR="$PROJECT_DIR/output/synthetic-dialogue"

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
    CUDA_VISIBLE_DEVICES=1 $VENV_PYTHON "$SCRIPT_DIR/05_validate_asr.py" 2>&1 | tee -a "$ASR_LOG" || true
    sleep "$VALIDATE_INTERVAL"
done &
ASR_PID=$!
echo "$(date '+%Y-%m-%d %H:%M:%S') ASR validator PID: $ASR_PID" | tee -a "$LOG"

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

    $VENV_PYTHON "$SCRIPT_DIR/run_pipeline_async.py" --max-concurrent 8 2>&1 | tee -a "$LOG" || true

    new_hours=$(get_hours)
    echo "$(date '+%Y-%m-%d %H:%M:%S') Pipeline done. ${current}h -> ${new_hours}h" | tee -a "$LOG"

    # Clean stage2+3 so next iteration generates fresh situations + dialogues
    # Stage4 audio is preserved (unique dialogue IDs per run)
    echo "$(date '+%Y-%m-%d %H:%M:%S') Cleaning stage2+3 for next iteration..." | tee -a "$LOG"
    rm -f "$OUTPUT_DIR"/stage2_situations/*.json
    rm -f "$OUTPUT_DIR"/stage3_dialogues/*.json
done

# Final ASR validation pass before exiting
echo "$(date '+%Y-%m-%d %H:%M:%S') Stopping ASR validator..." | tee -a "$LOG"
kill "$ASR_PID" 2>/dev/null || true
wait "$ASR_PID" 2>/dev/null || true

echo "$(date '+%Y-%m-%d %H:%M:%S') Running final ASR validation..." | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=1 $VENV_PYTHON "$SCRIPT_DIR/05_validate_asr.py" 2>&1 | tee -a "$LOG" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') All done. Final: $(get_hours)h ($(get_dialogue_count) dialogues)" | tee -a "$LOG"
