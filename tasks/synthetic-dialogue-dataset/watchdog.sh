#!/usr/bin/env bash
# watchdog.sh — Monitor pipeline, restart if idle or stuck.
# Runs in a loop, checks every 10 minutes.
# Usage: bash watchdog.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG="/tmp/pipeline-watchdog.log"
PIPELINE_LOG="/tmp/pipeline-loop.log"
CHECK_INTERVAL=600  # 10 minutes

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG"
}

check_and_fix() {
    # 1. Check if run_pipeline_async.py is running
    PIPELINE_PID=$(pgrep -f "run_pipeline_async.py" 2>/dev/null || echo "")
    
    if [ -z "$PIPELINE_PID" ]; then
        log "WARN: No run_pipeline_async.py process found"
        
        # Check if run_loop.sh is alive
        LOOP_PID=$(pgrep -f "run_loop.sh" 2>/dev/null || echo "")
        if [ -z "$LOOP_PID" ]; then
            log "ERROR: run_loop.sh not running! Restarting..."
            cd "$PROJECT_DIR"
            nohup bash "$SCRIPT_DIR/run_loop.sh" 1000 >> "$PIPELINE_LOG" 2>&1 &
            log "Started run_loop.sh: PID $!"
            return
        fi
        
        # run_loop is alive but no pipeline — maybe between cycles or stuck
        # Check if log has been written to in last 5 minutes
        if [ -f "$PIPELINE_LOG" ]; then
            LAST_MOD=$(stat -c %Y "$PIPELINE_LOG" 2>/dev/null || echo 0)
            NOW=$(date +%s)
            AGE=$((NOW - LAST_MOD))
            
            if [ "$AGE" -gt 300 ]; then
                log "ERROR: Pipeline log stale for ${AGE}s. Killing run_loop and restarting..."
                pkill -9 -f "run_loop.sh" 2>/dev/null || true
                pkill -9 -f "run_pipeline_async" 2>/dev/null || true
                pkill -9 -f "05_validate_asr" 2>/dev/null || true
                sleep 5
                cd "$PROJECT_DIR"
                nohup bash "$SCRIPT_DIR/run_loop.sh" 1000 >> "$PIPELINE_LOG" 2>&1 &
                log "Force-restarted run_loop.sh: PID $!"
            else
                log "OK: Between pipeline cycles (log updated ${AGE}s ago)"
            fi
        fi
        return
    fi
    
    # 2. Pipeline is running — check if it's actually producing output
    if [ -f "$PIPELINE_LOG" ]; then
        LAST_MOD=$(stat -c %Y "$PIPELINE_LOG" 2>/dev/null || echo 0)
        NOW=$(date +%s)
        AGE=$((NOW - LAST_MOD))
        
        if [ "$AGE" -gt 600 ]; then
            log "ERROR: Pipeline running but no output for ${AGE}s. Killing and restarting..."
            kill -9 "$PIPELINE_PID" 2>/dev/null || true
            # Also kill any stuck tee processes under run_loop
            pkill -9 -P $(pgrep -f "run_loop.sh" | head -1) tee 2>/dev/null || true
            log "Killed stuck pipeline PID $PIPELINE_PID"
        else
            log "OK: Pipeline running (PID $PIPELINE_PID, log updated ${AGE}s ago)"
        fi
    fi
    
    # 3. Check ASR validator is running
    ASR_PID=$(pgrep -f "05_validate_asr.py" 2>/dev/null || echo "")
    if [ -z "$ASR_PID" ]; then
        log "WARN: ASR validator not running (run_loop will restart it)"
    fi
    
    # 4. Quick GPU check
    GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 2>/dev/null || echo "0")
    if [ "$GPU1_UTIL" -lt 5 ] 2>/dev/null; then
        log "WARN: GPU 1 (vLLM) utilization: ${GPU1_UTIL}%"
    fi
}

log "=== Watchdog started ==="
log "Check interval: ${CHECK_INTERVAL}s"

while true; do
    check_and_fix
    sleep "$CHECK_INTERVAL"
done
