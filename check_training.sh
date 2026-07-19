#!/usr/bin/env bash
LOG="/home/kenpeter/work/small/training.log"
STATUS="/home/kenpeter/work/small/training_status.txt"

if [ ! -f "$LOG" ]; then
    echo "[$(date)] No training log found" > "$STATUS"
    exit 0
fi

# Latest step line (MegaTrain format: Step X/Y | Loss ...)
LATEST=$(grep '| Loss' "$LOG" | tail -1)
# GPU memory
GPU=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
# Duration
PID=$(pgrep -f 'pretrain_megatrain' 2>/dev/null | head -1)
START=$(grep 'Starting pretraining' "$LOG" | tail -1 | head -c 21 2>/dev/null)

{
    echo "=== Training Status ==="
    echo "Updated: $(date)"
    echo ""
    if [ -n "$LATEST" ]; then
        echo "$LATEST"
    else
        echo "No step logged yet"
    fi
    echo ""
    echo "GPU memory: ${GPU:-N/A} MB" 
    echo "PID: ${PID:-stopped}"
    echo "Started: $START"
    echo ""
    echo "=== Last 20 log lines ==="
    tail -20 "$LOG"
} > "$STATUS"

cat "$STATUS"
