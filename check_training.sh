#!/usr/bin/env bash
LOG="/home/kenpeter/work/small/training.log"
STATUS="/home/kenpeter/work/small/training_status.txt"

if [ ! -f "$LOG" ]; then
    echo "[$(date)] No training log found" > "$STATUS"
    exit 0
fi

# Latest step line
LATEST=$(grep "^step " "$LOG" | tail -1)
# Latest validation line
VAL=$(grep "Val loss" "$LOG" | tail -1)
# GPU memory
GPU=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

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
    if [ -n "$VAL" ]; then
        echo "$VAL"
    fi
    echo ""
    echo "GPU Memory: ${GPU:-N/A} MB used"
    echo "PID: $(pgrep -f 'python3 train.py' 2>/dev/null | head -1 || echo 'not running')"
    echo ""
    echo "=== Last 20 lines ==="
    tail -20 "$LOG"
} > "$STATUS"

cat "$STATUS"
