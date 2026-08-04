#!/usr/bin/env bash
# Combined status report: training + downloads + disk.
# Usage: ./check_status.sh
STATUS="/home/kenpeter/work/small/status_report.txt"
TRAIN_LOG="/home/kenpeter/work/small/training.log"
DOWNLOAD_LOG="/home/kenpeter/work/small/download.log"

{
    echo "=== Status Report: $(date) ==="
    echo ""

    # ── Training ──
    LATEST=$(grep '| Loss' "$TRAIN_LOG" 2>/dev/null | tail -1)
    PID=$(pgrep -f 'pretrain_megatrain' 2>/dev/null | head -1)
    GPU=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    START=$(grep 'Starting pretraining' "$TRAIN_LOG" 2>/dev/null | tail -1 | head -c 21)
    echo "── Training ──"
    if [ -n "$LATEST" ]; then
        echo "  $LATEST"
    else
        echo "  No training running"
    fi
    echo "  PID: ${PID:-stopped} | GPU: ${GPU:-N/A} | Started: ${START:-?}"

    # ── Download ──
    echo ""
    echo "── Data Download ──"
    if [ -f "$DOWNLOAD_LOG" ]; then
        tail -3 "$DOWNLOAD_LOG" 2>/dev/null
    else
        echo "  Not started"
    fi

    # ── Disk ──
    echo ""
    echo "── Disk ──"
    df -h / | tail -1 | awk '{print "  Used: "$3" / "$2" ("$5" full)"}'
} > "$STATUS"

cat "$STATUS"
