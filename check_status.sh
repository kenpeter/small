#!/usr/bin/env bash
STATUS="/home/kenpeter/work/small/status_report.txt"
TRAIN_LOG="/home/kenpeter/work/small/training.log"
DOWNLOAD_LOG="/tmp/download_relaxed.log"
UPLOAD_SFT_LOG="/tmp/upload_sft.log"
UPLOAD_PRETRAIN_LOG="/tmp/upload_pretrain.log"

{
    echo "=== Status Report: $(date) ==="
    echo ""

    # ── Training ──
    LATEST=$(grep '| Loss' "$TRAIN_LOG" 2>/dev/null | tail -1)
    PID=$(pgrep -f 'pretrain_megatrain' 2>/dev/null | head -1)
    GPU=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "── Training ──"
    if [ -n "$LATEST" ]; then
        echo "  $LATEST"
    else
        echo "  No training running"
    fi
    echo "  PID: ${PID:-stopped} | GPU: ${GPU:-N/A}"

    # ── Download ──
    echo ""
    echo "── Data Download ──"
    if [ -f "$DOWNLOAD_LOG" ]; then
        tail -3 "$DOWNLOAD_LOG" 2>/dev/null
    else
        echo "  Not started"
    fi

    # ── Uploads ──
    echo ""
    echo "── HF Uploads ──"
    if [ -f "$UPLOAD_SFT_LOG" ]; then
        SFT_LAST=$(tail -1 "$UPLOAD_SFT_LOG" 2>/dev/null)
        echo "  SFT: $SFT_LAST"
    fi
    if [ -f "$UPLOAD_PRETRAIN_LOG" ]; then
        PRETRAIN_LAST=$(tail -1 "$UPLOAD_PRETRAIN_LOG" 2>/dev/null)
        echo "  Pretrain: $PRETRAIN_LAST"
    fi

    # ── Staging data ──
    echo ""
    echo "── Staging ──"
    du -sh /home/kenpeter/work/data/_staging_multi/*/ 2>/dev/null | sort -rh | head -5
    if [ $? -ne 0 ]; then
        echo "  (empty)"
    fi

    # ── Disk ──
    echo ""
    echo "── Disk ──"
    df -h / | tail -1 | awk '{print "  Used: "$3" / "$2" ("$5" full)"}'
} > "$STATUS"

cat "$STATUS"
