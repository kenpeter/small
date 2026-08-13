#!/bin/bash
# Warm-restart extension: 30K → 60K (plateau breaker #1)
# Usage: fire at step 30,000 (after the G1 easy-tail finishes).
# LR warm-restart is automatic: cosine over 60K steps → at resume t=0.5 → LR ≈ 1.5e-4.
# Curriculum auto-replays fold 1 (hard-heavy at resume — expected, 2-fold mirror design).
set -e
unset ENABLE_CUDA_GRAPH
unset ENABLE_HYDRA_PIKIA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/kenpeter/work/small/venv/bin/activate
cd /home/kenpeter/work/small

echo "GPU status before start:"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit --format=csv,noheader

# ── Preflight checks (same as run_pretrain_gpu.sh) ──
if pgrep -f "pretrain_gpu.py" > /dev/null; then
  echo "🔴 ABORT: pretrain_gpu.py already running — refusing to double-launch (would OOM)."
  exit 1
fi
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE_MB" -lt 2000 ]; then
  echo "🔴 ABORT: only ${FREE_MB}MiB free VRAM (need ≥2000MiB)."
  exit 1
fi
FREE_DISK_GB=$(df -BG /home/kenpeter/work | awk 'NR==2 {print $4}' | tr -dc '0-9')
if [ "${FREE_DISK_GB:-0}" -lt 10 ]; then
  echo "🔴 ABORT: only ${FREE_DISK_GB}GB free disk (need ≥10GB)."
  exit 1
fi
if [ ! -f /home/kenpeter/work/checkpoints/megatrain_latest.pt ]; then
  echo "🔴 ABORT: warm-start checkpoint /home/kenpeter/work/checkpoints/megatrain_latest.pt missing."
  exit 1
fi
echo "✅ Preflight passed (free VRAM ${FREE_MB}MiB, disk ${FREE_DISK_GB}GB)"

echo "Starting 60K warm-restart extension at $(date)"
echo ""

exec python3 pretrain_gpu.py \
  --batch-size 4 \
  --grad-accum 8 \
  --num-steps 60000 \
  --log-interval 100 \
  --save-interval 1000 \
  --save-every-minutes 20 \
  --warmup-steps 400 \
  --lr 3e-4 \
  --curriculum \
  --liger \
  --cautious \
  --fused-ce \
  --compile \
  --swa-window 24 \
  --resume-from /home/kenpeter/work/checkpoints/megatrain_latest.pt \
  >> /home/kenpeter/work/train_small.log 2>&1
