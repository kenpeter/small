#!/bin/bash
# Pure-GPU 1B pretraining (replaces CPUMaster offload — 2.1x faster)
unset ENABLE_CUDA_GRAPH
unset ENABLE_HYDRA_PIKIA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/kenpeter/work/small/venv/bin/activate
cd /home/kenpeter/work/small

echo "GPU status before start:"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit --format=csv,noheader

# ── Preflight checks (abort on any failure — prevents double-launch OOM, etc.) ──
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

echo "Starting pure-GPU pretrain at $(date)"
echo ""

exec python3 pretrain_gpu.py \
  --batch-size 2 \
  --grad-accum 16 \
  --num-steps 15000 \
  --log-interval 400 \
  --save-interval 1000 \
  --save-every-minutes 20 \
  --warmup-steps 1000 \
  --lr 3e-4 \
  --curriculum \
  --compile \
  --init-from /home/kenpeter/work/checkpoints/megatrain_latest.pt \
  >> /home/kenpeter/work/train_small.log 2>&1
