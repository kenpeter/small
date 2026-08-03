#!/bin/bash
# Pure-GPU 1B pretraining (replaces CPUMaster offload — 2.1x faster)
unset ENABLE_CUDA_GRAPH
unset ENABLE_HYDRA_PIKIA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/kenpeter/work/small/venv/bin/activate
cd /home/kenpeter/work/small

echo "GPU status before start:"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit --format=csv,noheader
echo "Starting pure-GPU pretrain at $(date)"
echo ""

exec python3 pretrain_gpu.py \
  --batch-size 2 \
  --grad-accum 16 \
  --num-steps 15000 \
  --log-interval 400 \
  --save-interval 1000 \
  --warmup-steps 1000 \
  --lr 3e-4 \
  --init-from /home/kenpeter/work/checkpoints/megatrain_latest.pt \
  >> /home/kenpeter/work/train_small.log 2>&1
