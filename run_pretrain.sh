#!/bin/bash
unset ENABLE_CUDA_GRAPH
unset ENABLE_HYDRA_PIKIA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/kenpeter/work/small/venv/bin/activate
cd /home/kenpeter/work/small

echo "GPU status before start:"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit --format=csv,noheader
echo ""
echo "Tip: If you want a cooler run, cap power with: sudo nvidia-smi -pl 200"
echo "Starting pretrain at $(date)"
echo ""

exec python3 pretrain_megatrain.py \
  --batch-size 4 \
  --num-steps 50000 \
  --lr 4e-4 \
  --grad-accum 12 \
  --num-grad-slabs 12 \
  --log-interval 120 \
  --save-interval 2000 \
  > /tmp/pretrain.log 2>&1
