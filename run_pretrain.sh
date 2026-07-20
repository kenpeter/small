#!/bin/bash
unset ENABLE_CUDA_GRAPH
unset ENABLE_HYDRA_PIKIA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /home/kenpeter/work/small/venv/bin/activate
cd /home/kenpeter/work/small
exec python3 pretrain_megatrain.py \
  --batch-size 4 \
  --num-steps 50000 \
  --lr 4e-4 \
  --grad-accum 12 \
  --num-grad-slabs 12 \
  --log-interval 120 \
  --save-interval 2000 \
  > /tmp/pretrain.log 2>&1
