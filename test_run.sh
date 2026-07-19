#!/usr/bin/env bash
cd /home/kenpeter/work/small
source venv/bin/activate
export PYTHONUNBUFFERED=1
exec python3 -u pretrain_megatrain.py \
  --batch-size 4 \
  --grad-accum 12 \
  --lr 4e-4 \
  --log-interval 1 \
  --save-interval 100 \
  --num-steps 3 \
  --checkpoint-interval 4 \
  --num-grad-slabs 12 \
  > /home/kenpeter/work/small/training.log 2>&1
