#!/bin/bash
cd /home/kenpeter/work/small
source venv/bin/activate
MODEL="/home/kenpeter/.cache/huggingface/hub/models--Qwen--Qwen3-8B-AWQ/snapshots/4da05a8edb55c6046cce958586c33b61da07bb79"
BASE="http://localhost:8000/v1"
for i in 4 5 6 7; do
  echo "Launching chunk $i..."
  python3 reformat_data.py \
    --input "/home/kenpeter/work/data/_reformatted/_input_chunk_${i}.jsonl" \
    --output "/home/kenpeter/work/data/_reformatted_chunk_${i}" \
    --formats textbook,qa \
    --max-docs 800 --max-tokens 2048 --temperature 0.4 \
    --workers 2 \
    --api-base "$BASE" \
    --model "$MODEL" \
    --no-resume &
  sleep 3
done
wait
echo "All 8 agents launched"
