#!/bin/bash
# Wrapper to start downloads with correct environment
unset ENABLE_CUDA_GRAPH
export HF_XET_HIGH_PERFORMANCE=1
# HF_TOKEN should be set externally: export HF_TOKEN=hf_xxxxx
cd /home/kenpeter/work/small
source venv/bin/activate
exec python3 download_data_relaxed.py > /home/kenpeter/work/small/download.log 2>&1
