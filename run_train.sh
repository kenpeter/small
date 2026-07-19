#!/usr/bin/env bash
cd /home/kenpeter/work/small
source venv/bin/activate
export PYTHONUNBUFFERED=1
exec python3 -u train.py > training.log 2>&1
