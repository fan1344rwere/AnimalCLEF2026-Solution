#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com
pip install einops -q
cd /root/autodl-tmp
nohup python -u v22_supcon.py > run_v22.log 2>&1 &
echo "V22 launched PID=$!"
