#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
pip install einops -q
echo "einops installed"
cd /root/autodl-tmp
nohup python -u v21_foundation_ensemble.py > run_v21.log 2>&1 &
echo "V21 launched PID=$!"
