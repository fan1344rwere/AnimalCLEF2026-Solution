#!/bin/bash
# V15 安装 + 运行一键脚本
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

echo "=== 安装wildlife-tools ==="
pip install git+https://github.com/WildlifeDatasets/wildlife-tools 2>&1 | tail -3

echo "=== 验证 ==="
python3 -c "
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import ImageDataset
print('wildlife-tools OK')
import wildlife_tools.similarity as ws
print('similarity module:', [x for x in dir(ws) if not x.startswith('_')])
"

echo "=== 启动V15 ==="
cd /root/autodl-tmp
nohup python3 -u v15_wildlife_tools.py > run_v15.log 2>&1 &
echo "PID: $!"
echo "=== 监控: tail -f /root/autodl-tmp/run_v15.log ==="
