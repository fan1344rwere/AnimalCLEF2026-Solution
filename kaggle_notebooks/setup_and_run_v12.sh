#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AnimalCLEF2026 V12 — H800 环境安装 + 运行脚本
# ═══════════════════════════════════════════════════════════════
set -e

echo "══════════════════════════════════════════════"
echo "  AnimalCLEF2026 V12 安装脚本"
echo "══════════════════════════════════════════════"

# HuggingFace mirror for China
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

# ── Step 1: 基础依赖 ──
echo "[1/5] 安装基础依赖..."
pip install -q torch torchvision timm tqdm scikit-learn pandas numpy Pillow
pip install -q safetensors huggingface_hub

# ── Step 2: HDBSCAN ──
echo "[2/5] 安装HDBSCAN..."
pip install -q hdbscan || echo "WARN: hdbscan安装失败，将使用DBSCAN+Agglo替代"

# ── Step 3: EVA02 (open_clip) ──
echo "[3/5] 安装OpenCLIP (EVA02)..."
pip install -q open_clip_torch || echo "WARN: open_clip安装失败，将跳过EVA02"

# ── Step 4: ALIKED + LightGlue (关键！+20pp的武器) ──
echo "[4/5] 安装LightGlue (ALIKED局部匹配)..."
pip install -q lightglue 2>/dev/null || {
    echo "  lightglue pip安装失败，尝试从GitHub安装..."
    pip install -q git+https://github.com/cvg/LightGlue.git 2>/dev/null || {
        echo "  WARN: LightGlue安装失败，尝试kornia替代..."
        pip install -q kornia || echo "  WARN: kornia也安装失败，局部匹配将不可用"
    }
}

# ── Step 5: 验证安装 ──
echo "[5/5] 验证安装..."
python -c "
import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    g=torch.cuda.get_device_properties(0)
    print(f'GPU: {g.name} VRAM: {g.total_memory/1e9:.1f}GB')
try: import timm; print('timm OK')
except: print('timm MISSING')
try: import hdbscan; print('hdbscan OK')
except: print('hdbscan MISSING')
try: import open_clip; print('open_clip OK')
except: print('open_clip MISSING')
try: from lightglue import LightGlue, ALIKED; print('LightGlue+ALIKED OK ★')
except: print('LightGlue MISSING (will try kornia fallback)')
try: import kornia; print('kornia OK')
except: print('kornia MISSING')
print('All checks done!')
"

echo ""
echo "══════════════════════════════════════════════"
echo "  安装完成！运行方式："
echo ""
echo "  # 标准运行（约45-60分钟）:"
echo "  python -u v12_ultimate_h800.py /root/autodl-tmp/animal-clef-2026 /root/autodl-tmp/ov12 2>&1 | tee run_v12.log"
echo ""
echo "  # 快速测试（禁用TTA+局部匹配，约10分钟）:"
echo "  python -c \""
echo "import v12_ultimate_h800 as v12"
echo "v12.CFG.USE_TTA = False"
echo "v12.CFG.USE_LOCAL = False"
echo "v12.CFG.USE_EVA02 = False"
echo "v12.CFG.USE_ARCFACE = False"
echo "v12.main()"
echo "\""
echo "══════════════════════════════════════════════"
