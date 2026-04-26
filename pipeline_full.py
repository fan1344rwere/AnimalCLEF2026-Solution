# ================================================================
# AnimalCLEF2026 完整Pipeline (一个Cell，直接粘贴运行)
# GPU: RTX PRO 6000 95GB | 无网络 | 零安装
# ================================================================

import os, gc, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from safetensors.torch import load_file
import timm
warnings.filterwarnings("ignore")

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB, timm: {timm.__version__}")

# ======================== 路径 ========================
COMP_DIR    = "/kaggle/input/datasets/algebraictopology/animal"
BIO3_ROOT   = "/kaggle/input/datasets/algebraictopology/bio-3model"
WILDLIFE    = "/kaggle/input/datasets/wildlifedatasets/wildlifereid-10k"
DINOV3_ROOT = "/kaggle/input/models/pr4deepr/dinov3-vith16plus"

for n, p in {"比赛数据": COMP_DIR, "Bio3": BIO3_ROOT, "DINOv3": DINOV3_ROOT}.items():
    print(f"{'✅' if os.path.isdir(p) else '❌'} {n}: {p}")

# ======================== 数据 ========================
metadata = pd.read_csv(os.path.join(COMP_DIR, "metadata.csv"))
metadata = metadata[metadata["species"].notna()].reset_index(drop=True)  # 去掉species为NaN的行
sample_sub = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

# 找图像路径列
img_col = None
for c in ["path", "image_path", "file_path", "filename"]:
    if c in metadata.columns:
        img_col = c; break
assert img_col, f"找不到图像列! 列名: {list(metadata.columns)}"

# 测试第一张图能不能找到
test_img = str(metadata.iloc[0][img_col])
img_base = COMP_DIR
for try_base in [COMP_DIR, os.path.join(COMP_DIR, "images")]:
    if os.path.exists(os.path.join(try_base, test_img)):
        img_base = try_base; break
print(f"图像列: {img_col}, 基目录: {img_base}, 首图存在: {os.path.exists(os.path.join(img_base, test_img))}")

print(f"\n总图片: {len(metadata)}")
for sp in sorted(metadata["species"].unique()):
    m = metadata[metadata["species"]==sp]
    tr, te = m[m["split"]=="train"], m[m["split"]=="test"]
    print(f"  {sp}: train={len(tr)} ({tr['identity'].nunique()}ids), test={len(te)}")

# ======================== Dataset ========================
class DS(Dataset):
    def __init__(self, df, base, transform, col):
        self.df, self.base, self.transform, self.col = df.reset_index(drop=True), base, transform, col
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        p = os.path.join(self.base, str(self.df.iloc[i][self.col]))
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB", (224,224), (128,128,128))
        return self.transform(img), i

@torch.no_grad()
def extract(model, dl, desc=""):
    model.eval(); feats = []; t0 = time.time()
    for bi, (imgs, _) in enumerate(dl):
        out = model(imgs.to(device, dtype=torch.float16))
        if isinstance(out, (tuple, list)): out = out[0]
        if isinstance(out, dict):
            out = out.get("last_hidden_state", out.get("pooler_output", list(out.values())[0]))
        if out.dim() == 3: out = out[:, 0]
        feats.append(F.normalize(out.float(), dim=-1).cpu().numpy())
        if (bi+1) % 30 == 0: print(f"  [{desc}] {(bi+1)*dl.batch_size}/{len(dl.dataset)} ({time.time()-t0:.0f}s)")
    feats = np.concatenate(feats)
    print(f"  [{desc}] {feats.shape} done in {time.time()-t0:.0f}s")
    return feats

# ======================== 诊断: 打印完整目录树 ========================
print("\n=== 模型目录完整结构 ===")
for label, root in [("DINOv3", DINOV3_ROOT), ("Bio3", BIO3_ROOT)]:
    print(f"\n📁 {label}: {root}")
    for r, dirs, files in os.walk(root):
        depth = r.replace(root, "").count(os.sep)
        indent = "  " * (depth + 1)
        print(f"{indent}📁 {os.path.basename(r)}/")
        for f in files:
            sz = os.path.getsize(os.path.join(r, f))
            szs = f"{sz/1024**3:.1f}GB" if sz > 1024**3 else f"{sz/1024**2:.0f}MB" if sz > 1024**2 else f"{sz/1024:.0f}KB"
            print(f"{indent}  📄 {f} ({szs})")

# ======================== 找模型权重 ========================
def find_weights(root):
    """找最大的权重文件（.safetensors优先，其次.bin）"""
    all_weights = []
    for r, d, files in os.walk(root):
        for f in files:
            fp = os.path.join(r, f)
            sz = os.path.getsize(fp)
            if sz < 1_000_000:  # 跳过小于1MB的
                continue
            if f.endswith(".safetensors"):
                all_weights.append((fp, sz, 0))  # 优先级0(最高)
            elif f.endswith(".bin"):
                all_weights.append((fp, sz, 1))
    if not all_weights:
        return None
    # 按优先级排序，同优先级取最大
    all_weights.sort(key=lambda x: (x[2], -x[1]))
    return all_weights[0][0]

def find_config(root):
    for r, d, files in os.walk(root):
        if "config.json" in files:
            with open(os.path.join(r, "config.json")) as f:
                return json.load(f), r
    return None, root

# ======================== 提取特征 ========================
ALL_FEATURES = {}

# --- 1. DINOv3 ViT-H+ ---
print("\n" + "="*60 + "\n[1/3] DINOv3 ViT-H+\n" + "="*60)
d3w = find_weights(DINOV3_ROOT)
d3cfg, d3cfgdir = find_config(DINOV3_ROOT)
print(f"  weights: {d3w}")
if d3cfg:
    print(f"  config keys: {list(d3cfg.keys())}")
    print(f"  architecture: {d3cfg.get('architecture', d3cfg.get('model_name', d3cfg.get('model_type', '?')))}")

d3 = None
arch_name = None
if d3cfg:
    arch_name = d3cfg.get("architecture", d3cfg.get("model_name", None))

# 尝试1: config里的architecture
if arch_name and d3w:
    try:
        d3 = timm.create_model(arch_name, pretrained=False, num_classes=0, checkpoint_path=d3w)
        print(f"  ✅ Loaded: {arch_name}")
    except Exception as e:
        print(f"  ❌ config arch failed: {e}")

# 尝试2: timm所有dinov3 huge/h+变体
if d3 is None and d3w:
    # 列出timm中所有dinov3相关模型
    all_dinov3 = [m for m in timm.list_models("*dinov3*") if "huge" in m or "hplus" in m or "h_plus" in m or "vith" in m]
    print(f"  timm中的DINOv3 H+变体: {all_dinov3}")
    for name in all_dinov3:
        try:
            d3 = timm.create_model(name, pretrained=False, num_classes=0, checkpoint_path=d3w)
            print(f"  ✅ Loaded: {name}"); break
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            d3 = None

# 尝试3: 列出timm中所有dinov3模型供参考
if d3 is None:
    all_d3 = timm.list_models("*dinov3*")
    print(f"  timm中所有dinov3模型({len(all_d3)}个): {all_d3[:20]}")

if d3:
    d3 = d3.half().to(device).eval()
    try:
        cfg = timm.data.resolve_model_data_config(d3)
        tf = timm.data.create_transform(**cfg, is_training=False)
    except:
        tf = transforms.Compose([transforms.Resize(256, interpolation=3), transforms.CenterCrop(224),
              transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    dl = DataLoader(DS(metadata, img_base, tf, img_col), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    ALL_FEATURES["dinov3"] = extract(d3, dl, "DINOv3")
    del d3; gc.collect(); torch.cuda.empty_cache()
else:
    print("  ⚠️ DINOv3 跳过（请把上面timm模型列表截图给我，我来匹配正确名字）")

# --- 2. MegaDescriptor-L-384 ---
print("\n" + "="*60 + "\n[2/3] MegaDescriptor-L-384\n" + "="*60)
ml_root = os.path.join(BIO3_ROOT, "models", "megadesc-l384")
if not os.path.isdir(ml_root):
    # 递归搜索
    for r, d, f in os.walk(BIO3_ROOT):
        if "l384" in r.lower() or "l-384" in r.lower():
            ml_root = r; break
mlw = find_weights(ml_root)
mlcfg, mlcfgdir = find_config(ml_root)
print(f"  root: {ml_root}\n  weights: {mlw}")
print(f"  config arch: {mlcfg.get('architecture','?') if mlcfg else 'no config'}")

ml = None
arch = mlcfg.get("architecture", "") if mlcfg else ""
print(f"  config arch: {arch}")
# MegaDescriptor用timm加载，必须strict=False（它有额外的ArcFace头权重）
candidates = [arch] if arch else []
candidates += ["swin_large_patch4_window12_384", "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
               "swin_large_patch4_window12_384.ms_in22k"]
for name in candidates:
    if not name: continue
    try:
        ml = timm.create_model(name, pretrained=False, num_classes=0)
        if mlw.endswith(".safetensors"):
            sd = load_file(mlw)
        else:
            sd = torch.load(mlw, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        info = ml.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(info.unexpected_keys)
        print(f"  ✅ {name}: loaded {n_loaded}/{len(sd)} params (missing={len(info.missing_keys)}, extra={len(info.unexpected_keys)})")
        break
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        ml = None

if ml:
    ml = ml.half().to(device).eval()
    try:
        cfg = timm.data.resolve_model_data_config(ml)
        tf = timm.data.create_transform(**cfg, is_training=False)
    except:
        tf = transforms.Compose([transforms.Resize(384, interpolation=3), transforms.CenterCrop(384),
              transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    dl = DataLoader(DS(metadata, img_base, tf, img_col), batch_size=48, shuffle=False, num_workers=4, pin_memory=True)
    ALL_FEATURES["mega_l"] = extract(ml, dl, "MegaDesc-L")
    del ml; gc.collect(); torch.cuda.empty_cache()
else:
    print("  ⚠️ MegaDescriptor-L 跳过")

# --- 3. MegaDescriptor-DINOv2-518 ---
print("\n" + "="*60 + "\n[3/3] MegaDescriptor-DINOv2-518\n" + "="*60)
md_root = os.path.join(BIO3_ROOT, "models", "megadesc-dinov2-518")
if not os.path.isdir(md_root):
    for r, d, f in os.walk(BIO3_ROOT):
        if "dinov2" in r.lower() and "518" in r.lower():
            md_root = r; break
mdw = find_weights(md_root)
mdcfg, mdcfgdir = find_config(md_root)
print(f"  root: {md_root}\n  weights: {mdw}")
print(f"  config arch: {mdcfg.get('architecture','?') if mdcfg else 'no config'}")

md = None
arch = mdcfg.get("architecture", "") if mdcfg else ""
print(f"  config arch: {arch}")
candidates = [arch] if arch else []
candidates += ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_dinov2.lvd142m",
               "vit_large_patch14_reg4_dinov2.lvd142m_pc24_ft_in1k_518"]
for name in candidates:
    if not name: continue
    try:
        md = timm.create_model(name, pretrained=False, num_classes=0)
        if mdw.endswith(".safetensors"):
            sd = load_file(mdw)
        else:
            sd = torch.load(mdw, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        info = md.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(info.unexpected_keys)
        print(f"  ✅ {name}: loaded {n_loaded}/{len(sd)} params (missing={len(info.missing_keys)}, extra={len(info.unexpected_keys)})")
        break
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        md = None

if md:
    md = md.half().to(device).eval()
    try:
        cfg = timm.data.resolve_model_data_config(md)
        tf = timm.data.create_transform(**cfg, is_training=False)
    except:
        tf = transforms.Compose([transforms.Resize(518, interpolation=3), transforms.CenterCrop(518),
              transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    dl = DataLoader(DS(metadata, img_base, tf, img_col), batch_size=24, shuffle=False, num_workers=4, pin_memory=True)
    ALL_FEATURES["mega_d"] = extract(md, dl, "MegaDesc-D")
    del md; gc.collect(); torch.cuda.empty_cache()
else:
    print("  ⚠️ MegaDescriptor-DINOv2 跳过")

print(f"\n{'='*60}\n特征提取完成! 成功加载 {len(ALL_FEATURES)} 个backbone:")
for k, v in ALL_FEATURES.items(): print(f"  {k}: {v.shape}")
np.savez_compressed("/kaggle/working/features.npz", **ALL_FEATURES)

# ======================== 聚类 ========================
species_list = sorted(metadata["species"].unique())
n_models = len(ALL_FEATURES)
w = 1.0 / max(n_models, 1)
results = {}

for species in species_list:
    print(f"\n--- {species} ---")
    sp = metadata[metadata["species"]==species]
    tr, te = sp[sp["split"]=="train"], sp[sp["split"]=="test"]
    tr_idx, te_idx = tr.index.tolist(), te.index.tolist()

    # 融合测试集相似度
    te_sim = sum(w * (feats[te_idx] @ feats[te_idx].T) for feats in ALL_FEATURES.values())

    # 训练集grid search阈值
    if len(tr_idx) > 10:
        tr_sim = sum(w * (feats[tr_idx] @ feats[tr_idx].T) for feats in ALL_FEATURES.values())
        tr_dist = np.clip(1.0 - tr_sim, 0, None); np.fill_diagonal(tr_dist, 0)
        tr_dist = (tr_dist + tr_dist.T) / 2
        Z_tr = linkage(squareform(tr_dist, checks=False), method="average")
        tr_labels = tr["identity"].values

        best_ari, best_t = -1, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            ari = adjusted_rand_score(tr_labels, fcluster(Z_tr, t=t, criterion="distance"))
            if ari > best_ari: best_ari, best_t = ari, t
        print(f"  Train: ARI={best_ari:.4f}, threshold={best_t:.3f}")
    else:
        best_t = 0.55
        print(f"  No train data, default threshold={best_t}")

    # 测试集聚类
    te_dist = np.clip(1.0 - te_sim, 0, None); np.fill_diagonal(te_dist, 0)
    te_dist = (te_dist + te_dist.T) / 2
    labels = fcluster(linkage(squareform(te_dist, checks=False), method="average"), t=best_t, criterion="distance")
    n_cl = len(set(labels))
    print(f"  Test: {len(te_idx)} imgs → {n_cl} clusters")
    results[species] = (te_idx, labels)

# ======================== 提交 ========================
all_ids, all_labels, offset = [], [], 0
for sp in species_list:
    idx, lab = results[sp]
    ids = metadata.iloc[idx]["image_id"].values
    lab_off = lab + offset; offset = lab_off.max() + 1
    all_ids.extend(ids); all_labels.extend(lab_off)

sub = pd.DataFrame({"image_id": all_ids, "identity": all_labels})
sub_final = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
miss = sub_final["identity"].isna().sum()
if miss > 0:
    mx = sub_final["identity"].max()
    sub_final.loc[sub_final["identity"].isna(), "identity"] = range(int(mx)+1, int(mx)+1+miss)
sub_final["identity"] = sub_final["identity"].astype(int)
sub_final.to_csv("/kaggle/working/submission.csv", index=False)

print(f"\n{'='*60}")
print(f"✅ submission.csv saved! ({len(sub_final)} images, {sub_final['identity'].nunique()} clusters)")
print(f"{'='*60}")
