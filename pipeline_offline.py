## ================================================================
## AnimalCLEF2026 Pipeline — 无网络版 (关网环境可用)
## GPU: RTX Pro 6000 (48GB) | 3-backbone | 纯预装包
## 不需要pip install任何东西！
## ================================================================

# ============ Cell 1: Import (全部是Kaggle预装包) ============
import os, sys, gc, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# 检查预装包版本
import timm; print(f"timm: {timm.__version__}")
import safetensors; print(f"safetensors: {safetensors.__version__}")

# ============ Cell 2: 扫描Input路径 ============
input_root = "/kaggle/input"
print("=== Kaggle Input ===")
for d in sorted(os.listdir(input_root)):
    full = os.path.join(input_root, d)
    if os.path.isdir(full):
        print(f"\n📁 {d}/")
        for sub in sorted(os.listdir(full))[:15]:
            subsub = os.path.join(full, sub)
            tag = "📁" if os.path.isdir(subsub) else "📄"
            size = ""
            if os.path.isfile(subsub):
                s = os.path.getsize(subsub)
                size = f" ({s/1024**2:.1f}MB)" if s > 1024**2 else f" ({s/1024:.0f}KB)"
            print(f"  {tag} {sub}{size}")
            if os.path.isdir(subsub):
                for s3 in sorted(os.listdir(subsub))[:8]:
                    p3 = os.path.join(subsub, s3)
                    tag3 = "📁" if os.path.isdir(p3) else "📄"
                    sz = ""
                    if os.path.isfile(p3):
                        ss = os.path.getsize(p3)
                        sz = f" ({ss/1024**3:.1f}GB)" if ss > 1024**3 else f" ({ss/1024**2:.1f}MB)" if ss > 1024**2 else ""
                    print(f"    {tag3} {s3}{sz}")

# ============ Cell 3: 配置路径 (根据Cell 2输出调整!) ============
# --- 比赛数据 ---
COMP_DIR = None
for d in os.listdir(input_root):
    p = os.path.join(input_root, d)
    if os.path.isfile(os.path.join(p, "metadata.csv")) and os.path.isfile(os.path.join(p, "sample_submission.csv")):
        COMP_DIR = p
        break
# 如果自动检测失败，手动设置:
# COMP_DIR = "/kaggle/input/animal"

# --- 模型目录 ---
# 自动查找safetensors文件
def find_model_weights(start_dir):
    """递归查找包含safetensors或bin权重的目录"""
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if f.endswith(".safetensors") and os.path.getsize(os.path.join(root, f)) > 10*1024*1024:
                return root, f
            if f.endswith(".bin") and "pytorch_model" in f:
                return root, f
    return start_dir, None

# DINOv3 ViT-H+
DINOV3_ROOT = None
for d in os.listdir(input_root):
    if "dinov3" in d.lower():
        DINOV3_ROOT = os.path.join(input_root, d)
        break

# Bio-3model
BIO3_ROOT = None
for d in os.listdir(input_root):
    if "bio" in d.lower() and "model" in d.lower():
        BIO3_ROOT = os.path.join(input_root, d)
        break
    if "3model" in d.lower():
        BIO3_ROOT = os.path.join(input_root, d)
        break

print(f"\n=== 检测到的路径 ===")
print(f"比赛数据: {COMP_DIR}")
print(f"DINOv3:   {DINOV3_ROOT}")
print(f"Bio3Model: {BIO3_ROOT}")

# 查找具体模型权重
if DINOV3_ROOT:
    d3_dir, d3_file = find_model_weights(DINOV3_ROOT)
    print(f"DINOv3权重: {d3_dir}/{d3_file}")

if BIO3_ROOT:
    for model_name in ["megadesc-l384", "megadesc-dinov2-518"]:
        for root, dirs, files in os.walk(BIO3_ROOT):
            if model_name in root.lower() or model_name.replace("-","") in root.lower().replace("-",""):
                mdir, mfile = find_model_weights(root)
                print(f"{model_name}权重: {mdir}/{mfile}")
                break

# ============ Cell 4: 读取数据 ============
metadata = pd.read_csv(os.path.join(COMP_DIR, "metadata.csv"))
sample_sub = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

print(f"\n=== 数据概况 ===")
print(f"总图片: {len(metadata)}")
for sp in sorted(metadata["species"].unique()):
    m = metadata[metadata["species"] == sp]
    tr = m[m["split"]=="train"]
    te = m[m["split"]=="test"]
    nid = tr["identity"].nunique() if "identity" in tr.columns else 0
    print(f"  {sp}: train={len(tr)}, test={len(te)}, ids={nid}")

# 图像路径测试
test_row = metadata.iloc[0]
img_col = "path" if "path" in metadata.columns else "image_path" if "image_path" in metadata.columns else None
if img_col:
    test_path = os.path.join(COMP_DIR, test_row[img_col])
    if not os.path.exists(test_path):
        # 可能在images/子目录
        test_path = os.path.join(COMP_DIR, "images", test_row[img_col])
    print(f"\n图像列: {img_col}")
    print(f"示例路径: {test_path}, 存在: {os.path.exists(test_path)}")
else:
    print(f"\n列名: {list(metadata.columns)}")
    print("WARNING: 找不到图像路径列，请手动指定!")

# ============ Cell 5: Dataset ============
class AnimalDataset(Dataset):
    def __init__(self, df, comp_dir, transform, img_col="path"):
        self.df = df.reset_index(drop=True)
        self.comp_dir = comp_dir
        self.transform = transform
        self.img_col = img_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row[self.img_col])
        # 尝试多种路径
        for base in [self.comp_dir, os.path.join(self.comp_dir, "images")]:
            full = os.path.join(base, rel_path)
            if os.path.exists(full):
                img = Image.open(full).convert("RGB")
                return self.transform(img), idx
        # fallback
        return self.transform(Image.new("RGB", (224,224), (128,128,128))), idx

# ============ Cell 6: 通用特征提取 ============
@torch.no_grad()
def extract_features(model, dataloader, device, desc=""):
    model.eval()
    all_feats = []
    t0 = time.time()
    for bi, (imgs, idxs) in enumerate(dataloader):
        imgs = imgs.to(device, dtype=torch.float16)
        out = model(imgs)
        # 处理不同模型的输出格式
        if isinstance(out, dict):
            # transformers风格
            if "last_hidden_state" in out:
                feats = out["last_hidden_state"][:, 0]  # CLS token
            elif "pooler_output" in out:
                feats = out["pooler_output"]
            else:
                feats = list(out.values())[0]
                if feats.dim() == 3:
                    feats = feats[:, 0]
        elif isinstance(out, (tuple, list)):
            feats = out[0]
            if feats.dim() == 3:
                feats = feats[:, 0]
        else:
            feats = out
            if feats.dim() == 3:
                feats = feats[:, 0]

        feats = F.normalize(feats.float(), dim=-1)
        all_feats.append(feats.cpu().numpy())

        if (bi+1) % 20 == 0:
            n_done = (bi+1) * dataloader.batch_size
            elapsed = time.time() - t0
            speed = n_done / elapsed
            print(f"  [{desc}] {n_done}/{len(dataloader.dataset)} ({speed:.0f} img/s, {elapsed:.0f}s)")

    feats = np.concatenate(all_feats, axis=0)
    print(f"  [{desc}] Done: {feats.shape}, {time.time()-t0:.0f}s")
    return feats

# ============ Cell 7: 加载模型 + 提特征 ============
# 检测图像路径列名
img_col = "path" if "path" in metadata.columns else "image_path"
print(f"Using image column: {img_col}")

ALL_FEATURES = {}
from safetensors.torch import load_file

# -------- 7a: DINOv3 ViT-H+ --------
print("\n" + "="*60)
print("[1/3] DINOv3 ViT-H+")
print("="*60)

# 找权重文件
d3_dir, d3_file = find_model_weights(DINOV3_ROOT)
d3_weight_path = os.path.join(d3_dir, d3_file)
print(f"  Weight: {d3_weight_path} ({os.path.getsize(d3_weight_path)/1024**3:.1f}GB)")

# 读config获取架构名
config_candidates = [
    os.path.join(d3_dir, "config.json"),
    os.path.join(os.path.dirname(d3_dir), "config.json"),
]
d3_config = None
for cc in config_candidates:
    if os.path.exists(cc):
        with open(cc) as f:
            d3_config = json.load(f)
        print(f"  Config: {cc}")
        break

# 尝试多种加载方式
d3_model = None

# 方式1: timm (如果config里有architecture信息)
if d3_config and "architecture" in d3_config:
    arch = d3_config["architecture"]
    print(f"  Architecture from config: {arch}")
    try:
        d3_model = timm.create_model(arch, pretrained=False, num_classes=0)
        sd = load_file(d3_weight_path)
        d3_model.load_state_dict(sd, strict=False)
        print(f"  Loaded via timm: {arch}")
    except Exception as e:
        print(f"  timm load failed: {e}")
        d3_model = None

# 方式2: timm直接checkpoint_path
if d3_model is None:
    for name_try in [
        "vit_huge_patch16_gap_dinov3.lvd1689m",
        "vit_huge_patch16_dinov3.lvd1689m",
    ]:
        try:
            d3_model = timm.create_model(name_try, pretrained=False, num_classes=0,
                                          checkpoint_path=d3_weight_path)
            print(f"  Loaded via timm: {name_try}")
            break
        except Exception as e:
            print(f"  {name_try}: {e}")

# 方式3: transformers
if d3_model is None:
    try:
        from transformers import AutoModel
        d3_model = AutoModel.from_pretrained(d3_dir, trust_remote_code=True, local_files_only=True)
        print(f"  Loaded via transformers")
    except Exception as e:
        print(f"  transformers failed: {e}")

if d3_model is not None:
    d3_model = d3_model.half().to(device).eval()
    # 获取transform
    try:
        data_cfg = timm.data.resolve_model_data_config(d3_model)
        d3_transform = timm.data.create_transform(**data_cfg, is_training=False)
    except:
        d3_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    ds = AnimalDataset(metadata, COMP_DIR, d3_transform, img_col)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    ALL_FEATURES["dinov3"] = extract_features(d3_model, dl, device, "DINOv3-H+")
    del d3_model; gc.collect(); torch.cuda.empty_cache()
else:
    print("  !!! DINOv3加载失败，跳过")

# -------- 7b: MegaDescriptor-L-384 --------
print("\n" + "="*60)
print("[2/3] MegaDescriptor-L-384")
print("="*60)

mega_l_root = None
for root, dirs, files in os.walk(BIO3_ROOT):
    if "megadesc-l384" in root.lower().replace("_", "-"):
        mega_l_root = root
        break
    if "l384" in root.lower() or "l-384" in root.lower():
        mega_l_root = root
        break

if mega_l_root:
    ml_dir, ml_file = find_model_weights(mega_l_root)
    ml_weight = os.path.join(ml_dir, ml_file)
    print(f"  Weight: {ml_weight}")

    ml_model = None
    # 读config
    for cc in [os.path.join(ml_dir, "config.json"), os.path.join(mega_l_root, "config.json")]:
        if os.path.exists(cc):
            with open(cc) as f:
                cfg = json.load(f)
            arch = cfg.get("architecture", cfg.get("model_name", ""))
            if arch:
                try:
                    ml_model = timm.create_model(arch, pretrained=False, num_classes=0,
                                                  checkpoint_path=ml_weight)
                    print(f"  Loaded: {arch}")
                except Exception as e:
                    print(f"  {arch} failed: {e}")
            break

    # fallback: 尝试常见Swin-L名
    if ml_model is None:
        for name_try in ["swin_large_patch4_window12_384.ms_in22k_ft_in1k",
                         "swin_large_patch4_window12_384"]:
            try:
                ml_model = timm.create_model(name_try, pretrained=False, num_classes=0)
                sd = load_file(ml_weight)
                ml_model.load_state_dict(sd, strict=False)
                print(f"  Loaded fallback: {name_try}")
                break
            except Exception as e:
                print(f"  {name_try}: {e}")

    if ml_model is not None:
        ml_model = ml_model.half().to(device).eval()
        try:
            data_cfg = timm.data.resolve_model_data_config(ml_model)
            ml_transform = timm.data.create_transform(**data_cfg, is_training=False)
        except:
            ml_transform = transforms.Compose([
                transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        ds = AnimalDataset(metadata, COMP_DIR, ml_transform, img_col)
        dl = DataLoader(ds, batch_size=48, shuffle=False, num_workers=4, pin_memory=True)
        ALL_FEATURES["mega_l"] = extract_features(ml_model, dl, device, "MegaDesc-L")
        del ml_model; gc.collect(); torch.cuda.empty_cache()
    else:
        print("  !!! MegaDescriptor-L加载失败")

# -------- 7c: MegaDescriptor-DINOv2-518 --------
print("\n" + "="*60)
print("[3/3] MegaDescriptor-DINOv2-518")
print("="*60)

mega_d_root = None
for root, dirs, files in os.walk(BIO3_ROOT):
    if "dinov2-518" in root.lower().replace("_", "-") or "dinov2518" in root.lower():
        mega_d_root = root
        break

if mega_d_root:
    md_dir, md_file = find_model_weights(mega_d_root)
    md_weight = os.path.join(md_dir, md_file)
    print(f"  Weight: {md_weight}")

    md_model = None
    for cc in [os.path.join(md_dir, "config.json"), os.path.join(mega_d_root, "config.json")]:
        if os.path.exists(cc):
            with open(cc) as f:
                cfg = json.load(f)
            arch = cfg.get("architecture", cfg.get("model_name", ""))
            if arch:
                try:
                    md_model = timm.create_model(arch, pretrained=False, num_classes=0,
                                                  checkpoint_path=md_weight)
                    print(f"  Loaded: {arch}")
                except Exception as e:
                    print(f"  {arch} failed: {e}")
            break

    if md_model is None:
        for name_try in ["vit_large_patch14_reg4_dinov2.lvd142m",
                         "vit_large_patch14_dinov2.lvd142m"]:
            try:
                md_model = timm.create_model(name_try, pretrained=False, num_classes=0)
                sd = load_file(md_weight)
                md_model.load_state_dict(sd, strict=False)
                print(f"  Loaded fallback: {name_try}")
                break
            except Exception as e:
                print(f"  {name_try}: {e}")

    if md_model is not None:
        md_model = md_model.half().to(device).eval()
        try:
            data_cfg = timm.data.resolve_model_data_config(md_model)
            md_transform = timm.data.create_transform(**data_cfg, is_training=False)
        except:
            md_transform = transforms.Compose([
                transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        ds = AnimalDataset(metadata, COMP_DIR, md_transform, img_col)
        dl = DataLoader(ds, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)
        ALL_FEATURES["mega_d"] = extract_features(md_model, dl, device, "MegaDesc-D518")
        del md_model; gc.collect(); torch.cuda.empty_cache()
    else:
        print("  !!! MegaDescriptor-DINOv2加载失败")

# ============ 特征摘要 ============
print("\n" + "="*60)
print("特征提取完成!")
for name, feat in ALL_FEATURES.items():
    print(f"  {name}: {feat.shape}")
print("="*60)

# 缓存
np.savez_compressed("/kaggle/working/features.npz", **ALL_FEATURES)

# ============ Cell 8: 分物种聚类 ============
species_list = sorted(metadata["species"].unique())
n_models = len(ALL_FEATURES)
default_w = 1.0 / n_models  # 均匀权重

results = {}  # species -> (test_indices, labels)

for species in species_list:
    print(f"\n--- {species} ---")
    sp = metadata[metadata["species"] == species]
    sp_train = sp[sp["split"] == "train"]
    sp_test = sp[sp["split"] == "test"]
    train_idx = sp_train.index.tolist()
    test_idx = sp_test.index.tolist()

    # 融合相似度矩阵 (测试集)
    test_sim = np.zeros((len(test_idx), len(test_idx)), dtype=np.float32)
    for name, feats in ALL_FEATURES.items():
        f_test = feats[test_idx]
        test_sim += default_w * (f_test @ f_test.T)

    # 训练集grid search阈值
    if len(train_idx) > 10:
        train_sim = np.zeros((len(train_idx), len(train_idx)), dtype=np.float32)
        for name, feats in ALL_FEATURES.items():
            f_train = feats[train_idx]
            train_sim += default_w * (f_train @ f_train.T)

        train_labels = sp_train["identity"].values
        train_dist = np.clip(1.0 - train_sim, 0, None)
        np.fill_diagonal(train_dist, 0)
        train_dist = (train_dist + train_dist.T) / 2
        condensed_tr = squareform(train_dist, checks=False)
        Z_tr = linkage(condensed_tr, method="average")

        best_ari, best_t = -1, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            pred = fcluster(Z_tr, t=t, criterion="distance")
            ari = adjusted_rand_score(train_labels, pred)
            if ari > best_ari:
                best_ari, best_t = ari, t
        print(f"  Train grid search: best ARI={best_ari:.4f} at threshold={best_t:.3f}")
    else:
        best_t = 0.55
        print(f"  No training data, using default threshold={best_t:.3f}")

    # 测试集聚类
    test_dist = np.clip(1.0 - test_sim, 0, None)
    np.fill_diagonal(test_dist, 0)
    test_dist = (test_dist + test_dist.T) / 2
    condensed_te = squareform(test_dist, checks=False)
    Z_te = linkage(condensed_te, method="average")
    labels = fcluster(Z_te, t=best_t, criterion="distance")

    n_cl = len(set(labels))
    n_sing = sum(1 for c in set(labels) if list(labels).count(c) == 1)
    print(f"  Test: {len(test_idx)} imgs → {n_cl} clusters, {n_sing} singletons")
    results[species] = (test_idx, labels)

# ============ Cell 9: 生成submission ============
print("\n" + "="*60)
print("生成submission.csv")
print("="*60)

all_ids = []
all_labels = []
offset = 0
for species in species_list:
    test_idx, labels = results[species]
    image_ids = metadata.iloc[test_idx]["image_id"].values
    labels_offset = labels + offset
    offset = labels_offset.max() + 1
    all_ids.extend(image_ids)
    all_labels.extend(labels_offset)

sub = pd.DataFrame({"image_id": all_ids, "identity": all_labels})
sub_final = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")

missing = sub_final["identity"].isna().sum()
if missing > 0:
    print(f"WARNING: {missing} missing images!")
    max_l = sub_final["identity"].max()
    for i, idx in enumerate(sub_final[sub_final["identity"].isna()].index):
        sub_final.loc[idx, "identity"] = max_l + 1 + i

sub_final["identity"] = sub_final["identity"].astype(int)
sub_final.to_csv("/kaggle/working/submission.csv", index=False)

print(f"Saved: /kaggle/working/submission.csv")
print(f"Images: {len(sub_final)}")
print(f"Clusters: {sub_final['identity'].nunique()}")
print(sub_final.head(10))
print(f"\n{'='*60}")
print("DONE! 下载submission.csv到AnimalCLEF手动提交")
print(f"{'='*60}")
