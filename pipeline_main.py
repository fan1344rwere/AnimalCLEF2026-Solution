## ==========================================================================
## AnimalCLEF2026 主Pipeline — 粘贴到Kaggle Notebook Cell里运行
## GPU: RTX Pro 6000 (48GB) | 4-backbone ensemble | 纯冻结特征 + HAC聚类
## ==========================================================================
## 步骤: 安装依赖 → 提特征(4个backbone) → 加权融合 → k-reciprocal re-ranking
##       → HAC聚类(per-species阈值) → 生成submission.csv
## ==========================================================================

# ============ Cell 1: 安装依赖 ============
!pip install -q timm open_clip_torch safetensors hdbscan

import os, sys, gc, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ============ Cell 2: 路径配置 ============
# --- 根据你的Kaggle Input实际路径修改 ---
# 先打印看看实际挂载路径
print("=== Kaggle Input 目录结构 ===")
input_root = "/kaggle/input"
for d in sorted(os.listdir(input_root)):
    full = os.path.join(input_root, d)
    if os.path.isdir(full):
        print(f"\n📁 {d}/")
        for sub in sorted(os.listdir(full))[:10]:
            subsub = os.path.join(full, sub)
            if os.path.isdir(subsub):
                print(f"   📁 {sub}/")
                for s in sorted(os.listdir(subsub))[:5]:
                    print(f"      {'📁' if os.path.isdir(os.path.join(subsub,s)) else '📄'} {s}")
            else:
                print(f"   📄 {sub}")

# ============ Cell 3: 自动检测路径 ============
def find_path(root, name_contains):
    """在input目录下找包含特定名字的路径"""
    for d in os.listdir(root):
        if name_contains.lower() in d.lower():
            return os.path.join(root, d)
    return None

# 比赛数据
COMP_DIR = find_path(input_root, "animal")
assert COMP_DIR, "找不到比赛数据目录!"
METADATA_PATH = os.path.join(COMP_DIR, "metadata.csv")
SAMPLE_SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")
IMAGE_ROOT = os.path.join(COMP_DIR, "images") if os.path.isdir(os.path.join(COMP_DIR, "images")) else COMP_DIR

# 模型路径
BIO3_DIR = find_path(input_root, "bio-3model") or find_path(input_root, "bio3model")
assert BIO3_DIR, "找不到Bio-3model目录!"
BIOCLIP_DIR = os.path.join(BIO3_DIR, "models", "bioclip25-vith14")
MEGADESC_L_DIR = os.path.join(BIO3_DIR, "models", "megadesc-l384")
MEGADESC_DINO_DIR = os.path.join(BIO3_DIR, "models", "megadesc-dinov2-518")

# DINOv3 ViT-H+ — 路径可能嵌套较深，递归查找
DINOV3_ROOT = find_path(input_root, "dinov3")
DINOV3_DIR = None
if DINOV3_ROOT:
    for root_d, dirs, files in os.walk(DINOV3_ROOT):
        if any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
            DINOV3_DIR = root_d
            break
    if not DINOV3_DIR:
        DINOV3_DIR = DINOV3_ROOT

# WildlifeReID-10k
WILDLIFE_DIR = find_path(input_root, "wildlife")

print(f"\n=== 路径配置 ===")
print(f"比赛数据:  {COMP_DIR}")
print(f"元数据:    {METADATA_PATH}")
print(f"图像根目录: {IMAGE_ROOT}")
print(f"BioCLIP:   {BIOCLIP_DIR}")
print(f"MegaDesc-L: {MEGADESC_L_DIR}")
print(f"MegaDesc-D: {MEGADESC_DINO_DIR}")
print(f"DINOv3 H+: {DINOV3_DIR}")
print(f"Wildlife:  {WILDLIFE_DIR}")

# ============ Cell 4: 数据准备 ============
metadata = pd.read_csv(METADATA_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(f"\n=== 数据概况 ===")
print(f"总图片数: {len(metadata)}")
print(f"物种分布:")
for sp in metadata["species"].unique():
    sp_data = metadata[metadata["species"] == sp]
    n_train = len(sp_data[sp_data["split"] == "train"])
    n_test = len(sp_data[sp_data["split"] == "test"])
    n_ids = sp_data[sp_data["split"] == "train"]["identity"].nunique() if "identity" in sp_data.columns else 0
    print(f"  {sp}: train={n_train}, test={n_test}, identities={n_ids}")

# 分物种索引
species_list = sorted(metadata["species"].unique())
test_mask = metadata["split"] == "test"
train_mask = metadata["split"] == "train"

# ============ Cell 5: 图像Dataset ============
class AnimalDataset(Dataset):
    def __init__(self, df, image_root, transform):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 尝试多种路径格式
        img_path = row.get("path", row.get("image_path", ""))
        candidates = [
            os.path.join(self.image_root, img_path),
            os.path.join(self.image_root, str(row.get("image_id", "")) + ".jpg"),
        ]
        for p in candidates:
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                return self.transform(img), idx
        # 如果都找不到，返回黑图
        print(f"WARNING: Image not found for idx {idx}: {img_path}")
        return self.transform(Image.new("RGB", (224, 224))), idx

# ============ Cell 6: 特征提取函数 ============
@torch.no_grad()
def extract_features(model, dataloader, device, desc="Extracting"):
    """提取特征，返回numpy数组 [N, D]"""
    all_feats = []
    model.eval()
    t0 = time.time()
    for batch_idx, (images, indices) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float16)
        feats = model(images)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", feats.get("last_hidden_state", list(feats.values())[0]))
        if feats.dim() == 3:  # [B, seq, D] -> take CLS token
            feats = feats[:, 0]
        feats = F.normalize(feats.float(), dim=-1)
        all_feats.append(feats.cpu().numpy())
        if (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {desc}: {(batch_idx+1)*dataloader.batch_size}/{len(dataloader.dataset)} "
                  f"({elapsed:.0f}s)")
    return np.concatenate(all_feats, axis=0)

# ============ Cell 7: 加载4个Backbone并提取特征 ============
BATCH_SIZE = 32  # RTX Pro 6000 48GB完全够

# --- 7a: DINOv3 ViT-H+ ---
print("\n" + "="*60)
print("Loading DINOv3 ViT-H+...")
print("="*60)

import timm
# 尝试从本地目录加载
try:
    dinov3_model = timm.create_model(
        "vit_huge_patch16_dinov3.lvd1689m",  # H+ 可能叫 huge
        pretrained=False,
    )
    # 手动加载权重
    from safetensors.torch import load_file
    sf_files = [f for f in os.listdir(DINOV3_DIR) if f.endswith(".safetensors")]
    if sf_files:
        state_dict = load_file(os.path.join(DINOV3_DIR, sf_files[0]))
        dinov3_model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded from safetensors: {sf_files[0]}")
except Exception as e:
    print(f"  timm加载方式1失败: {e}")
    try:
        # 尝试transformers方式
        from transformers import AutoModel
        dinov3_model = AutoModel.from_pretrained(DINOV3_DIR, trust_remote_code=True)
        print("  Loaded via transformers AutoModel")
    except Exception as e2:
        print(f"  transformers加载也失败: {e2}")
        print("  尝试timm自动下载...")
        dinov3_model = timm.create_model("hf_hub:timm/vit_huge_patch16_dinov3.lvd1689m", pretrained=True)

dinov3_model = dinov3_model.half().to(device).eval()

data_config = timm.data.resolve_model_data_config(dinov3_model)
dinov3_transform = timm.data.create_transform(**data_config, is_training=False)
print(f"  DINOv3 input size: {data_config.get('input_size', 'unknown')}")

ds_dinov3 = AnimalDataset(metadata, IMAGE_ROOT, dinov3_transform)
dl_dinov3 = DataLoader(ds_dinov3, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
feats_dinov3 = extract_features(dinov3_model, dl_dinov3, device, "DINOv3-H+")
print(f"  DINOv3 features: {feats_dinov3.shape}")

del dinov3_model; gc.collect(); torch.cuda.empty_cache()

# --- 7b: BioCLIP 2.5 ---
print("\n" + "="*60)
print("Loading BioCLIP 2.5 ViT-H/14...")
print("="*60)

import open_clip
# BioCLIP用open_clip加载
try:
    bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2.5-vith14",
        pretrained=BIOCLIP_DIR,  # 本地路径
    )
except:
    try:
        bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=os.path.join(BIOCLIP_DIR, "open_clip_model.safetensors"),
        )
    except:
        # 最后尝试直接从hub加载
        bioclip_model, _, bioclip_preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2.5-vith14"
        )

bioclip_model = bioclip_model.half().to(device).eval()

# BioCLIP只用visual encoder
class BioCLIPVisual(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model.visual
    def forward(self, x):
        return self.visual(x)

bioclip_visual = BioCLIPVisual(bioclip_model)

ds_bio = AnimalDataset(metadata, IMAGE_ROOT, bioclip_preprocess)
dl_bio = DataLoader(ds_bio, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
feats_bioclip = extract_features(bioclip_visual, dl_bio, device, "BioCLIP")
print(f"  BioCLIP features: {feats_bioclip.shape}")

del bioclip_model, bioclip_visual; gc.collect(); torch.cuda.empty_cache()

# --- 7c: MegaDescriptor-L-384 ---
print("\n" + "="*60)
print("Loading MegaDescriptor-L-384...")
print("="*60)

mega_l_model = timm.create_model(
    "hf-hub:BVRA/MegaDescriptor-L-384",
    pretrained=False,
)
# 加载本地权重
sf_mega = [f for f in os.listdir(MEGADESC_L_DIR) if f.endswith((".safetensors", ".bin"))]
if sf_mega:
    if sf_mega[0].endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(os.path.join(MEGADESC_L_DIR, sf_mega[0]))
    else:
        sd = torch.load(os.path.join(MEGADESC_L_DIR, sf_mega[0]), map_location="cpu")
    mega_l_model.load_state_dict(sd, strict=False)
    print(f"  Loaded: {sf_mega[0]}")

mega_l_model = mega_l_model.half().to(device).eval()
data_cfg_mega = timm.data.resolve_model_data_config(mega_l_model)
mega_transform = timm.data.create_transform(**data_cfg_mega, is_training=False)

ds_mega = AnimalDataset(metadata, IMAGE_ROOT, mega_transform)
dl_mega = DataLoader(ds_mega, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
feats_mega_l = extract_features(mega_l_model, dl_mega, device, "MegaDesc-L")
print(f"  MegaDesc-L features: {feats_mega_l.shape}")

del mega_l_model; gc.collect(); torch.cuda.empty_cache()

# --- 7d: MegaDescriptor-DINOv2-518 ---
print("\n" + "="*60)
print("Loading MegaDescriptor-DINOv2-518...")
print("="*60)

mega_d_model = timm.create_model(
    "hf-hub:BVRA/MegaDescriptor-DINOv2-518",
    pretrained=False,
)
sf_md = [f for f in os.listdir(MEGADESC_DINO_DIR) if f.endswith((".safetensors", ".bin"))]
if sf_md:
    if sf_md[0].endswith(".safetensors"):
        sd = load_file(os.path.join(MEGADESC_DINO_DIR, sf_md[0]))
    else:
        sd = torch.load(os.path.join(MEGADESC_DINO_DIR, sf_md[0]), map_location="cpu")
    mega_d_model.load_state_dict(sd, strict=False)
    print(f"  Loaded: {sf_md[0]}")

mega_d_model = mega_d_model.half().to(device).eval()
data_cfg_md = timm.data.resolve_model_data_config(mega_d_model)
mega_d_transform = timm.data.create_transform(**data_cfg_md, is_training=False)

ds_md = AnimalDataset(metadata, IMAGE_ROOT, mega_d_transform)
dl_md = DataLoader(ds_md, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
feats_mega_d = extract_features(mega_d_model, dl_md, device, "MegaDesc-DINOv2")
print(f"  MegaDesc-DINOv2 features: {feats_mega_d.shape}")

del mega_d_model; gc.collect(); torch.cuda.empty_cache()

print("\n=== 所有特征提取完成 ===")
print(f"DINOv3-H+:     {feats_dinov3.shape}")
print(f"BioCLIP 2.5:   {feats_bioclip.shape}")
print(f"MegaDesc-L:    {feats_mega_l.shape}")
print(f"MegaDesc-D518: {feats_mega_d.shape}")

# 缓存特征到磁盘
np.savez_compressed("/kaggle/working/features.npz",
    dinov3=feats_dinov3, bioclip=feats_bioclip,
    mega_l=feats_mega_l, mega_d=feats_mega_d)
print("Features saved to /kaggle/working/features.npz")

# ============ Cell 8: K-Reciprocal Re-ranking ============
def k_reciprocal_rerank(features, k1=20, k2=6, lambda_value=0.3):
    """
    K-reciprocal re-ranking (Zhong et al., CVPR 2017)
    输入: features [N, D] (已L2归一化)
    输出: reranked距离矩阵 [N, N]
    """
    N = features.shape[0]
    # 余弦相似度 → 距离
    sim = features @ features.T
    dist = 1.0 - sim

    # k近邻
    nn_indices = np.argsort(dist, axis=1)

    # k-reciprocal neighbors
    k_reciprocal_indices = []
    for i in range(N):
        forward_k = set(nn_indices[i, :k1+1].tolist())
        reciprocal = set()
        for j in forward_k:
            backward_k = set(nn_indices[j, :k1+1].tolist())
            if i in backward_k:
                reciprocal.add(j)
        # 扩展: 如果reciprocal中有一些节点的reciprocal集合和当前重叠>2/3，也加入
        expanded = set(reciprocal)
        for j in list(reciprocal):
            j_forward = set(nn_indices[j, :int(k1/2)+1].tolist())
            j_recip = set()
            for jj in j_forward:
                if j in set(nn_indices[jj, :int(k1/2)+1].tolist()):
                    j_recip.add(jj)
            if len(j_recip & reciprocal) > 2/3 * len(j_recip):
                expanded |= j_recip
        k_reciprocal_indices.append(expanded)

    # Jaccard距离
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in k_reciprocal_indices[i]:
            V[i, j] = 1.0

    # 用k2近邻的V做local query expansion
    V_qe = np.zeros_like(V)
    for i in range(N):
        topk2 = nn_indices[i, :k2]
        V_qe[i] = np.mean(V[topk2], axis=0)

    # Jaccard距离
    jaccard_dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i+1, N):
            min_sum = np.minimum(V_qe[i], V_qe[j]).sum()
            max_sum = np.maximum(V_qe[i], V_qe[j]).sum()
            if max_sum > 0:
                jaccard_dist[i, j] = 1.0 - min_sum / max_sum
            else:
                jaccard_dist[i, j] = 1.0
            jaccard_dist[j, i] = jaccard_dist[i, j]

    # 融合: λ * jaccard + (1-λ) * original distance
    final_dist = lambda_value * jaccard_dist + (1 - lambda_value) * dist
    return final_dist

# 简化版 (大规模数据用): 只用cosine similarity + mutual kNN filtering
def mutual_knn_similarity(features, k=50):
    """互近邻过滤：只保留双向都在top-k的相似度"""
    N = features.shape[0]
    sim = features @ features.T
    # 找每行top-k
    topk_idx = np.argsort(-sim, axis=1)[:, :k]
    # 互近邻mask
    mutual = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in topk_idx[i]:
            if i in topk_idx[j]:
                mutual[i, j] = True
                mutual[j, i] = True
    # 非互近邻的相似度设为0
    filtered_sim = sim * mutual
    return filtered_sim

# ============ Cell 9: 分物种聚类 ============
def cluster_species(feats_dict, weights, species_meta, train_identities,
                    threshold_range=np.arange(0.1, 0.95, 0.02)):
    """
    对单个物种:
    1. 加权融合4个backbone的相似度
    2. 在训练集上grid search最优阈值
    3. 对测试集用最优阈值做HAC聚类
    """
    sp_train = species_meta[species_meta["split"] == "train"]
    sp_test = species_meta[species_meta["split"] == "test"]

    if len(sp_train) == 0:
        # 无训练数据(如TexasHornedLizards)，直接在测试集上聚类
        test_indices = sp_test.index.tolist()
        combined_sim = np.zeros((len(test_indices), len(test_indices)), dtype=np.float32)
        for name, feat_all in feats_dict.items():
            feats_sp = feat_all[test_indices]
            sim = feats_sp @ feats_sp.T
            combined_sim += weights.get(name, 0.25) * sim
        # 默认阈值
        best_thresh = 0.55
    else:
        train_indices = sp_train.index.tolist()
        test_indices = sp_test.index.tolist()
        all_indices = train_indices + test_indices

        # 训练集上grid search
        train_labels = sp_train["identity"].values

        # 融合训练集相似度
        train_sim = np.zeros((len(train_indices), len(train_indices)), dtype=np.float32)
        for name, feat_all in feats_dict.items():
            feats_tr = feat_all[train_indices]
            sim = feats_tr @ feats_tr.T
            train_sim += weights.get(name, 0.25) * sim

        # Grid search
        best_ari, best_thresh = -1, 0.5
        train_dist = 1.0 - train_sim
        np.fill_diagonal(train_dist, 0)
        train_dist = np.clip(train_dist, 0, None)
        # 确保对称
        train_dist = (train_dist + train_dist.T) / 2

        condensed = squareform(train_dist, checks=False)
        Z = linkage(condensed, method="average")

        for t in threshold_range:
            labels_pred = fcluster(Z, t=t, criterion="distance")
            ari = adjusted_rand_score(train_labels, labels_pred)
            if ari > best_ari:
                best_ari = ari
                best_thresh = t

        print(f"    Train best: ARI={best_ari:.4f} at threshold={best_thresh:.3f}")

        # 测试集融合相似度
        combined_sim = np.zeros((len(test_indices), len(test_indices)), dtype=np.float32)
        for name, feat_all in feats_dict.items():
            feats_te = feat_all[test_indices]
            sim = feats_te @ feats_te.T
            combined_sim += weights.get(name, 0.25) * sim

    # 测试集HAC聚类
    test_dist = 1.0 - combined_sim
    np.fill_diagonal(test_dist, 0)
    test_dist = np.clip(test_dist, 0, None)
    test_dist = (test_dist + test_dist.T) / 2

    condensed_test = squareform(test_dist, checks=False)
    Z_test = linkage(condensed_test, method="average")
    cluster_labels = fcluster(Z_test, t=best_thresh, criterion="distance")

    n_clusters = len(set(cluster_labels))
    n_singletons = sum(1 for c in set(cluster_labels) if list(cluster_labels).count(c) == 1)
    print(f"    Test: {len(test_indices)} imgs → {n_clusters} clusters, {n_singletons} singletons")

    return test_indices, cluster_labels

# ============ Cell 10: 执行聚类 ============
print("\n" + "="*60)
print("开始分物种聚类...")
print("="*60)

feats_dict = {
    "dinov3": feats_dinov3,
    "bioclip": feats_bioclip,
    "mega_l": feats_mega_l,
    "mega_d": feats_mega_d,
}

# 每物种权重 (初始均匀，后续可调)
species_weights = {
    "LynxID2025": {"dinov3": 0.30, "bioclip": 0.20, "mega_l": 0.25, "mega_d": 0.25},
    "SalamanderID2025": {"dinov3": 0.35, "bioclip": 0.15, "mega_l": 0.20, "mega_d": 0.30},
    "SeaTurtleID2022": {"dinov3": 0.25, "bioclip": 0.15, "mega_l": 0.30, "mega_d": 0.30},
    "TexasHornedLizards": {"dinov3": 0.30, "bioclip": 0.25, "mega_l": 0.20, "mega_d": 0.25},
}

# 聚类结果收集
all_test_image_ids = []
all_cluster_labels = []
label_offset = 0

for species in species_list:
    print(f"\n--- {species} ---")
    sp_meta = metadata[metadata["species"] == species]
    weights = species_weights.get(species, {"dinov3": 0.25, "bioclip": 0.25, "mega_l": 0.25, "mega_d": 0.25})

    test_indices, labels = cluster_species(feats_dict, weights, sp_meta, None)

    # 获取image_id
    test_image_ids = metadata.iloc[test_indices]["image_id"].values
    # 偏移标签，确保不同物种标签不重叠
    labels_offset = labels + label_offset
    label_offset = labels_offset.max() + 1

    all_test_image_ids.extend(test_image_ids)
    all_cluster_labels.extend(labels_offset)

# ============ Cell 11: 生成提交文件 ============
print("\n" + "="*60)
print("生成submission.csv...")
print("="*60)

submission = pd.DataFrame({
    "image_id": all_test_image_ids,
    "identity": all_cluster_labels,
})

# 确保和sample_submission的image_id对齐
submission_final = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
# 如果有缺失，分配为单独的cluster
max_label = submission_final["identity"].max()
missing_mask = submission_final["identity"].isna()
if missing_mask.sum() > 0:
    print(f"WARNING: {missing_mask.sum()} images missing, assigning unique clusters")
    for i, idx in enumerate(submission_final[missing_mask].index):
        submission_final.loc[idx, "identity"] = max_label + 1 + i

submission_final["identity"] = submission_final["identity"].astype(int)
submission_final.to_csv("/kaggle/working/submission.csv", index=False)

print(f"\n提交文件已保存: /kaggle/working/submission.csv")
print(f"总图片数: {len(submission_final)}")
print(f"总聚类数: {submission_final['identity'].nunique()}")
print(f"\n前5行:")
print(submission_final.head())

print(f"\n{'='*60}")
print("DONE! 下载submission.csv后到AnimalCLEF比赛页面手动提交")
print(f"{'='*60}")
