#!/usr/bin/env python3
"""
AnimalCLEF2026 V14 — Clean Clustering Solution
================================================
核心原则：
  1. 纯聚类，不做lookup/分类
  2. 训练集只用来调聚类阈值（grid search ARI）
  3. 多模型特征融合（MegaDescriptor + MiewID）
  4. TTA（原图 + 水平翻转）
  5. 每个物种独立处理

Pipeline:
  STAGE 1: 提取特征（两个模型 × TTA）
  STAGE 2: 对每个有训练集的物种，grid search最优聚类阈值
  STAGE 3: 用最优阈值聚类测试集
  STAGE 4: 生成submission.csv

预计运行时间: ~10分钟（5090 32GB）
"""

import os, sys, gc, time, json, warnings
warnings.filterwarnings("ignore")

# HuggingFace mirror for China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ═══════════════════════════════════════════════════════════
# CONFIG — keep it simple
# ═══════════════════════════════════════════════════════════
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/animal-clef-2026"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/ov14"
os.makedirs(OUT_DIR, exist_ok=True)

SPECIES_WITH_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
ALL_SPECIES = SPECIES_WITH_TRAIN + ["TexasHornedLizards"]

# Model weights for similarity fusion (tuned later if needed)
# MegaDescriptor: specialized for animal re-id
# MiewID: strong on different species
FUSION_WEIGHTS = {"mega": 0.5, "miew": 0.5}

# ═══════════════════════════════════════════════════════════
# DATASET — simple, correct
# ═══════════════════════════════════════════════════════════
class AnimalDataset(Dataset):
    """Load images with optional horizontal flip for TTA."""
    def __init__(self, df, root_dir, img_size, flip=False):
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.flip = flip
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root, row["path"])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384), (128, 128, 128))

        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return self.transform(img), int(row["image_id"])


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def extract_features(model, df, root_dir, img_size, batch_size=48, use_tta=True):
    """Extract L2-normalized features with optional TTA (orig + hflip average)."""

    # Original features
    ds = AnimalDataset(df, root_dir, img_size, flip=False)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    all_embs, all_ids = [], []
    for imgs, ids in tqdm(dl, desc=f"  feat@{img_size}", leave=False):
        emb = model(imgs.to(DEVICE))
        all_embs.append(emb.cpu())
        all_ids.extend(ids.numpy())
    orig_embs = torch.cat(all_embs)

    if use_tta:
        # Flipped features
        ds_flip = AnimalDataset(df, root_dir, img_size, flip=True)
        dl_flip = DataLoader(ds_flip, batch_size=batch_size, num_workers=4, pin_memory=True)
        flip_embs = []
        for imgs, _ in tqdm(dl_flip, desc=f"  tta@{img_size}", leave=False):
            emb = model(imgs.to(DEVICE))
            flip_embs.append(emb.cpu())
        flip_embs = torch.cat(flip_embs)

        # Average and re-normalize
        combined = (orig_embs + flip_embs) / 2.0
        features = F.normalize(combined, dim=-1).numpy()
    else:
        features = F.normalize(orig_embs, dim=-1).numpy()

    return features, np.array(all_ids)


def load_megadescriptor():
    """Load MegaDescriptor-L-384."""
    import timm
    print("[MODEL] Loading MegaDescriptor-L-384...", end=" ", flush=True)
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()
    print(f"OK (dim={model.num_features})")
    return model, 384


def load_miewid():
    """Load MiewID-MSV3."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    print("[MODEL] Loading MiewID-MSV3...", end=" ", flush=True)

    cfg_path = hf_hub_download("conservationxlabs/miewid-msv3", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    arch = cfg.get("architecture", "efficientnetv2_rw_m")

    model = timm.create_model(arch, pretrained=False, num_classes=0)
    wt_path = hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")
    state = {k: v for k, v in load_file(wt_path).items() if "classifier" not in k}
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()
    print(f"OK (dim={model.num_features})")
    return model, 440


# ═══════════════════════════════════════════════════════════
# SIMILARITY & CLUSTERING — the heart of the solution
# ═══════════════════════════════════════════════════════════
def compute_fused_similarity(feat_dict, indices):
    """Compute fused pairwise cosine similarity for given indices.

    feat_dict: {"mega": all_features_array, "miew": all_features_array}
    indices: array of row indices to select
    """
    sim_sum = None
    weight_sum = 0.0
    for model_name, all_feats in feat_dict.items():
        feats = normalize(all_feats[indices], axis=1)
        sim = feats @ feats.T
        w = FUSION_WEIGHTS.get(model_name, 0.5)
        if sim_sum is None:
            sim_sum = sim * w
        else:
            sim_sum += sim * w
        weight_sum += w
    return sim_sum / weight_sum


def find_best_threshold(sim_matrix, true_labels, method="agglomerative"):
    """Grid search for the best clustering threshold using ARI on training data.

    This is THE key step: we try many thresholds, cluster the training data,
    and pick the threshold that gives the best ARI vs ground truth.
    """
    dist = np.clip(1.0 - sim_matrix, 0, 2)
    np.fill_diagonal(dist, 0)

    best_ari = -1
    best_threshold = 0.5
    best_n_clusters = -1

    # Wide search first
    for threshold in np.arange(0.05, 1.50, 0.01):
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="precomputed",
                linkage="average"
            )
            pred_labels = clustering.fit_predict(dist)
            ari = adjusted_rand_score(true_labels, pred_labels)
            n_cl = len(set(pred_labels))

            if ari > best_ari:
                best_ari = ari
                best_threshold = threshold
                best_n_clusters = n_cl
        except Exception:
            continue

    return best_threshold, best_ari, best_n_clusters


def cluster_test_images(sim_matrix, threshold):
    """Apply Agglomerative Clustering with the given threshold."""
    dist = np.clip(1.0 - sim_matrix, 0, 2)
    np.fill_diagonal(dist, 0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average"
    )
    return clustering.fit_predict(dist)


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 60)
    print("  AnimalCLEF2026 V14 — Clean Clustering Solution")
    print("=" * 60)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f}GB")
    else:
        print("  WARNING: No GPU detected, running on CPU (will be slow)")

    # ── Load metadata ──
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    train_df = meta[meta.split == "train"].copy()
    test_df = meta[meta.split == "test"].copy()

    print(f"\n  Data: {len(train_df)} train, {len(test_df)} test")
    for sp in ALL_SPECIES:
        n_tr = len(train_df[train_df.dataset == sp])
        n_te = len(test_df[test_df.dataset == sp])
        n_id = train_df[train_df.dataset == sp].identity.nunique() if n_tr > 0 else 0
        print(f"    {sp:25s} train={n_tr:5d}  ids={n_id:4d}  test={n_te:4d}")

    # ══════════════════════════════════════════════════════
    # STAGE 1: Extract features with both models
    # ══════════════════════════════════════════════════════
    print(f"\n{'━' * 55}")
    print("STAGE 1: Feature Extraction (TTA: orig + hflip)")
    print(f"{'━' * 55}")

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    n_train = len(train_df)

    feat_dict = {}  # {"mega": features_array, "miew": features_array}

    # MegaDescriptor
    mega_model, mega_size = load_megadescriptor()
    mega_feats, mega_ids = extract_features(mega_model, all_df, DATA_DIR, mega_size, batch_size=48)
    feat_dict["mega"] = mega_feats
    print(f"  MegaDescriptor: {mega_feats.shape}")
    del mega_model; torch.cuda.empty_cache(); gc.collect()

    # MiewID
    miew_model, miew_size = load_miewid()
    miew_feats, miew_ids = extract_features(miew_model, all_df, DATA_DIR, miew_size, batch_size=32)
    feat_dict["miew"] = miew_feats
    print(f"  MiewID: {miew_feats.shape}")
    del miew_model; torch.cuda.empty_cache(); gc.collect()

    # Build image_id → index mapping
    all_image_ids = all_df.image_id.values
    id_to_idx = {int(all_image_ids[i]): i for i in range(len(all_image_ids))}

    feat_time = (time.time() - t0) / 60
    print(f"  Feature extraction done in {feat_time:.1f}min")

    # ══════════════════════════════════════════════════════
    # STAGE 2 & 3: Per-species threshold tuning + clustering
    # ══════════════════════════════════════════════════════
    print(f"\n{'━' * 55}")
    print("STAGE 2: Per-species Threshold Tuning + Clustering")
    print(f"{'━' * 55}")

    all_predictions = {}  # image_id → cluster_label_string

    # Keep track of tuned thresholds (for TexasHornedLizards fallback)
    tuned_thresholds = {}

    for species in ALL_SPECIES:
        print(f"\n  {'─' * 50}")
        print(f"  {species}")
        print(f"  {'─' * 50}")

        sp_test = test_df[test_df.dataset == species].reset_index(drop=True)
        test_indices = np.array([id_to_idx[int(iid)] for iid in sp_test.image_id.values])
        n_test = len(sp_test)

        if species in SPECIES_WITH_TRAIN:
            sp_train = train_df[train_df.dataset == species].reset_index(drop=True)
            train_indices = np.array([id_to_idx[int(iid)] for iid in sp_train.image_id.values])
            train_labels = sp_train.identity.values
            n_train_sp = len(sp_train)
            n_ids = len(set(train_labels))

            print(f"    Train: {n_train_sp} images, {n_ids} individuals")
            print(f"    Test:  {n_test} images")

            # ── Tune threshold on training data ──
            train_sim = compute_fused_similarity(feat_dict, train_indices)
            best_th, best_ari, best_ncl = find_best_threshold(train_sim, train_labels)
            tuned_thresholds[species] = best_th

            print(f"    Tuned threshold: {best_th:.3f}")
            print(f"    Train ARI: {best_ari:.4f} ({best_ncl} clusters for {n_ids} true ids)")

            # ── Cluster test data with tuned threshold ──
            test_sim = compute_fused_similarity(feat_dict, test_indices)
            test_labels = cluster_test_images(test_sim, best_th)
            n_test_cl = len(set(test_labels))
            print(f"    Test clusters: {n_test_cl}")

        else:
            # TexasHornedLizards — no training data
            print(f"    Test:  {n_test} images (NO training data)")

            test_sim = compute_fused_similarity(feat_dict, test_indices)

            # Use median of tuned thresholds from other species as starting point
            if tuned_thresholds:
                fallback_th = np.median(list(tuned_thresholds.values()))
            else:
                fallback_th = 0.50

            # Also do a heuristic search: try to find a threshold where
            # n_clusters is reasonable (expect ~30-120 for 274 images)
            best_th, best_score = fallback_th, -1
            dist = np.clip(1.0 - test_sim, 0, 2)
            np.fill_diagonal(dist, 0)

            for th in np.arange(0.10, 1.50, 0.01):
                try:
                    pred = AgglomerativeClustering(
                        n_clusters=None, distance_threshold=th,
                        metric="precomputed", linkage="average"
                    ).fit_predict(dist)
                    n_cl = len(set(pred))
                    # Heuristic: expect roughly 1 individual per 3-5 images
                    ratio = n_cl / n_test
                    if 0.15 <= ratio <= 0.60:
                        # Prefer ratio around 0.35 (similar to other species)
                        score = -abs(ratio - 0.35)
                        if score > best_score:
                            best_score = score
                            best_th = th
                except Exception:
                    continue

            test_labels = cluster_test_images(test_sim, best_th)
            n_test_cl = len(set(test_labels))
            print(f"    Using threshold: {best_th:.3f} (fallback median: {fallback_th:.3f})")
            print(f"    Test clusters: {n_test_cl}")

        # ── Store predictions ──
        for i in range(n_test):
            img_id = int(sp_test.iloc[i].image_id)
            all_predictions[img_id] = f"cluster_{species}_{test_labels[i]}"

    # ══════════════════════════════════════════════════════
    # STAGE 4: Generate submission
    # ══════════════════════════════════════════════════════
    print(f"\n{'━' * 55}")
    print("STAGE 4: Generate Submission")
    print(f"{'━' * 55}")

    sub = sample_sub.copy()
    missing = 0
    for i in range(len(sub)):
        img_id = int(sub.iloc[i].image_id)
        if img_id in all_predictions:
            sub.at[i, "cluster"] = all_predictions[img_id]
        else:
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing} images not assigned!")

    out_path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)

    total_time = (time.time() - t0) / 60
    print(f"\n  Output: {out_path}")
    print(f"  Rows: {len(sub)}  Total clusters: {sub.cluster.nunique()}")
    for sp in ALL_SPECIES:
        sp_sub = sub[sub.cluster.str.contains(sp)]
        print(f"    {sp:25s}  imgs={len(sp_sub):4d}  clusters={sp_sub.cluster.nunique()}")
    print(f"\n  Total time: {total_time:.1f}min")
    print("=" * 60)
    print("DONE!")


if __name__ == "__main__":
    main()
