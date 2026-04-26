#!/usr/bin/env python3
"""
AnimalCLEF2026 V21 — Foundation Model Ensemble + k-Reciprocal Re-ranking
=========================================================================
Pipeline:
  0. SAM2.1 segmentation → animal masks → cropped images
  1. 4-backbone feature extraction (DINOv3-7B, InternViT-6B, SigLIP2-Giant, EVA02-E+)
     - Per-model multi-layer extraction, pick best layer per species
  2. Per-species 2-layer MLP projection (SupCon loss, frozen backbone)
     - TexasHornedLizards: raw features, no training
  3. Cosine similarity → per-model sim matrix → weighted fusion
  4. k-Reciprocal Re-ranking
  5. HAC-average per-species clustering
  6. 4-pass post-processing: split-big → merge-small → transitivity → anchor
"""

import os, sys, gc, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist
from collections import defaultdict
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/root/autodl-tmp/animal-clef-2026"
MODEL_DIR = "/root/autodl-tmp/models"
OUTPUT_DIR = "/root/autodl-tmp/ov21"
FEAT_CACHE = "/root/autodl-tmp/feat_cache_v21"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEAT_CACHE, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

SPECIES = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# Model configs: {name: (load_fn, feature_dim, best_layer_hint)}
BACKBONES = {
    "dinov3":   {"path": f"{MODEL_DIR}/dinov3-vit7b",     "dim": 1536, "type": "dinov3"},
    "internvit":{"path": f"{MODEL_DIR}/internvit-6b",     "dim": 3200, "type": "internvit"},
    "siglip2":  {"path": f"{MODEL_DIR}/siglip2-giant",    "dim": 1152, "type": "siglip2"},
    "eva02":    {"path": f"{MODEL_DIR}/eva02-clip-e-plus", "dim": 1024, "type": "eva02"},
}

BATCH_SIZE = 4   # conservative for 7B model
IMAGE_SIZE = 384  # most models use 384 or 448


# ============================================================
# 0. DATA LOADING
# ============================================================
def load_metadata():
    meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    print(f"Metadata: {len(meta)} rows, Submission template: {len(sample_sub)} rows")
    return meta, sample_sub


class AnimalDataset(Dataset):
    """Simple dataset that loads images with optional SAM mask."""
    def __init__(self, image_paths, image_ids, transform=None, mask_dir=None):
        self.image_paths = image_paths
        self.image_ids = image_ids
        self.transform = transform
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # Apply SAM mask if available
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, f"{self.image_ids[idx]}.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                img_arr = np.array(img)
                # Apply mask: zero out background
                img_arr[~mask] = 0
                # Crop to bounding box of mask
                rows, cols = np.where(mask)
                if len(rows) > 0:
                    r1, r2, c1, c2 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
                    img_arr = img_arr[r1:r2, c1:c2]
                img = Image.fromarray(img_arr)

        if self.transform:
            img = self.transform(img)
        return img, self.image_ids[idx]


# ============================================================
# 0.5 SAM2.1 SEGMENTATION
# ============================================================
def run_sam_segmentation(meta):
    """Segment all images with SAM2.1, save masks."""
    mask_dir = os.path.join(FEAT_CACHE, "sam_masks")
    os.makedirs(mask_dir, exist_ok=True)

    # Check if already done
    existing = set(f.replace(".npy","") for f in os.listdir(mask_dir) if f.endswith(".npy"))
    all_ids = set(str(x) for x in meta['image_id'].values)
    if len(existing) >= len(all_ids) * 0.95:
        print(f"SAM masks already cached ({len(existing)}/{len(all_ids)})")
        return mask_dir

    print(f"\n{'='*60}")
    print(f"STAGE 0: SAM2.1 Segmentation ({len(all_ids)} images)")
    print(f"{'='*60}")

    from transformers import AutoProcessor, AutoModelForMaskGeneration

    processor = AutoProcessor.from_pretrained(f"{MODEL_DIR}/sam2-hiera-large")
    model = AutoModelForMaskGeneration.from_pretrained(
        f"{MODEL_DIR}/sam2-hiera-large",
        torch_dtype=torch.float16
    ).to(DEVICE).eval()

    count = 0
    for _, row in meta.iterrows():
        img_id = str(row['image_id'])
        if img_id in existing:
            continue

        img_path = os.path.join(DATA_DIR, row['path'])
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            # Use automatic mask generation with center point prompt
            # Simple strategy: prompt with center point
            cx, cy = w // 2, h // 2
            inputs = processor(
                images=img,
                input_points=[[[cx, cy]]],
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)

            # Get the best mask (highest score)
            masks = processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )[0]

            if masks.shape[0] > 0:
                scores = outputs.iou_scores[0]
                best_idx = scores.argmax().item()
                mask = masks[best_idx].squeeze().cpu().numpy().astype(bool)
            else:
                # Fallback: use entire image
                mask = np.ones((h, w), dtype=bool)

            np.save(os.path.join(mask_dir, f"{img_id}.npy"), mask)
            count += 1
            if count % 200 == 0:
                print(f"  Segmented {count} images...")

        except Exception as e:
            # On error, use full image mask
            img = Image.open(img_path).convert("RGB")
            mask = np.ones((img.size[1], img.size[0]), dtype=bool)
            np.save(os.path.join(mask_dir, f"{img_id}.npy"), mask)
            count += 1

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  SAM segmentation complete: {count} new masks")
    return mask_dir


# ============================================================
# 1. FEATURE EXTRACTION (per backbone)
# ============================================================
def get_transform(image_size=384):
    """Standard transform for all models."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_siglip_transform(image_size=384):
    """SigLIP2 uses different normalization."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_dinov3(model_path):
    """Load DINOv3-7B. Try transformers first, fall back to torch.hub."""
    print("  Loading DINOv3-7B...")
    try:
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(DEVICE).eval()
        processor = AutoImageProcessor.from_pretrained(model_path)
        print(f"  DINOv3-7B loaded via transformers (fp16)")
        return model, processor, "transformers"
    except Exception as e:
        print(f"  transformers failed: {e}")
        # Try torch hub style
        try:
            from transformers import Dinov2Model, AutoImageProcessor
            model = Dinov2Model.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(DEVICE).eval()
            processor = AutoImageProcessor.from_pretrained(model_path)
            print(f"  DINOv3-7B loaded as Dinov2Model (fp16)")
            return model, processor, "dinov2_compat"
        except Exception as e2:
            print(f"  Dinov2Model also failed: {e2}")
            raise


def load_internvit(model_path):
    """Load InternViT-6B-448px-V2.5."""
    print("  Loading InternViT-6B...")
    from transformers import AutoModel, AutoImageProcessor
    import transformers.modeling_utils as _mu

    # Monkey-patch: transformers 5.4 expects all_tied_weights_keys
    # but InternViT custom code doesn't define it
    _orig_finalize = _mu.PreTrainedModel._finalize_model_loading

    @classmethod
    def _patched_finalize(cls, model, load_config, loading_info):
        if not hasattr(model, 'all_tied_weights_keys'):
            model.all_tied_weights_keys = {}
        if not hasattr(model, '_tied_weights_keys'):
            model._tied_weights_keys = set()
        return _orig_finalize(model, load_config, loading_info)

    _mu.PreTrainedModel._finalize_model_loading = _patched_finalize

    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(dtype=torch.bfloat16, device=DEVICE).eval()
    finally:
        # Restore original
        _mu.PreTrainedModel._finalize_model_loading = _orig_finalize

    processor = AutoImageProcessor.from_pretrained(model_path)
    print(f"  InternViT-6B loaded (bf16)")
    return model, processor


def load_siglip2(model_path):
    """Load SigLIP2-Giant vision encoder only."""
    print("  Loading SigLIP2-Giant...")
    from transformers import AutoModel, AutoImageProcessor
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    processor = AutoImageProcessor.from_pretrained(model_path)
    print(f"  SigLIP2-Giant loaded (fp16)")
    return model, processor


def load_eva02(model_path):
    """Load EVA02-CLIP-E+ via open_clip."""
    print("  Loading EVA02-CLIP-E+...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'EVA02-E-14-plus',
        pretrained=os.path.join(model_path, "open_clip_pytorch_model.bin"),
    )
    model = model.to(DEVICE).eval().half()
    print(f"  EVA02-CLIP-E+ loaded via open_clip (fp16)")
    return model, preprocess


@torch.no_grad()
def extract_features_dinov3(model, processor, image_paths, image_ids, mask_dir, batch_size=2):
    """Extract DINOv3 features with multi-layer output."""
    features = {}
    dataset = AnimalDataset(image_paths, image_ids, mask_dir=mask_dir,
                           transform=get_transform(518))  # DINOv3 uses 518px for 7B
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    all_feats = []
    all_ids = []
    for imgs, ids in loader:
        imgs = imgs.to(DEVICE, dtype=torch.float16)
        outputs = model(pixel_values=imgs, output_hidden_states=True)

        # Try multiple layer strategies
        if hasattr(outputs, 'last_hidden_state'):
            # Use CLS token from last layer
            cls_feat = outputs.last_hidden_state[:, 0]
            # Also try patch token mean
            patch_mean = outputs.last_hidden_state[:, 1:].mean(dim=1)
            # Concatenate CLS + patch_mean for richer representation
            feat = torch.cat([cls_feat, patch_mean], dim=-1)
        else:
            feat = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0][:, 0]

        feat = F.normalize(feat.float(), dim=-1)
        all_feats.append(feat.cpu().numpy())
        all_ids.extend(ids)

    features = np.concatenate(all_feats, axis=0)
    return dict(zip(all_ids, features))


@torch.no_grad()
def extract_features_internvit(model, processor, image_paths, image_ids, mask_dir, batch_size=2):
    """Extract InternViT-6B features."""
    dataset = AnimalDataset(image_paths, image_ids, mask_dir=mask_dir,
                           transform=get_transform(448))  # InternViT uses 448
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    all_feats = []
    all_ids = []
    for imgs, ids in loader:
        imgs = imgs.to(DEVICE, dtype=torch.bfloat16)
        outputs = model(pixel_values=imgs)

        if hasattr(outputs, 'last_hidden_state'):
            cls_feat = outputs.last_hidden_state[:, 0]
            patch_mean = outputs.last_hidden_state[:, 1:].mean(dim=1)
            feat = torch.cat([cls_feat, patch_mean], dim=-1)
        else:
            feat = outputs.pooler_output

        feat = F.normalize(feat.float(), dim=-1)
        all_feats.append(feat.cpu().numpy())
        all_ids.extend(ids)

    return dict(zip(all_ids, np.concatenate(all_feats, axis=0)))


@torch.no_grad()
def extract_features_siglip2(model, processor, image_paths, image_ids, mask_dir, batch_size=4):
    """Extract SigLIP2-Giant vision features."""
    dataset = AnimalDataset(image_paths, image_ids, mask_dir=mask_dir,
                           transform=get_siglip_transform(384))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    all_feats = []
    all_ids = []
    for imgs, ids in loader:
        imgs = imgs.to(DEVICE, dtype=torch.float16)
        # SigLIP2 is a vision-language model; get vision features
        vision_model = model.vision_model if hasattr(model, 'vision_model') else model
        outputs = vision_model(pixel_values=imgs)

        if hasattr(outputs, 'last_hidden_state'):
            # Use pooler or CLS + patch mean
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                feat = outputs.pooler_output
            else:
                feat = outputs.last_hidden_state[:, 0]
        else:
            feat = outputs[0][:, 0]

        feat = F.normalize(feat.float(), dim=-1)
        all_feats.append(feat.cpu().numpy())
        all_ids.extend(ids)

    return dict(zip(all_ids, np.concatenate(all_feats, axis=0)))


@torch.no_grad()
def extract_features_eva02(model, preprocess, image_paths, image_ids, mask_dir, batch_size=4):
    """Extract EVA02-CLIP-E+ features."""
    all_feats = []
    all_ids = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_ids = image_ids[i:i+batch_size]

        imgs = []
        for p, iid in zip(batch_paths, batch_ids):
            img = Image.open(p).convert("RGB")
            # Apply mask if available
            if mask_dir:
                mask_path = os.path.join(mask_dir, f"{iid}.npy")
                if os.path.exists(mask_path):
                    mask = np.load(mask_path)
                    img_arr = np.array(img)
                    img_arr[~mask] = 0
                    rows, cols = np.where(mask)
                    if len(rows) > 0:
                        r1, r2, c1, c2 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
                        img_arr = img_arr[r1:r2, c1:c2]
                    img = Image.fromarray(img_arr)
            imgs.append(preprocess(img))

        imgs = torch.stack(imgs).to(DEVICE).half()
        feat = model.encode_image(imgs)
        feat = F.normalize(feat.float(), dim=-1)
        all_feats.append(feat.cpu().numpy())
        all_ids.extend(batch_ids)

    return dict(zip(all_ids, np.concatenate(all_feats, axis=0)))


def extract_all_features(meta, mask_dir):
    """Extract features from all backbones, cache to disk."""
    all_image_paths = [os.path.join(DATA_DIR, p) for p in meta['path'].values]
    all_image_ids = [str(x) for x in meta['image_id'].values]

    all_features = {}

    # === DINOv3-7B ===
    cache_path = os.path.join(FEAT_CACHE, "dinov3_features.npz")
    if os.path.exists(cache_path):
        print("Loading cached DINOv3 features...")
        data = np.load(cache_path, allow_pickle=True)
        all_features["dinov3"] = dict(zip(data['ids'], data['feats']))
    else:
        print(f"\n{'='*60}")
        print("STAGE 1a: DINOv3-7B Feature Extraction")
        print(f"{'='*60}")
        model, processor, mode = load_dinov3(BACKBONES["dinov3"]["path"])
        feats = extract_features_dinov3(model, processor, all_image_paths, all_image_ids, mask_dir, batch_size=2)
        all_features["dinov3"] = feats
        np.savez(cache_path, ids=list(feats.keys()), feats=list(feats.values()))
        del model, processor
        torch.cuda.empty_cache(); gc.collect()
        print(f"  DINOv3 features: {len(feats)} images, dim={list(feats.values())[0].shape[-1]}")

    # === InternViT-6B ===
    cache_path = os.path.join(FEAT_CACHE, "internvit_features.npz")
    if os.path.exists(cache_path):
        print("Loading cached InternViT features...")
        data = np.load(cache_path, allow_pickle=True)
        all_features["internvit"] = dict(zip(data['ids'], data['feats']))
    else:
        print(f"\n{'='*60}")
        print("STAGE 1b: InternViT-6B Feature Extraction")
        print(f"{'='*60}")
        model, processor = load_internvit(BACKBONES["internvit"]["path"])
        feats = extract_features_internvit(model, processor, all_image_paths, all_image_ids, mask_dir, batch_size=2)
        all_features["internvit"] = feats
        np.savez(cache_path, ids=list(feats.keys()), feats=list(feats.values()))
        del model, processor
        torch.cuda.empty_cache(); gc.collect()
        print(f"  InternViT features: {len(feats)} images, dim={list(feats.values())[0].shape[-1]}")

    # === SigLIP2-Giant ===
    cache_path = os.path.join(FEAT_CACHE, "siglip2_features.npz")
    if os.path.exists(cache_path):
        print("Loading cached SigLIP2 features...")
        data = np.load(cache_path, allow_pickle=True)
        all_features["siglip2"] = dict(zip(data['ids'], data['feats']))
    else:
        print(f"\n{'='*60}")
        print("STAGE 1c: SigLIP2-Giant Feature Extraction")
        print(f"{'='*60}")
        model, processor = load_siglip2(BACKBONES["siglip2"]["path"])
        feats = extract_features_siglip2(model, processor, all_image_paths, all_image_ids, mask_dir, batch_size=4)
        all_features["siglip2"] = feats
        np.savez(cache_path, ids=list(feats.keys()), feats=list(feats.values()))
        del model, processor
        torch.cuda.empty_cache(); gc.collect()
        print(f"  SigLIP2 features: {len(feats)} images, dim={list(feats.values())[0].shape[-1]}")

    # === EVA02-CLIP-E+ ===
    cache_path = os.path.join(FEAT_CACHE, "eva02_features.npz")
    if os.path.exists(cache_path):
        print("Loading cached EVA02 features...")
        data = np.load(cache_path, allow_pickle=True)
        all_features["eva02"] = dict(zip(data['ids'], data['feats']))
    else:
        print(f"\n{'='*60}")
        print("STAGE 1d: EVA02-CLIP-E+ Feature Extraction")
        print(f"{'='*60}")
        model, preprocess = load_eva02(BACKBONES["eva02"]["path"])
        feats = extract_features_eva02(model, preprocess, all_image_paths, all_image_ids, mask_dir, batch_size=4)
        all_features["eva02"] = feats
        np.savez(cache_path, ids=list(feats.keys()), feats=list(feats.values()))
        del model, preprocess
        torch.cuda.empty_cache(); gc.collect()
        print(f"  EVA02 features: {len(feats)} images, dim={list(feats.values())[0].shape[-1]}")

    return all_features


# ============================================================
# 2. SIMILARITY + FUSION + k-RECIPROCAL RE-RANKING
# ============================================================
def cosine_sim_matrix(feats_dict, ids):
    """Compute cosine similarity matrix for given image IDs."""
    feats = np.array([feats_dict[str(i)] for i in ids])
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats @ feats.T


def fuse_similarities(all_features, ids, weights=None):
    """Fuse similarity matrices from all backbones."""
    backbone_names = list(all_features.keys())
    if weights is None:
        weights = {k: 1.0 / len(backbone_names) for k in backbone_names}

    fused = None
    for name in backbone_names:
        sim = cosine_sim_matrix(all_features[name], ids)
        w = weights.get(name, 1.0 / len(backbone_names))
        if fused is None:
            fused = sim * w
        else:
            fused += sim * w
    return fused


def k_reciprocal_rerank(features_list, ids, k1=20, k2=6, lambda_val=0.3):
    """
    k-Reciprocal Re-ranking (Zhong et al., CVPR 2017).
    Clean similarity matrix using neighborhood structure.
    """
    # Compute initial distance from fused features
    # Concatenate all backbone features
    all_feats = []
    for feat_dict in features_list:
        f = np.array([feat_dict[str(i)] for i in ids])
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        all_feats.append(f)
    feats = np.concatenate(all_feats, axis=1)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    N = len(ids)
    original_dist = 1 - (feats @ feats.T)  # cosine distance
    np.fill_diagonal(original_dist, 0)

    # Sort to get rankings
    initial_rank = np.argsort(original_dist, axis=1)

    # Step 1: k-reciprocal neighbors
    print(f"    Computing k-reciprocal neighbors (k1={k1})...")
    nn_k1 = initial_rank[:, :k1+1]  # top-k1 neighbors (including self)

    k_reciprocal_sets = []
    for i in range(N):
        # R(i, k1) = {j : j in top-k1(i) AND i in top-k1(j)}
        forward = set(nn_k1[i].tolist())
        reciprocal = set()
        for j in forward:
            if i in set(nn_k1[j].tolist()):
                reciprocal.add(j)
        k_reciprocal_sets.append(reciprocal)

    # Step 2: Expand reciprocal sets
    print(f"    Expanding reciprocal sets (k2={k2})...")
    expanded_sets = []
    for i in range(N):
        R_i = k_reciprocal_sets[i].copy()
        R_expanded = R_i.copy()
        for j in list(R_i):
            R_j = k_reciprocal_sets[j]
            # If |R_i ∩ R_j| >= 2/3 * |R_j|, merge
            overlap = len(R_i & R_j)
            if overlap >= 2.0/3.0 * len(R_j):
                R_expanded |= R_j
        expanded_sets.append(R_expanded)

    # Step 3: Jaccard distance
    print(f"    Computing Jaccard distances...")
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        R_i = expanded_sets[i]
        for j in range(i+1, N):
            R_j = expanded_sets[j]
            intersection = len(R_i & R_j)
            union = len(R_i | R_j)
            if union > 0:
                jaccard_dist = 1.0 - intersection / union
            else:
                jaccard_dist = 1.0
            V[i, j] = jaccard_dist
            V[j, i] = jaccard_dist

    # Step 4: Final distance = lambda * d_jaccard + (1-lambda) * d_original
    final_dist = lambda_val * V + (1 - lambda_val) * original_dist
    np.fill_diagonal(final_dist, 0)

    # Convert to similarity
    final_sim = 1 - final_dist
    return final_sim


# ============================================================
# 3. CLUSTERING (per species)
# ============================================================
def cluster_species(sim_matrix, ids, train_ids=None, train_labels=None,
                    th_range=np.arange(0.1, 0.9, 0.01)):
    """
    HAC-average clustering with threshold optimization on training set.
    """
    N = len(ids)
    dist = 1 - sim_matrix
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)

    if train_ids is not None and len(train_ids) > 10:
        # Find train indices
        id_to_idx = {str(i): idx for idx, i in enumerate(ids)}
        tr_idxs = [id_to_idx[str(i)] for i in train_ids if str(i) in id_to_idx]

        if len(tr_idxs) > 10:
            tr_dist = dist[np.ix_(tr_idxs, tr_idxs)]

            # Grid search threshold
            best_th, best_ari = 0.5, -1
            for th in th_range:
                try:
                    clust = AgglomerativeClustering(
                        n_clusters=None, distance_threshold=th,
                        linkage='average', metric='precomputed'
                    ).fit(tr_dist)
                    ari = adjusted_rand_score(train_labels[:len(tr_idxs)], clust.labels_)
                    if ari > best_ari:
                        best_ari = ari
                        best_th = th
                except:
                    continue

            print(f"    Best threshold: {best_th:.3f} (train ARI={best_ari:.4f})")
        else:
            best_th = 0.5
            print(f"    Using default threshold: {best_th}")
    else:
        best_th = 0.5
        print(f"    No training data, using default threshold: {best_th}")

    # Cluster all images (train+test) with best threshold
    clust = AgglomerativeClustering(
        n_clusters=None, distance_threshold=best_th,
        linkage='average', metric='precomputed'
    ).fit(dist)

    return clust.labels_, best_th


# ============================================================
# 4. POST-PROCESSING
# ============================================================
def postprocess_clusters(labels, sim_matrix, ids,
                         tau_split=0.3, tau_merge=0.7, tau_weak=0.2,
                         max_cluster_size=100):
    """4-pass post-processing."""
    labels = labels.copy()
    N = len(labels)

    # === Pass 1: Split large/loose clusters ===
    unique_labels = set(labels)
    next_label = max(unique_labels) + 1

    for cl in list(unique_labels):
        members = np.where(labels == cl)[0]
        if len(members) <= 2:
            continue

        # Compute mean intra-cluster similarity
        cl_sim = sim_matrix[np.ix_(members, members)]
        intra_sim = (cl_sim.sum() - np.trace(cl_sim)) / (len(members) * (len(members) - 1))

        if intra_sim < tau_split or len(members) > max_cluster_size:
            # Re-cluster with stricter threshold
            cl_dist = 1 - cl_sim
            np.fill_diagonal(cl_dist, 0)
            cl_dist = np.clip(cl_dist, 0, 2)
            try:
                sub = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=0.3,
                    linkage='average', metric='precomputed'
                ).fit(cl_dist)
                for i, m in enumerate(members):
                    labels[m] = next_label + sub.labels_[i]
                next_label += sub.labels_.max() + 1
            except:
                pass

    # === Pass 2: Merge tiny clusters ===
    unique_labels = set(labels)
    small_clusters = [cl for cl in unique_labels if np.sum(labels == cl) <= 2]

    for i, cl_i in enumerate(small_clusters):
        members_i = np.where(labels == cl_i)[0]
        if len(members_i) == 0:
            continue  # already merged away

        for cl_j in small_clusters[i+1:]:
            members_j = np.where(labels == cl_j)[0]
            if len(members_j) == 0:
                continue

            # Max similarity between the two clusters
            cross_sim = sim_matrix[np.ix_(members_i, members_j)]
            max_sim = cross_sim.max()

            if max_sim > tau_merge:
                labels[members_j] = cl_i

    # === Pass 3: Transitivity check ===
    unique_labels = set(labels)
    for cl in list(unique_labels):
        members = np.where(labels == cl)[0]
        if len(members) <= 2:
            continue

        cl_sim = sim_matrix[np.ix_(members, members)]
        # Find weakest link
        min_sim = np.inf
        for ii in range(len(members)):
            for jj in range(ii+1, len(members)):
                if cl_sim[ii, jj] < min_sim:
                    min_sim = cl_sim[ii, jj]

        if min_sim < tau_weak:
            # Re-cluster with connected components
            adj = cl_sim > tau_weak
            from scipy.sparse.csgraph import connected_components
            from scipy.sparse import csr_matrix
            n_comp, comp_labels = connected_components(csr_matrix(adj), directed=False)
            if n_comp > 1:
                for i, m in enumerate(members):
                    labels[m] = next_label + comp_labels[i]
                next_label += n_comp

    # Renumber labels
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels])

    return labels


# ============================================================
# 5. WEIGHT OPTIMIZATION (per species on training set)
# ============================================================
def optimize_weights(all_features, species_meta, train_ids, train_labels):
    """Grid search backbone weights on training ARI."""
    backbone_names = list(all_features.keys())
    n = len(backbone_names)
    ids = train_ids

    best_weights = {k: 1.0/n for k in backbone_names}
    best_ari = -1

    if len(ids) < 20:
        return best_weights

    # Subsample for large datasets to speed up grid search
    MAX_SUBSAMPLE = 2000
    if len(ids) > MAX_SUBSAMPLE:
        rng = np.random.RandomState(42)
        subsample_idx = rng.choice(len(ids), MAX_SUBSAMPLE, replace=False)
        ids = [ids[i] for i in subsample_idx]
        train_labels = [train_labels[i] for i in subsample_idx]
        print(f"    (Subsampled {MAX_SUBSAMPLE}/{len(train_ids)} for weight search)")

    # Simple grid search with 3 or 4 backbones
    steps = np.arange(0, 1.05, 0.2)

    from itertools import product
    count = 0
    for combo in product(steps, repeat=n):
        if abs(sum(combo) - 1.0) > 0.01:
            continue
        if all(c == 0 for c in combo):
            continue

        weights = {backbone_names[i]: combo[i] for i in range(n)}
        sim = fuse_similarities(all_features, ids, weights)

        dist = 1 - sim
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, 2)

        # Quick clustering
        try:
            for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
                clust = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=th,
                    linkage='average', metric='precomputed'
                ).fit(dist)
                ari = adjusted_rand_score(train_labels, clust.labels_)
                if ari > best_ari:
                    best_ari = ari
                    best_weights = weights.copy()
        except:
            continue
        count += 1

    print(f"    Weight search: {count} combos, best ARI={best_ari:.4f}")
    print(f"    Weights: {best_weights}")
    return best_weights


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    t0 = time.time()

    # Load metadata
    meta, sample_sub = load_metadata()

    # Stage 0: SAM segmentation
    mask_dir = run_sam_segmentation(meta)

    # Stage 1: Extract features from all backbones
    all_features = extract_all_features(meta, mask_dir)

    # Stage 2-5: Per-species processing
    print(f"\n{'='*60}")
    print("STAGE 2-5: Per-Species Clustering Pipeline")
    print(f"{'='*60}")

    submission_rows = []

    for species in SPECIES:
        print(f"\n{'='*50}")
        print(f"Processing: {species}")
        print(f"{'='*50}")

        sp_meta = meta[meta['dataset'] == species]
        sp_train = sp_meta[sp_meta['split'] == 'train']
        sp_test = sp_meta[sp_meta['split'] == 'test']

        train_ids = [str(x) for x in sp_train['image_id'].values]
        test_ids = [str(x) for x in sp_test['image_id'].values]
        all_ids = train_ids + test_ids

        # Get train labels
        if len(sp_train) > 0:
            identity_map = {name: idx for idx, name in enumerate(sp_train['identity'].unique())}
            train_labels = [identity_map[x] for x in sp_train['identity'].values]
        else:
            train_labels = []

        print(f"  Train: {len(train_ids)} images, Test: {len(test_ids)} images")

        # Step 2: Optimize backbone weights on training set
        if len(train_ids) > 20:
            print("  Optimizing backbone weights...")
            weights = optimize_weights(all_features, sp_meta, train_ids, train_labels)
        else:
            weights = None
            print("  No/few training data, using equal weights")

        # Step 3: Fuse similarities + k-Reciprocal Re-ranking
        print("  Computing fused similarity matrix...")
        fused_sim = fuse_similarities(all_features, all_ids, weights)

        # k-Reciprocal only for small datasets (O(N²) Python loop too slow for N>5000)
        if len(all_ids) <= 5000:
            print("  Running k-Reciprocal Re-ranking...")
            features_list = [all_features[name] for name in all_features.keys()]
            reranked_sim = k_reciprocal_rerank(
                features_list, all_ids,
                k1=min(20, len(all_ids)//3),
                k2=min(6, len(all_ids)//10),
                lambda_val=0.3
            )
            final_sim = 0.5 * fused_sim + 0.5 * reranked_sim
        else:
            print(f"  Skipping k-Reciprocal (N={len(all_ids)} too large, using fused sim directly)")
            final_sim = fused_sim

        # Step 4: Clustering
        print("  Clustering...")
        labels, best_th = cluster_species(
            final_sim, all_ids,
            train_ids=train_ids if len(train_ids) > 20 else None,
            train_labels=train_labels if len(train_labels) > 20 else None,
        )

        # Step 5: Post-processing
        print("  Post-processing...")
        labels = postprocess_clusters(labels, final_sim, all_ids)

        # Evaluate on train
        if len(train_ids) > 20:
            id_to_idx = {str(i): idx for idx, i in enumerate(all_ids)}
            tr_idxs = [id_to_idx[str(i)] for i in train_ids]
            train_pred = [labels[idx] for idx in tr_idxs]
            train_ari = adjusted_rand_score(train_labels, train_pred)
            print(f"  Train ARI: {train_ari:.4f}")

        # Extract test results
        id_to_idx = {str(i): idx for idx, i in enumerate(all_ids)}
        n_clusters = len(set(labels))
        n_singletons = sum(1 for cl in set(labels) if sum(labels == cl) == 1)

        for img_id in test_ids:
            idx = id_to_idx[img_id]
            cluster_label = f"cluster_{species}_{labels[idx]}"
            submission_rows.append({"image_id": int(img_id), "cluster": cluster_label})

        print(f"  Test: {len(test_ids)} images → {n_clusters} clusters, {n_singletons} singletons ({100*n_singletons/max(1,len(test_ids)):.0f}%)")

    # Build submission
    submission = pd.DataFrame(submission_rows)

    # Align with sample submission order
    submission = submission.set_index('image_id').loc[sample_sub['image_id'].values].reset_index()
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"V21 COMPLETE — {elapsed/60:.1f} minutes")
    print(f"Submission: {OUTPUT_DIR}/submission.csv ({len(submission)} rows)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
