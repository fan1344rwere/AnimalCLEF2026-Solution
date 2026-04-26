#!/usr/bin/env python3
"""
AnimalCLEF2026 V30b — WildlifeReID-10k Pre-trained SupCon
============================================================
PRAGMATIC VERSION:
- Wildlife features: SigLIP2 only (can load, 1536-dim)
- Competition features: use cached DINOv3+SigLIP2+MegaDesc from V23
- SupCon: train on wildlife SigLIP2 features (3497 ids)
- Transfer: project competition SigLIP2 features, ensemble with MegaDesc
"""
import os, sys, gc, time, csv, warnings, json
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
from collections import defaultdict, Counter
from safetensors.torch import load_file
import timm

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory/1024**3:.1f}GB")

# ============================================================
# CONFIG
# ============================================================
COMP_DIR = "/root/autodl-tmp/animal-clef-2026"
WILDLIFE_META = "/root/autodl-tmp/wildlife10k_good.csv"
WILDLIFE_BASE = "/root/autodl-tmp/wildlifereid10k_full"
COMP_FEAT_CACHE = "/root/autodl-tmp/feat_cache_v23"
WILDLIFE_FEAT_CACHE = "/root/autodl-tmp/feat_cache_wildlife"
OUTPUT_DIR = "/root/autodl-tmp/ov30"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WILDLIFE_FEAT_CACHE, exist_ok=True)

SUPCON_EPOCHS = 50
SUPCON_LR = 3e-4
SUPCON_WD = 1e-3
SUPCON_TEMP = 0.07
PK_P = 64
PK_K = 4

SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# ============================================================
# DATASET & FEATURE EXTRACTION
# ============================================================
class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new("RGB", (384, 384), (128, 128, 128))
        return self.transform(img), idx

@torch.no_grad()
def extract_features(model, dataloader, desc=""):
    model.eval()
    all_feats = []
    t0 = time.time()
    for bi, (imgs, _) in enumerate(dataloader):
        out = model(imgs.to(device, dtype=torch.float16))
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("last_hidden_state", out.get("pooler_output", list(out.values())[0]))
        if out.dim() == 3:
            out = out[:, 0]
        feats = F.normalize(out.float(), dim=-1)
        all_feats.append(feats.cpu().numpy())
        if (bi + 1) % 50 == 0:
            elapsed = time.time() - t0
            done = (bi + 1) * dataloader.batch_size
            total = len(dataloader.dataset)
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{desc}] {done}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    feats = np.concatenate(all_feats)
    print(f"  [{desc}] Done: {feats.shape} in {time.time()-t0:.0f}s")
    return feats

# ============================================================
# PHASE 1: EXTRACT FEATURES FOR WILDLIFE (DINOv3 + SigLIP2)
# Uses transformers.AutoModel (V21-proven method, handles sharded safetensors)
# ============================================================
def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_siglip_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

@torch.no_grad()
def extract_hf_features(model, dataloader, mode="dinov3", desc=""):
    """Extract features using HuggingFace model (transformers API)."""
    model.eval()
    all_feats = []
    t0 = time.time()
    for bi, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device, dtype=torch.float16)
        if mode == "siglip2":
            vision_model = model.vision_model if hasattr(model, 'vision_model') else model
            outputs = vision_model(pixel_values=imgs)
        else:
            outputs = model(pixel_values=imgs, output_hidden_states=False)

        if hasattr(outputs, 'last_hidden_state'):
            cls_feat = outputs.last_hidden_state[:, 0]
            patch_mean = outputs.last_hidden_state[:, 1:].mean(dim=1)
            feat = torch.cat([cls_feat, patch_mean], dim=-1)
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            feat = outputs.pooler_output
        else:
            feat = outputs[0][:, 0]

        feat = F.normalize(feat.float(), dim=-1)
        all_feats.append(feat.cpu().numpy())
        if (bi + 1) % 100 == 0:
            elapsed = time.time() - t0
            done = (bi + 1) * dataloader.batch_size
            total = len(dataloader.dataset)
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"  [{desc}] {done}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    feats = np.concatenate(all_feats)
    print(f"  [{desc}] Done: {feats.shape} in {time.time()-t0:.0f}s")
    return feats

def phase1():
    print("=" * 60)
    print("PHASE 1: Extract DINOv3 + SigLIP2 features for WildlifeReID-10k")
    print("=" * 60)
    from transformers import AutoModel

    wildlife_df = pd.read_csv(WILDLIFE_META)
    paths = wildlife_df["full_path"].tolist()
    print(f"Wildlife images: {len(paths)}")

    # ---- DINOv3-7B: original shards were deleted to free disk ----
    # Competition data has DINOv3 cached from V23; wildlife does not.
    # Use SigLIP2 only for wildlife SupCon training.
    print("DINOv3-7B: shards deleted (disk freed earlier), skipped for wildlife")
    print("  Competition has V23 DINOv3 cache → will use in Phase 3 ensemble")

    # ---- SigLIP2-Giant via transformers ----
    siglip_cache = os.path.join(WILDLIFE_FEAT_CACHE, "siglip2_features.npz")
    if os.path.exists(siglip_cache):
        print(f"SigLIP2: cached")
        return

    siglip_dir = "/root/autodl-tmp/models/siglip2-giant"
    print(f"\nLoading SigLIP2-Giant via transformers from {siglip_dir}...")
    model = AutoModel.from_pretrained(
        siglip_dir, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    print(f"  SigLIP2-Giant loaded! GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    ds = ImageDataset(paths, get_siglip_transform(384))
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    feats = extract_hf_features(model, dl, mode="siglip2", desc="SigLIP2")
    np.savez_compressed(siglip_cache, feats=feats)
    print(f"  Saved: {siglip_cache} ({feats.shape})")
    del model, feats; gc.collect(); torch.cuda.empty_cache()

# ============================================================
# PHASE 2: TRAIN SUPCON ON WILDLIFE
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        N = features.shape[0]
        sim = features @ features.T / self.temperature
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        self_mask = 1.0 - torch.eye(N, device=features.device)
        pos_mask = pos_mask * self_mask
        n_pos = pos_mask.sum(dim=1)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim) * self_mask
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        log_prob = sim - log_sum_exp
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        valid = n_pos > 0
        return -mean_log_prob_pos[valid].mean()

class PKSampler:
    def __init__(self, labels, P=64, K=4):
        self.P, self.K = P, K
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.valid_labels = [l for l, idxs in self.label_to_indices.items() if len(idxs) >= 2]
        print(f"  PKSampler: {len(self.valid_labels)} valid labels (of {len(self.label_to_indices)} total)")
    def sample(self):
        selected = np.random.choice(self.valid_labels,
                                    size=min(self.P, len(self.valid_labels)), replace=False)
        indices = []
        for label in selected:
            pool = self.label_to_indices[label]
            chosen = np.random.choice(pool, size=self.K, replace=len(pool) < self.K)
            indices.extend(chosen)
        return indices

def phase2():
    print("\n" + "=" * 60)
    print("PHASE 2: Train SupCon on WildlifeReID-10k (SigLIP2 features)")
    print("=" * 60)

    wildlife_df = pd.read_csv(WILDLIFE_META)
    identities = wildlife_df["identity"].values
    unique_ids = sorted(set(identities))
    id_to_label = {id_: i for i, id_ in enumerate(unique_ids)}
    labels = np.array([id_to_label[x] for x in identities])
    print(f"Identities: {len(unique_ids)}, Images: {len(labels)}")

    # Load SigLIP2 features
    siglip_f = np.load(os.path.join(WILDLIFE_FEAT_CACHE, "siglip2_features.npz"))["feats"].astype(np.float32)
    feats = siglip_f
    feat_dim = feats.shape[1]
    print(f"SigLIP2 features: {feats.shape}")
    del siglip_f; gc.collect()

    # Train
    proj = ProjectionHead(feat_dim, 1024, 256, 0.3).to(device)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=SUPCON_LR, weight_decay=SUPCON_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS)
    criterion = SupConLoss(temperature=SUPCON_TEMP)
    sampler = PKSampler(labels, P=PK_P, K=PK_K)

    feats_t = torch.from_numpy(feats).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)

    n_iters = max(len(feats) // (PK_P * PK_K), 30)
    print(f"Training: {SUPCON_EPOCHS} epochs, {n_iters} iters/epoch, batch={PK_P}×{PK_K}={PK_P*PK_K}")

    for epoch in range(SUPCON_EPOCHS):
        proj.train()
        losses = []
        for _ in range(n_iters):
            idx = sampler.sample()
            z = proj(feats_t[idx])
            loss = criterion(z, labels_t[idx])
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{SUPCON_EPOCHS}: loss={np.mean(losses):.4f}")

    torch.save(proj.state_dict(), os.path.join(OUTPUT_DIR, "supcon_proj_wildlife.pth"))
    print(f"Saved projection head")

    del feats_t, labels_t
    gc.collect()
    torch.cuda.empty_cache()
    return proj, feat_dim

# ============================================================
# PHASE 3: CLUSTER COMPETITION DATA
# ============================================================
def hac_cluster(sim_matrix, threshold):
    dist = np.clip(1.0 - sim_matrix, 0, None)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    return fcluster(Z, t=threshold, criterion='distance') - 1

def phase3(proj, feat_dim):
    print("\n" + "=" * 60)
    print("PHASE 3: Zero-shot transfer → HAC clustering")
    print("=" * 60)

    comp_meta = pd.read_csv(os.path.join(COMP_DIR, "metadata.csv"))
    sample_sub = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

    # Load competition SigLIP2 (same backbone as wildlife SupCon training)
    comp_siglip = np.load(os.path.join(COMP_FEAT_CACHE, "siglip2_features.npz"))["feats"].astype(np.float32)
    print(f"Competition SigLIP2: {comp_siglip.shape}")

    # Load MegaDescriptor + DINOv3 for ensemble
    comp_mega = np.load(os.path.join(COMP_FEAT_CACHE, "megadesc_features.npz"))["feats"].astype(np.float32)
    comp_dinov3 = np.load(os.path.join(COMP_FEAT_CACHE, "dinov3_features.npz"))["feats"].astype(np.float32)
    print(f"Competition MegaDesc: {comp_mega.shape}, DINOv3: {comp_dinov3.shape}")

    # Project competition SigLIP2 through wildlife-trained SupCon head
    proj.eval()
    with torch.no_grad():
        comp_projected = proj(torch.from_numpy(comp_siglip).float().to(device)).cpu().numpy()
    print(f"Projected: {comp_projected.shape}")
    del comp_siglip

    # Normalize for cosine sim
    comp_mega_norm = comp_mega / (np.linalg.norm(comp_mega, axis=1, keepdims=True) + 1e-8)
    comp_dinov3_norm = comp_dinov3 / (np.linalg.norm(comp_dinov3, axis=1, keepdims=True) + 1e-8)
    del comp_mega, comp_dinov3

    submission_dict = {}
    thresholds_found = []

    for species in SPECIES_ORDER:
        print(f"\n--- {species} ---")
        sp_mask = comp_meta["dataset"] == species
        sp_meta = comp_meta[sp_mask].reset_index(drop=True)
        sp_indices = np.where(sp_mask.values)[0]

        train_mask = sp_meta["split"] == "train"
        test_mask = sp_meta["split"] == "test"
        train_idx = np.where(train_mask.values)[0]
        test_idx = np.where(test_mask.values)[0]
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

        # 3-way similarity: SupCon projected + MegaDesc + DINOv3
        sp_proj = comp_projected[sp_indices]
        sp_mega = comp_mega_norm[sp_indices]
        sp_dinov3 = comp_dinov3_norm[sp_indices]
        sim_supcon = sp_proj @ sp_proj.T
        sim_mega = sp_mega @ sp_mega.T
        sim_dinov3 = sp_dinov3 @ sp_dinov3.T

        # Ensemble weights
        if species == "TexasHornedLizards":
            w_sc, w_mega, w_dino = 0.3, 0.4, 0.3
        elif species == "SeaTurtleID2022":
            w_sc, w_mega, w_dino = 0.5, 0.3, 0.2
        else:
            w_sc, w_mega, w_dino = 0.4, 0.35, 0.25

        sim_fused = w_sc * sim_supcon + w_mega * sim_mega + w_dino * sim_dinov3
        print(f"  Ensemble: supcon={w_sc}, mega={w_mega}, dinov3={w_dino}")

        if len(train_idx) > 0:
            train_labels_raw = sp_meta.loc[train_mask, "identity"].values
            uid = sorted(set(train_labels_raw))
            id_map = {id_: i for i, id_ in enumerate(uid)}
            train_labels = np.array([id_map[x] for x in train_labels_raw])

            sim_train = sim_fused[np.ix_(train_idx, train_idx)]
            best_ari, best_th = -1, 0.5
            for th in np.arange(0.05, 0.95, 0.005):
                pred = hac_cluster(sim_train, th)
                ari = adjusted_rand_score(train_labels, pred)
                if ari > best_ari:
                    best_ari, best_th = ari, th
            print(f"  Train ARI: {best_ari:.4f}, threshold: {best_th:.3f}")
            thresholds_found.append(best_th)
        else:
            # Use median of other species' thresholds
            if thresholds_found:
                best_th = np.median(thresholds_found)
            else:
                best_th = 0.55
            print(f"  No train data, using threshold: {best_th:.3f}")

        # Cluster test data
        sim_test = sim_fused[np.ix_(test_idx, test_idx)]
        test_labels = hac_cluster(sim_test, best_th)

        # Map to submission
        test_image_ids = sp_meta.loc[test_mask, "image_id"].values
        unique_cl = sorted(set(test_labels))
        cl_map = {c: f"cluster_{species}_{i}" for i, c in enumerate(unique_cl)}
        for img_id, cl in zip(test_image_ids, test_labels):
            submission_dict[str(img_id)] = cl_map[cl]

        n_cl = len(unique_cl)
        counts = Counter(test_labels)
        singletons = sum(1 for v in counts.values() if v == 1)
        max_cl = max(counts.values()) if counts else 0
        print(f"  Clusters: {n_cl}, singletons: {singletons} ({singletons/max(n_cl,1)*100:.0f}%), max: {max_cl}")

    # Generate submission
    print("\n" + "=" * 60)
    print("Generating submission")
    sub_ids = [str(x) for x in sample_sub["image_id"].values]
    sub_clusters = []
    missing = 0
    for img_id in sub_ids:
        if img_id in submission_dict:
            sub_clusters.append(submission_dict[img_id])
        else:
            sub_clusters.append(f"cluster_unknown_{missing}")
            missing += 1

    submission = pd.DataFrame({"image_id": sample_sub["image_id"].values, "cluster": sub_clusters})
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")
    print(f"   {len(submission)} images, {submission['cluster'].nunique()} clusters, {missing} missing")

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("AnimalCLEF2026 V30b: WildlifeReID-10k SupCon (DINOv3-7B + SigLIP2)")
    print("=" * 70)
    phase1()
    proj, feat_dim = phase2()
    phase3(proj, feat_dim)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("DONE!")
