#!/usr/bin/env python3
"""
AnimalCLEF2026 V30 — WildlifeReID-10k Pre-trained SupCon + Zero-shot Transfer
==============================================================================
Phase 1: Extract DINOv3-7B + SigLIP2 features for WildlifeReID-10k (71K imgs)
Phase 2: Train SupCon projection head on 3,497 identities
Phase 3: Zero-shot transfer to competition data → HAC clustering
Phase 4: Generate submission.csv
"""
import os, sys, gc, time, csv, warnings
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

# SupCon config
PROJ_HIDDEN = 1024
PROJ_OUT = 256
SUPCON_TEMP = 0.07
SUPCON_LR = 3e-4
SUPCON_WD = 1e-3
SUPCON_EPOCHS = 40
DROPOUT_RATE = 0.3
PK_P = 64  # identities per batch
PK_K = 4   # images per identity

SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# ============================================================
# DATASET
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

# ============================================================
# FEATURE EXTRACTION
# ============================================================
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
            out = out[:, 0]  # CLS token
        feats = F.normalize(out.float(), dim=-1)
        all_feats.append(feats.cpu().numpy())
        if (bi + 1) % 50 == 0:
            elapsed = time.time() - t0
            done = (bi + 1) * dataloader.batch_size
            total = len(dataloader.dataset)
            eta = elapsed / done * (total - done)
            print(f"  [{desc}] {done}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    feats = np.concatenate(all_feats)
    print(f"  [{desc}] Done: {feats.shape} in {time.time()-t0:.0f}s")
    return feats

def load_model_and_transform(model_name, model_dir):
    """Load a model from local dir"""
    if model_name == "dinov3":
        # DINOv3-7B: find weights and try to load
        weights_path = None
        for r, d, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".safetensors") and os.path.getsize(os.path.join(r, f)) > 1e9:
                    weights_path = os.path.join(r, f)
                    break
            if weights_path:
                break

        # Try loading as timm model
        import json
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            arch = cfg.get("architecture", "")
        else:
            arch = ""

        # Try various DINOv3 model names
        candidates = [arch] if arch else []
        candidates += [m for m in timm.list_models("*dinov3*") if "7b" in m.lower() or "huge" in m.lower() or "hplus" in m.lower()]
        candidates += [m for m in timm.list_models("*dinov3*")]

        model = None
        for name in candidates:
            if not name:
                continue
            try:
                model = timm.create_model(name, pretrained=False, num_classes=0, checkpoint_path=weights_path)
                print(f"  Loaded DINOv3 as: {name}")
                break
            except:
                continue

        if model is None:
            raise RuntimeError(f"Cannot load DINOv3 from {model_dir}")

        model = model.half().to(device).eval()
        try:
            data_cfg = timm.data.resolve_model_data_config(model)
            transform = timm.data.create_transform(**data_cfg, is_training=False)
        except:
            transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return model, transform

    elif model_name == "siglip2":
        weights_path = None
        for r, d, files in os.walk(model_dir):
            for f in files:
                if f.endswith(".safetensors") and os.path.getsize(os.path.join(r, f)) > 1e9:
                    weights_path = os.path.join(r, f)
                    break
            if weights_path:
                break

        # SigLIP2 via timm
        import json
        config_path = os.path.join(model_dir, "config.json")
        arch = ""
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            arch = cfg.get("architecture", cfg.get("model_name", ""))

        candidates = [arch] if arch else []
        candidates += [m for m in timm.list_models("*siglip2*") if "giant" in m.lower()]
        candidates += [m for m in timm.list_models("*siglip*") if "giant" in m.lower()]

        model = None
        for name in candidates:
            if not name:
                continue
            try:
                model = timm.create_model(name, pretrained=False, num_classes=0, checkpoint_path=weights_path)
                print(f"  Loaded SigLIP2 as: {name}")
                break
            except:
                continue

        if model is None:
            raise RuntimeError(f"Cannot load SigLIP2 from {model_dir}")

        model = model.half().to(device).eval()
        try:
            data_cfg = timm.data.resolve_model_data_config(model)
            transform = timm.data.create_transform(**data_cfg, is_training=False)
        except:
            transform = transforms.Compose([
                transforms.Resize(384, interpolation=3),
                transforms.CenterCrop(384),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return model, transform

# ============================================================
# PHASE 1: EXTRACT WILDLIFE FEATURES
# ============================================================
def phase1_extract_wildlife():
    print("=" * 60)
    print("PHASE 1: Extract features for WildlifeReID-10k")
    print("=" * 60)

    # Load wildlife metadata
    wildlife_df = pd.read_csv(WILDLIFE_META)
    paths = wildlife_df["full_path"].tolist()
    print(f"Wildlife images: {len(paths)}")

    for model_name, model_dir in [
        ("dinov3", "/root/autodl-tmp/models/dinov3-vit7b"),
        ("siglip2", "/root/autodl-tmp/models/siglip2-giant"),
    ]:
        cache_path = os.path.join(WILDLIFE_FEAT_CACHE, f"{model_name}_features.npz")
        if os.path.exists(cache_path):
            print(f"\n{model_name}: cached, skipping")
            continue

        print(f"\nExtracting {model_name}...")
        model, transform = load_model_and_transform(model_name, model_dir)
        ds = ImageDataset(paths, transform)
        dl = DataLoader(ds, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)
        feats = extract_features(model, dl, model_name)

        np.savez_compressed(cache_path, feats=feats, paths=np.array(paths))
        print(f"  Saved: {cache_path} ({feats.shape})")

        del model
        gc.collect()
        torch.cuda.empty_cache()

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

def phase2_train_supcon():
    print("\n" + "=" * 60)
    print("PHASE 2: Train SupCon on WildlifeReID-10k")
    print("=" * 60)

    # Load wildlife features
    wildlife_df = pd.read_csv(WILDLIFE_META)
    identities = wildlife_df["identity"].values

    # Encode labels
    unique_ids = sorted(set(identities))
    id_to_label = {id_: i for i, id_ in enumerate(unique_ids)}
    labels = np.array([id_to_label[x] for x in identities])
    print(f"Identities: {len(unique_ids)}, Images: {len(labels)}")

    # Load and concatenate features
    features_list = []
    for name in ["dinov3", "siglip2"]:
        cache_path = os.path.join(WILDLIFE_FEAT_CACHE, f"{name}_features.npz")
        data = np.load(cache_path)
        feats = data["feats"].astype(np.float32)
        features_list.append(feats)
        print(f"  {name}: {feats.shape}")

    concat_feats = np.concatenate(features_list, axis=1)
    total_dim = concat_feats.shape[1]
    print(f"  Concatenated: {total_dim} dim")

    # Train SupCon
    proj = ProjectionHead(total_dim, PROJ_HIDDEN, PROJ_OUT, DROPOUT_RATE).to(device)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=SUPCON_LR, weight_decay=SUPCON_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS)
    criterion = SupConLoss(temperature=SUPCON_TEMP)
    sampler = PKSampler(labels, P=PK_P, K=PK_K)

    feats_t = torch.from_numpy(concat_feats).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)

    n_iters = max(len(concat_feats) // (PK_P * PK_K), 20)
    print(f"\nTraining: {SUPCON_EPOCHS} epochs, {n_iters} iters/epoch")
    print(f"  batch: P={PK_P} ids × K={PK_K} imgs = {PK_P*PK_K}")

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
            avg_loss = np.mean(losses) if losses else 0
            print(f"  Epoch {epoch+1}/{SUPCON_EPOCHS}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    # Save model
    proj_path = os.path.join(OUTPUT_DIR, "supcon_proj_wildlife.pth")
    torch.save(proj.state_dict(), proj_path)
    print(f"\nSaved projection head: {proj_path}")

    del feats_t, labels_t
    gc.collect()
    torch.cuda.empty_cache()

    return proj, total_dim

# ============================================================
# PHASE 3: ZERO-SHOT TRANSFER TO COMPETITION DATA
# ============================================================
def hac_cluster(sim_matrix, threshold):
    dist = np.clip(1.0 - sim_matrix, 0, None)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=threshold, criterion='distance')
    return labels - 1

def phase3_cluster(proj, total_dim):
    print("\n" + "=" * 60)
    print("PHASE 3: Zero-shot transfer to competition data")
    print("=" * 60)

    # Load competition features (already cached from V23)
    comp_meta = pd.read_csv(os.path.join(COMP_DIR, "metadata.csv"))
    sample_sub = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

    comp_feats = {}
    for name in ["dinov3", "siglip2"]:
        cache = np.load(os.path.join(COMP_FEAT_CACHE, f"{name}_features.npz"))
        comp_feats[name] = cache["feats"].astype(np.float32)
        print(f"  Competition {name}: {comp_feats[name].shape}")

    # Concatenate in same order as wildlife
    comp_concat = np.concatenate([comp_feats["dinov3"], comp_feats["siglip2"]], axis=1)
    print(f"  Competition concatenated: {comp_concat.shape}")
    assert comp_concat.shape[1] == total_dim, f"Dim mismatch: {comp_concat.shape[1]} vs {total_dim}"

    # Project through trained SupCon head
    proj.eval()
    with torch.no_grad():
        comp_projected = proj(torch.from_numpy(comp_concat).float().to(device)).cpu().numpy()
    print(f"  Projected: {comp_projected.shape}")

    # Also get MegaDescriptor features for ensemble
    mega_cache = np.load(os.path.join(COMP_FEAT_CACHE, "megadesc_features.npz"))
    mega_feats = mega_cache["feats"].astype(np.float32)
    print(f"  MegaDescriptor: {mega_feats.shape}")

    # Per-species clustering
    submission_dict = {}

    for species in SPECIES_ORDER:
        print(f"\n--- {species} ---")
        sp_mask = comp_meta["dataset"] == species
        sp_meta = comp_meta[sp_mask].reset_index(drop=True)
        sp_indices = np.where(sp_mask.values)[0]

        train_mask = sp_meta["split"] == "train"
        test_mask = sp_meta["split"] == "test"
        train_idx = np.where(train_mask.values)[0]
        test_idx = np.where(test_mask.values)[0]
        n_train = len(train_idx)
        n_test = len(test_idx)
        print(f"  Train: {n_train}, Test: {n_test}")

        # Get projected features for this species
        sp_proj = comp_projected[sp_indices]

        # Also get MegaDescriptor features
        sp_mega = mega_feats[sp_indices]

        # Ensemble: SupCon projected + MegaDescriptor
        # Compute similarity from both, weighted average
        sim_supcon = sp_proj @ sp_proj.T
        sp_mega_norm = sp_mega / (np.linalg.norm(sp_mega, axis=1, keepdims=True) + 1e-8)
        sim_mega = sp_mega_norm @ sp_mega_norm.T

        # Weight: more SupCon for species with training data, equal for Texas
        if species == "TexasHornedLizards":
            alpha = 0.5  # equal weight
        else:
            alpha = 0.7  # more SupCon

        sim_fused = alpha * sim_supcon + (1 - alpha) * sim_mega

        if n_train > 0:
            # Grid search threshold on training data
            train_labels_raw = sp_meta.loc[train_mask, "identity"].values
            unique_train_ids = sorted(set(train_labels_raw))
            id_map = {id_: i for i, id_ in enumerate(unique_train_ids)}
            train_labels = np.array([id_map[x] for x in train_labels_raw])

            sim_train = sim_fused[np.ix_(train_idx, train_idx)]
            best_ari, best_th = -1, 0.5
            for th in np.arange(0.05, 0.95, 0.01):
                pred = hac_cluster(sim_train, th)
                ari = adjusted_rand_score(train_labels, pred)
                if ari > best_ari:
                    best_ari, best_th = ari, th
            print(f"  Train ARI: {best_ari:.4f}, threshold: {best_th:.3f}")
        else:
            # No training data — use median threshold from other species
            best_th = 0.55  # will be overridden after other species
            print(f"  No train data, default threshold: {best_th}")

        # Cluster TEST data
        sim_test = sim_fused[np.ix_(test_idx, test_idx)]
        test_labels = hac_cluster(sim_test, best_th)

        # Map to submission
        test_image_ids = sp_meta.loc[test_mask, "image_id"].values
        unique_clusters = sorted(set(test_labels))
        cluster_map = {c: f"cluster_{species}_{i}" for i, c in enumerate(unique_clusters)}

        for img_id, cl in zip(test_image_ids, test_labels):
            submission_dict[str(img_id)] = cluster_map[cl]

        n_cl = len(unique_clusters)
        counts = Counter(test_labels)
        singletons = sum(1 for v in counts.values() if v == 1)
        max_cl = max(counts.values()) if counts else 0
        print(f"  Clusters: {n_cl}, singletons: {singletons} ({singletons/max(n_cl,1)*100:.0f}%), max: {max_cl}")

    # Generate submission
    print("\n" + "=" * 60)
    print("Generating submission")
    print("=" * 60)

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
    print(f"Saved: {out_path}")
    print(f"Total: {len(submission)} images, {submission['cluster'].nunique()} clusters")
    if missing > 0:
        print(f"WARNING: {missing} missing images!")

# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("=" * 70)
    print("AnimalCLEF2026 V30: WildlifeReID-10k Pre-trained SupCon")
    print("=" * 70)

    # Phase 1: Extract wildlife features
    phase1_extract_wildlife()

    # Phase 2: Train SupCon on wildlife
    proj, total_dim = phase2_train_supcon()

    # Phase 3: Cluster competition data
    phase3_cluster(proj, total_dim)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print("DONE!")

if __name__ == "__main__":
    main()
