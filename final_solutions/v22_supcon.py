#!/usr/bin/env python3
"""
AnimalCLEF2026 V22 — SupCon Projection Heads on Cached Features + MegaDescriptor
==================================================================================
Key insight: V21 proved raw foundation features have ~0 individual discriminability.
SupCon projection learns to REORGANIZE the feature space: same individual → close, different → far.
MegaDescriptor is brought back as 5th backbone (it already has Re-ID signal: 0.86 ARI on SeaTurtle).

Pipeline:
  1. Load V21 cached features (DINOv3/InternViT/SigLIP2/EVA02)
  2. Extract MegaDescriptor features (re-download, ~5min)
  3. Per-species: train SupCon projection head on concatenated 5-backbone features
  4. Project all features → cosine similarity → HAC clustering
  5. Post-processing → submission
"""

import os, sys, gc, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
from PIL import Image
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/root/autodl-tmp/animal-clef-2026"
MODEL_DIR = "/root/autodl-tmp/models"
OUTPUT_DIR = "/root/autodl-tmp/ov22"
FEAT_CACHE = "/root/autodl-tmp/feat_cache_v21"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# SupCon hyperparams
PROJ_HIDDEN = 1024
PROJ_OUT = 256
SUPCON_TEMP = 0.07
SUPCON_LR = 5e-4
SUPCON_EPOCHS = 50
SUPCON_BATCH = 512

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("PHASE 1: Loading data and cached features")
print("=" * 60)

meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
print(f"Metadata: {len(meta)} rows, Submission: {len(sample_sub)} rows")

# All image IDs as strings
all_image_ids = [str(x) for x in meta['image_id'].values]
id_to_row = {str(row['image_id']): row for _, row in meta.iterrows()}

# Load cached foundation features
cached_backbones = {}
for name in ["dinov3", "internvit", "siglip2", "eva02"]:
    path = os.path.join(FEAT_CACHE, f"{name}_features.npz")
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        ids = [str(x) for x in data['ids']]
        feats = np.array(data['feats'], dtype=np.float32)
        cached_backbones[name] = dict(zip(ids, feats))
        print(f"  {name}: {len(ids)} images, dim={feats.shape[1]}")
    else:
        print(f"  WARNING: {name} not found, skipping")

# ============================================================
# 2. MEGADESCRIPTOR FEATURE EXTRACTION
# ============================================================
mega_cache = os.path.join(FEAT_CACHE, "megadesc_features.npz")
if os.path.exists(mega_cache):
    print("Loading cached MegaDescriptor features...")
    data = np.load(mega_cache, allow_pickle=True)
    cached_backbones["megadesc"] = dict(zip(
        [str(x) for x in data['ids']], np.array(data['feats'], dtype=np.float32)
    ))
    print(f"  megadesc: {len(cached_backbones['megadesc'])} images, dim={list(cached_backbones['megadesc'].values())[0].shape[0]}")
else:
    print("\n" + "=" * 60)
    print("PHASE 1b: Extracting MegaDescriptor-L-384 features")
    print("=" * 60)

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"

    import timm
    from torchvision import transforms

    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    model = model.to(DEVICE).eval().half()

    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_paths = [os.path.join(DATA_DIR, p) for p in meta['path'].values]
    mega_feats = []
    mega_ids = []

    with torch.no_grad():
        for i in range(0, len(all_paths), 32):
            batch_imgs = []
            for p in all_paths[i:i+32]:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(transform(img))
            batch = torch.stack(batch_imgs).to(DEVICE).half()
            feat = model(batch)
            feat = F.normalize(feat.float(), dim=-1)
            mega_feats.append(feat.cpu().numpy())
            mega_ids.extend(all_image_ids[i:i+32])
            if (i // 32) % 100 == 0:
                print(f"  MegaDesc: {i}/{len(all_paths)}")

    mega_feats = np.concatenate(mega_feats, axis=0)
    cached_backbones["megadesc"] = dict(zip(mega_ids, mega_feats))
    np.savez(mega_cache, ids=mega_ids, feats=mega_feats)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  MegaDescriptor: {len(mega_ids)} images, dim={mega_feats.shape[1]}")

backbone_names = list(cached_backbones.keys())
print(f"\nTotal backbones: {len(backbone_names)}: {backbone_names}")


# ============================================================
# 3. SUPCON PROJECTION HEAD
# ============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(features, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    features: (N, D) L2-normalized
    labels: (N,) identity labels
    """
    N = features.shape[0]
    device = features.device

    # Pairwise similarity
    sim = features @ features.T / temperature  # (N, N)

    # Positive mask: same identity, exclude self
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)

    # Check: skip if no positive pairs in this batch
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Self-mask (exclude diagonal) — avoid inplace ops
    self_mask = 1.0 - torch.eye(N, device=device)

    # Log-softmax: for each row, softmax over all non-self entries
    logits_max = sim.max(dim=1, keepdim=True)[0].detach()
    logits = sim - logits_max
    exp_logits = torch.exp(logits) * self_mask  # zero out self WITHOUT inplace

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean over positive pairs
    pos_count = pos_mask.sum(dim=1)
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

    # Only count samples with at least 1 positive
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return -mean_log_prob[valid].mean()


# ============================================================
# 4. IDENTITY-BALANCED BATCH SAMPLER
# ============================================================
class PKSampler:
    """Sample P identities, K images per identity → batch = P*K."""

    def __init__(self, labels, P=32, K=4):
        self.labels = np.array(labels)
        self.P = P
        self.K = K

        # Build index: identity → list of indices
        self.id_to_idx = defaultdict(list)
        for i, l in enumerate(self.labels):
            self.id_to_idx[l].append(i)

        # Only identities with >= K images (or >= 2 for K=2)
        min_k = min(K, 2)
        self.valid_ids = [k for k, v in self.id_to_idx.items() if len(v) >= min_k]
        self.n_batches = max(1, len(self.valid_ids) // P)

    def __iter__(self):
        rng = np.random.RandomState()
        for _ in range(self.n_batches):
            # Sample P identities
            chosen_ids = rng.choice(self.valid_ids, min(self.P, len(self.valid_ids)), replace=False)
            batch = []
            for cid in chosen_ids:
                indices = self.id_to_idx[cid]
                if len(indices) >= self.K:
                    chosen = rng.choice(indices, self.K, replace=False)
                else:
                    chosen = rng.choice(indices, self.K, replace=True)
                batch.extend(chosen.tolist())
            yield batch

    def __len__(self):
        return self.n_batches


# ============================================================
# 5. TRAIN SUPCON PER SPECIES
# ============================================================
def train_supcon_for_species(species, train_ids, train_labels, all_ids):
    """Train a SupCon projection head for one species, return projected features for all images."""
    print(f"\n  Training SupCon projection head for {species}...")

    # Concatenate all backbone features for this species
    # Build feature matrix: rows = images, cols = concatenated backbone features
    backbone_dims = {}
    for bname in backbone_names:
        sample_feat = list(cached_backbones[bname].values())[0]
        backbone_dims[bname] = len(sample_feat)

    total_dim = sum(backbone_dims.values())
    print(f"    Concatenated dim: {total_dim} ({backbone_dims})")

    # Build train feature matrix
    train_feats = np.zeros((len(train_ids), total_dim), dtype=np.float32)
    offset = 0
    for bname in backbone_names:
        dim = backbone_dims[bname]
        for i, tid in enumerate(train_ids):
            if tid in cached_backbones[bname]:
                f = np.array(cached_backbones[bname][tid], dtype=np.float32)
                # L2 normalize per backbone before concatenation
                norm = np.linalg.norm(f) + 1e-8
                train_feats[i, offset:offset+dim] = f / norm
        offset += dim

    train_labels_arr = np.array(train_labels)

    # Count identities with 2+ images
    from collections import Counter
    id_counts = Counter(train_labels_arr)
    n_valid = sum(1 for c in id_counts.values() if c >= 2)
    print(f"    Train: {len(train_ids)} images, {len(id_counts)} identities, {n_valid} with 2+ images")

    if n_valid < 5:
        print(f"    Too few identities with pairs, skipping SupCon")
        return None

    # Setup model + optimizer
    proj = ProjectionHead(total_dim, PROJ_HIDDEN, PROJ_OUT).to(DEVICE)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=SUPCON_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS)

    # PK Sampler
    P = min(32, n_valid)
    K = min(4, min(id_counts[k] for k in id_counts if id_counts[k] >= 2))
    K = max(K, 2)
    sampler = PKSampler(train_labels_arr, P=P, K=K)

    train_feats_tensor = torch.tensor(train_feats, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels_arr, dtype=torch.long)

    # Training loop
    best_loss = float('inf')
    best_state = None

    for epoch in range(SUPCON_EPOCHS):
        proj.train()
        epoch_loss = 0
        n_batches = 0

        for batch_idx in sampler:
            batch_feats = train_feats_tensor[batch_idx].to(DEVICE)
            batch_labels = train_labels_tensor[batch_idx].to(DEVICE)

            projected = proj(batch_feats)
            loss = supcon_loss(projected, batch_labels, temperature=SUPCON_TEMP)

            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{SUPCON_EPOCHS}: loss={avg_loss:.4f} (best={best_loss:.4f})")

    if best_state is not None:
        proj.load_state_dict(best_state)
    print(f"    Best SupCon loss: {best_loss:.4f}")

    # Project ALL features (train + test)
    proj.eval()
    all_feats = np.zeros((len(all_ids), total_dim), dtype=np.float32)
    offset = 0
    for bname in backbone_names:
        dim = backbone_dims[bname]
        for i, aid in enumerate(all_ids):
            if aid in cached_backbones[bname]:
                f = np.array(cached_backbones[bname][aid], dtype=np.float32)
                norm = np.linalg.norm(f) + 1e-8
                all_feats[i, offset:offset+dim] = f / norm
        offset += dim

    with torch.no_grad():
        # Process in batches to avoid memory issues
        projected_all = []
        for i in range(0, len(all_ids), 1024):
            batch = torch.tensor(all_feats[i:i+1024], dtype=torch.float32).to(DEVICE)
            p = proj(batch).cpu().numpy()
            projected_all.append(p)
        projected_all = np.concatenate(projected_all, axis=0)

    # Evaluate on train
    train_projected = projected_all[:len(train_ids)]
    train_sim = train_projected @ train_projected.T
    np.fill_diagonal(train_sim, 0)

    best_ari = -1
    for th in np.arange(0.05, 0.95, 0.02):
        try:
            dist = 1 - train_sim
            np.fill_diagonal(dist, 0)
            clust = AgglomerativeClustering(
                n_clusters=None, distance_threshold=th,
                linkage='average', metric='precomputed'
            ).fit(np.clip(dist, 0, 2))
            ari = adjusted_rand_score(train_labels, clust.labels_)
            if ari > best_ari:
                best_ari = ari
        except:
            continue

    print(f"    SupCon projected train ARI: {best_ari:.4f}")

    # Save projection head weights for potential transfer
    proj_state = {k: v.cpu().clone() for k, v in proj.state_dict().items()} if best_state else None

    del proj
    torch.cuda.empty_cache()

    return projected_all, total_dim, proj_state


# ============================================================
# 6. RAW SIMILARITY BASELINE (for comparison / fallback)
# ============================================================
def compute_raw_similarity(all_ids, weights=None):
    """Compute weighted cosine similarity from raw (unprojected) features."""
    if weights is None:
        weights = {k: 1.0/len(backbone_names) for k in backbone_names}

    N = len(all_ids)
    fused = np.zeros((N, N), dtype=np.float32)

    for bname in backbone_names:
        w = weights.get(bname, 0)
        if w <= 0:
            continue
        feats = np.array([
            np.array(cached_backbones[bname].get(aid, np.zeros(1)), dtype=np.float32)
            for aid in all_ids
        ])
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim = feats @ feats.T
        fused += sim * w

    return fused


# ============================================================
# 7. CLUSTERING
# ============================================================
def cluster_with_threshold_search(sim_matrix, train_indices, train_labels):
    """HAC with threshold grid search on train subset."""
    dist = 1.0 - sim_matrix
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)

    if train_indices is not None and len(train_indices) > 20 and train_labels is not None:
        tr_dist = dist[np.ix_(train_indices, train_indices)]
        best_th, best_ari = 0.5, -1

        for th in np.arange(0.05, 0.95, 0.02):
            try:
                clust = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=th,
                    linkage='average', metric='precomputed'
                ).fit(tr_dist)
                ari = adjusted_rand_score(train_labels, clust.labels_)
                if ari > best_ari:
                    best_ari = ari
                    best_th = th
            except:
                continue

        print(f"    Best threshold: {best_th:.3f} (train ARI={best_ari:.4f})")
    else:
        best_th = 0.5

    # Cluster all
    clust = AgglomerativeClustering(
        n_clusters=None, distance_threshold=best_th,
        linkage='average', metric='precomputed'
    ).fit(dist)

    return clust.labels_, best_th


# ============================================================
# 8. MAIN
# ============================================================
def main():
    t0 = time.time()

    print(f"\n{'='*60}")
    print("PHASE 2-3: Per-Species SupCon + Clustering")
    print(f"{'='*60}")

    submission_rows = []
    species_results = {}
    saved_proj_heads = {}  # Save trained projection heads for transfer

    for species in SPECIES_ORDER:
        print(f"\n{'='*50}")
        print(f"Processing: {species}")
        print(f"{'='*50}")

        sp_meta = meta[meta['dataset'] == species]
        sp_train = sp_meta[sp_meta['split'] == 'train']
        sp_test = sp_meta[sp_meta['split'] == 'test']

        train_ids = [str(x) for x in sp_train['image_id'].values]
        test_ids = [str(x) for x in sp_test['image_id'].values]
        all_ids = train_ids + test_ids

        # Train labels
        if len(sp_train) > 0:
            identity_map = {name: idx for idx, name in enumerate(sp_train['identity'].unique())}
            train_labels = [identity_map[x] for x in sp_train['identity'].values]
        else:
            train_labels = []

        print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

        # === SupCon or raw ===
        if len(train_ids) > 50 and len(set(train_labels)) > 5:
            result = train_supcon_for_species(species, train_ids, train_labels, all_ids)

            if result is not None:
                projected, total_dim, proj_state = result
                saved_proj_heads[species] = (total_dim, proj_state)  # Save for transfer
                # Use projected features
                sim_matrix = projected @ projected.T
                # Also compute raw MegaDescriptor similarity for ensemble
                mega_sim = np.zeros((len(all_ids), len(all_ids)), dtype=np.float32)
                mega_feats = np.array([
                    np.array(cached_backbones["megadesc"].get(aid, np.zeros(1)), dtype=np.float32)
                    for aid in all_ids
                ])
                mega_feats = mega_feats / (np.linalg.norm(mega_feats, axis=1, keepdims=True) + 1e-8)
                mega_sim = mega_feats @ mega_feats.T

                # Ensemble: projected + raw MegaDescriptor
                # Search best alpha on train
                train_idx = list(range(len(train_ids)))
                best_alpha, best_ari = 0.5, -1
                for alpha in np.arange(0.0, 1.05, 0.1):
                    blended = alpha * sim_matrix + (1 - alpha) * mega_sim
                    tr_sim = blended[np.ix_(train_idx, train_idx)]
                    tr_dist = np.clip(1 - tr_sim, 0, 2)
                    np.fill_diagonal(tr_dist, 0)
                    for th in np.arange(0.1, 0.9, 0.05):
                        try:
                            c = AgglomerativeClustering(
                                n_clusters=None, distance_threshold=th,
                                linkage='average', metric='precomputed'
                            ).fit(tr_dist)
                            ari = adjusted_rand_score(train_labels, c.labels_)
                            if ari > best_ari:
                                best_ari = ari
                                best_alpha = alpha
                        except:
                            continue

                print(f"  Ensemble alpha={best_alpha:.1f} (SupCon:{best_alpha:.0%} + MegaDesc:{1-best_alpha:.0%}), best train ARI={best_ari:.4f}")
                final_sim = best_alpha * sim_matrix + (1 - best_alpha) * mega_sim
            else:
                # SupCon failed, use raw MegaDescriptor only
                print("  Falling back to raw MegaDescriptor")
                final_sim = compute_raw_similarity(all_ids, {"megadesc": 1.0})
        else:
            # No training data (TexasHornedLizards) → transfer projection head from best species
            print("  No training data, using transferred projection head")

            # Pick the best trained projection head (prefer SeaTurtle > Salamander > Lynx)
            transfer_from = None
            for sp_candidate in ["SeaTurtleID2022", "SalamanderID2025", "LynxID2025"]:
                if sp_candidate in saved_proj_heads and saved_proj_heads[sp_candidate][1] is not None:
                    transfer_from = sp_candidate
                    break

            if transfer_from:
                print(f"  Transferring projection head from {transfer_from}")
                total_dim, proj_state = saved_proj_heads[transfer_from]

                # Build concatenated features for this species
                backbone_dims = {}
                for bname in backbone_names:
                    sample_feat = list(cached_backbones[bname].values())[0]
                    backbone_dims[bname] = len(sample_feat)

                all_feats = np.zeros((len(all_ids), total_dim), dtype=np.float32)
                offset = 0
                for bname in backbone_names:
                    dim = backbone_dims[bname]
                    for i, aid in enumerate(all_ids):
                        if aid in cached_backbones[bname]:
                            f = np.array(cached_backbones[bname][aid], dtype=np.float32)
                            norm = np.linalg.norm(f) + 1e-8
                            all_feats[i, offset:offset+dim] = f / norm
                    offset += dim

                # Load projection head and project
                proj = ProjectionHead(total_dim, PROJ_HIDDEN, PROJ_OUT).to(DEVICE)
                proj.load_state_dict(proj_state)
                proj.eval()

                with torch.no_grad():
                    projected_all = []
                    for i in range(0, len(all_ids), 1024):
                        batch = torch.tensor(all_feats[i:i+1024], dtype=torch.float32).to(DEVICE)
                        p = proj(batch).cpu().numpy()
                        projected_all.append(p)
                    projected_all = np.concatenate(projected_all, axis=0)

                final_sim = projected_all @ projected_all.T
                del proj
                torch.cuda.empty_cache()
            else:
                # Fallback: MegaDescriptor only
                print("  No projection head available, using MegaDescriptor only")
                final_sim = compute_raw_similarity(all_ids, {"megadesc": 1.0})

        # === Clustering ===
        print("  Clustering...")
        train_idx = list(range(len(train_ids)))
        labels, best_th = cluster_with_threshold_search(
            final_sim, train_idx if len(train_ids) > 20 else None,
            train_labels if len(train_labels) > 20 else None
        )

        # Evaluate
        if len(train_ids) > 20:
            train_pred = labels[:len(train_ids)]
            final_train_ari = adjusted_rand_score(train_labels, train_pred)
            print(f"  Final Train ARI: {final_train_ari:.4f}")
        else:
            final_train_ari = None

        # Stats
        test_labels = labels[len(train_ids):]
        n_clusters = len(set(test_labels))
        n_singletons = sum(1 for cl in set(test_labels) if sum(test_labels == cl) == 1)
        print(f"  Test: {len(test_ids)} → {n_clusters} clusters, {n_singletons} singletons ({100*n_singletons/max(1,len(test_ids)):.0f}%)")

        species_results[species] = {
            "train_ari": final_train_ari,
            "n_clusters": n_clusters,
            "n_singletons": n_singletons,
        }

        # Build submission rows
        for i, img_id in enumerate(test_ids):
            cluster_label = f"cluster_{species}_{test_labels[i]}"
            submission_rows.append({"image_id": int(img_id), "cluster": cluster_label})

    # === Build submission ===
    submission = pd.DataFrame(submission_rows)
    submission = submission.set_index('image_id').loc[sample_sub['image_id'].values].reset_index()
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"V22 COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    for sp, r in species_results.items():
        ari_str = f"{r['train_ari']:.4f}" if r['train_ari'] is not None else "N/A"
        print(f"  {sp}: train ARI={ari_str}, test clusters={r['n_clusters']}, singletons={r['n_singletons']}")

    print(f"\nSubmission: {OUTPUT_DIR}/submission.csv ({len(submission)} rows)")
    print("=" * 60)


if __name__ == "__main__":
    main()
