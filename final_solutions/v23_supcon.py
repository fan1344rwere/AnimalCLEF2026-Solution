#!/usr/bin/env python3
"""
AnimalCLEF2026 V23 — SupCon on ALL 5 backbones (proven V22 approach + better features)
=======================================================================================
V22 got 0.901/0.908 train ARI with 5-backbone concat SupCon per-species.
V23 uses the same approach with stronger features:
  - DINOv3-7B (4096) instead of old dinov3 (1536)
  - + MegaDescriptor-DINOv2-518 (1024)

Pipeline:
  1. Load ALL 5 cached features (dinov3, siglip2, eva02, megadesc_l384, megadesc_dinov2)
  2. Per-species: concat ALL 5 → train SupCon projection head (50 epochs)
  3. Ensemble: SupCon projected sim + raw mega sims
  4. HAC clustering + post-processing
  5. TexasHornedLizards: transfer head from SeaTurtleID2022
"""

import os, sys, gc, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict, Counter
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE = "/root/autodl-tmp/\u6700\u7ec8\u5723\u6218"
DATA_DIR = f"{BASE}/animalclef/animal-clef-2026"
OUTPUT_DIR = f"{BASE}/animalclef/ov23"
FEAT_CACHE = f"{BASE}/animalclef/feat_cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# ALL 5 backbones for SupCon concat
ALL_BACKBONES = ["dinov3", "siglip2", "eva02", "megadesc_l384", "megadesc_dinov2"]

# SupCon hyperparams (same as proven V22)
PROJ_HIDDEN = 1024
PROJ_OUT = 256
SUPCON_TEMP = 0.07
SUPCON_LR = 5e-4
SUPCON_EPOCHS = 50

# ============================================================
# 1. LOAD DATA + FEATURES
# ============================================================
print("=" * 60)
print("PHASE 1: Loading cached features")
print("=" * 60)

meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
comp_ids = [str(x) for x in meta['image_id'].values]
print(f"Competition: {len(meta)} images, Submission: {len(sample_sub)} rows")

# Load all 5 backbone features
comp_features = {}
for bname in ALL_BACKBONES:
    cache_path = os.path.join(FEAT_CACHE, f"{bname}_features.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        feats = data['feats'].astype(np.float32)
        ids = [str(x) for x in data['ids']]
        id_to_idx = {str(i): idx for idx, i in enumerate(ids)}
        ordered = np.zeros((len(comp_ids), feats.shape[1]), dtype=np.float32)
        for ci, cid in enumerate(comp_ids):
            if cid in id_to_idx:
                ordered[ci] = feats[id_to_idx[cid]]
        comp_features[bname] = ordered
        print(f"  {bname}: {ordered.shape}")
    else:
        print(f"  WARNING: {bname} not found")

backbone_names = list(comp_features.keys())
backbone_dims = {b: comp_features[b].shape[1] for b in backbone_names}
total_dim = sum(backbone_dims.values())
print(f"\nBackbones: {backbone_names}")
print(f"Dims: {backbone_dims}, total={total_dim}")


# ============================================================
# 2. SUPCON PROJECTION HEAD
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
    """Supervised Contrastive Loss."""
    N = features.shape[0]
    device = features.device
    sim = features @ features.T / temperature
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    self_mask = 1.0 - torch.eye(N, device=device)
    logits_max = sim.max(dim=1, keepdim=True)[0].detach()
    logits = sim - logits_max
    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1)
    mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return -mean_log_prob[valid].mean()


class PKSampler:
    def __init__(self, labels, P=32, K=4):
        self.labels = np.array(labels)
        self.P, self.K = P, K
        self.id_to_idx = defaultdict(list)
        for i, l in enumerate(self.labels):
            self.id_to_idx[l].append(i)
        min_k = min(K, 2)
        self.valid_ids = [k for k, v in self.id_to_idx.items() if len(v) >= min_k]
        self.n_batches = max(1, len(self.valid_ids) // P)

    def __iter__(self):
        rng = np.random.RandomState()
        for _ in range(self.n_batches):
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
# 3. BUILD CONCAT FEATURES
# ============================================================
def build_concat_features(image_ids):
    """Build L2-normed concatenated feature matrix for given image IDs."""
    id_to_global = {cid: i for i, cid in enumerate(comp_ids)}
    indices = [id_to_global[sid] for sid in image_ids]
    parts = []
    for bname in backbone_names:
        f = comp_features[bname][indices]
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        parts.append(f)
    return np.concatenate(parts, axis=1).astype(np.float32)


# ============================================================
# 4. TRAIN SUPCON PER SPECIES
# ============================================================
def train_supcon(species, train_ids, train_labels, all_ids):
    """Train SupCon projection for one species. Returns projected features for all."""
    print(f"\n  Training SupCon for {species}...")

    train_feats = build_concat_features(train_ids)
    all_feats = build_concat_features(all_ids)
    train_labels_arr = np.array(train_labels)

    id_counts = Counter(train_labels_arr)
    n_valid = sum(1 for c in id_counts.values() if c >= 2)
    print(f"    {len(train_ids)} imgs, {len(id_counts)} ids, {n_valid} with 2+ imgs, dim={total_dim}")

    if n_valid < 5:
        print(f"    Too few paired identities, returning None")
        return None, None

    proj = ProjectionHead(total_dim, PROJ_HIDDEN, PROJ_OUT).to(DEVICE)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=SUPCON_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SUPCON_EPOCHS)

    P = min(32, n_valid)
    K = min(4, min(id_counts[k] for k in id_counts if id_counts[k] >= 2))
    K = max(K, 2)
    sampler = PKSampler(train_labels_arr, P=P, K=K)

    feats_t = torch.tensor(train_feats, dtype=torch.float32).to(DEVICE)
    labels_t = torch.tensor(train_labels_arr, dtype=torch.long).to(DEVICE)

    best_loss = float('inf')
    best_state = None

    for epoch in range(SUPCON_EPOCHS):
        proj.train()
        epoch_loss = 0
        n_batches = 0
        for batch_idx in sampler:
            batch_feats = feats_t[batch_idx]
            batch_labels = labels_t[batch_idx]
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
    print(f"    Best loss: {best_loss:.4f}")

    # Project ALL features
    proj.eval()
    all_feats_t = torch.tensor(all_feats, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        projected_all = []
        for i in range(0, len(all_feats_t), 1024):
            p = proj(all_feats_t[i:i+1024]).cpu().numpy()
            projected_all.append(p)
        projected_all = np.concatenate(projected_all, axis=0)

    # Eval on train
    train_proj = projected_all[:len(train_ids)]
    tr_sim = train_proj @ train_proj.T
    best_ari = -1
    for th in np.arange(0.05, 0.95, 0.02):
        try:
            dist = np.clip(1 - tr_sim, 0, 2)
            np.fill_diagonal(dist, 0)
            c = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                        linkage='average', metric='precomputed').fit(dist)
            ari = adjusted_rand_score(train_labels, c.labels_)
            if ari > best_ari:
                best_ari = ari
        except:
            continue
    print(f"    SupCon projected train ARI: {best_ari:.4f}")

    proj_state = {k: v.cpu().clone() for k, v in proj.state_dict().items()}
    del proj, feats_t, labels_t, all_feats_t
    torch.cuda.empty_cache()

    return projected_all, proj_state


# ============================================================
# 5. CLUSTERING
# ============================================================
def cluster_with_search(sim_matrix, train_indices, train_labels, max_search=2000):
    """HAC with threshold grid search (subsampled for large sets)."""
    dist = np.clip(1 - sim_matrix, 0, 2)
    np.fill_diagonal(dist, 0)

    if train_indices is not None and len(train_indices) > 20 and train_labels is not None:
        # Subsample for speed
        if len(train_indices) > max_search:
            rng = np.random.RandomState(42)
            sub_idx = sorted(rng.choice(len(train_indices), max_search, replace=False).tolist())
            tr_indices = [train_indices[i] for i in sub_idx]
            tr_labels = [train_labels[i] for i in sub_idx]
        else:
            tr_indices = train_indices
            tr_labels = train_labels

        tr_dist = dist[np.ix_(tr_indices, tr_indices)]
        best_th, best_ari = 0.5, -1
        for th in np.arange(0.05, 0.95, 0.02):
            try:
                c = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                            linkage='average', metric='precomputed').fit(tr_dist)
                ari = adjusted_rand_score(tr_labels, c.labels_)
                if ari > best_ari:
                    best_ari = ari
                    best_th = th
            except:
                continue
        print(f"    Best threshold: {best_th:.3f} (train ARI={best_ari:.4f})")
    else:
        best_th = 0.5

    clust = AgglomerativeClustering(n_clusters=None, distance_threshold=best_th,
                                    linkage='average', metric='precomputed').fit(dist)
    return clust.labels_, best_th


def postprocess(labels, sim_matrix, tau_split=0.3, tau_merge=0.7, tau_weak=0.2, max_size=100):
    """4-pass post-processing."""
    labels = labels.copy()
    next_label = max(set(labels)) + 1

    # Pass 1: Split large/loose
    for cl in list(set(labels)):
        members = np.where(labels == cl)[0]
        if len(members) <= 2:
            continue
        cl_sim = sim_matrix[np.ix_(members, members)]
        intra = (cl_sim.sum() - np.trace(cl_sim)) / (len(members) * (len(members) - 1))
        if intra < tau_split or len(members) > max_size:
            cl_dist = np.clip(1 - cl_sim, 0, 2)
            np.fill_diagonal(cl_dist, 0)
            try:
                sub = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3,
                                              linkage='average', metric='precomputed').fit(cl_dist)
                for i, m in enumerate(members):
                    labels[m] = next_label + sub.labels_[i]
                next_label += sub.labels_.max() + 1
            except:
                pass

    # Pass 2: Merge tiny
    small = [cl for cl in set(labels) if np.sum(labels == cl) <= 2]
    for i, cl_i in enumerate(small):
        mi = np.where(labels == cl_i)[0]
        if len(mi) == 0:
            continue
        for cl_j in small[i+1:]:
            mj = np.where(labels == cl_j)[0]
            if len(mj) == 0:
                continue
            if sim_matrix[np.ix_(mi, mj)].max() > tau_merge:
                labels[mj] = cl_i

    # Pass 3: Transitivity
    for cl in list(set(labels)):
        members = np.where(labels == cl)[0]
        if len(members) <= 2:
            continue
        cl_sim = sim_matrix[np.ix_(members, members)]
        min_sim = cl_sim[np.triu_indices(len(members), k=1)].min() if len(members) > 1 else 1.0
        if min_sim < tau_weak:
            adj = cl_sim > tau_weak
            from scipy.sparse.csgraph import connected_components
            from scipy.sparse import csr_matrix
            n_comp, comp_labels = connected_components(csr_matrix(adj), directed=False)
            if n_comp > 1:
                for i, m in enumerate(members):
                    labels[m] = next_label + comp_labels[i]
                next_label += n_comp

    # Renumber
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels])


# ============================================================
# 6. MAIN
# ============================================================
def main():
    t0 = time.time()

    print(f"\n{'='*60}")
    print("PHASE 2: Per-Species SupCon + Clustering")
    print(f"{'='*60}")

    submission_rows = []
    species_results = {}
    saved_proj_states = {}

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

        if len(sp_train) > 0:
            identity_map = {name: idx for idx, name in enumerate(sp_train['identity'].unique())}
            train_labels = [identity_map[x] for x in sp_train['identity'].values]
        else:
            train_labels = []

        print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

        # SupCon or transfer
        if len(train_ids) > 50 and len(set(train_labels)) > 5:
            projected, proj_state = train_supcon(species, train_ids, train_labels, all_ids)

            if projected is not None:
                saved_proj_states[species] = proj_state
                sim_supcon = projected @ projected.T

                # Raw mega similarities for ensemble
                id_to_global = {cid: i for i, cid in enumerate(comp_ids)}
                sp_indices = [id_to_global[sid] for sid in all_ids]

                mega_sims = {}
                for mn in ["megadesc_l384", "megadesc_dinov2"]:
                    if mn in comp_features:
                        f = comp_features[mn][sp_indices]
                        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                        mega_sims[mn] = f @ f.T

                # Search best alpha on train (subsampled)
                train_local = list(range(len(train_ids)))
                MAX_S = 2000
                if len(train_local) > MAX_S:
                    rng = np.random.RandomState(42)
                    s_idx = sorted(rng.choice(len(train_local), MAX_S, replace=False).tolist())
                    s_labels = [train_labels[i] for i in s_idx]
                else:
                    s_idx = train_local
                    s_labels = train_labels

                best_ari, best_alpha, best_th_e = -1, 0.5, 0.5
                for alpha in np.arange(0.3, 1.01, 0.1):
                    mega_w = (1 - alpha) / 2
                    blended = alpha * sim_supcon
                    for mn in mega_sims:
                        blended = blended + mega_w * mega_sims[mn]

                    tr_sim = blended[np.ix_(s_idx, s_idx)]
                    tr_dist = np.clip(1 - tr_sim, 0, 2)
                    np.fill_diagonal(tr_dist, 0)

                    for th in np.arange(0.05, 0.90, 0.03):
                        try:
                            c = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                                        linkage='average', metric='precomputed').fit(tr_dist)
                            ari = adjusted_rand_score(s_labels, c.labels_)
                            if ari > best_ari:
                                best_ari = ari
                                best_alpha = alpha
                                best_th_e = th
                        except:
                            continue

                mega_w = (1 - best_alpha) / 2
                print(f"  Ensemble: supcon={best_alpha:.2f}, each_mega={mega_w:.2f}, th={best_th_e:.3f}, ARI={best_ari:.4f}")

                final_sim = best_alpha * sim_supcon
                for mn in mega_sims:
                    final_sim = final_sim + mega_w * mega_sims[mn]
            else:
                # SupCon failed, use raw mega only
                id_to_global = {cid: i for i, cid in enumerate(comp_ids)}
                sp_indices = [id_to_global[sid] for sid in all_ids]
                final_sim = np.zeros((len(all_ids), len(all_ids)))
                for mn in ["megadesc_l384", "megadesc_dinov2"]:
                    if mn in comp_features:
                        f = comp_features[mn][sp_indices]
                        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                        final_sim += 0.5 * (f @ f.T)
                best_th_e = 0.5
        else:
            # TexasHornedLizards: transfer
            print("  No training data, using transferred head")
            transfer_from = None
            for sp in ["SeaTurtleID2022", "SalamanderID2025", "LynxID2025"]:
                if sp in saved_proj_states:
                    transfer_from = sp
                    break

            id_to_global = {cid: i for i, cid in enumerate(comp_ids)}
            sp_indices = [id_to_global[sid] for sid in all_ids]

            if transfer_from:
                print(f"  Transferring from {transfer_from}")
                proj = ProjectionHead(total_dim, PROJ_HIDDEN, PROJ_OUT).to(DEVICE)
                proj.load_state_dict(saved_proj_states[transfer_from])
                proj.eval()

                all_feats = build_concat_features(all_ids)
                with torch.no_grad():
                    projected = []
                    for i in range(0, len(all_feats), 1024):
                        batch = torch.tensor(all_feats[i:i+1024], dtype=torch.float32).to(DEVICE)
                        projected.append(proj(batch).cpu().numpy())
                    projected = np.concatenate(projected, axis=0)

                final_sim = 0.5 * (projected @ projected.T)
                for mn in ["megadesc_l384", "megadesc_dinov2"]:
                    if mn in comp_features:
                        f = comp_features[mn][sp_indices]
                        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                        final_sim += 0.25 * (f @ f.T)
                del proj; torch.cuda.empty_cache()
            else:
                final_sim = np.zeros((len(all_ids), len(all_ids)))
                for mn in ["megadesc_l384", "megadesc_dinov2"]:
                    if mn in comp_features:
                        f = comp_features[mn][sp_indices]
                        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
                        final_sim += 0.5 * (f @ f.T)
            best_th_e = 0.5

        # Cluster
        print("  Clustering...")
        train_local = list(range(len(train_ids)))
        labels, best_th = cluster_with_search(
            final_sim,
            train_local if len(train_ids) > 20 else None,
            train_labels if len(train_labels) > 20 else None,
        )

        # Post-process
        labels = postprocess(labels, final_sim)

        # Eval
        if len(train_ids) > 20:
            train_pred = labels[:len(train_ids)]
            final_ari = adjusted_rand_score(train_labels, train_pred)
            print(f"  Final Train ARI: {final_ari:.4f}")
        else:
            final_ari = None

        test_labels = labels[len(train_ids):]
        n_clusters = len(set(test_labels))
        n_sing = sum(1 for cl in set(test_labels) if sum(test_labels == cl) == 1)
        print(f"  Test: {len(test_ids)} -> {n_clusters} clusters, {n_sing} singletons ({100*n_sing/max(1,len(test_ids)):.0f}%)")

        species_results[species] = {"ari": final_ari, "clusters": n_clusters, "singletons": n_sing}

        for i, img_id in enumerate(test_ids):
            submission_rows.append({"image_id": int(img_id), "cluster": f"cluster_{species}_{test_labels[i]}"})

    # Build submission
    submission = pd.DataFrame(submission_rows)
    submission = submission.set_index('image_id').loc[sample_sub['image_id'].values].reset_index()
    submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"V23 COMPLETE - {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    for sp, r in species_results.items():
        ari_str = f"{r['ari']:.4f}" if r['ari'] is not None else "N/A"
        print(f"  {sp}: ARI={ari_str}, clusters={r['clusters']}, singletons={r['singletons']}")
    print(f"\nSubmission: {OUTPUT_DIR}/submission.csv ({len(submission)} rows)")


if __name__ == "__main__":
    main()
