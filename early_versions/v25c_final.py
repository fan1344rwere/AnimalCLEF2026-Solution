#!/usr/bin/env python3
"""
AnimalCLEF2026 V25 — Hybrid: HAC for 3 species + Visual-guided for TexasHorned
================================================================================
Key lessons from V22-V24:
  - V22 (HAC, 0.24012) > V24 (t-SNE+HDBSCAN, 0.14697) => HAC is better for this task
  - SupCon Phase A works well (Lynx 0.58, Salamander 0.61, SeaTurtle 0.97)
  - Pseudo-labels destroy quality => disabled
  - t-SNE+HDBSCAN underperforms HAC => go back to HAC with threshold grid search
  - TexasHorned needs visual constraints, not pure unsupervised

Strategy:
  - Lynx/Salamander/SeaTurtle: SupCon + HAC + cosine threshold grid search (like V22)
  - TexasHorned: DINOv3 features + visual constraints (consecutive-ID pairs, spot density)
"""

import os, sys, gc, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict, Counter
from PIL import Image
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/root/autodl-tmp/animal-clef-2026"
MODEL_DIR = "/root/autodl-tmp/models"
OUTPUT_DIR = "/root/autodl-tmp/ov25c"
FEAT_CACHE = "/root/autodl-tmp/feat_cache_v23"  # reuse cached features
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory/1024**3:.1f}GB")

SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# SupCon hyperparams — same as V24 Phase A (proven to work)
PROJ_HIDDEN = 1024
PROJ_OUT = 256
SUPCON_TEMP = 0.1
SUPCON_LR = 2e-4
SUPCON_WD = 1e-3
SUPCON_EPOCHS = 30
DROPOUT_RATE = 0.3

# HAC threshold search range (per species, based on V16/V22 experience)
HAC_THRESHOLD_RANGE = {
    "LynxID2025": np.arange(0.30, 0.80, 0.02),
    "SalamanderID2025": np.arange(0.05, 0.50, 0.02),
    "SeaTurtleID2022": np.arange(0.30, 0.80, 0.02),
}

# ============================================================
# LOAD CACHED FEATURES
# ============================================================
def load_cached_features(meta):
    """Load pre-extracted features from V23/V24 cache."""
    all_ids = [str(x) for x in meta['image_id'].values]
    features = {}
    for name in ["dinov3", "siglip2", "megadesc"]:
        cache_path = os.path.join(FEAT_CACHE, f"{name}_features.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            features[name] = np.array(data['feats'], dtype=np.float32)
            print(f"  {name}: {features[name].shape}")
        else:
            print(f"  WARNING: {name} cache not found!")
    return features, all_ids


# ============================================================
# PROJECTION HEAD + SUPCON (same as V24)
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
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        N = features.shape[0]
        sim = features @ features.T / self.temperature
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        self_mask = 1.0 - torch.eye(N, device=device)
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
    def __init__(self, labels, P=16, K=4):
        self.P, self.K = P, K
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.valid_labels = [l for l, idxs in self.label_to_indices.items() if len(idxs) >= 2]
        if not self.valid_labels:
            self.valid_labels = list(self.label_to_indices.keys())

    def sample(self):
        selected = np.random.choice(self.valid_labels,
                                    size=min(self.P, len(self.valid_labels)), replace=False)
        indices = []
        for label in selected:
            pool = self.label_to_indices[label]
            chosen = np.random.choice(pool, size=self.K,
                                      replace=len(pool) < self.K)
            indices.extend(chosen)
        return indices


def train_supcon(features, labels, in_dim, n_epochs=30):
    """Train SupCon projection head. Returns (projected_features, projection_model)."""
    n_ids = len(set(labels))
    proj = ProjectionHead(in_dim, PROJ_HIDDEN, PROJ_OUT, DROPOUT_RATE).to(DEVICE)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=SUPCON_LR, weight_decay=SUPCON_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = SupConLoss(temperature=SUPCON_TEMP)
    sampler = PKSampler(labels, P=min(16, n_ids), K=4)

    features_t = torch.from_numpy(features).float().to(DEVICE)
    labels_t = torch.from_numpy(labels).long().to(DEVICE)

    for epoch in range(n_epochs):
        proj.train()
        losses = []
        for _ in range(max(len(features) // (16 * 4), 10)):
            idx = sampler.sample()
            z = proj(features_t[idx])
            loss = criterion(z, labels_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={np.mean(losses):.4f}")

    # Project all features
    proj.eval()
    with torch.no_grad():
        projected = proj(features_t).cpu().numpy()
    return projected, proj  # Return BOTH projected features AND the model


# ============================================================
# HAC WITH THRESHOLD GRID SEARCH (V22-style, proven to work)
# ============================================================
def hac_cluster(sim_matrix, threshold):
    """Agglomerative clustering with cosine distance threshold."""
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.maximum(dist_matrix, 0)
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)

    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=threshold, criterion='distance')
    return labels - 1  # 0-indexed


def grid_search_threshold(sim_matrix, true_labels, thresholds):
    """Find optimal HAC threshold on training data."""
    best_ari = -1
    best_th = thresholds[0]
    for th in thresholds:
        pred = hac_cluster(sim_matrix, th)
        ari = adjusted_rand_score(true_labels, pred)
        if ari > best_ari:
            best_ari = ari
            best_th = th
    return best_th, best_ari


# ============================================================
# TEXAS HORNED LIZARDS: VISUAL-GUIDED CLUSTERING
# ============================================================
def cluster_texas_horned(all_concat_feats, feats_dinov3, image_ids):
    """
    Cluster TexasHornedLizards using concatenated features + visual constraints.

    Key fix: DINOv3 alone has too-high similarity between all images (same species).
    Using MegaDescriptor component helps (has Re-ID signal).
    """
    print("  Visual-guided clustering for TexasHornedLizards")
    n = len(all_concat_feats)

    # Use concatenated features (includes MegaDesc Re-ID signal)
    feats_norm = all_concat_feats / (np.linalg.norm(all_concat_feats, axis=1, keepdims=True) + 1e-8)
    sim = feats_norm @ feats_norm.T

    # Visual constraint: consecutive image IDs likely same individual
    id_array = np.array(image_ids)
    for i in range(n):
        for j in range(i+1, min(i+3, n)):
            if abs(int(id_array[j]) - int(id_array[i])) <= 2:
                old_sim = sim[i, j]
                boosted = min(1.0, old_sim * 1.2 + 0.10)
                sim[i, j] = boosted
                sim[j, i] = boosted

    # Find threshold giving 40-80 clusters (visual estimate)
    # Search from HIGH threshold down (more splitting first)
    best_th = None
    for th in np.arange(0.95, 0.10, -0.01):
        pred = hac_cluster(sim, th)
        n_cl = len(set(pred))
        if 40 <= n_cl <= 80:
            best_th = th
            break

    if best_th is None:
        # Fallback: just pick threshold giving closest to 60 clusters
        results = []
        for th in np.arange(0.10, 0.99, 0.01):
            pred = hac_cluster(sim, th)
            n_cl = len(set(pred))
            results.append((th, n_cl, abs(n_cl - 60)))
        results.sort(key=lambda x: x[2])
        best_th = results[0][0]

    labels = hac_cluster(sim, best_th)
    n_clusters = len(set(labels))
    print(f"    Threshold: {best_th:.2f}, Clusters: {n_clusters}")
    return labels


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("AnimalCLEF2026 V25: HAC + Visual-guided TexasHorned")
    print("=" * 70)
    t_start = time.time()

    meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    print(f"Metadata: {len(meta)} rows, Submission: {len(sample_sub)} rows")

    # Load cached features
    print("\nLoading cached features...")
    features, all_ids = load_cached_features(meta)
    backbone_names = sorted(features.keys())
    concat_feats = np.concatenate([features[name] for name in backbone_names], axis=1)
    total_dim = concat_feats.shape[1]
    print(f"Concatenated: {total_dim} dim from {backbone_names}")

    submission_dict = {}

    for species in SPECIES_ORDER:
        print(f"\n{'='*60}")
        print(f"Species: {species}")
        print(f"{'='*60}")

        sp_mask = meta['dataset'] == species
        sp_meta = meta[sp_mask].reset_index(drop=True)
        sp_indices = np.where(sp_mask.values)[0]
        sp_feats = concat_feats[sp_indices]

        train_mask = sp_meta['split'] == 'train'
        test_mask = sp_meta['split'] == 'test'
        train_indices = np.where(train_mask.values)[0]
        test_indices = np.where(test_mask.values)[0]
        n_train = train_mask.sum()
        n_test = test_mask.sum()
        print(f"  Train: {n_train}, Test: {n_test}")

        if species == "TexasHornedLizards":
            # --- Visual-guided clustering ---
            dinov3_feats = features["dinov3"][sp_indices]
            test_image_ids = sp_meta['image_id'].values
            test_labels = cluster_texas_horned(sp_feats, dinov3_feats, test_image_ids)

        elif n_train > 0:
            # --- SupCon + HAC with threshold grid search ---
            train_feats = sp_feats[train_indices]
            test_feats = sp_feats[test_indices]
            all_sp_feats = sp_feats

            # Encode labels
            train_identities = sp_meta.loc[train_mask, 'identity'].values
            unique_ids = sorted(set(train_identities))
            id_to_label = {id_: i for i, id_ in enumerate(unique_ids)}
            train_labels = np.array([id_to_label[x] for x in train_identities])
            print(f"  Identities: {len(unique_ids)}")

            # SupCon projection — train ONCE, use for both grid search and final
            print("  Training SupCon projection...")
            projected_train, proj_model = train_supcon(train_feats, train_labels, total_dim, SUPCON_EPOCHS)

            # Cosine similarity on projected training features
            sim_train = projected_train @ projected_train.T

            # Grid search threshold
            thresholds = HAC_THRESHOLD_RANGE[species]
            best_th, best_ari = grid_search_threshold(sim_train, train_labels, thresholds)
            print(f"  Best threshold: {best_th:.2f}, Train ARI: {best_ari:.4f}")

            # Project ALL features using the SAME model (no retraining!)
            proj_model.eval()
            with torch.no_grad():
                all_proj = proj_model(torch.from_numpy(all_sp_feats).float().to(DEVICE)).cpu().numpy()
            del proj_model; gc.collect(); torch.cuda.empty_cache()

            # Verify on train — use train_indices NOT [:n_train]!
            train_proj = all_proj[train_indices]
            sim_train2 = train_proj @ train_proj.T
            _, train_ari2 = grid_search_threshold(sim_train2, train_labels, [best_th])
            print(f"  Train ARI (verify): {train_ari2:.4f}")

            # Cluster test data — use test_indices NOT [n_train:]!
            test_proj = all_proj[test_indices]
            sim_test = test_proj @ test_proj.T
            test_labels = hac_cluster(sim_test, best_th)
            test_image_ids = sp_meta.loc[test_mask, 'image_id'].values

        else:
            continue

        # Map to submission
        if species != "TexasHornedLizards":
            test_image_ids = sp_meta.loc[test_mask, 'image_id'].values
        else:
            test_image_ids = sp_meta['image_id'].values

        unique_clusters = sorted(set(test_labels))
        cluster_map = {c: f"cluster_{species}_{i}" for i, c in enumerate(unique_clusters)}
        for img_id, cl in zip(test_image_ids, test_labels):
            submission_dict[str(img_id)] = cluster_map[cl]

        # Stats
        n_clusters = len(unique_clusters)
        counts = Counter(test_labels)
        singletons = sum(1 for v in counts.values() if v == 1)
        max_cl = max(counts.values()) if counts else 0
        print(f"  Final: {len(test_image_ids)} imgs → {n_clusters} clusters, "
              f"singletons={singletons} ({singletons/max(n_clusters,1)*100:.0f}%), max={max_cl}")

    # Generate submission
    print(f"\n{'='*60}")
    print("Generating Submission")
    print(f"{'='*60}")
    sub_ids = [str(x) for x in sample_sub['image_id'].values]
    sub_clusters = []
    missing = 0
    for img_id in sub_ids:
        if img_id in submission_dict:
            sub_clusters.append(submission_dict[img_id])
        else:
            sub_clusters.append(f"cluster_unknown_{missing}")
            missing += 1

    submission = pd.DataFrame({'image_id': sample_sub['image_id'].values, 'cluster': sub_clusters})
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Total: {len(submission)} images, {submission['cluster'].nunique()} clusters")
    if missing > 0:
        print(f"WARNING: {missing} missing images!")

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
