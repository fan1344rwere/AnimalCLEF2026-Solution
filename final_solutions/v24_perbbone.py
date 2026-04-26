#!/usr/bin/env python3
"""
V24 — Per-backbone SupCon + Wildlife (exactly what V22 should have been)
========================================================================
Per backbone × per species:
  - Non-mega (dinov3/siglip2/eva02): train SupCon on wildlife + competition train
  - Mega (megadesc_l384/megadesc_dinov2): train SupCon on competition train only
Then fuse 5 similarity matrices → HAC clustering
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict, Counter
warnings.filterwarnings("ignore")

BASE = "/root/autodl-tmp/\u6700\u7ec8\u5723\u6218"
DATA_DIR = f"{BASE}/animalclef/animal-clef-2026"
OUTPUT_DIR = f"{BASE}/animalclef/ov24"
FEAT_CACHE = f"{BASE}/animalclef/feat_cache"
WILD_FEAT_CACHE = f"{BASE}/animalclef/feat_cache_wildlife"
WILDLIFE_META = f"{BASE}/wildlifereid10k_full/metadata.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]
ALL_BACKBONES = ["dinov3", "siglip2", "eva02", "megadesc_l384", "megadesc_dinov2"]
WILD_BACKBONES = ["dinov3", "siglip2", "eva02"]
SUPCON_TEMP = 0.07
SUPCON_EPOCHS = 50
SUPCON_LR = 5e-4
PROJ_OUT = 256
MAX_WILD = 15000  # subsample wildlife per training run

# ============================================================
# Load data
# ============================================================
print("=" * 60)
print("Loading features")
print("=" * 60)
meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
comp_ids = [str(x) for x in meta['image_id'].values]

comp_features = {}
for b in ALL_BACKBONES:
    p = os.path.join(FEAT_CACHE, f"{b}_features.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        feats = d['feats'].astype(np.float32)
        ids = [str(x) for x in d['ids']]
        idx_map = {str(i): j for j, i in enumerate(ids)}
        ordered = np.zeros((len(comp_ids), feats.shape[1]), dtype=np.float32)
        for ci, cid in enumerate(comp_ids):
            if cid in idx_map:
                ordered[ci] = feats[idx_map[cid]]
        comp_features[b] = ordered
        print(f"  comp {b}: {ordered.shape}")

wild_features = {}
for b in WILD_BACKBONES:
    p = os.path.join(WILD_FEAT_CACHE, f"{b}_features.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        wild_features[b] = d['feats'].astype(np.float32)
        print(f"  wild {b}: {wild_features[b].shape}")

wildlife_df = pd.read_csv(WILDLIFE_META)
wild_identities = wildlife_df['identity'].values
wild_unique_ids = sorted(set(wild_identities))
wild_id_to_label = {id_: i for i, id_ in enumerate(wild_unique_ids)}
wild_labels_all = np.array([wild_id_to_label[x] for x in wild_identities])
print(f"Wildlife: {len(wildlife_df)} imgs, {len(wild_unique_ids)} identities")

# ============================================================
# SupCon components
# ============================================================
class ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        h = min(1024, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h, out_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def supcon_loss(features, labels, temperature=0.07):
    N = features.shape[0]
    device = features.device
    sim = features @ features.T / temperature
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    self_mask = 1.0 - torch.eye(N, device=device)
    mx = sim.max(dim=1, keepdim=True)[0].detach()
    logits = sim - mx
    exp_l = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_l.sum(dim=1, keepdim=True) + 1e-8)
    pc = pos_mask.sum(dim=1)
    mlp = (pos_mask * log_prob).sum(dim=1) / (pc + 1e-8)
    valid = pc > 0
    return -mlp[valid].mean() if valid.sum() > 0 else torch.tensor(0.0, device=device, requires_grad=True)

class PKSampler:
    def __init__(self, labels, P=32, K=4):
        self.P, self.K = P, K
        self.id_to_idx = defaultdict(list)
        for i, l in enumerate(labels):
            self.id_to_idx[l].append(i)
        self.valid_ids = [k for k, v in self.id_to_idx.items() if len(v) >= 2]
        self.n_batches = max(1, len(self.valid_ids) // P)
    def __iter__(self):
        rng = np.random.RandomState()
        for _ in range(self.n_batches):
            chosen = rng.choice(self.valid_ids, min(self.P, len(self.valid_ids)), replace=False)
            batch = []
            for cid in chosen:
                pool = self.id_to_idx[cid]
                if len(pool) >= self.K:
                    batch.extend(rng.choice(pool, self.K, replace=False).tolist())
                else:
                    batch.extend(rng.choice(pool, self.K, replace=True).tolist())
            yield batch
    def __len__(self):
        return self.n_batches

def train_proj(feats, labels, in_dim, epochs=50, lr=5e-4):
    """Train a single projection head with SupCon."""
    label_arr = np.array(labels)
    id_counts = Counter(label_arr)
    n_valid = sum(1 for c in id_counts.values() if c >= 2)
    if n_valid < 5:
        return None

    proj = ProjHead(in_dim, PROJ_OUT).to(DEVICE)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    P = min(32, n_valid)
    K = min(4, min(c for c in id_counts.values() if c >= 2))
    K = max(K, 2)
    sampler = PKSampler(label_arr, P=P, K=K)

    feats_t = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
    labels_t = torch.tensor(label_arr, dtype=torch.long).to(DEVICE)

    best_loss, best_state = float('inf'), None
    for epoch in range(epochs):
        proj.train()
        losses = []
        for batch_idx in sampler:
            z = proj(feats_t[batch_idx])
            loss = supcon_loss(z, labels_t[batch_idx], SUPCON_TEMP)
            if loss.item() > 0:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())
        sched.step()
        avg = np.mean(losses) if losses else float('inf')
        if avg < best_loss and avg > 0:
            best_loss = avg
            best_state = {k: v.clone() for k, v in proj.state_dict().items()}
        if (epoch+1) % 10 == 0:
            print(f"      Ep {epoch+1}/{epochs}: loss={avg:.4f} (best={best_loss:.4f})")

    if best_state:
        proj.load_state_dict(best_state)
    del feats_t, labels_t; torch.cuda.empty_cache()
    return proj

def project_features(proj, feats):
    """Project features through trained head."""
    proj.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), 2048):
            batch = torch.tensor(feats[i:i+2048], dtype=torch.float32).to(DEVICE)
            out.append(proj(batch).cpu().numpy())
    return np.concatenate(out, axis=0)

# ============================================================
# Main pipeline
# ============================================================
def main():
    t0 = time.time()
    print(f"\n{'='*60}")
    print("Per-backbone, per-species SupCon + Wildlife")
    print(f"{'='*60}")

    submission_rows = []
    species_results = {}
    saved_proj_states = {}  # {(species, backbone): proj_state}

    for species in SPECIES_ORDER:
        print(f"\n{'='*50}")
        print(f"{species}")
        print(f"{'='*50}")

        sp_meta = meta[meta['dataset'] == species]
        sp_train = sp_meta[sp_meta['split'] == 'train']
        sp_test = sp_meta[sp_meta['split'] == 'test']
        train_ids = [str(x) for x in sp_train['image_id'].values]
        test_ids = [str(x) for x in sp_test['image_id'].values]
        all_ids = train_ids + test_ids

        id_to_global = {cid: i for i, cid in enumerate(comp_ids)}
        sp_indices = [id_to_global[sid] for sid in all_ids]
        train_local = list(range(len(train_ids)))

        if len(sp_train) > 0:
            id_map = {name: idx for idx, name in enumerate(sp_train['identity'].unique())}
            train_labels = [id_map[x] for x in sp_train['identity'].values]
        else:
            train_labels = []

        print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

        # Per-backbone: train SupCon, get similarity
        backbone_sims = {}

        for bname in ALL_BACKBONES:
            if bname not in comp_features:
                continue
            dim = comp_features[bname].shape[1]

            # Get competition train features (L2-normed)
            comp_train_f = comp_features[bname][[id_to_global[tid] for tid in train_ids]]
            comp_train_f = comp_train_f / (np.linalg.norm(comp_train_f, axis=1, keepdims=True) + 1e-8)

            if len(train_ids) > 50 and len(set(train_labels)) > 5:
                # Combine with wildlife for non-mega backbones
                if bname in wild_features and bname in WILD_BACKBONES:
                    wf = wild_features[bname]
                    wf = wf / (np.linalg.norm(wf, axis=1, keepdims=True) + 1e-8)
                    # Subsample wildlife
                    if len(wf) > MAX_WILD:
                        rng = np.random.RandomState(42)
                        idx = rng.choice(len(wf), MAX_WILD, replace=False)
                        wf = wf[idx]
                        wl = wild_labels_all[idx]
                    else:
                        wl = wild_labels_all
                    # Offset wildlife labels to avoid collision
                    max_comp_label = max(train_labels) + 1
                    wl_offset = wl + max_comp_label

                    all_feats = np.concatenate([comp_train_f, wf], axis=0).astype(np.float32)
                    all_labels = np.concatenate([train_labels, wl_offset])
                    print(f"    {bname} (dim={dim}): comp={len(comp_train_f)} + wild={len(wf)} = {len(all_feats)}")
                else:
                    all_feats = comp_train_f.astype(np.float32)
                    all_labels = np.array(train_labels)
                    print(f"    {bname} (dim={dim}): comp={len(comp_train_f)} only")

                proj = train_proj(all_feats, all_labels, dim, epochs=SUPCON_EPOCHS, lr=SUPCON_LR)

                if proj is not None:
                    saved_proj_states[(species, bname)] = {k: v.cpu().clone() for k, v in proj.state_dict().items()}
                    # Project all species images
                    sp_feats = comp_features[bname][sp_indices]
                    sp_feats = sp_feats / (np.linalg.norm(sp_feats, axis=1, keepdims=True) + 1e-8)
                    projected = project_features(proj, sp_feats.astype(np.float32))
                    backbone_sims[bname] = projected @ projected.T
                    del proj; torch.cuda.empty_cache()
                else:
                    # Fallback: raw cosine sim
                    sp_feats = comp_features[bname][sp_indices]
                    sp_feats = sp_feats / (np.linalg.norm(sp_feats, axis=1, keepdims=True) + 1e-8)
                    backbone_sims[bname] = sp_feats @ sp_feats.T
            else:
                # No training data (Texas) → transfer or raw
                transferred = False
                for donor in ["SeaTurtleID2022", "SalamanderID2025", "LynxID2025"]:
                    key = (donor, bname)
                    if key in saved_proj_states:
                        proj = ProjHead(dim, PROJ_OUT).to(DEVICE)
                        proj.load_state_dict(saved_proj_states[key])
                        sp_feats = comp_features[bname][sp_indices]
                        sp_feats = sp_feats / (np.linalg.norm(sp_feats, axis=1, keepdims=True) + 1e-8)
                        projected = project_features(proj, sp_feats.astype(np.float32))
                        backbone_sims[bname] = projected @ projected.T
                        del proj; torch.cuda.empty_cache()
                        transferred = True
                        break
                if not transferred:
                    sp_feats = comp_features[bname][sp_indices]
                    sp_feats = sp_feats / (np.linalg.norm(sp_feats, axis=1, keepdims=True) + 1e-8)
                    backbone_sims[bname] = sp_feats @ sp_feats.T

        # Fuse similarities with weight search
        bnames_avail = list(backbone_sims.keys())
        n_b = len(bnames_avail)
        print(f"  Fusing {n_b} backbone similarities...")

        if len(train_ids) > 20:
            # Subsample for speed
            MAX_S = 2000
            if len(train_local) > MAX_S:
                rng = np.random.RandomState(42)
                s_idx = sorted(rng.choice(len(train_local), MAX_S, replace=False).tolist())
                s_labels = [train_labels[i] for i in s_idx]
            else:
                s_idx = train_local
                s_labels = train_labels

            best_ari, best_weights, best_th = -1, [1.0/n_b]*n_b, 0.5

            # Equal weight baseline
            fused_eq = sum(backbone_sims[b] for b in bnames_avail) / n_b
            tr_d = np.clip(1 - fused_eq[np.ix_(s_idx, s_idx)], 0, 2)
            np.fill_diagonal(tr_d, 0)
            for th in np.arange(0.05, 0.90, 0.02):
                try:
                    c = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                                linkage='average', metric='precomputed').fit(tr_d)
                    ari = adjusted_rand_score(s_labels, c.labels_)
                    if ari > best_ari:
                        best_ari = ari; best_weights = [1.0/n_b]*n_b; best_th = th
                except: pass

            # Try mega-heavy weights
            for mega_w in [0.3, 0.5, 0.7]:
                non_mega_w = (1.0 - mega_w * 2) / max(1, n_b - 2)
                if non_mega_w < 0: continue
                weights = []
                for b in bnames_avail:
                    if 'mega' in b:
                        weights.append(mega_w)
                    else:
                        weights.append(non_mega_w)
                # Normalize
                ws = sum(weights)
                weights = [w/ws for w in weights]

                fused = sum(w * backbone_sims[b] for w, b in zip(weights, bnames_avail))
                tr_d = np.clip(1 - fused[np.ix_(s_idx, s_idx)], 0, 2)
                np.fill_diagonal(tr_d, 0)
                for th in np.arange(0.05, 0.90, 0.02):
                    try:
                        c = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                                    linkage='average', metric='precomputed').fit(tr_d)
                        ari = adjusted_rand_score(s_labels, c.labels_)
                        if ari > best_ari:
                            best_ari = ari; best_weights = weights; best_th = th
                    except: pass

            print(f"  Best ARI={best_ari:.4f}, th={best_th:.3f}")
            print(f"  Weights: {dict(zip(bnames_avail, [f'{w:.3f}' for w in best_weights]))}")
            final_sim = sum(w * backbone_sims[b] for w, b in zip(best_weights, bnames_avail))
        else:
            final_sim = sum(backbone_sims[b] for b in bnames_avail) / n_b
            best_th = 0.5

        # Cluster
        dist = np.clip(1 - final_sim, 0, 2)
        np.fill_diagonal(dist, 0)

        if len(train_ids) > 20:
            # Fine-tune threshold on full train
            tr_d = dist[np.ix_(train_local, train_local)]
            bt, ba = best_th, -1
            for th in np.arange(max(0.05, best_th-0.1), min(0.95, best_th+0.1), 0.01):
                try:
                    c = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                                linkage='average', metric='precomputed').fit(tr_d)
                    ari = adjusted_rand_score(train_labels, c.labels_)
                    if ari > ba: ba = ari; bt = th
                except: pass
            best_th = bt
            print(f"  Refined threshold: {best_th:.3f} (ARI={ba:.4f})")

        clust = AgglomerativeClustering(n_clusters=None, distance_threshold=best_th,
                                        linkage='average', metric='precomputed').fit(dist)
        labels = clust.labels_

        # Eval
        if len(train_ids) > 20:
            ari = adjusted_rand_score(train_labels, labels[:len(train_ids)])
            print(f"  Final Train ARI: {ari:.4f}")
        else:
            ari = None

        test_labels = labels[len(train_ids):]
        nc = len(set(test_labels))
        ns = sum(1 for cl in set(test_labels) if sum(test_labels == cl) == 1)
        print(f"  Test: {len(test_ids)} -> {nc} clusters, {ns} singletons")

        species_results[species] = {"ari": ari, "clusters": nc}
        for i, img_id in enumerate(test_ids):
            submission_rows.append({"image_id": int(img_id), "cluster": f"cluster_{species}_{test_labels[i]}"})

    # Submission
    sub = pd.DataFrame(submission_rows)
    sub = sub.set_index('image_id').loc[sample_sub['image_id'].values].reset_index()
    sub.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"V24 COMPLETE - {elapsed/60:.1f} min")
    for sp, r in species_results.items():
        a = f"{r['ari']:.4f}" if r['ari'] else "N/A"
        print(f"  {sp}: ARI={a}, clusters={r['clusters']}")
    print(f"Submission: {OUTPUT_DIR}/submission.csv ({len(sub)} rows)")

if __name__ == "__main__":
    main()
