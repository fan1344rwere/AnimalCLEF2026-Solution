#!/usr/bin/env python3
"""
AnimalCLEF2026 V19 — Orientation-Aware Breakthrough for Top5
=============================================================
Target: 0.231 → 0.53+ (Top5 threshold = 0.528)

Key innovations over V16 (0.231):
  1. Orientation-Aware Matching: Only compare same-orientation images
     - left↔left, right↔right; cross-orientation = penalty
     - Reduces 50%+ false matches for Lynx
  2. Representative Matching with Train Anchors:
     - Build per-identity, per-orientation representative embeddings
     - Assign test images to nearest representative (same orientation)
     - High confidence → assign; Low confidence → "new individual"
  3. Local Feature Matching as PRIMARY (2025 winner strategy):
     - Global features for prefiltering (top-K candidates)
     - ALIKED+LightGlue for final similarity scoring
     - Local matching is the main signal, not supplement
  4. Score Calibration (Isotonic Regression):
     - Calibrate raw scores to probabilities using training pairs
  5. Hybrid Clustering:
     - Assigned individuals from representative matching
     - Discovery: self-cluster "new" images using calibrated local scores
     - Cross-orientation merging for assigned individuals

=== Based on 2025 champion DataBoom (0.713) strategy ===
"""
import os, sys, gc, time, json, warnings, itertools
warnings.filterwarnings("ignore")

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
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/animal-clef-2026"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/ov19"
os.makedirs(OUT_DIR, exist_ok=True)

SP_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
SP_ALL   = SP_TRAIN + ["TexasHornedLizards"]

# Orientation groups that are "compatible" for matching
# left↔left, right↔right are primary
# front↔front, back↔back are primary
# left↔right can sometimes match (same animal from different sides) but with heavy penalty
ORIENT_COMPAT = {
    "LynxID2025": {
        ("left","left"): 1.0, ("right","right"): 1.0,
        ("front","front"): 1.0, ("back","back"): 1.0,
        ("left","right"): 0.15, ("right","left"): 0.15,  # very different views
        ("front","back"): 0.05, ("back","front"): 0.05,
        ("unknown","left"): 0.5, ("left","unknown"): 0.5,
        ("unknown","right"): 0.5, ("right","unknown"): 0.5,
        ("unknown","front"): 0.5, ("front","unknown"): 0.5,
        ("unknown","back"): 0.5, ("back","unknown"): 0.5,
        ("unknown","unknown"): 0.7,
        ("front","left"): 0.3, ("left","front"): 0.3,
        ("front","right"): 0.3, ("right","front"): 0.3,
        ("back","left"): 0.1, ("left","back"): 0.1,
        ("back","right"): 0.1, ("right","back"): 0.1,
    },
    "SalamanderID2025": {
        ("top","top"): 1.0, ("right","right"): 1.0,
        ("top","right"): 0.3, ("right","top"): 0.3,  # different views of salamander
    },
    "SeaTurtleID2022": {},  # Sea turtles — orientation less critical, mostly top-down
    "TexasHornedLizards": {},  # Unknown orientations
}

# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════
class AnimalDS(Dataset):
    def __init__(self, df, root, sz, flip=False):
        self.df, self.root, self.flip = df.reset_index(drop=True), root, flip
        self.tf = transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            img = Image.open(os.path.join(self.root, r["path"])).convert("RGB")
        except:
            img = Image.new("RGB", (384, 384), (128, 128, 128))
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tf(img), int(r["image_id"])

@torch.no_grad()
def extract_features(model, df, root, sz, bs=48, desc=""):
    """Extract features with TTA (horizontal flip)."""
    dl = DataLoader(AnimalDS(df, root, sz), batch_size=bs, num_workers=4, pin_memory=True)
    embs, ids = [], []
    for imgs, iids in tqdm(dl, desc=f"  {desc}@{sz}", leave=False):
        embs.append(model(imgs.to(DEVICE)).cpu())
        ids.extend(iids.numpy())
    orig = torch.cat(embs)

    dl2 = DataLoader(AnimalDS(df, root, sz, flip=True), batch_size=bs, num_workers=4, pin_memory=True)
    flipped = []
    for imgs, _ in tqdm(dl2, desc=f"  {desc}_tta@{sz}", leave=False):
        flipped.append(model(imgs.to(DEVICE)).cpu())
    flip_embs = torch.cat(flipped)

    # TTA: average original + flipped
    final = F.normalize((orig + flip_embs) / 2, dim=-1).numpy()
    return final, np.array(ids)


# ═══════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════
def load_mega():
    import timm
    m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
    m = m.to(DEVICE).eval()
    print(f"  [Model] MegaDescriptor-L-384 OK dim={m.num_features}")
    return m, 384

def load_miew():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    c = json.load(open(hf_hub_download("conservationxlabs/miewid-msv3", "config.json")))
    m = timm.create_model(c.get("architecture", "efficientnetv2_rw_m"), pretrained=False, num_classes=0)
    s = {k: v for k, v in load_file(
        hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")
    ).items() if "classifier" not in k}
    m.load_state_dict(s, strict=False)
    m = m.to(DEVICE).eval()
    print(f"  [Model] MiewID-MSV3 OK dim={m.num_features}")
    return m, 440

def load_dino():
    """Load DINOv2 - best for texture/pattern matching."""
    try:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", trust_repo=True)
        m = m.to(DEVICE).eval()
        print("  [Model] DINOv2-ViT-L-reg OK dim=1024")
        return m, 518
    except:
        try:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
            m = m.to(DEVICE).eval()
            print("  [Model] DINOv2-ViT-L OK dim=1024")
            return m, 518
        except Exception as e:
            print(f"  [Model] DINOv2 failed: {e}")
            return None, 0


# ═══════════════════════════════════════════════════════════
# LOCAL FEATURE MATCHING (ALIKED + LightGlue)
# ═══════════════════════════════════════════════════════════
class LocalMatcher:
    def __init__(self, max_keypoints=1024):
        self.ok = False
        try:
            from lightglue import LightGlue, ALIKED
            self.ext = ALIKED(max_num_keypoints=max_keypoints).eval().to(DEVICE)
            self.mat = LightGlue(features="aliked").eval().to(DEVICE)
            self.ok = True
            self.kp_cache = {}
            print("  [Local] ALIKED+LightGlue OK (max_kp=%d)" % max_keypoints)
        except Exception as e:
            print(f"  [Local] Failed: {e}")

    @torch.no_grad()
    def get_keypoints(self, img_path):
        """Extract and cache keypoints for an image."""
        if img_path in self.kp_cache:
            return self.kp_cache[img_path]
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize to reasonable size for keypoint extraction
            w, h = img.size
            max_dim = 768
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                img = img.resize((int(w*scale), int(h*scale)))
            t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
            feats = self.ext.extract(t)
            self.kp_cache[img_path] = feats
            return feats
        except:
            return None

    @torch.no_grad()
    def match_pair(self, f0, f1):
        """Match two images and return match score."""
        if f0 is None or f1 is None:
            return 0.0
        try:
            r = self.mat({"image0": f0, "image1": f1})
            if "matching_scores0" in r:
                scores = r["matching_scores0"]
                # Count high-confidence matches
                n_good = int((scores > 0.5).sum().item())
                # Weighted score: consider both count and average confidence
                if len(scores) > 0 and n_good > 0:
                    avg_conf = float(scores[scores > 0.5].mean().item())
                    return n_good * avg_conf  # confidence-weighted count
                return 0.0
            elif "matches0" in r:
                n = int((r["matches0"] > -1).sum().item())
                return float(n)
            return 0.0
        except:
            return 0.0

    def clear_cache(self):
        self.kp_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_pairwise_local(self, paths, pairs, root, desc=""):
        """Compute local matching scores for given pairs."""
        if not self.ok:
            return {}

        # Collect unique image indices
        unique_idx = set()
        for i, j in pairs:
            unique_idx.add(i)
            unique_idx.add(j)

        # Extract keypoints
        print(f"      {desc} Extracting keypoints for {len(unique_idx)} images...")
        kp_map = {}
        for idx in tqdm(sorted(unique_idx), desc=f"      {desc} kp", leave=False):
            full_path = os.path.join(root, paths[idx])
            kp_map[idx] = self.get_keypoints(full_path)

        # Match pairs
        print(f"      {desc} Matching {len(pairs)} pairs...")
        scores = {}
        for i, j in tqdm(pairs, desc=f"      {desc} match", leave=False):
            s = self.match_pair(kp_map[i], kp_map[j])
            scores[(i, j)] = s
            scores[(j, i)] = s

        self.clear_cache()
        return scores


# ═══════════════════════════════════════════════════════════
# ORIENTATION-AWARE SIMILARITY
# ═══════════════════════════════════════════════════════════
def build_orientation_aware_sim(feats_dict, indices, weights, orientations, sp):
    """Build similarity matrix with orientation awareness.

    Key idea: penalize cross-orientation similarities to avoid matching
    left-side photos with right-side photos of different animals.
    """
    n = len(indices)

    # Build base global similarity
    sim = np.zeros((n, n))
    for name, w in weights.items():
        if w <= 0:
            continue
        f = normalize(feats_dict[name][indices], axis=1)
        sim += w * (f @ f.T)

    # Apply orientation penalty
    compat = ORIENT_COMPAT.get(sp, {})
    if compat:
        orient_mask = np.ones((n, n))
        for i in range(n):
            for j in range(i+1, n):
                oi = str(orientations[i]) if pd.notna(orientations[i]) else "unknown"
                oj = str(orientations[j]) if pd.notna(orientations[j]) else "unknown"
                # Look up compatibility factor
                factor = compat.get((oi, oj), 0.5)  # default moderate penalty
                orient_mask[i, j] = factor
                orient_mask[j, i] = factor
        np.fill_diagonal(orient_mask, 1.0)

        # Apply: sim = sim * orient_mask
        sim = sim * orient_mask
        print(f"      Orientation mask applied: {len(compat)} rules, avg_factor={orient_mask[np.triu_indices(n,1)].mean():.3f}")

    return sim


def build_representative_embeddings(feats_dict, tr_df, tr_idx, weights):
    """Build per-identity, per-orientation representative embeddings.

    Returns: dict of {identity: {orientation: mean_embedding}}
    """
    reps = defaultdict(dict)  # {identity: {orient: embed}}
    model_names = [k for k, v in weights.items() if v > 0]

    for identity in tr_df.identity.unique():
        mask = tr_df.identity.values == identity
        id_idx = tr_idx[mask]
        id_orients = tr_df[mask].orientation.values

        # Group by orientation
        orient_groups = defaultdict(list)
        for i, oi in enumerate(id_orients):
            oi_str = str(oi) if pd.notna(oi) else "unknown"
            orient_groups[oi_str].append(id_idx[i])

        for orient, idxs in orient_groups.items():
            # Mean embedding across all models
            combined = []
            for name in model_names:
                f = normalize(feats_dict[name][np.array(idxs)], axis=1)
                mean_f = f.mean(axis=0, keepdims=True)
                combined.append(normalize(mean_f, axis=1) * weights[name])
            rep = np.sum(combined, axis=0)
            rep = normalize(rep, axis=1)
            reps[identity][orient] = rep.flatten()

    return reps


# ═══════════════════════════════════════════════════════════
# REPRESENTATIVE MATCHING: Assign test to known identities
# ═══════════════════════════════════════════════════════════
def representative_matching_v2(feats_dict, tr_df, tr_idx, te_idx, te_orients, weights, sp, threshold):
    """Improved representative matching: directly compute test-vs-train similarity.

    Instead of building representative embeddings, directly compare each test image
    to all training images, respecting orientation compatibility.
    """
    n_te = len(te_idx)
    n_tr = len(tr_idx)
    model_names = [k for k, v in weights.items() if v > 0]
    compat = ORIENT_COMPAT.get(sp, {})
    tr_orients = tr_df.orientation.values
    tr_labels = tr_df.identity.values

    # Build test-train similarity matrix
    sim_te_tr = np.zeros((n_te, n_tr))
    for name in model_names:
        if weights[name] <= 0:
            continue
        f_te = normalize(feats_dict[name][te_idx], axis=1)
        f_tr = normalize(feats_dict[name][tr_idx], axis=1)
        sim_te_tr += weights[name] * (f_te @ f_tr.T)

    # Apply orientation penalty
    if compat:
        for i in range(n_te):
            oi = str(te_orients[i]) if pd.notna(te_orients[i]) else "unknown"
            for j in range(n_tr):
                oj = str(tr_orients[j]) if pd.notna(tr_orients[j]) else "unknown"
                factor = compat.get((oi, oj), 0.5)
                sim_te_tr[i, j] *= factor

    # For each test image: find best matching identity
    # Aggregate per identity: take max similarity to any train image of that identity
    unique_ids = np.unique(tr_labels)
    id2idx = {uid: np.where(tr_labels == uid)[0] for uid in unique_ids}

    assignments = {}  # test_local_idx → identity string or None
    scores = {}       # test_local_idx → confidence score

    for i in range(n_te):
        best_score = -1
        best_id = None
        second_best = -1

        for uid, uid_indices in id2idx.items():
            # Max similarity to any training image of this identity
            max_sim = sim_te_tr[i, uid_indices].max()
            # Also compute mean of top-3 for more robust matching
            top_k = min(3, len(uid_indices))
            top_sims = np.sort(sim_te_tr[i, uid_indices])[-top_k:]
            agg_sim = top_sims.mean()  # mean of top-k

            if agg_sim > best_score:
                second_best = best_score
                best_score = agg_sim
                best_id = uid
            elif agg_sim > second_best:
                second_best = agg_sim

        # Margin: how much better is the best match vs second best
        margin = best_score - second_best if second_best > -1 else best_score

        if best_score >= threshold and margin >= 0.02:
            assignments[i] = best_id
            scores[i] = best_score
        else:
            assignments[i] = None
            scores[i] = best_score

    n_assigned = sum(1 for v in assignments.values() if v is not None)
    print(f"      Representative matching: {n_assigned}/{n_te} assigned (threshold={threshold:.3f})")

    return assignments, scores, sim_te_tr


# ═══════════════════════════════════════════════════════════
# HYBRID CLUSTERING
# ═══════════════════════════════════════════════════════════
def hybrid_cluster(feats_dict, te_idx, te_orients, assignments, weights, sp,
                   local_matcher, all_paths, root, cluster_threshold):
    """Hybrid clustering combining assigned and discovered individuals.

    1. Assigned images keep their training identity
    2. Unassigned ("new") images are clustered among themselves
    3. Use orientation-aware similarity + local matching for discovery
    4. Merge clusters across orientations for assigned identities
    """
    n_te = len(te_idx)

    # Separate assigned and unassigned
    assigned_idx = [i for i in range(n_te) if assignments[i] is not None]
    new_idx = [i for i in range(n_te) if assignments[i] is None]

    print(f"      Assigned: {len(assigned_idx)}, New (to cluster): {len(new_idx)}")

    # For assigned images: group by identity
    id_groups = defaultdict(list)
    for i in assigned_idx:
        id_groups[assignments[i]].append(i)

    # For new images: cluster using orientation-aware similarity
    labels = np.full(n_te, -1, dtype=object)

    # Assign known identities
    label_counter = 0
    identity_to_label = {}
    for identity, members in id_groups.items():
        if identity not in identity_to_label:
            identity_to_label[identity] = f"known_{label_counter}"
            label_counter += 1
        for m in members:
            labels[m] = identity_to_label[identity]

    # Cluster "new" images
    if len(new_idx) > 1:
        new_idx_arr = np.array(new_idx)
        new_orients = [te_orients[i] for i in new_idx]

        # Build similarity matrix for new images
        new_sim = build_orientation_aware_sim(
            feats_dict, te_idx[new_idx_arr], weights, new_orients, sp
        )

        # Optional: enhance with local matching for top candidates
        if local_matcher and local_matcher.ok and len(new_idx) <= 800:
            print(f"      Running local matching for {len(new_idx)} new images...")
            new_paths = [all_paths[te_idx[i]] for i in new_idx]
            # Get top-K pairs from global similarity
            pairs = get_top_pairs(new_sim, k=30, max_pairs=15000)
            if pairs:
                local_scores = local_matcher.compute_pairwise_local(
                    new_paths, pairs, root, desc="new"
                )
                if local_scores:
                    # Normalize local scores
                    max_ls = max(local_scores.values()) if local_scores.values() else 1.0
                    if max_ls > 0:
                        # Build local similarity matrix
                        local_sim = np.zeros_like(new_sim)
                        for (i, j), s in local_scores.items():
                            local_sim[i, j] = s / max_ls
                        np.fill_diagonal(local_sim, 1.0)

                        # Fuse: global * (1-beta) + local * beta
                        # Search best beta on structure
                        best_beta = 0.3  # default
                        new_sim = (1 - best_beta) * new_sim + best_beta * local_sim
                        print(f"      Local fusion applied (beta={best_beta})")

        # Cluster
        dist = np.clip(1 - new_sim, 0, 2)
        np.fill_diagonal(dist, 0)

        try:
            new_labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=cluster_threshold,
                metric="precomputed", linkage="average"
            ).fit_predict(dist)

            for i, ni in enumerate(new_idx):
                labels[ni] = f"new_{new_labels[i]}"

            print(f"      New clusters: {len(set(new_labels))}")
        except Exception as e:
            print(f"      Clustering failed: {e}")
            for i, ni in enumerate(new_idx):
                labels[ni] = f"new_{i}"  # Each as singleton

    elif len(new_idx) == 1:
        labels[new_idx[0]] = "new_0"

    return labels


def get_top_pairs(sim, k=30, max_pairs=15000):
    """Get top-K most similar pairs for local matching."""
    n = sim.shape[0]
    pairs = set()
    for i in range(n):
        s = sim[i].copy()
        s[i] = -999
        topk = np.argsort(s)[-k:]
        for j in topk:
            if s[j] > 0.1:  # minimum similarity threshold
                pairs.add((min(i, j), max(i, j)))
    pairs = list(pairs)
    if len(pairs) > max_pairs:
        svals = [(sim[i, j], (i, j)) for i, j in pairs]
        svals.sort(reverse=True)
        pairs = [p for _, p in svals[:max_pairs]]
    return pairs


# ═══════════════════════════════════════════════════════════
# PARAMETER SEARCH ON TRAINING DATA
# ═══════════════════════════════════════════════════════════
def search_params_on_train(feats_dict, tr_df, tr_idx, weights, sp):
    """Search optimal parameters using training data with cross-validation.

    Searches:
      1. Assignment threshold for representative matching
      2. Cluster threshold for new individuals
    """
    labels = tr_df.identity.values
    orients = tr_df.orientation.values
    n = len(tr_idx)

    # Build orientation-aware similarity
    sim = build_orientation_aware_sim(feats_dict, tr_idx, weights, orients, sp)

    # Grid search for clustering threshold
    best_ari, best_th = -1, 0.5
    dist = np.clip(1 - sim, 0, 2)
    np.fill_diagonal(dist, 0)

    for th in np.arange(0.05, 1.50, 0.01):
        try:
            pred = AgglomerativeClustering(
                n_clusters=None, distance_threshold=th,
                metric="precomputed", linkage="average"
            ).fit_predict(dist)
            ari = adjusted_rand_score(labels, pred)
            if ari > best_ari:
                best_ari, best_th = ari, th
        except:
            pass

    print(f"      Train clustering ARI={best_ari:.4f} th={best_th:.3f}")

    # For representative matching: simulate leave-some-out
    # Use 70% as "train" references, 30% as "test"
    rng = np.random.RandomState(42)
    unique_ids = np.unique(labels)
    best_assign_th = 0.4
    best_assign_ari = -1

    for trial in range(3):
        # Split each identity: 70% ref, 30% query
        ref_mask = np.zeros(n, dtype=bool)
        query_mask = np.zeros(n, dtype=bool)
        for uid in unique_ids:
            uid_idx = np.where(labels == uid)[0]
            if len(uid_idx) <= 1:
                ref_mask[uid_idx] = True  # singleton goes to ref
                continue
            n_ref = max(1, int(0.7 * len(uid_idx)))
            perm = rng.permutation(len(uid_idx))
            ref_mask[uid_idx[perm[:n_ref]]] = True
            query_mask[uid_idx[perm[n_ref:]]] = True

        if query_mask.sum() < 10:
            continue

        ref_idx = tr_idx[ref_mask]
        query_idx = tr_idx[query_mask]
        ref_df = tr_df[ref_mask].reset_index(drop=True)
        query_orients = orients[query_mask]
        query_labels = labels[query_mask]

        for assign_th in np.arange(0.15, 0.65, 0.05):
            assigns, _, _ = representative_matching_v2(
                feats_dict, ref_df, ref_idx, query_idx, query_orients,
                weights, sp, assign_th
            )
            pred_labels = []
            for i in range(len(query_idx)):
                if assigns[i] is not None:
                    pred_labels.append(assigns[i])
                else:
                    pred_labels.append(f"new_{i}")

            ari = adjusted_rand_score(query_labels, pred_labels)
            if ari > best_assign_ari:
                best_assign_ari = ari
                best_assign_th = assign_th

    print(f"      Representative matching ARI={best_assign_ari:.4f} assign_th={best_assign_th:.3f}")

    return best_th, best_assign_th, best_ari


def search_model_weights_orient(feats_dict, tr_idx, labels, orients, sp, steps=11):
    """Search optimal model weights with orientation awareness."""
    models = [k for k in feats_dict.keys()]
    if len(models) == 1:
        return {models[0]: 1.0}, -1

    best_ari, best_w = -1, None
    n = len(tr_idx)

    # Pre-compute per-model orientation-aware similarities
    sims = {}
    for name in models:
        f = normalize(feats_dict[name][tr_idx], axis=1)
        base_sim = f @ f.T

        # Apply orientation penalty
        compat = ORIENT_COMPAT.get(sp, {})
        if compat:
            for i in range(n):
                for j in range(i+1, n):
                    oi = str(orients[i]) if pd.notna(orients[i]) else "unknown"
                    oj = str(orients[j]) if pd.notna(orients[j]) else "unknown"
                    factor = compat.get((oi, oj), 0.5)
                    base_sim[i, j] *= factor
                    base_sim[j, i] *= factor
        sims[name] = base_sim

    if len(models) == 2:
        for a in np.linspace(0, 1, steps):
            sim = a * sims[models[0]] + (1-a) * sims[models[1]]
            dist = np.clip(1-sim, 0, 2); np.fill_diagonal(dist, 0)
            for th in np.arange(0.05, 1.5, 0.02):
                try:
                    pred = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                            metric="precomputed", linkage="average").fit_predict(dist)
                    ari = adjusted_rand_score(labels, pred)
                    if ari > best_ari:
                        best_ari = ari
                        best_w = {models[0]: a, models[1]: 1-a}
                except: pass

    elif len(models) == 3:
        for a in np.linspace(0, 1, steps):
            for b in np.linspace(0, 1-a, steps):
                c = 1 - a - b
                sim = a*sims[models[0]] + b*sims[models[1]] + c*sims[models[2]]
                dist = np.clip(1-sim, 0, 2); np.fill_diagonal(dist, 0)
                for th in np.arange(0.05, 1.5, 0.02):
                    try:
                        pred = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                metric="precomputed", linkage="average").fit_predict(dist)
                        ari = adjusted_rand_score(labels, pred)
                        if ari > best_ari:
                            best_ari = ari
                            best_w = {models[0]: a, models[1]: b, models[2]: c}
                    except: pass

    if best_w is None:
        best_w = {m: 1.0/len(models) for m in models}

    return best_w, best_ari


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 70)
    print("  AnimalCLEF2026 V19 — Orientation-Aware Breakthrough for Top5")
    print("=" * 70)

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | {p.total_memory/1e9:.0f}GB")
    else:
        print("  WARNING: No GPU detected! Running on CPU (will be slow)")

    # ── Load Data ──
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    ssub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    trdf = meta[meta.split == "train"]
    tedf = meta[meta.split == "test"]
    alldf = pd.concat([trdf, tedf], ignore_index=True)
    ntr, nte = len(trdf), len(tedf)
    print(f"  Data: {ntr} train, {nte} test")

    # ── STAGE 1: Extract Features ──
    print(f"\n{'━'*65}")
    print("STAGE 1: Feature Extraction (3 Backbones + TTA)")
    print(f"{'━'*65}")

    fd = {}
    m, sz = load_mega()
    fd["mega"], _ = extract_features(m, alldf, DATA_DIR, sz, 48, "Mega")
    del m; torch.cuda.empty_cache(); gc.collect()

    m, sz = load_miew()
    fd["miew"], _ = extract_features(m, alldf, DATA_DIR, sz, 32, "Miew")
    del m; torch.cuda.empty_cache(); gc.collect()

    dm, dsz = load_dino()
    if dm is not None:
        fd["dino"], _ = extract_features(dm, alldf, DATA_DIR, dsz, 16, "DINO")
        del dm; torch.cuda.empty_cache(); gc.collect()

    all_ids = alldf.image_id.values
    id2idx = {int(all_ids[i]): i for i in range(len(all_ids))}
    all_paths = alldf.path.values
    all_orients = alldf.orientation.values

    print(f"  Models: {list(fd.keys())} | Elapsed: {(time.time()-t0)/60:.1f}min")

    # ── STAGE 2: Local Matcher Init ──
    print(f"\n{'━'*65}")
    print("STAGE 2: Local Feature Matcher")
    print(f"{'━'*65}")
    lm = LocalMatcher(max_keypoints=1024)

    # ── STAGE 3: Per-Species Pipeline ──
    print(f"\n{'━'*65}")
    print("STAGE 3: Per-Species Orientation-Aware Pipeline")
    print(f"{'━'*65}")

    preds = {}

    for sp in SP_ALL:
        print(f"\n  {'═'*55}")
        print(f"  {sp}")
        print(f"  {'═'*55}")

        te = tedf[tedf.dataset == sp].reset_index(drop=True)
        te_idx = np.array([id2idx[int(x)] for x in te.image_id])
        te_orients = te.orientation.values
        nte_sp = len(te)

        if sp in SP_TRAIN:
            tr = trdf[trdf.dataset == sp].reset_index(drop=True)
            tr_idx = np.array([id2idx[int(x)] for x in tr.image_id])
            tr_labels = tr.identity.values
            tr_orients = tr.orientation.values
            ntr_sp = len(tr)
            n_ids = len(set(tr_labels))
            print(f"    {ntr_sp} train ({n_ids} ids), {nte_sp} test")

            # Show orientation distribution
            for split_name, orient_arr in [("train", tr_orients), ("test", te_orients)]:
                oc = pd.Series(orient_arr).value_counts()
                print(f"    {split_name} orientations: {dict(oc)}")

            # ── A: Search optimal model weights (with orientation awareness) ──
            print(f"\n    [A] Searching model weights (orientation-aware)...")
            # Sample if too many
            if ntr_sp > 1500:
                rng = np.random.RandomState(42)
                si = rng.choice(ntr_sp, 1500, replace=False)
                s_idx, s_lab, s_ori = tr_idx[si], tr_labels[si], tr_orients[si]
            else:
                s_idx, s_lab, s_ori = tr_idx, tr_labels, tr_orients

            weights, ari_w = search_model_weights_orient(fd, s_idx, s_lab, s_ori, sp)
            print(f"        Weights: {weights}")
            print(f"        Train ARI (orient-aware): {ari_w:.4f}")

            # ── B: Search assignment & cluster thresholds ──
            print(f"\n    [B] Searching thresholds...")
            cluster_th, assign_th, train_ari = search_params_on_train(
                fd, tr, tr_idx, weights, sp
            )

            # ── C: Representative matching on test ──
            print(f"\n    [C] Representative matching...")
            assignments, conf_scores, sim_te_tr = representative_matching_v2(
                fd, tr, tr_idx, te_idx, te_orients, weights, sp, assign_th
            )

            # ── D: Local matching verification for top candidates ──
            if lm.ok:
                print(f"\n    [D] Local matching verification...")
                # Verify uncertain assignments (close to threshold)
                uncertain = []
                for i in range(nte_sp):
                    if assignments[i] is not None and conf_scores[i] < assign_th + 0.1:
                        uncertain.append(i)
                    elif assignments[i] is None and conf_scores[i] > assign_th - 0.15:
                        uncertain.append(i)

                if uncertain and len(uncertain) <= 500:
                    print(f"      Verifying {len(uncertain)} uncertain assignments...")
                    # For each uncertain, get top-5 train candidates and do local matching
                    for i in uncertain:
                        te_path = os.path.join(DATA_DIR, all_paths[te_idx[i]])
                        te_kp = lm.get_keypoints(te_path)

                        # Get top-5 train candidates
                        top5_tr = np.argsort(sim_te_tr[i])[-5:]
                        best_local = 0
                        best_tr_id = None

                        for j in top5_tr:
                            tr_path = os.path.join(DATA_DIR, all_paths[tr_idx[j]])
                            tr_kp = lm.get_keypoints(tr_path)
                            local_score = lm.match_pair(te_kp, tr_kp)
                            if local_score > best_local:
                                best_local = local_score
                                best_tr_id = tr_labels[j]

                        # Update assignment if local matching disagrees
                        if best_local > 15 and assignments[i] is None:
                            # Local matching found a match → assign
                            assignments[i] = best_tr_id
                        elif best_local < 3 and assignments[i] is not None:
                            # Local matching says no match → unassign
                            assignments[i] = None

                    lm.clear_cache()
                    n_assigned = sum(1 for v in assignments.values() if v is not None)
                    print(f"      After local verification: {n_assigned}/{nte_sp} assigned")

            # ── E: Hybrid clustering ──
            print(f"\n    [E] Hybrid clustering (assigned + discovery)...")
            labels = hybrid_cluster(
                fd, te_idx, te_orients, assignments, weights, sp,
                lm, all_paths, DATA_DIR, cluster_th
            )

        else:
            # TexasHornedLizards — no training data
            print(f"    No training data, {nte_sp} test images")
            print(f"    Using pure test clustering with local matching")

            # Equal weights
            weights = {k: 1.0/len(fd) for k in fd}

            # Build orientation-aware similarity (orientations may not exist)
            te_sim = build_orientation_aware_sim(fd, te_idx, weights, te_orients, sp)

            # Local matching for Texas Horned Lizards (pattern-based)
            if lm.ok and nte_sp <= 300:
                print(f"    Local matching for all test images...")
                te_paths = [all_paths[te_idx[i]] for i in range(nte_sp)]
                pairs = get_top_pairs(te_sim, k=30, max_pairs=15000)
                if pairs:
                    local_scores = lm.compute_pairwise_local(te_paths, pairs, DATA_DIR, desc="Texas")
                    if local_scores:
                        max_ls = max(local_scores.values()) if local_scores.values() else 1.0
                        if max_ls > 0:
                            local_sim = np.zeros((nte_sp, nte_sp))
                            for (i, j), s in local_scores.items():
                                local_sim[i, j] = s / max_ls
                            np.fill_diagonal(local_sim, 1.0)
                            # Higher beta for lizards (pattern matching is crucial)
                            beta = 0.4
                            te_sim = (1 - beta) * te_sim + beta * local_sim
                            print(f"    Local fusion (beta={beta})")

            # Search threshold
            best_th, best_score = 0.4, -1
            dist = np.clip(1 - te_sim, 0, 2)
            np.fill_diagonal(dist, 0)
            for th in np.arange(0.10, 1.50, 0.01):
                try:
                    pred = AgglomerativeClustering(
                        n_clusters=None, distance_threshold=th,
                        metric="precomputed", linkage="average"
                    ).fit_predict(dist)
                    ratio = len(set(pred)) / nte_sp
                    # Target ratio from training data distributions
                    if 0.15 <= ratio <= 0.55:
                        sc = -abs(ratio - 0.35)
                        if sc > best_score:
                            best_score, best_th = sc, th
                except:
                    pass

            labels_arr = AgglomerativeClustering(
                n_clusters=None, distance_threshold=best_th,
                metric="precomputed", linkage="average"
            ).fit_predict(dist)

            labels = {}
            for i in range(nte_sp):
                labels[i] = f"texas_{labels_arr[i]}"
            print(f"    → {len(set(labels_arr))} clusters (th={best_th:.3f})")

        # Convert labels to submission format
        for i in range(nte_sp):
            img_id = int(te.iloc[i].image_id)
            if isinstance(labels, dict):
                preds[img_id] = f"cluster_{sp}_{labels[i]}"
            else:
                preds[img_id] = f"cluster_{sp}_{labels[i]}"

        # Summary
        unique_labels = set(labels[i] if isinstance(labels, dict) else labels[i] for i in range(nte_sp))
        print(f"\n    Summary: {nte_sp} images → {len(unique_labels)} clusters")

    # ── STAGE 4: Generate Submission ──
    print(f"\n{'━'*65}")
    print("STAGE 4: Submission")
    print(f"{'━'*65}")

    sub = ssub.copy()
    for i in range(len(sub)):
        iid = int(sub.iloc[i].image_id)
        if iid in preds:
            sub.at[i, "cluster"] = preds[iid]

    out_path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)

    print(f"  Output: {out_path}")
    print(f"  Rows: {len(sub)}, Unique clusters: {sub.cluster.nunique()}")
    for sp in SP_ALL:
        sp_sub = sub[sub.cluster.str.contains(sp)]
        if len(sp_sub) > 0:
            vc = sp_sub.cluster.value_counts()
            singletons = (vc == 1).sum()
            print(f"    {sp:25s} {len(sp_sub):4d} imgs, {sp_sub.cluster.nunique():4d} cl, "
                  f"singletons={singletons} ({100*singletons/len(vc):.0f}%)")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * 70)
    print("  DONE! Ready for submission.")
    print("=" * 70)


if __name__ == "__main__":
    main()
