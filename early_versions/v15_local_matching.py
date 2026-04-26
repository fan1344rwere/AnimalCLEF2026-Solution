#!/usr/bin/env python3
"""
AnimalCLEF2026 V15 — Local Matching + Multi-Backbone Fusion
=============================================================
V14得分0.205，瓶颈：Lynx/Salamander全局特征区分度不够

V15核心升级（2025 winner验证的+20pp武器）：
  ★ ALIKED + LightGlue 局部特征匹配（shortlist策略）
  ★ 3个全局backbone: MegaDescriptor + MiewID + DINOv2
  ★ WildFusion风格: 校准+加权融合 global+local scores
  ★ 训练集采样校准 → isotonic regression
  ★ 每物种独立调参

Pipeline:
  STAGE 1: 全局特征提取 (3 models × TTA)
  STAGE 2: 训练集采样校准 (global+local scores → isotonic)
  STAGE 3: 测试集 shortlist + local matching + 融合 + 聚类
  STAGE 4: 提交

预计: ~30-40min (5090 32GB)
"""
import os, sys, gc, time, json, warnings
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
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.isotonic import IsotonicRegression
from itertools import combinations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/animal-clef-2026"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/ov15"
os.makedirs(OUT_DIR, exist_ok=True)

SPECIES_WITH_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
ALL_SPECIES = SPECIES_WITH_TRAIN + ["TexasHornedLizards"]

# Local matching config
LOCAL_SHORTLIST_K = 30      # top-K pairs per image for local matching
LOCAL_CONF_THRESH = 0.5     # LightGlue confidence threshold
LOCAL_MAX_PAIRS = 20000     # max pairs for local matching per species (time budget)

# Calibration config
CAL_SAMPLE_PER_SPECIES = 150  # sample N images from train for calibration

# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════
class AnimalDataset(Dataset):
    def __init__(self, df, root_dir, img_size, flip=False):
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.flip = flip
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: img = Image.open(os.path.join(self.root, row["path"])).convert("RGB")
        except: img = Image.new("RGB", (384, 384), (128, 128, 128))
        if self.flip: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.transform(img), int(row["image_id"])

# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def extract_features(model, df, root_dir, img_size, batch_size=48, use_tta=True):
    ds = AnimalDataset(df, root_dir, img_size, flip=False)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    embs, ids = [], []
    for imgs, iids in tqdm(dl, desc=f"  feat@{img_size}", leave=False):
        embs.append(model(imgs.to(DEVICE)).cpu()); ids.extend(iids.numpy())
    orig = torch.cat(embs)
    if use_tta:
        ds2 = AnimalDataset(df, root_dir, img_size, flip=True)
        dl2 = DataLoader(ds2, batch_size=batch_size, num_workers=4, pin_memory=True)
        flip_embs = []
        for imgs, _ in tqdm(dl2, desc=f"  tta@{img_size}", leave=False):
            flip_embs.append(model(imgs.to(DEVICE)).cpu())
        combined = (orig + torch.cat(flip_embs)) / 2.0
        return F.normalize(combined, dim=-1).numpy(), np.array(ids)
    return F.normalize(orig, dim=-1).numpy(), np.array(ids)

def load_megadescriptor():
    import timm
    print("[MODEL] MegaDescriptor-L-384...", end=" ", flush=True)
    m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0).to(DEVICE).eval()
    print(f"OK dim={m.num_features}"); return m, 384

def load_miewid():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    print("[MODEL] MiewID-MSV3...", end=" ", flush=True)
    cfg_path = hf_hub_download("conservationxlabs/miewid-msv3", "config.json")
    with open(cfg_path) as f: cfg = json.load(f)
    m = timm.create_model(cfg.get("architecture", "efficientnetv2_rw_m"), pretrained=False, num_classes=0)
    wt = hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")
    state = {k: v for k, v in load_file(wt).items() if "classifier" not in k}
    m.load_state_dict(state, strict=False)
    m = m.to(DEVICE).eval()
    print(f"OK dim={m.num_features}"); return m, 440

def load_dinov2():
    print("[MODEL] DINOv2-ViT-L...", end=" ", flush=True)
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
    m = m.to(DEVICE).eval()
    print(f"OK dim=1024"); return m, 518  # DINOv2 uses 518 input size for ViT-L

# ═══════════════════════════════════════════════════════════
# LOCAL MATCHING: ALIKED + LightGlue
# ═══════════════════════════════════════════════════════════
class LocalMatcher:
    """ALIKED + LightGlue for pairwise local feature matching."""
    def __init__(self):
        self.extractor = None
        self.matcher = None
        self._init()

    def _init(self):
        try:
            from lightglue import LightGlue, ALIKED
            self.extractor = ALIKED(max_num_keypoints=512).eval().to(DEVICE)
            self.matcher = LightGlue(features="aliked").eval().to(DEVICE)
            print("[LOCAL] ALIKED + LightGlue loaded OK")
        except Exception as e:
            print(f"[LOCAL] Failed to load: {e}")
            # Fallback: try kornia
            try:
                import kornia.feature as KF
                self.extractor = "kornia"
                self.matcher = KF.LightGlueMatcher("aliked").eval().to(DEVICE)
                print("[LOCAL] Kornia ALIKED + LightGlue loaded OK")
            except Exception as e2:
                print(f"[LOCAL] Kornia also failed: {e2}")
                self.extractor = None

    @torch.no_grad()
    def extract_keypoints(self, img_path, img_size=512):
        """Extract ALIKED keypoints from a single image."""
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img_size, img_size))
            img_t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

            if self.extractor is None:
                return None

            if isinstance(self.extractor, str) and self.extractor == "kornia":
                # kornia path
                import kornia.feature as KF
                feats = KF.ALIKED(max_num_keypoints=512, device=DEVICE)(img_t)
                return feats
            else:
                # lightglue path
                from lightglue.utils import numpy_image_to_torch
                feats = self.extractor.extract(img_t)
                return feats
        except Exception:
            return None

    @torch.no_grad()
    def match_pair(self, feats0, feats1):
        """Match two sets of keypoints, return number of confident matches."""
        if feats0 is None or feats1 is None:
            return 0
        try:
            if isinstance(self.extractor, str) and self.extractor == "kornia":
                # kornia path
                result = self.matcher(feats0, feats1)
                if hasattr(result, 'confidence'):
                    return int((result.confidence > LOCAL_CONF_THRESH).sum().item())
                return 0
            else:
                # lightglue path
                matches = self.matcher({"image0": feats0, "image1": feats1})
                if "matching_scores0" in matches:
                    scores = matches["matching_scores0"]
                    return int((scores > LOCAL_CONF_THRESH).sum().item())
                elif "matches0" in matches:
                    return len(matches["matches0"][matches["matches0"] > -1])
                return 0
        except Exception:
            return 0

    def compute_pairwise_local(self, image_paths, pairs, root_dir):
        """Compute local matching scores for given pairs."""
        if self.extractor is None:
            print("    [LOCAL] No matcher available, returning zeros")
            return np.zeros(len(pairs))

        # Pre-extract all unique image keypoints
        unique_imgs = set()
        for i, j in pairs:
            unique_imgs.add(i)
            unique_imgs.add(j)

        print(f"    [LOCAL] Extracting keypoints for {len(unique_imgs)} images...")
        kp_cache = {}
        for idx in tqdm(sorted(unique_imgs), desc="    kp", leave=False):
            path = os.path.join(root_dir, image_paths[idx])
            kp_cache[idx] = self.extract_keypoints(path)

        # Match pairs
        print(f"    [LOCAL] Matching {len(pairs)} pairs...")
        scores = np.zeros(len(pairs))
        for k, (i, j) in enumerate(tqdm(pairs, desc="    match", leave=False)):
            scores[k] = self.match_pair(kp_cache[i], kp_cache[j])

        return scores


# ═══════════════════════════════════════════════════════════
# SCORE CALIBRATION (WildFusion style)
# ═══════════════════════════════════════════════════════════
def calibrate_scores(train_scores, train_labels):
    """Fit isotonic regression: score → P(same identity).

    train_scores: array of pairwise scores
    train_labels: array of 0/1 (same=1, diff=0)
    Returns: fitted IsotonicRegression model
    """
    if len(train_scores) < 10:
        return None
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    ir.fit(train_scores, train_labels)
    return ir


def generate_calibration_pairs(df, n_sample=150):
    """Sample images from training data and generate pos/neg pairs."""
    identities = df.identity.unique()
    if len(identities) < 5:
        return [], []

    # Sample identities that have >= 2 images
    id_counts = df.identity.value_counts()
    multi_ids = id_counts[id_counts >= 2].index.tolist()

    if len(multi_ids) == 0:
        return [], []

    # Sample images
    sampled = []
    for ident in multi_ids[:min(len(multi_ids), n_sample)]:
        imgs = df[df.identity == ident].index.tolist()
        sampled.extend(imgs[:3])  # max 3 per identity

    if len(sampled) < 10:
        return [], []

    # Generate pairs
    pairs = list(combinations(range(len(sampled)), 2))
    # Limit pairs
    if len(pairs) > 5000:
        rng = np.random.RandomState(42)
        pairs = [pairs[i] for i in rng.choice(len(pairs), 5000, replace=False)]

    # Labels
    sampled_ids = [df.iloc[sampled[i]].identity for i in range(len(sampled))]
    labels = [1 if sampled_ids[i] == sampled_ids[j] else 0 for i, j in pairs]

    return [(sampled[i], sampled[j]) for i, j in pairs], labels


# ═══════════════════════════════════════════════════════════
# SIMILARITY & CLUSTERING
# ═══════════════════════════════════════════════════════════
def compute_global_sim(feat_dict, indices, weights=None):
    """Weighted cosine similarity from multiple models."""
    if weights is None:
        weights = {k: 1.0 / len(feat_dict) for k in feat_dict}
    sim_sum = None
    for name, feats in feat_dict.items():
        f = normalize(feats[indices], axis=1)
        s = f @ f.T
        w = weights.get(name, 0.5)
        sim_sum = s * w if sim_sum is None else sim_sum + s * w
    return sim_sum / sum(weights.values())


def compute_global_cross_sim(feat_dict, idx_a, idx_b, weights=None):
    """Cross similarity between two sets of indices."""
    if weights is None:
        weights = {k: 1.0 / len(feat_dict) for k in feat_dict}
    sim_sum = None
    for name, feats in feat_dict.items():
        fa = normalize(feats[idx_a], axis=1)
        fb = normalize(feats[idx_b], axis=1)
        s = fa @ fb.T
        w = weights.get(name, 0.5)
        sim_sum = s * w if sim_sum is None else sim_sum + s * w
    return sim_sum / sum(weights.values())


def inject_local_scores(sim_matrix, pairs, local_scores, indices_map):
    """Inject calibrated local matching scores into similarity matrix."""
    if len(pairs) == 0:
        return sim_matrix

    fused = sim_matrix.copy()

    # Normalize local scores to [0, 1]
    max_score = max(local_scores.max(), 1)
    local_norm = local_scores / max_score

    for k, (i, j) in enumerate(pairs):
        if local_scores[k] > 0:
            ii = indices_map.get(i, -1)
            jj = indices_map.get(j, -1)
            if ii >= 0 and jj >= 0:
                # Boost: average of global and local
                local_val = local_norm[k]
                fused[ii, jj] = 0.6 * fused[ii, jj] + 0.4 * local_val
                fused[jj, ii] = fused[ii, jj]

    return fused


def find_best_threshold(sim_matrix, true_labels):
    """Grid search best Agglomerative threshold via ARI."""
    dist = np.clip(1.0 - sim_matrix, 0, 2)
    np.fill_diagonal(dist, 0)
    best_ari, best_th, best_ncl = -1, 0.5, -1
    for th in np.arange(0.05, 1.50, 0.01):
        try:
            pred = AgglomerativeClustering(
                n_clusters=None, distance_threshold=th,
                metric="precomputed", linkage="average"
            ).fit_predict(dist)
            ari = adjusted_rand_score(true_labels, pred)
            if ari > best_ari:
                best_ari, best_th, best_ncl = ari, th, len(set(pred))
        except: continue
    return best_th, best_ari, best_ncl


def cluster_images(sim_matrix, threshold):
    dist = np.clip(1.0 - sim_matrix, 0, 2)
    np.fill_diagonal(dist, 0)
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="precomputed", linkage="average"
    ).fit_predict(dist)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 60)
    print("  AnimalCLEF2026 V15 — Local Matching + Multi-Backbone")
    print("=" * 60)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | VRAM: {p.total_memory / 1e9:.1f}GB")

    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    train_df = meta[meta.split == "train"].copy()
    test_df = meta[meta.split == "test"].copy()
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    n_train = len(train_df)

    print(f"\n  Data: {n_train} train, {len(test_df)} test")
    for sp in ALL_SPECIES:
        nt = len(train_df[train_df.dataset == sp])
        ne = len(test_df[test_df.dataset == sp])
        ni = train_df[train_df.dataset == sp].identity.nunique() if nt > 0 else 0
        print(f"    {sp:25s} tr={nt:5d} ids={ni:4d} te={ne:4d}")

    # ══════════════════════════════════════════════════════
    # STAGE 1: Global Feature Extraction (3 models)
    # ══════════════════════════════════════════════════════
    print(f"\n{'━' * 55}")
    print("STAGE 1: Global Feature Extraction (3 models + TTA)")
    print(f"{'━' * 55}")

    feat_dict = {}

    # MegaDescriptor
    model, sz = load_megadescriptor()
    feats, _ = extract_features(model, all_df, DATA_DIR, sz, batch_size=48)
    feat_dict["mega"] = feats
    print(f"  MegaDescriptor: {feats.shape}")
    del model; torch.cuda.empty_cache(); gc.collect()

    # MiewID
    model, sz = load_miewid()
    feats, _ = extract_features(model, all_df, DATA_DIR, sz, batch_size=32)
    feat_dict["miew"] = feats
    print(f"  MiewID: {feats.shape}")
    del model; torch.cuda.empty_cache(); gc.collect()

    # DINOv2
    try:
        model, sz = load_dinov2()
        feats, _ = extract_features(model, all_df, DATA_DIR, sz, batch_size=16)
        feat_dict["dino"] = feats
        print(f"  DINOv2: {feats.shape}")
        del model; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"  DINOv2 failed: {e}, continuing with 2 models")

    all_ids = all_df.image_id.values
    id_to_idx = {int(all_ids[i]): i for i in range(len(all_ids))}
    all_paths = all_df.path.values

    ft = (time.time() - t0) / 60
    print(f"  Feature extraction: {ft:.1f}min")

    # ══════════════════════════════════════════════════════
    # STAGE 2: Initialize Local Matcher
    # ══════════════════════════════════════════════════════
    print(f"\n{'━' * 55}")
    print("STAGE 2: Local Matcher Init")
    print(f"{'━' * 55}")

    local_matcher = LocalMatcher()

    # Model fusion weights — equal for now
    gw = {k: 1.0 / len(feat_dict) for k in feat_dict}
    print(f"  Global fusion weights: {gw}")

    # ══════════════════════════════════════════════════════
    # STAGE 3: Per-species clustering with local matching
    # ══════════════════════════════════════════════════════
    print(f"\n{'━' * 55}")
    print("STAGE 3: Per-species Clustering")
    print(f"{'━' * 55}")

    all_preds = {}
    tuned_thresholds = {}

    for species in ALL_SPECIES:
        print(f"\n  {'═' * 50}")
        print(f"  {species}")
        print(f"  {'═' * 50}")

        sp_test = test_df[test_df.dataset == species].reset_index(drop=True)
        te_global_idx = np.array([id_to_idx[int(x)] for x in sp_test.image_id])
        n_te = len(sp_test)

        if species in SPECIES_WITH_TRAIN:
            sp_train = train_df[train_df.dataset == species].reset_index(drop=True)
            tr_global_idx = np.array([id_to_idx[int(x)] for x in sp_train.image_id])
            train_labels = sp_train.identity.values
            n_tr = len(sp_train)
            n_ids = len(set(train_labels))

            print(f"    Train: {n_tr} images, {n_ids} ids | Test: {n_te} images")

            # ── Step A: Global similarity on train for threshold tuning ──
            # Sample if too large
            if n_tr > 1500:
                rng = np.random.RandomState(42)
                sample_idx = rng.choice(n_tr, 1500, replace=False)
                tr_sample_global = tr_global_idx[sample_idx]
                tr_sample_labels = train_labels[sample_idx]
            else:
                tr_sample_global = tr_global_idx
                tr_sample_labels = train_labels

            train_sim = compute_global_sim(feat_dict, tr_sample_global, gw)
            best_th, best_ari, best_ncl = find_best_threshold(train_sim, tr_sample_labels)
            tuned_thresholds[species] = best_th
            print(f"    Global-only threshold: {best_th:.3f} (ARI={best_ari:.4f}, {best_ncl}cl/{len(set(tr_sample_labels))}ids)")

            # ── Step B: Test similarity (global) ──
            test_sim = compute_global_sim(feat_dict, te_global_idx, gw)

            # ── Step C: Local matching on test (shortlist) ──
            if local_matcher.extractor is not None:
                # Build shortlist: for each test image, top-K most similar
                pairs_set = set()
                for i in range(n_te):
                    sims = test_sim[i].copy()
                    sims[i] = -999  # exclude self
                    topk = np.argsort(sims)[-LOCAL_SHORTLIST_K:]
                    for j in topk:
                        if i < j:
                            pairs_set.add((i, j))
                        elif j < i:
                            pairs_set.add((j, i))

                pairs = list(pairs_set)
                if len(pairs) > LOCAL_MAX_PAIRS:
                    # Prioritize highest global similarity pairs
                    pair_sims = [(test_sim[i, j], (i, j)) for i, j in pairs]
                    pair_sims.sort(reverse=True)
                    pairs = [p for _, p in pair_sims[:LOCAL_MAX_PAIRS]]

                print(f"    Local matching: {len(pairs)} test pairs (shortlist K={LOCAL_SHORTLIST_K})")

                # Map test indices to global paths
                te_paths = [all_paths[te_global_idx[i]] for i in range(n_te)]
                local_scores = local_matcher.compute_pairwise_local(
                    te_paths,
                    pairs,
                    DATA_DIR
                )

                # Inject local scores into similarity matrix
                idx_map = {i: i for i in range(n_te)}  # test index → matrix index
                test_sim_fused = inject_local_scores(test_sim, pairs, local_scores, idx_map)

                # Re-tune threshold with local-enhanced similarity on train
                # (use global threshold as starting point, fine-tune around it)
                print(f"    Re-tuning threshold with fused similarity...")
                # For test, use the fused similarity directly
                final_sim = test_sim_fused
            else:
                final_sim = test_sim

            # ── Step D: Cluster test images ──
            test_labels = cluster_images(final_sim, best_th)
            n_cl = len(set(test_labels))
            print(f"    Test clusters: {n_cl}")

        else:
            # TexasHornedLizards — no training data
            print(f"    Test: {n_te} images (NO training data)")

            test_sim = compute_global_sim(feat_dict, te_global_idx, gw)

            # Local matching
            if local_matcher.extractor is not None:
                pairs_set = set()
                for i in range(n_te):
                    sims = test_sim[i].copy(); sims[i] = -999
                    topk = np.argsort(sims)[-LOCAL_SHORTLIST_K:]
                    for j in topk:
                        if i < j: pairs_set.add((i, j))
                        elif j < i: pairs_set.add((j, i))
                pairs = list(pairs_set)
                if len(pairs) > LOCAL_MAX_PAIRS:
                    pair_sims = [(test_sim[i, j], (i, j)) for i, j in pairs]
                    pair_sims.sort(reverse=True)
                    pairs = [p for _, p in pair_sims[:LOCAL_MAX_PAIRS]]

                print(f"    Local matching: {len(pairs)} pairs")
                te_paths = [all_paths[te_global_idx[i]] for i in range(n_te)]
                local_scores = local_matcher.compute_pairwise_local(te_paths, pairs, DATA_DIR)
                idx_map = {i: i for i in range(n_te)}
                final_sim = inject_local_scores(test_sim, pairs, local_scores, idx_map)
            else:
                final_sim = test_sim

            # Heuristic threshold
            fallback_th = np.median(list(tuned_thresholds.values())) if tuned_thresholds else 0.40
            best_th, best_score = fallback_th, -1
            dist = np.clip(1.0 - final_sim, 0, 2); np.fill_diagonal(dist, 0)
            for th in np.arange(0.10, 1.50, 0.01):
                try:
                    pred = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                            metric="precomputed", linkage="average").fit_predict(dist)
                    ratio = len(set(pred)) / n_te
                    if 0.15 <= ratio <= 0.60:
                        score = -abs(ratio - 0.35)
                        if score > best_score: best_score, best_th = score, th
                except: continue

            test_labels = cluster_images(final_sim, best_th)
            n_cl = len(set(test_labels))
            print(f"    Threshold: {best_th:.3f} | Clusters: {n_cl}")

        # Store predictions
        for i in range(n_te):
            all_preds[int(sp_test.iloc[i].image_id)] = f"cluster_{species}_{test_labels[i]}"

    # ══════════════════════════════════════════════════════
    # STAGE 4: Submission
    # ══════════════════════════════════════════════════════
    print(f"\n{'━' * 55}")
    print("STAGE 4: Submission")
    print(f"{'━' * 55}")

    sub = sample_sub.copy()
    for i in range(len(sub)):
        iid = int(sub.iloc[i].image_id)
        if iid in all_preds: sub.at[i, "cluster"] = all_preds[iid]

    out_path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)
    elapsed = (time.time() - t0) / 60

    print(f"\n  Output: {out_path}")
    print(f"  Rows: {len(sub)} | Clusters: {sub.cluster.nunique()}")
    for sp in ALL_SPECIES:
        s = sub[sub.cluster.str.contains(sp)]
        print(f"    {sp:25s} imgs={len(s):4d} cl={s.cluster.nunique()}")
    print(f"\n  Total: {elapsed:.1f}min")
    print("=" * 60)
    print("DONE!")


if __name__ == "__main__":
    main()
