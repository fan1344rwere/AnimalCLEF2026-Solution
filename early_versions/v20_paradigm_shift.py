#!/usr/bin/env python3
"""
AnimalCLEF2026 V20 — Complete Paradigm Shift
=============================================
ABANDON everything from V14-V19. Fresh architecture.

Core philosophy: 2025 champion DataBoom (0.713) used LOCAL matching as PRIMARY.
We've been using it as supplement. That's why we're stuck at 0.23.

Architecture:
  1. Global features ONLY for candidate shortlisting (top-K retrieval)
  2. ALIKED+LightGlue local matching = THE similarity signal
  3. Isotonic regression score calibration on training pairs
  4. Train+Test unified graph → Louvain community detection
  5. Label propagation from training anchors

NO AgglomerativeClustering. NO orientation penalty. NO representative matching.
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
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
from sklearn.isotonic import IsotonicRegression
from collections import defaultdict
import scipy.sparse as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/animal-clef-2026"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/ov20"
os.makedirs(OUT_DIR, exist_ok=True)

SP_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
SP_ALL   = SP_TRAIN + ["TexasHornedLizards"]

# How many global candidates to retrieve for local matching
TOP_K_GLOBAL = 40   # per image, retrieve top-40 candidates
# Max total local matching pairs per species (budget control)
MAX_LOCAL_PAIRS = 30000

# ═══════════════════════════════════════════════════════════
class AnimalDS(Dataset):
    def __init__(self, df, root, sz, flip=False):
        self.df, self.root, self.flip = df.reset_index(drop=True), root, flip
        self.tf = transforms.Compose([
            transforms.Resize((sz, sz)), transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try: img = Image.open(os.path.join(self.root, r["path"])).convert("RGB")
        except: img = Image.new("RGB", (384, 384), (128, 128, 128))
        if self.flip: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tf(img), int(r["image_id"])

@torch.no_grad()
def extract(model, df, root, sz, bs=48, desc=""):
    dl = DataLoader(AnimalDS(df, root, sz), batch_size=bs, num_workers=4, pin_memory=True)
    e, ids = [], []
    for imgs, iids in tqdm(dl, desc=f"  {desc}@{sz}", leave=False):
        e.append(model(imgs.to(DEVICE)).cpu()); ids.extend(iids.numpy())
    orig = torch.cat(e)
    dl2 = DataLoader(AnimalDS(df, root, sz, flip=True), batch_size=bs, num_workers=4, pin_memory=True)
    f = []
    for imgs, _ in tqdm(dl2, desc=f"  {desc}_tta@{sz}", leave=False):
        f.append(model(imgs.to(DEVICE)).cpu())
    return F.normalize((orig + torch.cat(f))/2, dim=-1).numpy(), np.array(ids)

def load_mega():
    import timm
    m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0).to(DEVICE).eval()
    print(f"  [M] MegaDescriptor dim={m.num_features}"); return m, 384

def load_miew():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    c = json.load(open(hf_hub_download("conservationxlabs/miewid-msv3", "config.json")))
    m = timm.create_model(c.get("architecture", "efficientnetv2_rw_m"), pretrained=False, num_classes=0)
    s = {k: v for k, v in load_file(hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")).items() if "classifier" not in k}
    m.load_state_dict(s, strict=False); m = m.to(DEVICE).eval()
    print(f"  [M] MiewID dim={m.num_features}"); return m, 440

def load_dino():
    try:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", trust_repo=True).to(DEVICE).eval()
        print("  [M] DINOv2-L-reg dim=1024"); return m, 518
    except:
        try:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True).to(DEVICE).eval()
            print("  [M] DINOv2-L dim=1024"); return m, 518
        except Exception as e:
            print(f"  [M] DINOv2 fail: {e}"); return None, 0


# ═══════════════════════════════════════════════════════════
# LOCAL FEATURE MATCHER — This is now the PRIMARY signal
# ═══════════════════════════════════════════════════════════
class LocalMatcher:
    def __init__(self, max_kp=1024):
        self.ok = False
        try:
            from lightglue import LightGlue, ALIKED
            self.ext = ALIKED(max_num_keypoints=max_kp).eval().to(DEVICE)
            self.mat = LightGlue(features="aliked").eval().to(DEVICE)
            self.ok = True
            self.cache = {}
            print(f"  [L] ALIKED+LightGlue OK (kp={max_kp})")
        except Exception as e:
            print(f"  [L] FAILED: {e}")

    @torch.no_grad()
    def get_kp(self, path):
        if path in self.cache:
            return self.cache[path]
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            mx = 768
            if max(w, h) > mx:
                s = mx / max(w, h)
                img = img.resize((int(w*s), int(h*s)))
            t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
            f = self.ext.extract(t)
            self.cache[path] = f
            return f
        except:
            return None

    @torch.no_grad()
    def match(self, f0, f1):
        if f0 is None or f1 is None: return 0.0
        try:
            r = self.mat({"image0": f0, "image1": f1})
            if "matching_scores0" in r:
                sc = r["matching_scores0"]
                good = sc[sc > 0.5]
                if len(good) > 0:
                    return float(len(good)) * float(good.mean().item())
                return 0.0
            elif "matches0" in r:
                return float((r["matches0"] > -1).sum().item())
            return 0.0
        except:
            return 0.0

    def batch_match(self, pairs, paths, root, desc=""):
        """Match all pairs and return dict of scores."""
        if not self.ok:
            return {}
        # Extract unique keypoints
        unique = set()
        for i, j in pairs:
            unique.add(i); unique.add(j)
        print(f"    [{desc}] Extracting {len(unique)} keypoints...", flush=True)
        kp = {}
        for idx in tqdm(sorted(unique), desc=f"    kp", leave=False):
            kp[idx] = self.get_kp(os.path.join(root, paths[idx]))

        print(f"    [{desc}] Matching {len(pairs)} pairs...", flush=True)
        scores = {}
        for i, j in tqdm(pairs, desc=f"    match", leave=False):
            s = self.match(kp[i], kp[j])
            scores[(i, j)] = s
            scores[(j, i)] = s

        self.cache.clear()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return scores

# ═══════════════════════════════════════════════════════════
# SCORE CALIBRATION — Learn what "same individual" means
# ═══════════════════════════════════════════════════════════
def calibrate_scores(local_scores, pairs, labels, indices_in_species):
    """Use training pairs to calibrate local match scores.

    Build isotonic regression: local_score → P(same_individual)
    Training signal: pairs where we KNOW if same/different identity.
    """
    if not local_scores:
        return None

    # Collect training pairs with labels
    X, y = [], []
    for (i, j), score in local_scores.items():
        if i >= len(labels) or j >= len(labels):
            continue  # skip test-test pairs for calibration
        if i < len(labels) and j < len(labels):
            same = 1 if labels[i] == labels[j] else 0
            X.append(score)
            y.append(same)

    if len(X) < 50:
        print(f"    Too few calibration pairs ({len(X)}), skipping calibration")
        return None

    X, y = np.array(X), np.array(y)
    # Isotonic regression: monotonically increasing mapping score→probability
    ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    ir.fit(X, y)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"    Calibration: {len(X)} pairs ({n_pos} same, {n_neg} diff)")
    print(f"    Score range: [{X.min():.1f}, {X.max():.1f}]")

    return ir


def apply_calibration(local_scores, calibrator):
    """Apply calibrated scores."""
    if calibrator is None:
        # Simple normalization
        if not local_scores:
            return {}
        mx = max(local_scores.values())
        if mx <= 0:
            return {k: 0.0 for k in local_scores}
        return {k: v/mx for k, v in local_scores.items()}

    calibrated = {}
    for k, v in local_scores.items():
        calibrated[k] = float(calibrator.predict([v])[0])
    return calibrated


# ═══════════════════════════════════════════════════════════
# GRAPH CLUSTERING — Louvain community detection
# ═══════════════════════════════════════════════════════════
def louvain_cluster(n, edges, resolution=1.0):
    """Louvain community detection on weighted graph.

    edges: list of (i, j, weight)
    Returns: labels array
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i, j, w in edges:
            if w > 0.01 and i != j:
                G.add_edge(i, j, weight=w)

        communities = louvain_communities(G, weight='weight', resolution=resolution, seed=42)
        labels = np.zeros(n, dtype=int)
        for cidx, community in enumerate(communities):
            for node in community:
                labels[node] = cidx
        return labels

    except ImportError:
        print("    WARNING: networkx not available, falling back to simple threshold clustering")
        return threshold_cluster(n, edges)


def leiden_cluster(n, edges, resolution=1.0):
    """Try Leiden clustering (better than Louvain). Fallback to Louvain."""
    try:
        import igraph as ig
        import leidenalg

        G = ig.Graph()
        G.add_vertices(n)
        e_list, w_list = [], []
        for i, j, w in edges:
            if w > 0.01 and i != j:
                e_list.append((i, j))
                w_list.append(w)
        G.add_edges(e_list)

        partition = leidenalg.find_partition(
            G, leidenalg.RBConfigurationVertexPartition,
            weights=w_list, resolution_parameter=resolution, seed=42
        )
        return np.array(partition.membership)
    except ImportError:
        return louvain_cluster(n, edges, resolution)


def threshold_cluster(n, edges):
    """Simple connected-component clustering."""
    from scipy.sparse.csgraph import connected_components
    row, col, data = [], [], []
    for i, j, w in edges:
        if w > 0.1 and i != j:
            row.append(i); col.append(j); data.append(w)
            row.append(j); col.append(i); data.append(w)
    if not row:
        return np.arange(n)
    M = sp.csr_matrix((data, (row, col)), shape=(n, n))
    _, labels = connected_components(M, directed=False)
    return labels


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def get_top_k_pairs(sim, k, max_pairs):
    """Get top-K most similar pairs from global similarity matrix."""
    n = sim.shape[0]
    pairs = set()
    for i in range(n):
        s = sim[i].copy()
        s[i] = -999
        topk = np.argsort(s)[-k:]
        for j in topk:
            if s[j] > -0.5:  # very loose threshold
                pairs.add((min(i,j), max(i,j)))
    pairs = list(pairs)
    if len(pairs) > max_pairs:
        svals = [(sim[i,j], (i,j)) for i,j in pairs]
        svals.sort(reverse=True)
        pairs = [p for _,p in svals[:max_pairs]]
    return pairs


def search_resolution(n, edges, labels_true, train_mask):
    """Search best Louvain resolution using training data."""
    best_ari, best_res = -1, 1.0
    for res in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        pred = louvain_cluster(n, edges, resolution=res)
        # Evaluate only on training portion
        if train_mask is not None and train_mask.sum() > 0:
            ari = adjusted_rand_score(labels_true[train_mask], pred[train_mask])
        else:
            # No training data: prefer resolution that gives reasonable cluster count
            ratio = len(set(pred)) / n
            ari = -abs(ratio - 0.35)

        if ari > best_ari:
            best_ari, best_res = ari, res

    return best_res, best_ari


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V20 — LOCAL-PRIMARY PIPELINE (Complete Paradigm Shift)")
    print("=" * 70)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | {p.total_memory/1e9:.0f}GB")

    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    ssub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    trdf = meta[meta.split == "train"]
    tedf = meta[meta.split == "test"]
    alldf = pd.concat([trdf, tedf], ignore_index=True)
    print(f"  {len(trdf)} train + {len(tedf)} test = {len(alldf)} total")

    # ── STAGE 1: Global Features (for candidate retrieval ONLY) ──
    print(f"\n{'━'*65}")
    print("STAGE 1: Global Features (candidate retrieval only)")
    print(f"{'━'*65}")

    fd = {}
    m, sz = load_mega()
    fd["mega"], _ = extract(m, alldf, DATA_DIR, sz, 48, "Mega")
    del m; torch.cuda.empty_cache(); gc.collect()

    m, sz = load_miew()
    fd["miew"], _ = extract(m, alldf, DATA_DIR, sz, 32, "Miew")
    del m; torch.cuda.empty_cache(); gc.collect()

    dm, dsz = load_dino()
    if dm:
        fd["dino"], _ = extract(dm, alldf, DATA_DIR, dsz, 16, "DINO")
        del dm; torch.cuda.empty_cache(); gc.collect()

    all_ids = alldf.image_id.values
    id2idx = {int(all_ids[i]): i for i in range(len(all_ids))}
    all_paths = alldf.path.values
    print(f"  Features: {list(fd.keys())} | {(time.time()-t0)/60:.1f}min")

    # ── STAGE 2: Local Matcher ──
    print(f"\n{'━'*65}")
    print("STAGE 2: Local Feature Matcher (THE primary signal)")
    print(f"{'━'*65}")
    lm = LocalMatcher(max_kp=1024)

    # ── STAGE 3: Per-species LOCAL-PRIMARY pipeline ──
    print(f"\n{'━'*65}")
    print("STAGE 3: Per-species Graph-based Pipeline")
    print(f"{'━'*65}")

    preds = {}

    for sp in SP_ALL:
        print(f"\n  {'═'*55}")
        print(f"  {sp}")
        print(f"  {'═'*55}")

        te = tedf[tedf.dataset == sp].reset_index(drop=True)
        te_gidx = np.array([id2idx[int(x)] for x in te.image_id])  # global index
        nte = len(te)

        has_train = sp in SP_TRAIN
        if has_train:
            tr = trdf[trdf.dataset == sp].reset_index(drop=True)
            tr_gidx = np.array([id2idx[int(x)] for x in tr.image_id])
            tr_labels = tr.identity.values
            ntr = len(tr)
            n_ids = len(set(tr_labels))
            print(f"  {ntr} train ({n_ids} ids) + {nte} test")

            # Combined: train first, then test
            combined_gidx = np.concatenate([tr_gidx, te_gidx])
            combined_paths = [all_paths[i] for i in combined_gidx]
            n_combined = len(combined_gidx)
            train_mask = np.zeros(n_combined, dtype=bool)
            train_mask[:ntr] = True

            # Labels: train have real labels, test have -1
            combined_labels = np.full(n_combined, "unknown", dtype=object)
            combined_labels[:ntr] = tr_labels
        else:
            print(f"  No training data, {nte} test images")
            combined_gidx = te_gidx
            combined_paths = [all_paths[i] for i in combined_gidx]
            n_combined = nte
            train_mask = None
            combined_labels = None
            ntr = 0

        # ── Step A: Global similarity for candidate retrieval ──
        print(f"  [A] Global candidate retrieval (top-{TOP_K_GLOBAL})...")
        # Equal-weight average of all models for retrieval
        global_sim = np.zeros((n_combined, n_combined))
        for name in fd:
            f = normalize(fd[name][combined_gidx], axis=1)
            global_sim += (f @ f.T) / len(fd)

        # Get top-K candidate pairs
        pairs = get_top_k_pairs(global_sim, TOP_K_GLOBAL, MAX_LOCAL_PAIRS)
        print(f"    {len(pairs)} candidate pairs (from {n_combined} images)")

        # ── Step B: Local matching (PRIMARY signal) ──
        if lm.ok and len(pairs) > 0:
            print(f"  [B] Local matching (PRIMARY signal)...")
            raw_scores = lm.batch_match(pairs, combined_paths, DATA_DIR, desc=sp[:8])

            # ── Step C: Score calibration ──
            if has_train:
                print(f"  [C] Score calibration (isotonic regression)...")
                calibrator = calibrate_scores(
                    raw_scores, pairs, combined_labels, None
                )
                cal_scores = apply_calibration(raw_scores, calibrator)
            else:
                print(f"  [C] No training data, normalizing scores...")
                cal_scores = apply_calibration(raw_scores, None)

            # Also add global similarity as secondary signal
            # Fused score = alpha * local_calibrated + (1-alpha) * global
            alpha = 0.7  # local is PRIMARY

            # ── Step D: Build graph edges ──
            print(f"  [D] Building similarity graph...")
            edges = []
            for (i, j), local_w in cal_scores.items():
                if i < j:  # avoid duplicates
                    g_w = max(0, global_sim[i, j])
                    fused = alpha * local_w + (1 - alpha) * g_w
                    if fused > 0.02:
                        edges.append((i, j, fused))

            # Also add strong global-only edges that might have been missed
            for i in range(n_combined):
                for j in range(i+1, n_combined):
                    if (i, j) not in cal_scores and global_sim[i,j] > 0.85:
                        edges.append((i, j, global_sim[i,j] * 0.5))

            print(f"    {len(edges)} graph edges")

            # ── Step E: Search resolution & cluster ──
            print(f"  [E] Graph clustering (Louvain)...")
            if has_train:
                best_res, train_ari = search_resolution(
                    n_combined, edges, combined_labels, train_mask
                )
                print(f"    Best resolution={best_res:.1f}, train ARI={train_ari:.4f}")
            else:
                # Try a few resolutions, pick one giving reasonable cluster count
                best_res, best_ratio_diff = 1.0, 999
                for res in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                    pred = louvain_cluster(n_combined, edges, resolution=res)
                    ratio = len(set(pred)) / n_combined
                    diff = abs(ratio - 0.35)
                    if diff < best_ratio_diff:
                        best_ratio_diff = diff
                        best_res = res
                print(f"    Best resolution={best_res:.1f} (target ratio ~0.35)")

            final_labels = louvain_cluster(n_combined, edges, resolution=best_res)

        else:
            # Fallback: just use global similarity with threshold
            print(f"  [B-E] No local matcher, using global-only fallback...")
            dist = np.clip(1 - global_sim, 0, 2)
            np.fill_diagonal(dist, 0)
            from sklearn.cluster import AgglomerativeClustering

            best_th, best_ari = 0.5, -1
            if has_train:
                for th in np.arange(0.1, 1.5, 0.01):
                    try:
                        pred = AgglomerativeClustering(
                            n_clusters=None, distance_threshold=th,
                            metric="precomputed", linkage="average"
                        ).fit_predict(dist)
                        ari = adjusted_rand_score(combined_labels[train_mask], pred[train_mask])
                        if ari > best_ari:
                            best_ari, best_th = ari, th
                    except: pass
            else:
                best_th = 0.4

            final_labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=best_th,
                metric="precomputed", linkage="average"
            ).fit_predict(dist)

        # Extract test labels (test images start at index ntr)
        te_labels = final_labels[ntr:]
        n_clusters = len(set(te_labels))
        singletons = sum(1 for l in set(te_labels) if list(te_labels).count(l) == 1)

        print(f"  → {nte} images → {n_clusters} clusters, singletons={singletons} ({100*singletons/max(n_clusters,1):.0f}%)")

        # Store predictions
        for i in range(nte):
            preds[int(te.iloc[i].image_id)] = f"cluster_{sp}_{final_labels[ntr + i]}"

        # Evaluate on training data if available
        if has_train:
            tr_pred = final_labels[:ntr]
            tr_ari = adjusted_rand_score(tr_labels, tr_pred)
            print(f"  Train ARI = {tr_ari:.4f}")

    # ── STAGE 4: Submission ──
    print(f"\n{'━'*65}")
    print("STAGE 4: Submission")
    print(f"{'━'*65}")
    sub = ssub.copy()
    for i in range(len(sub)):
        iid = int(sub.iloc[i].image_id)
        if iid in preds:
            sub.at[i, "cluster"] = preds[iid]

    out = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out, index=False)
    print(f"  Output: {out}")
    print(f"  Rows: {len(sub)}, Clusters: {sub.cluster.nunique()}")
    for sp in SP_ALL:
        s = sub[sub.cluster.str.contains(sp)]
        if len(s) > 0:
            vc = s.cluster.value_counts()
            sing = (vc == 1).sum()
            print(f"    {sp:25s} {len(s):4d} imgs, {s.cluster.nunique():4d} cl, "
                  f"max_cl={vc.max()}, sing={sing}({100*sing/len(vc):.0f}%)")

    print(f"\n  Time: {(time.time()-t0)/60:.1f}min")
    print("=" * 70)
    print("  DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
