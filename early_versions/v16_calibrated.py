#!/usr/bin/env python3
"""
AnimalCLEF2026 V16 — Calibrated Fusion for Top5
=================================================
V14=0.205 (correct base), V15=0.068 (broken local injection)

V16策略：
  1. V14验证过的基础：MegaDescriptor + MiewID + TTA
  2. ★ 每物种搜索最优模型权重 α (train ARI grid search)
  3. ★ 局部匹配单独建相似度矩阵，不污染全局矩阵
  4. ★ 训练集搜索最优融合权重 β = global*(1-β) + local*β
  5. ★ 阈值在融合后的相似度上搜索
  6. DINOv3-ViT-H+作为第3个backbone（如果可用）
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
from sklearn.preprocessing import normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/animal-clef-2026"
OUT_DIR  = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/ov16"
os.makedirs(OUT_DIR, exist_ok=True)

SP_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
SP_ALL = SP_TRAIN + ["TexasHornedLizards"]

LOCAL_K = 30
LOCAL_MAX = 15000

# ═══════════════════════════════════════════════════════════
class AnimalDS(Dataset):
    def __init__(self, df, root, sz, flip=False):
        self.df, self.root, self.flip = df.reset_index(drop=True), root, flip
        self.tf = transforms.Compose([transforms.Resize((sz,sz)), transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try: img = Image.open(os.path.join(self.root, r["path"])).convert("RGB")
        except: img = Image.new("RGB",(384,384),(128,128,128))
        if self.flip: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tf(img), int(r["image_id"])

@torch.no_grad()
def extract(model, df, root, sz, bs=48):
    dl = DataLoader(AnimalDS(df, root, sz), batch_size=bs, num_workers=4, pin_memory=True)
    e, ids = [], []
    for imgs, iids in tqdm(dl, desc=f"  @{sz}", leave=False):
        e.append(model(imgs.to(DEVICE)).cpu()); ids.extend(iids.numpy())
    orig = torch.cat(e)
    dl2 = DataLoader(AnimalDS(df, root, sz, flip=True), batch_size=bs, num_workers=4, pin_memory=True)
    f = []
    for imgs, _ in tqdm(dl2, desc=f"  tta@{sz}", leave=False):
        f.append(model(imgs.to(DEVICE)).cpu())
    return F.normalize((orig + torch.cat(f))/2, dim=-1).numpy(), np.array(ids)

def load_mega():
    import timm
    m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0).to(DEVICE).eval()
    print(f"[M] MegaDescriptor OK dim={m.num_features}"); return m, 384

def load_miew():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    c = json.load(open(hf_hub_download("conservationxlabs/miewid-msv3","config.json")))
    m = timm.create_model(c.get("architecture","efficientnetv2_rw_m"), pretrained=False, num_classes=0)
    s = {k:v for k,v in load_file(hf_hub_download("conservationxlabs/miewid-msv3","model.safetensors")).items() if "classifier" not in k}
    m.load_state_dict(s, strict=False); m = m.to(DEVICE).eval()
    print(f"[M] MiewID OK dim={m.num_features}"); return m, 440

def load_dino():
    try:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", trust_repo=True)
        m = m.to(DEVICE).eval(); print("[M] DINOv2-ViT-L-reg OK dim=1024"); return m, 518
    except:
        try:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
            m = m.to(DEVICE).eval(); print("[M] DINOv2-ViT-L OK dim=1024"); return m, 518
        except Exception as e:
            print(f"[M] DINOv2 failed: {e}"); return None, 0

# ═══════════════════════════════════════════════════════════
# LOCAL MATCHING
# ═══════════════════════════════════════════════════════════
class LocalMatcher:
    def __init__(self):
        self.ok = False
        try:
            from lightglue import LightGlue, ALIKED
            self.ext = ALIKED(max_num_keypoints=512).eval().to(DEVICE)
            self.mat = LightGlue(features="aliked").eval().to(DEVICE)
            self.ok = True; print("[L] ALIKED+LightGlue OK")
        except Exception as e:
            print(f"[L] Failed: {e}")

    @torch.no_grad()
    def kp(self, path):
        try:
            img = Image.open(path).convert("RGB").resize((512,512))
            t = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
            return self.ext.extract(t)
        except: return None

    @torch.no_grad()
    def match(self, f0, f1):
        if f0 is None or f1 is None: return 0
        try:
            r = self.mat({"image0": f0, "image1": f1})
            if "matching_scores0" in r:
                return int((r["matching_scores0"] > 0.5).sum().item())
            elif "matches0" in r:
                return int((r["matches0"] > -1).sum().item())
            return 0
        except: return 0

    def build_local_sim(self, paths, n, pairs, root):
        """Build a sparse local similarity matrix from pairwise matching."""
        if not self.ok: return None

        # Extract keypoints
        unique = set()
        for i,j in pairs: unique.add(i); unique.add(j)
        print(f"      kp: {len(unique)} images...", flush=True)
        cache = {}
        for idx in tqdm(sorted(unique), desc="      kp", leave=False):
            cache[idx] = self.kp(os.path.join(root, paths[idx]))

        # Match
        print(f"      match: {len(pairs)} pairs...", flush=True)
        raw = np.zeros(len(pairs))
        for k,(i,j) in enumerate(tqdm(pairs, desc="      match", leave=False)):
            raw[k] = self.match(cache[i], cache[j])

        # Build symmetric similarity matrix, normalized to [0,1]
        mx = max(raw.max(), 1)
        local_sim = np.zeros((n, n))
        for k,(i,j) in enumerate(pairs):
            v = raw[k] / mx
            local_sim[i,j] = v
            local_sim[j,i] = v
        np.fill_diagonal(local_sim, 1.0)

        return local_sim

# ═══════════════════════════════════════════════════════════
# CORE: search optimal params on training data
# ═══════════════════════════════════════════════════════════
def search_model_weights(feats_dict, indices, labels, steps=21):
    """Search optimal model weights for global similarity fusion."""
    models = list(feats_dict.keys())
    if len(models) == 1:
        return {models[0]: 1.0}, -1, -1

    best_ari, best_w, best_th = -1, None, 0.5
    n = len(indices)

    # Pre-compute per-model similarities
    sims = {}
    for name in models:
        f = normalize(feats_dict[name][indices], axis=1)
        sims[name] = f @ f.T

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
                        best_ari = ari; best_w = {models[0]:a, models[1]:1-a}; best_th = th
                except: pass
    elif len(models) == 3:
        for a in np.linspace(0, 1, 11):
            for b in np.linspace(0, 1-a, 11):
                c = 1 - a - b
                sim = a*sims[models[0]] + b*sims[models[1]] + c*sims[models[2]]
                dist = np.clip(1-sim, 0, 2); np.fill_diagonal(dist, 0)
                for th in np.arange(0.05, 1.5, 0.02):
                    try:
                        pred = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                metric="precomputed", linkage="average").fit_predict(dist)
                        ari = adjusted_rand_score(labels, pred)
                        if ari > best_ari:
                            best_ari = ari; best_w = {models[0]:a, models[1]:b, models[2]:c}; best_th = th
                    except: pass

    return best_w, best_ari, best_th


def search_local_fusion(global_sim, local_sim, labels, base_th):
    """Search optimal fusion weight β and threshold for global+local."""
    if local_sim is None:
        return 0.0, base_th, -1

    best_ari, best_beta, best_th = -1, 0.0, base_th
    for beta in np.arange(0, 0.81, 0.05):
        fused = (1-beta)*global_sim + beta*local_sim
        dist = np.clip(1-fused, 0, 2); np.fill_diagonal(dist, 0)
        for th in np.arange(max(0.05, base_th-0.3), min(1.5, base_th+0.3), 0.02):
            try:
                pred = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                        metric="precomputed", linkage="average").fit_predict(dist)
                ari = adjusted_rand_score(labels, pred)
                if ari > best_ari:
                    best_ari = ari; best_beta = beta; best_th = th
            except: pass
    return best_beta, best_th, best_ari


def cluster(sim, th):
    dist = np.clip(1-sim, 0, 2); np.fill_diagonal(dist, 0)
    return AgglomerativeClustering(n_clusters=None, distance_threshold=th,
            metric="precomputed", linkage="average").fit_predict(dist)


def make_global_sim(feats_dict, indices, weights):
    sim = None
    for name, w in weights.items():
        f = normalize(feats_dict[name][indices], axis=1)
        s = f @ f.T
        sim = s*w if sim is None else sim + s*w
    return sim


def get_shortlist_pairs(sim, k, max_pairs):
    n = sim.shape[0]
    pairs = set()
    for i in range(n):
        s = sim[i].copy(); s[i] = -999
        for j in np.argsort(s)[-k:]:
            pairs.add((min(i,j), max(i,j)))
    pairs = list(pairs)
    if len(pairs) > max_pairs:
        svals = [(sim[i,j], (i,j)) for i,j in pairs]
        svals.sort(reverse=True)
        pairs = [p for _,p in svals[:max_pairs]]
    return pairs


# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*60)
    print("  AnimalCLEF2026 V16 — Calibrated Fusion")
    print("="*60)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | {p.total_memory/1e9:.0f}GB")

    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    ssub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    trdf = meta[meta.split=="train"]; tedf = meta[meta.split=="test"]
    alldf = pd.concat([trdf, tedf], ignore_index=True)
    ntr = len(trdf)
    print(f"  {ntr} train, {len(tedf)} test")

    # ── STAGE 1: Features ──
    print(f"\n{'━'*55}\nSTAGE 1: Features\n{'━'*55}")
    fd = {}

    m, sz = load_mega()
    fd["mega"], _ = extract(m, alldf, DATA_DIR, sz, 48)
    del m; torch.cuda.empty_cache(); gc.collect()

    m, sz = load_miew()
    fd["miew"], _ = extract(m, alldf, DATA_DIR, sz, 32)
    del m; torch.cuda.empty_cache(); gc.collect()

    dm, dsz = load_dino()
    if dm is not None:
        fd["dino"], _ = extract(dm, alldf, DATA_DIR, dsz, 16)
        del dm; torch.cuda.empty_cache(); gc.collect()

    all_ids = alldf.image_id.values
    id2idx = {int(all_ids[i]):i for i in range(len(all_ids))}
    all_paths = alldf.path.values

    print(f"  Models: {list(fd.keys())} | {(time.time()-t0)/60:.1f}min")

    # ── STAGE 2: Local matcher ──
    print(f"\n{'━'*55}\nSTAGE 2: Local Matcher\n{'━'*55}")
    lm = LocalMatcher()

    # ── STAGE 3: Per-species ──
    print(f"\n{'━'*55}\nSTAGE 3: Per-species Calibrated Pipeline\n{'━'*55}")

    preds = {}

    for sp in SP_ALL:
        print(f"\n  {'═'*48}\n  {sp}\n  {'═'*48}")
        te = tedf[tedf.dataset==sp].reset_index(drop=True)
        te_idx = np.array([id2idx[int(x)] for x in te.image_id])
        nte = len(te)

        if sp in SP_TRAIN:
            tr = trdf[trdf.dataset==sp].reset_index(drop=True)
            tr_idx = np.array([id2idx[int(x)] for x in tr.image_id])
            tr_labels = tr.identity.values

            # Sample train if too big for weight search
            if len(tr_idx) > 1200:
                rng = np.random.RandomState(42)
                si = rng.choice(len(tr_idx), 1200, replace=False)
                s_idx, s_lab = tr_idx[si], tr_labels[si]
            else:
                s_idx, s_lab = tr_idx, tr_labels

            # ── A: Search optimal model weights on train ──
            print(f"    [A] Model weight search ({len(s_idx)} samples)...")
            w, ari_g, th_g = search_model_weights(fd, s_idx, s_lab)
            print(f"        weights={w}")
            print(f"        global ARI={ari_g:.4f} th={th_g:.3f}")

            # ── B: Local matching on train sample for fusion calibration ──
            if lm.ok:
                print(f"    [B] Local matching calibration...")
                tr_global_sim = make_global_sim(fd, s_idx, w)
                tr_pairs = get_shortlist_pairs(tr_global_sim, min(LOCAL_K, 20), min(LOCAL_MAX, 8000))
                tr_paths = [all_paths[s_idx[i]] for i in range(len(s_idx))]
                tr_local_sim = lm.build_local_sim(tr_paths, len(s_idx), tr_pairs, DATA_DIR)
                beta, th_f, ari_f = search_local_fusion(tr_global_sim, tr_local_sim, s_lab, th_g)
                print(f"        fused: β={beta:.2f} th={th_f:.3f} ARI={ari_f:.4f} (was {ari_g:.4f})")

                if ari_f <= ari_g:
                    beta = 0.0; th_f = th_g
                    print(f"        → local hurts, using global only")
            else:
                beta, th_f = 0.0, th_g

            # ── C: Cluster test ──
            print(f"    [C] Test clustering...")
            te_global = make_global_sim(fd, te_idx, w)

            if beta > 0 and lm.ok:
                te_pairs = get_shortlist_pairs(te_global, LOCAL_K, LOCAL_MAX)
                te_paths = [all_paths[te_idx[i]] for i in range(nte)]
                te_local = lm.build_local_sim(te_paths, nte, te_pairs, DATA_DIR)
                te_sim = (1-beta)*te_global + beta*te_local
            else:
                te_sim = te_global

            labels = cluster(te_sim, th_f)
            print(f"    → {len(set(labels))} clusters")

        else:
            # TexasHornedLizards — no train
            print(f"    No training data, {nte} test images")
            # Use average weights from other species or equal
            w = {k: 1.0/len(fd) for k in fd}
            te_global = make_global_sim(fd, te_idx, w)

            if lm.ok:
                te_pairs = get_shortlist_pairs(te_global, LOCAL_K, LOCAL_MAX)
                te_paths = [all_paths[te_idx[i]] for i in range(nte)]
                te_local = lm.build_local_sim(te_paths, nte, te_pairs, DATA_DIR)
                # Try multiple betas
                best_th, best_score, best_beta = 0.4, -1, 0.0
                for beta in [0.0, 0.1, 0.2, 0.3]:
                    sim = (1-beta)*te_global + beta*te_local if te_local is not None else te_global
                    dist = np.clip(1-sim, 0, 2); np.fill_diagonal(dist, 0)
                    for th in np.arange(0.10, 1.50, 0.01):
                        try:
                            pred = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                    metric="precomputed", linkage="average").fit_predict(dist)
                            ratio = len(set(pred)) / nte
                            if 0.15 <= ratio <= 0.55:
                                sc = -abs(ratio - 0.35)
                                if sc > best_score: best_score, best_th, best_beta = sc, th, beta
                        except: pass
                te_sim = (1-best_beta)*te_global + best_beta*te_local if te_local is not None else te_global
                print(f"    β={best_beta:.1f} th={best_th:.3f}")
            else:
                te_sim = te_global
                best_th = 0.40
                dist = np.clip(1-te_sim, 0, 2); np.fill_diagonal(dist, 0)
                for th in np.arange(0.10, 1.50, 0.01):
                    try:
                        pred = AgglomerativeClustering(n_clusters=None, distance_threshold=th,
                                metric="precomputed", linkage="average").fit_predict(dist)
                        ratio = len(set(pred)) / nte
                        if 0.15 <= ratio <= 0.55:
                            sc = -abs(ratio - 0.35)
                            if sc > -abs(0.35-0.35): best_th = th; break
                    except: pass

            labels = cluster(te_sim, best_th)
            print(f"    → {len(set(labels))} clusters")

        for i in range(nte):
            preds[int(te.iloc[i].image_id)] = f"cluster_{sp}_{labels[i]}"

    # ── STAGE 4: Submit ──
    print(f"\n{'━'*55}\nSTAGE 4: Submission\n{'━'*55}")
    sub = ssub.copy()
    for i in range(len(sub)):
        iid = int(sub.iloc[i].image_id)
        if iid in preds: sub.at[i,"cluster"] = preds[iid]
    out = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out, index=False)
    print(f"  {out}")
    print(f"  Rows={len(sub)} Clusters={sub.cluster.nunique()}")
    for sp in SP_ALL:
        s = sub[sub.cluster.str.contains(sp)]
        print(f"    {sp:25s} {len(s):4d} imgs, {s.cluster.nunique()} cl")
    print(f"  Time: {(time.time()-t0)/60:.1f}min")
    print("DONE!")

if __name__ == "__main__":
    main()
