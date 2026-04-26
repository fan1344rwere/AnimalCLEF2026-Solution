#!/usr/bin/env python
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AnimalCLEF2026 V12 — H800 ULTIMATE SOLUTION                              ║
║  Target ARI ≥ 0.60+ | Based on 2025 Winner Analysis                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Core Strategy (validated by 2025 AnimalCLEF winner papers):
  ★ ALIKED + LightGlue local matching (the +20pp weapon for Lynx/Salamander)
  ★ MegaDescriptor + MiewID + EVA02 triple-backbone global features
  ★ WildFusion-style isotonic calibration for principled score fusion
  ★ 5-crop + HFlip TTA (10 views) for robust test features
  ★ ArcFace fine-tuning for SeaTurtle (already high ARI → push to 0.95)
  ★ Orientation-aware similarity for Lynx/Salamander
  ★ Ensemble clustering: DBSCAN + Agglo + HDBSCAN → consensus
  ★ Train-anchored clustering for species with training data
  ★ Pseudo-label refinement (2 rounds)

INSTALL (run on AutoDL server first):
  pip install torch torchvision timm tqdm scikit-learn pandas numpy Pillow
  pip install safetensors huggingface_hub hdbscan
  pip install open_clip_torch    # for EVA02
  pip install lightglue           # for ALIKED + LightGlue
  # If lightglue fails: pip install git+https://github.com/cvg/LightGlue.git

USAGE:
  python -u v12_ultimate_h800.py [DATA_DIR] [OUTPUT_DIR]
  # Default: DATA_DIR=/root/autodl-tmp/animal-clef-2026, OUTPUT_DIR=/root/autodl-tmp/ov12
"""

import os, sys, gc, warnings, time as _t, json
from collections import defaultdict
from pathlib import Path

# ── HuggingFace mirror for China servers ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ── Optional dependencies (graceful fallback) ──
try:
    import hdbscan as _hdbscan; HAS_HDBSCAN = True
except ImportError: HAS_HDBSCAN = False; print("[WARN] hdbscan not found")

try:
    from lightglue import LightGlue as _LG, ALIKED as _ALIKED_Ex
    HAS_LIGHTGLUE = True
except ImportError:
    HAS_LIGHTGLUE = False
    print("[WARN] lightglue not found, trying kornia ALIKED fallback...")

try:
    import open_clip; HAS_OPENCLIP = True
except ImportError: HAS_OPENCLIP = False; print("[WARN] open_clip not found")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — H800 (80 GB VRAM) optimised
# ═══════════════════════════════════════════════════════════════════════════
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMEAN = [0.485, 0.456, 0.406]; ISTD = [0.229, 0.224, 0.225]

class CFG:
    # ── Paths ──
    DATA_DIR   = "/root/autodl-tmp/animal-clef-2026"
    OUTPUT_DIR = "/root/autodl-tmp/ov12"

    # ── Backbones ──
    USE_MEGA = True;   MEGA_HUB = "hf-hub:BVRA/MegaDescriptor-L-384";  MEGA_SZ = 384
    USE_MIEW = True;   MIEW_SZ  = 440
    USE_EVA02 = True;  EVA02_SZ = 336  # EVA02-L-14-336

    # ── Inference ──
    BS = 48; WORKERS = 6
    USE_TTA = True; TTA_NCROPS = 5; TTA_HFLIP = True  # → 10 views

    # ── Local Matching ──
    USE_LOCAL = True
    LOCAL_TOPK = 30        # top-K candidates for ALIKED local matching
    LOCAL_WEIGHT = 0.35    # weight of local score in final fusion

    # ── Fusion weights (global similarity) ──
    W_MEGA = 0.40; W_MIEW = 0.35; W_EVA02 = 0.25

    # ── ArcFace Fine-tuning ──
    USE_ARCFACE = True
    FT_EPOCHS = 15; FT_LR = 3e-4; ARC_S = 64.0; ARC_M = 0.5
    WARMUP_EP = 2; WD = 1e-4
    FT_BLEND = 0.50   # weight of fine-tuned features in fusion

    # ── Clustering ──
    DBSCAN_MIN = 2
    USE_ENSEMBLE = True

    # ── Pseudo-label ──
    PSEUDO_ROUNDS = 2

    # ── Species ──
    DS_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
    DS_ALL   = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]


# ═══════════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════════
def _orient(img, row):
    """Orientation normalization for salamanders (from competition data)."""
    if row.get("species") == "salamander" and pd.notna(row.get("orientation")):
        o = str(row["orientation"]).lower()
        if o == "right":  img = img.rotate(-90, expand=True)
        elif o == "left": img = img.rotate(90, expand=True)
    return img

def _mk_tf(sz):
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(IMEAN, ISTD)])

class InferDS(Dataset):
    def __init__(s, df, root, tf):
        s.df = df.reset_index(drop=True); s.root = root; s.tf = tf
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]
        try: img = Image.open(os.path.join(s.root, r["path"])).convert("RGB")
        except: img = Image.new("RGB", (384, 384), (128, 128, 128))
        img = _orient(img, r)
        return s.tf(img), int(r["image_id"])

class TTADS(Dataset):
    """10-view TTA: 5-crop × horizontal flip."""
    def __init__(s, df, root, sz, flip=True):
        s.df = df.reset_index(drop=True); s.root = root; s.sz = sz; s.flip = flip
        s.norm = transforms.Normalize(IMEAN, ISTD); s.tt = transforms.ToTensor()
    def __len__(s): return len(s.df)
    def _crops(s, img):
        sz = s.sz; w, h = img.size
        r = (sz * 1.15) / min(w, h)
        nw, nh = int(w * r), int(h * r)
        img = img.resize((nw, nh), Image.BICUBIC)
        cx, cy = nw // 2, nh // 2
        cc = [
            img.crop((cx - sz // 2, cy - sz // 2, cx - sz // 2 + sz, cy - sz // 2 + sz)),
            img.crop((0, 0, sz, sz)),
            img.crop((nw - sz, 0, nw, sz)),
            img.crop((0, nh - sz, sz, nh)),
            img.crop((nw - sz, nh - sz, nw, nh))]
        out = []
        for c in cc:
            c = c.resize((sz, sz), Image.BICUBIC)
            out.append(c)
            if s.flip:
                out.append(c.transpose(Image.FLIP_LEFT_RIGHT))
        return out
    def __getitem__(s, i):
        r = s.df.iloc[i]
        try: img = Image.open(os.path.join(s.root, r["path"])).convert("RGB")
        except: img = Image.new("RGB", (s.sz * 2, s.sz * 2), (128, 128, 128))
        img = _orient(img, r)
        crops = s._crops(img)
        ts = [s.norm(s.tt(c)) for c in crops]
        return torch.stack(ts), int(r["image_id"])

class TrainDS(Dataset):
    def __init__(s, df, root, sz, lcol="label"):
        s.df = df.reset_index(drop=True); s.root = root; s.lcol = lcol
        s.tf = transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(.7, 1.), ratio=(.8, 1.2)),
            transforms.RandomHorizontalFlip(.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(.3, .3, .2, .1),
            transforms.RandomGrayscale(.05),
            transforms.GaussianBlur(5, sigma=(.1, 2.)),
            transforms.ToTensor(),
            transforms.Normalize(IMEAN, ISTD),
            transforms.RandomErasing(p=.2, scale=(.02, .15))])
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]
        try: img = Image.open(os.path.join(s.root, r["path"])).convert("RGB")
        except: img = Image.new("RGB", (384, 384), (128, 128, 128))
        img = _orient(img, r)
        return s.tf(img), int(r[s.lcol])


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_mega():
    import timm
    print("[MODEL] MegaDescriptor-L-384 ...", end=" ", flush=True)
    m = timm.create_model(CFG.MEGA_HUB, pretrained=True, num_classes=0).to(DEV).eval()
    print(f"OK (dim={m.num_features})")
    return m

def load_miew():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    print("[MODEL] MiewID-msv3 ...", end=" ", flush=True)
    cfg_path = hf_hub_download("conservationxlabs/miewid-msv3", "config.json")
    with open(cfg_path) as f: cfg = json.load(f)
    arch = cfg.get("architecture", cfg.get("model_name", "efficientnetv2_rw_m"))
    m = timm.create_model(arch, pretrained=False, num_classes=0)
    wt_path = hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")
    state = load_file(wt_path)
    state = {k: v for k, v in state.items() if "classifier" not in k}
    m.load_state_dict(state, strict=False)
    m = m.eval().to(DEV)
    print(f"OK (dim={m.num_features})")
    return m

def load_eva02():
    if not HAS_OPENCLIP:
        print("[MODEL] EVA02 skipped (open_clip not installed)")
        return None
    print("[MODEL] EVA02-L-14-336 ...", end=" ", flush=True)
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'EVA02-L-14-336', pretrained='merged2b_s6b_b61k')
        model = model.eval().to(DEV)
        print("OK (dim=768)")
        return model
    except Exception as e:
        print(f"FAIL: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — with TTA
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_features(model, df, root, sz, tta=False, is_eva02=False):
    """Extract features from a model. Supports TTA and EVA02 special handling."""
    if tta and CFG.USE_TTA:
        nv = CFG.TTA_NCROPS * (2 if CFG.TTA_HFLIP else 1)
        ds = TTADS(df, root, sz, flip=CFG.TTA_HFLIP)
        bs = max(1, CFG.BS // nv)
        loader = DataLoader(ds, batch_size=bs, num_workers=CFG.WORKERS, pin_memory=True)
        embs, ids = [], []
        for crops, iids in tqdm(loader, desc=f"  TTA@{sz}({nv}v)", leave=False):
            B, N, C, H, W = crops.shape
            if is_eva02:
                out = model.encode_image(crops.view(B * N, C, H, W).to(DEV))
            else:
                out = model(crops.view(B * N, C, H, W).to(DEV))
            out = F.normalize(out.view(B, N, -1).mean(1), dim=-1)
            embs.append(out.cpu().numpy())
            ids.extend(iids.numpy())
    else:
        tf = _mk_tf(sz)
        ds = InferDS(df, root, tf)
        loader = DataLoader(ds, batch_size=CFG.BS, num_workers=CFG.WORKERS, pin_memory=True)
        embs, ids = [], []
        for imgs, iids in tqdm(loader, desc=f"  @{sz}", leave=False):
            if is_eva02:
                out = model.encode_image(imgs.to(DEV))
            else:
                out = model(imgs.to(DEV))
            out = F.normalize(out, dim=-1)
            embs.append(out.cpu().numpy())
            ids.extend(iids.numpy())
    return np.concatenate(embs), np.array(ids)


# ═══════════════════════════════════════════════════════════════════════════
# ★ ALIKED + LIGHTGLUE LOCAL MATCHING (the +20pp weapon)
# ═══════════════════════════════════════════════════════════════════════════
class LocalMatcher:
    """ALIKED keypoint extraction + LightGlue matching."""

    def __init__(self):
        self.available = False
        if HAS_LIGHTGLUE:
            try:
                self.extractor = _ALIKED_Ex(
                    max_num_keypoints=2048,
                    detection_threshold=0.01,
                    resize=512
                ).eval().to(DEV)
                self.matcher = _LG(features="aliked").eval().to(DEV)
                self.available = True
                print("[LOCAL] ALIKED + LightGlue loaded OK")
            except Exception as e:
                print(f"[LOCAL] LightGlue init failed: {e}")

        if not self.available:
            # Fallback: try kornia
            try:
                import kornia.feature as KF
                self.kn_matcher = KF.LoFTR(pretrained='outdoor').eval().to(DEV)
                self.available = True
                self._use_loftr = True
                print("[LOCAL] Fallback: kornia LoFTR loaded OK")
            except Exception as e:
                self._use_loftr = False
                print(f"[LOCAL] No local matcher available: {e}")

    def _load_img(self, path, sz=512):
        """Load image as tensor [1, 3, H, W]."""
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (sz, sz), (128, 128, 128))
        tf = transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor()])
        return tf(img).unsqueeze(0)

    @torch.no_grad()
    def match_pair(self, path1, path2, root):
        """Return local matching score (0-1) between two images."""
        if not self.available:
            return 0.0
        try:
            p1 = os.path.join(root, path1)
            p2 = os.path.join(root, path2)

            if hasattr(self, '_use_loftr') and self._use_loftr:
                # LoFTR fallback
                img1 = self._load_img(p1).to(DEV)
                img2 = self._load_img(p2).to(DEV)
                gray1 = img1.mean(dim=1, keepdim=True)
                gray2 = img2.mean(dim=1, keepdim=True)
                result = self.kn_matcher({"image0": gray1, "image1": gray2})
                n_matches = len(result["confidence"])
                return min(1.0, n_matches / 80.0)
            else:
                # ALIKED + LightGlue (preferred)
                from lightglue.utils import load_image
                img0 = load_image(p1).to(DEV)
                img1 = load_image(p2).to(DEV)
                feats0 = self.extractor.extract(img0)
                feats1 = self.extractor.extract(img1)
                matches01 = self.matcher({"image0": feats0, "image1": feats1})
                # Extract number of matches
                if "matches" in matches01:
                    if isinstance(matches01["matches"], list):
                        n = len(matches01["matches"])
                    else:
                        n = matches01["matches"].shape[0] if matches01["matches"].dim() == 2 else len(matches01["matches"])
                elif "matches0" in matches01:
                    m = matches01["matches0"]
                    n = (m > -1).sum().item() if m.dim() == 1 else m.shape[0]
                else:
                    n = 0
                return min(1.0, n / 50.0)
        except Exception as e:
            return 0.0

    def compute_pairwise_local(self, df, root, global_sim, topk=30):
        """
        For each image, match against top-K global-similar candidates.
        Returns enhanced similarity matrix (sparse local + dense global).
        """
        if not self.available:
            print("  [LOCAL] Not available, returning global sim")
            return global_sim

        n = len(df)
        local_scores = np.zeros((n, n), dtype=np.float32)

        total_pairs = 0
        print(f"  [LOCAL] Matching {n} images × top-{topk}...", flush=True)
        for i in tqdm(range(n), desc="  ALIKED", leave=False):
            topk_idx = np.argsort(-global_sim[i])[:topk + 1]
            for j in topk_idx:
                if j == i: continue
                if local_scores[i, j] > 0 or local_scores[j, i] > 0:
                    # Already computed (symmetric)
                    if local_scores[i, j] == 0:
                        local_scores[i, j] = local_scores[j, i]
                    continue
                score = self.match_pair(df.iloc[i]["path"], df.iloc[j]["path"], root)
                local_scores[i, j] = score
                local_scores[j, i] = score
                total_pairs += 1

        print(f"  [LOCAL] Computed {total_pairs} pairs")
        return local_scores


# ═══════════════════════════════════════════════════════════════════════════
# ★ ISOTONIC CALIBRATION (WildFusion-style)
# ═══════════════════════════════════════════════════════════════════════════
def build_calibration_pairs(feats, identities, n_pos=3000, n_neg=6000):
    """Build pos/neg pairs from training data for isotonic calibration."""
    feat = normalize(feats, axis=1)
    id_to_idx = defaultdict(list)
    for i, ident in enumerate(identities):
        id_to_idx[ident].append(i)

    rng = np.random.RandomState(42)
    pos_scores, neg_scores = [], []

    # Positive pairs (same individual)
    for ident, idxs in id_to_idx.items():
        if len(idxs) < 2: continue
        pairs = min(10, len(idxs) * (len(idxs) - 1) // 2)
        for _ in range(pairs):
            a, b = rng.choice(idxs, 2, replace=False)
            pos_scores.append(float(feat[a] @ feat[b]))
    if len(pos_scores) > n_pos:
        pos_scores = list(rng.choice(pos_scores, n_pos, replace=False))

    # Negative pairs (different individual)
    all_ids = [k for k, v in id_to_idx.items() if len(v) > 0]
    for _ in range(n_neg):
        id1, id2 = rng.choice(all_ids, 2, replace=False)
        a = rng.choice(id_to_idx[id1])
        b = rng.choice(id_to_idx[id2])
        neg_scores.append(float(feat[a] @ feat[b]))

    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    return scores, labels


def train_calibrator(scores, labels):
    """Train isotonic regression calibrator."""
    ir = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    ir.fit(scores, labels)
    return ir


def calibrate_similarity_matrix(sim, calibrator):
    """Apply isotonic calibration to a similarity matrix."""
    n = sim.shape[0]
    flat = sim.ravel()
    cal_flat = calibrator.predict(flat)
    return cal_flat.reshape(n, n)


# ═══════════════════════════════════════════════════════════════════════════
# ORIENTATION-AWARE SIMILARITY ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════
def orientation_adjust(sim, df, boost=1.15, penalty=0.85):
    """Same orientation → boost, different orientation → penalty."""
    if "orientation" not in df.columns:
        return sim
    oris = df["orientation"].fillna("unknown").str.lower().values
    n = len(oris)
    adj = sim.copy()
    for i in range(n):
        if oris[i] in ("unknown", "nan", ""): continue
        for j in range(i + 1, n):
            if oris[j] in ("unknown", "nan", ""): continue
            if oris[i] == oris[j]:
                adj[i, j] *= boost
                adj[j, i] *= boost
            else:
                adj[i, j] *= penalty
                adj[j, i] *= penalty
    return np.clip(adj, -1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# ARCFACE HEAD + FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════
class ArcHead(nn.Module):
    def __init__(s, d, n, sc=64., m=.5):
        super().__init__(); s.s = sc; s.m = m
        s.W = nn.Parameter(torch.empty(n, d)); nn.init.xavier_uniform_(s.W)
        s.cm = np.cos(m); s.sm = np.sin(m)
        s.th = np.cos(np.pi - m); s.mm = np.sin(np.pi - m) * m
    def forward(s, x, y=None):
        cos = F.linear(F.normalize(x), F.normalize(s.W))
        if y is None: return cos * s.s
        sin = (1 - cos.pow(2).clamp(0, 1)).sqrt()
        phi = cos * s.cm - sin * s.sm
        phi = torch.where(cos > s.th, phi, cos - s.mm)
        oh = torch.zeros_like(cos).scatter_(1, y.unsqueeze(1), 1.)
        return (oh * phi + (1 - oh) * cos) * s.s


def finetune_arcface(tdf, root, dsn, c=CFG):
    """ArcFace fine-tuning with 3-stage progressive unfreezing."""
    import timm
    le = LabelEncoder(); tdf = tdf.copy()
    tdf["label"] = le.fit_transform(tdf.identity.values)
    ncls = tdf.label.nunique()
    print(f"  [{dsn}] ArcFace: {ncls} classes, {len(tdf)} images")

    bb = timm.create_model(c.MEGA_HUB, pretrained=True, num_classes=0).to(DEV)
    edim = bb.num_features
    hd = ArcHead(edim, ncls, c.ARC_S, c.ARC_M).to(DEV)
    ds = TrainDS(tdf, root, c.MEGA_SZ)

    ccnt = tdf.label.value_counts().to_dict()
    sw = [1. / ccnt[int(r.label)] for _, r in tdf.iterrows()]
    loader = DataLoader(ds, batch_size=c.BS, sampler=WeightedRandomSampler(sw, len(sw)),
                        num_workers=c.WORKERS, pin_memory=True, drop_last=True)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    nps = list(bb.named_parameters())
    top25 = {n for n, _ in nps[int(len(nps) * .75):]}

    def _ep(opt, freeze_bb=False, tag=""):
        bb.train(); hd.train()
        ls, co, to = [], 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEV), labels.to(DEV)
            if freeze_bb:
                with torch.no_grad(): emb = F.normalize(bb(imgs), dim=-1)
            else:
                emb = F.normalize(bb(imgs), dim=-1)
            logits = hd(emb, labels); loss = crit(logits, labels)
            opt.zero_grad(); loss.backward()
            if not freeze_bb: torch.nn.utils.clip_grad_norm_(bb.parameters(), 1.)
            opt.step()
            ls.append(loss.item())
            co += (logits.argmax(1) == labels).sum().item(); to += labels.size(0)
        print(f"    {tag}: loss={np.mean(ls):.4f} acc={100 * co / to:.1f}%")

    # Stage 1: Head only (warmup)
    for p in bb.parameters(): p.requires_grad = False
    o1 = torch.optim.AdamW(hd.parameters(), lr=c.FT_LR, weight_decay=c.WD)
    for e in range(c.WARMUP_EP):
        _ep(o1, freeze_bb=True, tag=f"S1-e{e + 1}")

    # Stage 2: Top 25% of backbone + head
    for n, p in bb.named_parameters(): p.requires_grad = (n in top25)
    o2 = torch.optim.AdamW([
        {"params": [p for n, p in bb.named_parameters() if p.requires_grad], "lr": c.FT_LR * .1},
        {"params": hd.parameters(), "lr": c.FT_LR}], weight_decay=c.WD)
    mid = min(5, c.FT_EPOCHS - c.WARMUP_EP - 2)
    for e in range(mid): _ep(o2, tag=f"S2-e{e + 1}")

    # Stage 3: Full fine-tune with cosine schedule
    for p in bb.parameters(): p.requires_grad = True
    rem = c.FT_EPOCHS - c.WARMUP_EP - mid
    o3 = torch.optim.AdamW([
        {"params": bb.parameters(), "lr": c.FT_LR * .03},
        {"params": hd.parameters(), "lr": c.FT_LR * .3}], weight_decay=c.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(o3, T_max=max(rem, 1))
    for e in range(rem):
        _ep(o3, tag=f"S3-e{e + 1}"); sch.step()

    bb.eval()
    return bb, le


@torch.no_grad()
def extract_finetuned(model, df, root, sz):
    """Extract features from a fine-tuned model."""
    tf = _mk_tf(sz)
    ds = InferDS(df, root, tf)
    loader = DataLoader(ds, batch_size=CFG.BS, num_workers=CFG.WORKERS, pin_memory=True)
    embs = []
    for imgs, _ in tqdm(loader, desc="    ft-extract", leave=False):
        embs.append(F.normalize(model(imgs.to(DEV)), dim=-1).cpu().numpy())
    return np.concatenate(embs)


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERING METHODS
# ═══════════════════════════════════════════════════════════════════════════
def cc_cluster(sim, threshold):
    """Connected-component clustering from similarity matrix."""
    n = sim.shape[0]; parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[a] = b
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold: union(i, j)
    raw = [find(i) for i in range(n)]
    uq = {}; out = np.zeros(n, dtype=int); cnt = 0
    for i in range(n):
        r = find(i)
        if r not in uq: uq[r] = cnt; cnt += 1
        out[i] = uq[r]
    return out


def fix_noise(labels):
    """Assign noise points (-1) to unique clusters."""
    labels = labels.copy()
    ns = labels.max() + 1 if labels.max() >= 0 else 0
    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = ns; ns += 1
    return labels


def autotune_clustering(dist, sim, yt, label=""):
    """Auto-tune DBSCAN + Agglomerative + CC + HDBSCAN on training data."""
    results = {}

    # DBSCAN sweep
    ba, be = -1, .35
    for eps in np.arange(.06, 1.20, .015):
        pred = fix_noise(DBSCAN(eps=eps, min_samples=CFG.DBSCAN_MIN,
                                metric="precomputed").fit_predict(dist))
        a = adjusted_rand_score(yt, pred)
        if a > ba: ba, be = a, eps
    results["dbscan_eps"] = be; results["dbscan_ari"] = ba
    print(f"  [{label}] DBSCAN: eps={be:.3f} ARI={ba:.4f}")

    # Agglomerative sweep (average linkage)
    ba2, bd = -1, .5
    for dt in np.arange(.10, 1.20, .015):
        try:
            pred = AgglomerativeClustering(n_clusters=None, distance_threshold=dt,
                                           metric="precomputed", linkage="average").fit_predict(dist)
            a = adjusted_rand_score(yt, pred)
            if a > ba2: ba2, bd = a, dt
        except: pass
    results["agglo_dt"] = bd; results["agglo_ari"] = ba2
    print(f"  [{label}] Agglo:  dt={bd:.3f} ARI={ba2:.4f}")

    # Connected Components sweep
    ba3, bt = -1, .6
    for th in np.arange(.35, .90, .015):
        pred = cc_cluster(sim, th)
        a = adjusted_rand_score(yt, pred)
        if a > ba3: ba3, bt = a, th
    results["cc_thresh"] = bt; results["cc_ari"] = ba3
    print(f"  [{label}] CC:     th={bt:.3f} ARI={ba3:.4f}")

    # HDBSCAN
    if HAS_HDBSCAN:
        ba4, bm = -1, 3
        for ms in range(2, 10):
            try:
                cl = _hdbscan.HDBSCAN(min_cluster_size=ms, metric='precomputed')
                pred = fix_noise(cl.fit_predict(dist))
                a = adjusted_rand_score(yt, pred)
                if a > ba4: ba4, bm = a, ms
            except: pass
        results["hdbscan_min"] = bm; results["hdbscan_ari"] = ba4
        print(f"  [{label}] HDBSCAN: min_cs={bm} ARI={ba4:.4f}")

    return results


def cluster_ensemble(dist, sim, params, label=""):
    """Consensus clustering from multiple algorithms."""
    n = dist.shape[0]
    preds = []
    names = []

    # DBSCAN
    p = fix_noise(DBSCAN(eps=params["dbscan_eps"], min_samples=CFG.DBSCAN_MIN,
                         metric="precomputed").fit_predict(dist))
    preds.append(p); names.append("DBSCAN")

    # Agglomerative
    try:
        p2 = AgglomerativeClustering(n_clusters=None, distance_threshold=params["agglo_dt"],
                                      metric="precomputed", linkage="average").fit_predict(dist)
        preds.append(p2); names.append("Agglo")
    except:
        preds.append(p.copy()); names.append("Agglo(fb)")

    # CC
    p3 = cc_cluster(sim, params["cc_thresh"])
    preds.append(p3); names.append("CC")

    # HDBSCAN
    if HAS_HDBSCAN and "hdbscan_min" in params:
        try:
            cl = _hdbscan.HDBSCAN(min_cluster_size=params["hdbscan_min"], metric='precomputed')
            p4 = fix_noise(cl.fit_predict(dist))
            preds.append(p4); names.append("HDBSCAN")
        except: pass

    # Consensus: ≥ majority agree → same cluster
    threshold = len(preds) // 2 + 1
    cooccur = np.zeros((n, n), dtype=int)
    for pred in preds:
        for i in range(n):
            for j in range(i + 1, n):
                if pred[i] == pred[j]:
                    cooccur[i, j] += 1
                    cooccur[j, i] += 1
    consensus_sim = (cooccur >= threshold).astype(float) + np.eye(n)
    final = cc_cluster(consensus_sim, 0.5)

    counts = [len(set(p)) for p in preds]
    print(f"  [{label}] Ensemble: " +
          " ".join(f"{nm}={c}" for nm, c in zip(names, counts)) +
          f" → Consensus={len(set(final))}")
    return final


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP THRESHOLD AUTO-TUNE
# ═══════════════════════════════════════════════════════════════════════════
def autotune_lookup(sim_matrix, yt, dsn):
    """Find best threshold for lookup on train-vs-train similarity."""
    n = len(yt)
    np.fill_diagonal(sim_matrix, -1)
    ba, bt = -1, .5
    for th in np.arange(.25, .85, .01):
        pred = np.full(n, -1)
        for i in range(n):
            j = sim_matrix[i].argmax()
            if sim_matrix[i, j] >= th:
                pred[i] = yt[j]
            else:
                pred[i] = n + i
        a = adjusted_rand_score(yt, pred)
        if a > ba: ba, bt = a, th
    print(f"  [{dsn}] Lookup: th={bt:.2f} ARI={ba:.4f}")
    return bt, ba


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t0 = _t.time()
    CFG.DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else CFG.DATA_DIR
    CFG.OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else CFG.OUTPUT_DIR
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  AnimalCLEF2026 V12 — H800 ULTIMATE SOLUTION")
    print("=" * 70)
    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        print(f"GPU: {g.name} | VRAM: {g.total_memory / 1e9:.1f} GB")
    print(f"HDBSCAN: {HAS_HDBSCAN} | LightGlue: {HAS_LIGHTGLUE} | OpenCLIP: {HAS_OPENCLIP}")

    # ── Load data ──
    meta = pd.read_csv(os.path.join(CFG.DATA_DIR, "metadata.csv"))
    ssub = pd.read_csv(os.path.join(CFG.DATA_DIR, "sample_submission.csv"))
    trdf = meta[meta.split == "train"].copy()
    tedf = meta[meta.split == "test"].copy()
    print(f"\nData: {len(trdf)} train, {len(tedf)} test")
    for d in CFG.DS_ALL:
        tr = trdf[trdf.dataset == d]; te = tedf[tedf.dataset == d]
        print(f"  {d:25s} train={len(tr):5d} ids={tr.identity.nunique() if len(tr) else 0:4d} test={len(te):4d}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE A: Load Triple Backbones + Extract Global Features
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}\nSTAGE A: Triple-Backbone Global Feature Extraction\n{'━' * 60}")

    all_tr_feats = {}  # {model_name: (feats, ids)}
    all_te_feats = {}

    # MegaDescriptor
    if CFG.USE_MEGA:
        mega = load_mega()
        print("  Extracting MegaDescriptor features...")
        mega_tr, mega_tr_ids = extract_features(mega, trdf, CFG.DATA_DIR, CFG.MEGA_SZ, tta=False)
        mega_te, mega_te_ids = extract_features(mega, tedf, CFG.DATA_DIR, CFG.MEGA_SZ, tta=CFG.USE_TTA)
        all_tr_feats["mega"] = (mega_tr, mega_tr_ids)
        all_te_feats["mega"] = (mega_te, mega_te_ids)
        print(f"  Mega: train={mega_tr.shape} test={mega_te.shape}")
        del mega; torch.cuda.empty_cache(); gc.collect()

    # MiewID
    if CFG.USE_MIEW:
        miew = load_miew()
        print("  Extracting MiewID features...")
        miew_tr, _ = extract_features(miew, trdf, CFG.DATA_DIR, CFG.MIEW_SZ, tta=False)
        miew_te, _ = extract_features(miew, tedf, CFG.DATA_DIR, CFG.MIEW_SZ, tta=CFG.USE_TTA)
        all_tr_feats["miew"] = (miew_tr, mega_tr_ids)
        all_te_feats["miew"] = (miew_te, mega_te_ids)
        print(f"  Miew: train={miew_tr.shape} test={miew_te.shape}")
        del miew; torch.cuda.empty_cache(); gc.collect()

    # EVA02
    if CFG.USE_EVA02:
        eva02 = load_eva02()
        if eva02 is not None:
            print("  Extracting EVA02 features...")
            eva_tr, _ = extract_features(eva02, trdf, CFG.DATA_DIR, CFG.EVA02_SZ, tta=False, is_eva02=True)
            eva_te, _ = extract_features(eva02, tedf, CFG.DATA_DIR, CFG.EVA02_SZ, tta=CFG.USE_TTA, is_eva02=True)
            all_tr_feats["eva02"] = (eva_tr, mega_tr_ids)
            all_te_feats["eva02"] = (eva_te, mega_te_ids)
            print(f"  EVA02: train={eva_tr.shape} test={eva_te.shape}")
            del eva02; torch.cuda.empty_cache(); gc.collect()
        else:
            CFG.USE_EVA02 = False

    print(f"  Global features done in {(_t.time() - t0) / 60:.1f} min")

    # Build index maps
    tr_i2x = {int(mega_tr_ids[i]): i for i in range(len(mega_tr_ids))}
    te_i2x = {int(mega_te_ids[i]): i for i in range(len(mega_te_ids))}

    # ══════════════════════════════════════════════════════════════════════
    # STAGE B: ALIKED Local Matcher Init
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}\nSTAGE B: Local Matcher Init\n{'━' * 60}")
    local_matcher = LocalMatcher() if CFG.USE_LOCAL else None

    # ══════════════════════════════════════════════════════════════════════
    # STAGE C: ArcFace Fine-tuning (SeaTurtle only)
    # ══════════════════════════════════════════════════════════════════════
    ft_data = {}
    if CFG.USE_ARCFACE:
        print(f"\n{'━' * 60}\nSTAGE C: ArcFace Fine-Tuning\n{'━' * 60}")
        for dsn in ["SeaTurtleID2022"]:  # Only SeaTurtle (high base ARI)
            dtr = trdf[trdf.dataset == dsn].copy()
            dte = tedf[tedf.dataset == dsn].copy()
            if len(dtr) < 50: continue
            try:
                ftm, le = finetune_arcface(dtr, CFG.DATA_DIR, dsn)
                ft_tr = extract_finetuned(ftm, dtr, CFG.DATA_DIR, CFG.MEGA_SZ)
                ft_te = extract_finetuned(ftm, dte, CFG.DATA_DIR, CFG.MEGA_SZ)
                ft_data[dsn] = {"tr": ft_tr, "te": ft_te}
                print(f"  [{dsn}] ArcFace features: train={ft_tr.shape} test={ft_te.shape}")
                del ftm; torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                print(f"  [{dsn}] ArcFace FAILED: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE D: Species-Specific Processing
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}\nSTAGE D: Species-Specific Clustering\n{'━' * 60}")

    all_preds = {}

    for dsn in CFG.DS_ALL:
        print(f"\n{'═' * 55}\n  {dsn}\n{'═' * 55}")

        ds_te = tedf[tedf.dataset == dsn].reset_index(drop=True)
        teix = [te_i2x[int(x)] for x in ds_te.image_id.values]
        n_te = len(ds_te)

        # ── Compute weighted global similarity for test images ──
        sim_parts_te = []
        weights = []
        model_names = []
        for mname, w_attr in [("mega", "W_MEGA"), ("miew", "W_MIEW"), ("eva02", "W_EVA02")]:
            if mname in all_te_feats:
                f = normalize(all_te_feats[mname][0][teix], axis=1)
                sim_parts_te.append(f @ f.T)
                weights.append(getattr(CFG, w_attr))
                model_names.append(mname)

        wt = sum(weights)
        te_global_sim = sum(s * (w / wt) for s, w in zip(sim_parts_te, weights))
        print(f"  Global sim: {'+'.join(model_names)} (weighted)")

        if dsn in CFG.DS_TRAIN:
            ds_tr = trdf[trdf.dataset == dsn].reset_index(drop=True)
            trix = [tr_i2x[int(x)] for x in ds_tr.image_id.values]
            train_ids = ds_tr.identity.values
            n_tr = len(ds_tr)
            le = LabelEncoder(); yt = le.fit_transform(train_ids)

            # ── Build train-train similarity for calibration ──
            sim_parts_tr = []
            for mname, w_attr in [("mega", "W_MEGA"), ("miew", "W_MIEW"), ("eva02", "W_EVA02")]:
                if mname in all_tr_feats:
                    f = normalize(all_tr_feats[mname][0][trix], axis=1)
                    sim_parts_tr.append(f @ f.T)

            tr_global_sim = sum(s * (w / wt) for s, w in zip(sim_parts_tr, weights))

            # ── Build cross similarity (test × train) ──
            cross_sim_parts = []
            for mname, w_attr in [("mega", "W_MEGA"), ("miew", "W_MIEW"), ("eva02", "W_EVA02")]:
                if mname in all_te_feats and mname in all_tr_feats:
                    te_f = normalize(all_te_feats[mname][0][teix], axis=1)
                    tr_f = normalize(all_tr_feats[mname][0][trix], axis=1)
                    cross_sim_parts.append(te_f @ tr_f.T)

            cross_sim = sum(s * (w / wt) for s, w in zip(cross_sim_parts, weights))

            # ── Isotonic Calibration ──
            print(f"  Training isotonic calibrator...")
            # Use the first available model's train features for calibration
            primary_tr = None
            for mname in ["mega", "miew", "eva02"]:
                if mname in all_tr_feats:
                    primary_tr = normalize(all_tr_feats[mname][0][trix], axis=1)
                    break
            cal_scores, cal_labels = build_calibration_pairs(primary_tr, train_ids)
            calibrator = train_calibrator(cal_scores, cal_labels)
            print(f"  Calibrator trained on {len(cal_scores)} pairs")

            # Calibrate the train similarity
            tr_cal_sim = calibrate_similarity_matrix(tr_global_sim, calibrator)

            # ── Orientation adjustment (Lynx/Salamander) ──
            if dsn in ["LynxID2025", "SalamanderID2025"]:
                te_global_sim = orientation_adjust(te_global_sim, ds_te)
                tr_cal_sim = orientation_adjust(tr_cal_sim, ds_tr)
                print(f"  Orientation-adjusted")

            # ── ALIKED Local Matching (Lynx & Salamander — the key weapon!) ──
            local_te_sim = None
            if dsn in ["LynxID2025", "SalamanderID2025"] and local_matcher and local_matcher.available:
                print(f"  Running ALIKED local matching for {dsn}...")
                local_te_sim = local_matcher.compute_pairwise_local(
                    ds_te, CFG.DATA_DIR, te_global_sim, topk=CFG.LOCAL_TOPK)

            # ── Fuse: global + local + fine-tuned ──
            if local_te_sim is not None:
                # Calibrate local scores too (scale to same range)
                local_max = local_te_sim.max()
                if local_max > 0:
                    local_te_sim_norm = local_te_sim / local_max
                else:
                    local_te_sim_norm = local_te_sim
                te_fused_sim = (1 - CFG.LOCAL_WEIGHT) * te_global_sim + CFG.LOCAL_WEIGHT * local_te_sim_norm
                print(f"  Fused: global({1 - CFG.LOCAL_WEIGHT:.0%}) + local({CFG.LOCAL_WEIGHT:.0%})")
            else:
                te_fused_sim = te_global_sim

            # ── Blend with ArcFace fine-tuned features (SeaTurtle) ──
            if dsn in ft_data:
                ft_te_f = normalize(ft_data[dsn]["te"], axis=1)
                ft_sim = ft_te_f @ ft_te_f.T
                te_fused_sim = CFG.FT_BLEND * ft_sim + (1 - CFG.FT_BLEND) * te_fused_sim
                print(f"  Blended with ArcFace features ({CFG.FT_BLEND:.0%})")

            # ── Strategy: Train-anchored clustering ──
            # Build combined train+test similarity and cluster together
            # The train labels anchor the clustering

            # Combined features for anchored approach
            combined_sim_parts = []
            for mname, w_attr in [("mega", "W_MEGA"), ("miew", "W_MIEW"), ("eva02", "W_EVA02")]:
                if mname in all_te_feats and mname in all_tr_feats:
                    te_f = normalize(all_te_feats[mname][0][teix], axis=1)
                    tr_f = normalize(all_tr_feats[mname][0][trix], axis=1)
                    all_f = np.vstack([tr_f, te_f])
                    combined_sim_parts.append(all_f @ all_f.T)

            combined_sim = sum(s * (w / wt) for s, w in zip(combined_sim_parts, weights))

            # Orientation adjust combined
            if dsn in ["LynxID2025", "SalamanderID2025"]:
                combined_df = pd.concat([ds_tr, ds_te], ignore_index=True)
                combined_sim = orientation_adjust(combined_sim, combined_df)

            combined_dist = np.clip(1 - combined_sim, 0, 2)

            # Auto-tune on train portion
            tr_dist = combined_dist[:n_tr, :n_tr]
            tr_sim_for_tune = combined_sim[:n_tr, :n_tr]
            params = autotune_clustering(tr_dist, tr_sim_for_tune, yt, label=dsn)

            # Apply ensemble clustering to combined train+test
            if CFG.USE_ENSEMBLE and n_te > 10:
                full_labels = cluster_ensemble(combined_dist, combined_sim, params, label=dsn)
            else:
                # Use best single method
                best_method = max(
                    [(params.get("dbscan_ari", -1), "dbscan"),
                     (params.get("agglo_ari", -1), "agglo"),
                     (params.get("cc_ari", -1), "cc"),
                     (params.get("hdbscan_ari", -1), "hdbscan")],
                    key=lambda x: x[0])[1]
                if best_method == "dbscan":
                    full_labels = fix_noise(DBSCAN(eps=params["dbscan_eps"],
                                                   min_samples=CFG.DBSCAN_MIN,
                                                   metric="precomputed").fit_predict(combined_dist))
                elif best_method == "agglo":
                    full_labels = AgglomerativeClustering(
                        n_clusters=None, distance_threshold=params["agglo_dt"],
                        metric="precomputed", linkage="average").fit_predict(combined_dist)
                elif best_method == "cc":
                    full_labels = cc_cluster(combined_sim, params["cc_thresh"])
                else:
                    full_labels = fix_noise(DBSCAN(eps=params["dbscan_eps"],
                                                   min_samples=CFG.DBSCAN_MIN,
                                                   metric="precomputed").fit_predict(combined_dist))

            te_labels_anchored = full_labels[n_tr:]

            # ── Also try: lookup + cluster unknowns approach ──
            # (May be better for SeaTurtle which has high train ARI)
            # Lookup: use cross_sim to match test → train
            lookup_th, lookup_ari = autotune_lookup(tr_cal_sim.copy(), yt, dsn)

            if dsn == "SeaTurtleID2022" or lookup_ari > 0.4:
                # Calibrate cross similarity
                cross_cal = calibrator.predict(cross_sim.ravel()).reshape(cross_sim.shape)
                mx_sim = cross_cal.max(axis=1)
                mx_idx = cross_cal.argmax(axis=1)

                known = {}
                for i in range(n_te):
                    if mx_sim[i] >= lookup_th:
                        known[i] = train_ids[mx_idx[i]]

                print(f"  Lookup matched: {len(known)}/{n_te} ({100 * len(known) / n_te:.1f}%)")

                # Cluster unknowns using te_fused_sim
                unk = sorted(set(range(n_te)) - set(known))
                if len(unk) > 1:
                    unk_sim = te_fused_sim[np.ix_(unk, unk)]
                    unk_dist = np.clip(1 - unk_sim, 0, 2)

                    # Tune on unknown subset (no labels, heuristic)
                    best_eps, best_score = 0.40, -1
                    for eps in np.arange(0.15, 0.80, 0.02):
                        pred = fix_noise(DBSCAN(eps=eps, min_samples=2,
                                                metric="precomputed").fit_predict(unk_dist))
                        n_cl = len(set(pred))
                        n_per_cl = n_te / max(1, len(set(yt)))  # expected images per individual
                        expected_n_cl = max(5, len(unk) / max(2, n_per_cl))
                        score = -abs(n_cl - expected_n_cl) / max(1, expected_n_cl)
                        if score > best_score: best_score, best_eps = score, eps

                    unk_labels = fix_noise(DBSCAN(eps=best_eps, min_samples=2,
                                                  metric="precomputed").fit_predict(unk_dist))
                elif len(unk) == 1:
                    unk_labels = np.array([0])
                else:
                    unk_labels = np.array([])

                # Build lookup labels
                te_labels_lookup = np.zeros(n_te, dtype=int)
                id_to_cl = {}; next_cl = 0
                for i, ident in known.items():
                    if ident not in id_to_cl: id_to_cl[ident] = next_cl; next_cl += 1
                    te_labels_lookup[i] = id_to_cl[ident]
                base = next_cl
                for pos, ti in enumerate(unk):
                    te_labels_lookup[ti] = base + unk_labels[pos]

                n_cl_lookup = len(set(te_labels_lookup))
                n_cl_anchor = len(set(te_labels_anchored))
                print(f"  Lookup approach: {n_cl_lookup} clusters")
                print(f"  Anchored approach: {n_cl_anchor} clusters")

                # Validate: check which approach gives better train ARI
                # For anchored: check train portion
                anchor_train_ari = adjusted_rand_score(yt, full_labels[:n_tr])
                print(f"  Anchored train ARI: {anchor_train_ari:.4f}")
                print(f"  Lookup train ARI: {lookup_ari:.4f}")

                if lookup_ari > anchor_train_ari:
                    te_labels = te_labels_lookup
                    print(f"  → Using LOOKUP approach")
                else:
                    te_labels = te_labels_anchored
                    print(f"  → Using ANCHORED approach")
            else:
                te_labels = te_labels_anchored
                print(f"  → Using ANCHORED approach (lookup ARI too low)")

        else:
            # TexasHornedLizards: no training data → pure clustering
            dist = np.clip(1 - te_global_sim, 0, 2)

            # ALIKED local matching for lizards too if available
            if local_matcher and local_matcher.available and n_te < 500:
                print(f"  Running ALIKED for Lizards...")
                local_sim = local_matcher.compute_pairwise_local(
                    ds_te, CFG.DATA_DIR, te_global_sim, topk=min(CFG.LOCAL_TOPK, n_te - 1))
                lmax = local_sim.max()
                if lmax > 0:
                    local_sim_norm = local_sim / lmax
                    te_fused = (1 - CFG.LOCAL_WEIGHT) * te_global_sim + CFG.LOCAL_WEIGHT * local_sim_norm
                    dist = np.clip(1 - te_fused, 0, 2)

            # Heuristic tuning (no labels)
            best_eps, best_score = 0.40, -1
            for eps in np.arange(0.15, 0.80, 0.01):
                pred = fix_noise(DBSCAN(eps=eps, min_samples=2,
                                        metric="precomputed").fit_predict(dist))
                n_cl = len(set(pred))
                n_noise = (pred == -1).sum() if -1 in pred else 0
                if 10 <= n_cl <= 200:
                    score = 1 - n_noise / max(1, n_te) - abs(n_cl / n_te - 0.3)
                    if score > best_score: best_score, best_eps = score, eps

            te_labels = fix_noise(DBSCAN(eps=best_eps, min_samples=2,
                                         metric="precomputed").fit_predict(dist))
            print(f"  eps={best_eps:.2f} clusters={len(set(te_labels))}")

        n_cl = len(set(te_labels))
        print(f"  → Final: {n_cl} clusters for {n_te} images")

        for i in range(n_te):
            all_preds[int(ds_te.iloc[i].image_id)] = f"cluster_{dsn}_{te_labels[i]}"

    # ══════════════════════════════════════════════════════════════════════
    # STAGE E: Pseudo-Label Refinement
    # ══════════════════════════════════════════════════════════════════════
    if CFG.PSEUDO_ROUNDS > 0:
        print(f"\n{'━' * 60}\nSTAGE E: Pseudo-Label Refinement ({CFG.PSEUDO_ROUNDS} rounds)\n{'━' * 60}")

        # Use primary features
        primary_key = "mega" if "mega" in all_te_feats else list(all_te_feats.keys())[0]

        for rnd in range(CFG.PSEUDO_ROUNDS):
            changed = 0
            for dsn in CFG.DS_ALL:
                ds_te = tedf[tedf.dataset == dsn].reset_index(drop=True)
                teix = [te_i2x[int(x)] for x in ds_te.image_id.values]
                feat = normalize(all_te_feats[primary_key][0][teix], axis=1)

                # Build cluster centroids
                cm = defaultdict(list)
                for i in range(len(ds_te)):
                    cl = all_preds.get(int(ds_te.iloc[i].image_id), "?")
                    cm[cl].append(i)

                centroids = {}
                for cl, idx in cm.items():
                    if len(idx) > 0:
                        centroids[cl] = normalize(feat[idx].mean(axis=0, keepdims=True), axis=1)

                for i in range(len(ds_te)):
                    iid = int(ds_te.iloc[i].image_id)
                    cur = all_preds.get(iid)
                    if cur is None or cur not in centroids: continue
                    csim = float(feat[i:i + 1] @ centroids[cur].T)
                    best_cl, best_s = cur, csim
                    for cl, cent in centroids.items():
                        if cl == cur or dsn not in cl: continue
                        s = float(feat[i:i + 1] @ cent.T)
                        if s > best_s + 0.05:
                            best_s, best_cl = s, cl
                    if best_cl != cur:
                        all_preds[iid] = best_cl
                        changed += 1
            print(f"  Round {rnd + 1}: {changed} reassigned")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE F: Generate Submission
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}\nSTAGE F: Submission\n{'━' * 60}")
    sub = ssub.copy()
    for i in range(len(sub)):
        iid = int(sub.iloc[i].image_id)
        if iid in all_preds:
            sub.at[i, "cluster"] = all_preds[iid]

    out_path = os.path.join(CFG.OUTPUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False)
    elapsed = (_t.time() - t0) / 60

    print(f"\n  Saved: {out_path}")
    print(f"  Rows: {len(sub)}  Clusters: {sub.cluster.nunique()}")
    for d in CFG.DS_ALL:
        s = sub[sub.cluster.str.contains(d)]
        print(f"    {d:25s} imgs={len(s):4d} cl={s.cluster.nunique()}")
    print(f"\n  Total time: {elapsed:.1f} min")
    print("=" * 70)
    print("DONE!")

    return sub


if __name__ == "__main__":
    main()
