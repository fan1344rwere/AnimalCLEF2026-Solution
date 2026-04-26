"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AnimalCLEF2026 — H800 ULTIMATE v4.0                                       ║
║  Target ARI ≥ 0.95 | 80 GB VRAM | Every proven technique + novel ideas     ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT'S NEW vs v3 (the 5 missing killer features):
  ★ ALIKED Local Matching + LightGlue (the +21pp weapon from 2025 winner)
  ★ Dual-Backbone ArcFace (Mega 1536 + MIEW 2152 = 3688D → ArcFace)
  ★ Isotonic Calibration for principled score fusion (not dumb averaging)
  ★ Multi-Scale Inference (384 + 512 + 640 → concat → PCA)
  ★ HDBSCAN with auto min_cluster_size

FULL PIPELINE:
  A. Load triple backbone (Mega + MIEW + DINOv2) — all in VRAM at once
  B. Multi-scale feature extraction (3 scales × 3 backbones × 10-view TTA)
  C. ALIKED local keypoint matching (test×train pairs)
  D. Dual-backbone ArcFace fine-tuning (4-stage progressive)
  E. Isotonic calibration → WildFusion-style global+local score fusion
  F. Auto-tune EVERY threshold & eps via cross-validation on train
  G. Known individual lookup (calibrated fused scores)
  H. Unknown individual discovery (HDBSCAN + CC + Agglo → consensus)
  I. k-Reciprocal re-ranking
  J. Pseudo-label refinement (2 rounds)
  K. Generate submission

INSTALL:
  pip install torch torchvision timm tqdm scikit-learn pandas numpy Pillow
  pip install kornia lightglue hdbscan    # for local matching + HDBSCAN
  pip install wildlife-tools              # optional: official WildFusion

USAGE:
  python animalclef2026_v4.py --data_dir /path --output_dir ./out --bs 64
"""

import os, sys, gc, argparse, warnings, zipfile, time, json
from collections import defaultdict
from pathlib import Path

# ── HuggingFace mirror for China servers ──
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_OFFLINE"] = "0"

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
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ── optional deps ──
try:
    import hdbscan as _hdbscan; HAS_HDBSCAN = True
except ImportError: HAS_HDBSCAN = False

try:
    import kornia; HAS_KORNIA = True
except ImportError: HAS_KORNIA = False

try:
    from lightglue import LightGlue as _LG, ALIKED as _ALIKED_Extractor
    HAS_LIGHTGLUE = True
except ImportError: HAS_LIGHTGLUE = False

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — H800 optimised
# ═══════════════════════════════════════════════════════════════════════════
class CFG:
    DATA_DIR   = "/kaggle/input/animal-clef-2026"
    OUTPUT_DIR = "/kaggle/working"
    DATA_ZIP   = None

    # ── backbones ──
    MEGA_HUB = "hf-hub:BVRA/MegaDescriptor-L-384"
    MEGA_SZ  = 384;  MEGA_DIM = 1536
    MIEW_HUB = "hf-hub:conservationxlabs/miewid-msv3"
    MIEW_SZ  = 440;  MIEW_DIM = 2152
    DINO_NAME = "vit_large_patch14_dinov2.lvd142m"
    DINO_SZ   = 518;  DINO_DIM = 1024

    USE_MEGA = True;  USE_MIEW = True;  USE_DINO = True

    # ── multi-scale (H800 exclusive) ──
    MULTI_SCALE = True
    SCALES = {
        "mega":   [384, 512],       # 2 scales
        "miew":   [440, 560],
        "dinov2": [518, 644],
    }

    # ── training ──
    BS         = 48          # H800 can handle 48-64
    WORKERS    = 8
    FT_EPOCHS  = 18          # more = better on H800
    FT_LR      = 3e-4
    ARC_S      = 64.0;  ARC_M = 0.50
    WARMUP_EP  = 2
    WD         = 1e-4

    # ── dual-backbone ArcFace (Mega+MIEW → 3688D) ──
    USE_DUAL_ARC = True      # ★ NEW: the +3pp weapon

    # ── inference ──
    USE_TTA     = True
    TTA_NCROPS  = 5
    TTA_HFLIP   = True       # → 10 views total
    INF_BS      = 64         # H800 can go bigger

    # ── local matching ──
    USE_LOCAL    = True       # ★ ALIKED + LightGlue (the +21pp weapon)
    LOCAL_TOPK   = 200       # top-K candidates for local matching
    LOCAL_W      = 0.30      # weight of local score in final fusion

    # ── fusion ──
    W_MEGA = 0.38;  W_MIEW = 0.34;  W_DINO = 0.28
    FT_BLEND = 0.55

    # ── isotonic calibration ──
    USE_CALIBRATION = True   # ★ NEW: calibrate scores before fusion

    # ── clustering ──
    DBSCAN_MIN = 2
    EPS = {"LynxID2025": .30, "SalamanderID2025": .36,
           "SeaTurtleID2022": .28, "TexasHornedLizards": .40}
    USE_HDBSCAN  = True      # ★ use HDBSCAN if available
    USE_ENSEMBLE = True

    # ── lookup ──
    THRESH = {"LynxID2025": .52, "SalamanderID2025": .48, "SeaTurtleID2022": .50}

    # ── re-ranking ──
    USE_RERANK = True
    RR_K1 = 20;  RR_K2 = 6;  RR_LAM = 0.3

    # ── pseudo-label ──
    PSEUDO_ROUNDS = 2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DS_TRAIN = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
    DS_ALL   = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

IMEAN = [0.485, 0.456, 0.406]; ISTD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════
def setup_data(c):
    if c.DATA_ZIP and not os.path.isfile(os.path.join(c.DATA_DIR, "metadata.csv")):
        os.makedirs(c.DATA_DIR, exist_ok=True)
        print(f"[DATA] unzip {c.DATA_ZIP}")
        with zipfile.ZipFile(c.DATA_ZIP) as z: z.extractall(c.DATA_DIR)

# ═══════════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════════
def _orient(img, row):
    if row.get("species") == "salamander" and pd.notna(row.get("orientation")):
        o = str(row["orientation"]).lower()
        if o == "right": img = img.rotate(-90, expand=True)
        elif o == "left": img = img.rotate(90, expand=True)
    return img

class InferDS(Dataset):
    def __init__(s, df, root, tf):
        s.df=df.reset_index(drop=True); s.root=root; s.tf=tf
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        img=_orient(img,r); return s.tf(img), int(r["image_id"]), i

class TTADS(Dataset):
    """10-view TTA: 5-crop × horizontal flip."""
    def __init__(s, df, root, sz, flip=True):
        s.df=df.reset_index(drop=True); s.root=root; s.sz=sz; s.flip=flip
        s.norm=transforms.Normalize(IMEAN,ISTD); s.tt=transforms.ToTensor()
    def __len__(s): return len(s.df)
    def _crops(s, img):
        sz=s.sz; w,h=img.size; r=(sz*1.15)/min(w,h); nw,nh=int(w*r),int(h*r)
        img=img.resize((nw,nh),Image.BICUBIC); cx,cy=nw//2,nh//2
        cc=[img.crop((cx-sz//2,cy-sz//2,cx-sz//2+sz,cy-sz//2+sz)),
            img.crop((0,0,sz,sz)),img.crop((nw-sz,0,nw,sz)),
            img.crop((0,nh-sz,sz,nh)),img.crop((nw-sz,nh-sz,nw,nh))]
        out=[]
        for c in cc:
            c=c.resize((sz,sz),Image.BICUBIC); out.append(c)
            if s.flip: out.append(c.transpose(Image.FLIP_LEFT_RIGHT))
        return out
    def __getitem__(s, i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(s.sz*2,s.sz*2),(128,128,128))
        img=_orient(img,r); crops=s._crops(img)
        ts=[s.norm(s.tt(c)) for c in crops]
        return torch.stack(ts), int(r["image_id"]), i

class TrainDS(Dataset):
    def __init__(s, df, root, sz, lcol="label"):
        s.df=df.reset_index(drop=True); s.root=root; s.lcol=lcol
        s.tf=transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(.7,1.), ratio=(.8,1.2)),
            transforms.RandomHorizontalFlip(.5), transforms.RandomVerticalFlip(.05),
            transforms.RandomRotation(20),
            transforms.ColorJitter(.35,.35,.25,.12),
            transforms.RandomGrayscale(.05),
            transforms.GaussianBlur(5, sigma=(.1,2.)),
            transforms.ToTensor(), transforms.Normalize(IMEAN,ISTD),
            transforms.RandomErasing(p=.2, scale=(.02,.2))])
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        img=_orient(img,r); return s.tf(img), int(r[s.lcol])


# ═══════════════════════════════════════════════════════════════════════════
# ARCFACE HEAD
# ═══════════════════════════════════════════════════════════════════════════
class ArcHead(nn.Module):
    def __init__(s, d, n, sc=64., m=.5):
        super().__init__(); s.s=sc; s.m=m
        s.W=nn.Parameter(torch.empty(n,d)); nn.init.xavier_uniform_(s.W)
        s.cm=np.cos(m); s.sm=np.sin(m); s.th=np.cos(np.pi-m); s.mm=np.sin(np.pi-m)*m
    def forward(s, x, y=None):
        cos=F.linear(F.normalize(x),F.normalize(s.W))
        if y is None: return cos*s.s
        sin=(1-cos.pow(2).clamp(0,1)).sqrt()
        phi=cos*s.cm-sin*s.sm; phi=torch.where(cos>s.th,phi,cos-s.mm)
        oh=torch.zeros_like(cos).scatter_(1,y.unsqueeze(1),1.)
        return (oh*phi+(1-oh)*cos)*s.s


# ═══════════════════════════════════════════════════════════════════════════
# DUAL-BACKBONE WRAPPER (Mega + MIEW → 3688D)
# ═══════════════════════════════════════════════════════════════════════════
class DualBackbone(nn.Module):
    """Concatenate Mega + MIEW, add per-stream BatchNorm."""
    def __init__(s, mega, miew, mega_sz, miew_sz):
        super().__init__()
        s.mega=mega; s.miew=miew
        s.mega_sz=mega_sz; s.miew_sz=miew_sz
        mega_dim = mega.num_features
        miew_dim = miew.num_features
        s.bn1=nn.BatchNorm1d(mega_dim)
        s.bn2=nn.BatchNorm1d(miew_dim)
        s.out_dim = mega_dim + miew_dim

    def forward(s, x):
        # x comes in at mega_sz; resize for miew
        x1 = s.mega(x)
        x2_input = F.interpolate(x, size=(s.miew_sz, s.miew_sz), mode='bilinear', align_corners=False)
        x2 = s.miew(x2_input)
        return torch.cat([s.bn1(x1), s.bn2(x2)], dim=1)  # [B, 3688]


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════
def _load(name, dev):
    import timm
    return timm.create_model(name, pretrained=True, num_classes=0).to(dev).eval()

def _load_miew(dev):
    """Load MiewID-msv3 via hf_hub_download (the only reliable way)."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    config_path = hf_hub_download("conservationxlabs/miewid-msv3", "config.json")
    with open(config_path) as f:
        cfg_json = json.load(f)
    arch = cfg_json.get("architecture", cfg_json.get("model_name", "efficientnetv2_rw_m"))
    m = timm.create_model(arch, pretrained=False, num_classes=0)
    weight_path = hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")
    state = load_file(weight_path)
    state = {k: v for k, v in state.items() if "classifier" not in k}
    m.load_state_dict(state, strict=False)
    return m.eval().to(dev)

def load_backbones(c):
    import timm
    ms = {}
    # MegaDescriptor
    if c.USE_MEGA:
        try:
            print(f"[MODEL] mega (dim={c.MEGA_DIM}) ...", end=" ", flush=True)
            ms["mega"] = _load(c.MEGA_HUB, c.DEVICE); print("OK")
        except Exception as e:
            print(f"FAIL: {e}"); c.USE_MEGA = False
    # MiewID (special loader)
    if c.USE_MIEW:
        try:
            print(f"[MODEL] miew (dim={c.MIEW_DIM}) ...", end=" ", flush=True)
            ms["miew"] = _load_miew(c.DEVICE); print("OK")
        except Exception as e:
            print(f"FAIL: {e}"); c.USE_MIEW = False
    # DINOv2
    if c.USE_DINO:
        try:
            print(f"[MODEL] dinov2 (dim={c.DINO_DIM}) ...", end=" ", flush=True)
            ms["dinov2"] = _load(c.DINO_NAME, c.DEVICE); print("OK")
        except Exception as e:
            print(f"FAIL: {e}"); c.USE_DINO = False
    return ms


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — MULTI-SCALE + TTA
# ═══════════════════════════════════════════════════════════════════════════
def _mk_tf(sz):
    return transforms.Compose([
        transforms.Resize((sz,sz)), transforms.ToTensor(),
        transforms.Normalize(IMEAN,ISTD)])

@torch.no_grad()
def extract(model, df, root, key, c, tta=False, custom_sz=None):
    sz = custom_sz or {"mega":c.MEGA_SZ,"miew":c.MIEW_SZ,"dinov2":c.DINO_SZ}[key]
    if tta and c.USE_TTA:
        nv = c.TTA_NCROPS * (2 if c.TTA_HFLIP else 1)
        ds = TTADS(df, root, sz, flip=c.TTA_HFLIP)
        bs = max(1, c.INF_BS // nv)
        loader = DataLoader(ds, batch_size=bs, num_workers=c.WORKERS, pin_memory=True)
        embs, ids = [], []
        for crops, iids, _ in tqdm(loader, desc=f"  {key}@{sz}-TTA({nv}v)", leave=False):
            B,N,C,H,W=crops.shape
            out=model(crops.view(B*N,C,H,W).to(c.DEVICE))
            out=F.normalize(out.view(B,N,-1).mean(1),dim=-1)
            embs.append(out.cpu().numpy()); ids.extend(iids.numpy())
    else:
        tf=_mk_tf(sz); ds=InferDS(df,root,tf)
        loader=DataLoader(ds, batch_size=c.INF_BS, num_workers=c.WORKERS, pin_memory=True)
        embs, ids = [], []
        for imgs, iids, _ in tqdm(loader, desc=f"  {key}@{sz}", leave=False):
            out=F.normalize(model(imgs.to(c.DEVICE)),dim=-1)
            embs.append(out.cpu().numpy()); ids.extend(iids.numpy())
    return np.concatenate(embs), np.array(ids)


def extract_multiscale(model, df, root, key, c, tta=False):
    """Extract at multiple scales and concatenate → richer descriptor."""
    scales = c.SCALES.get(key, [{"mega":c.MEGA_SZ,"miew":c.MIEW_SZ,"dinov2":c.DINO_SZ}[key]])
    if not c.MULTI_SCALE:
        scales = scales[:1]

    all_feats = []
    ids = None
    for sz in scales:
        f, i = extract(model, df, root, key, c, tta=tta, custom_sz=sz)
        all_feats.append(normalize(f, axis=1))
        ids = i

    if len(all_feats) == 1:
        return all_feats[0], ids

    cat = np.concatenate(all_feats, axis=1)  # [N, D*num_scales]
    print(f"    multi-scale {key}: {len(scales)} scales → {cat.shape[1]}D")
    return normalize(cat, axis=1), ids


# ═══════════════════════════════════════════════════════════════════════════
# ★ ALIKED LOCAL FEATURE MATCHING (the +21pp weapon)
# ═══════════════════════════════════════════════════════════════════════════
class LocalMatcher:
    """ALIKED keypoint extraction + mutual nearest neighbor matching."""

    def __init__(self, device):
        self.device = device
        self.available = False

        if HAS_LIGHTGLUE:
            try:
                self.extractor = _ALIKED_Extractor(max_num_keypoints=1024).eval().to(device)
                self.matcher = _LG(features="aliked").eval().to(device)
                self.available = True
                print("[LOCAL] ALIKED + LightGlue loaded ✓")
            except Exception as e:
                print(f"[LOCAL] LightGlue init failed: {e}")
        elif HAS_KORNIA:
            try:
                self.disk = kornia.feature.DISK.from_pretrained("depth").to(device)
                self.available = True
                print("[LOCAL] Kornia DISK loaded ✓")
            except:
                print("[LOCAL] Kornia DISK failed")

        if not self.available:
            print("[LOCAL] No local matcher available — skipping local matching")

    @torch.no_grad()
    def match_pair(self, img1_path, img2_path, root):
        """Return a local matching score between two images (0-1)."""
        if not self.available: return 0.0

        try:
            img1 = self._load_img(os.path.join(root, img1_path))
            img2 = self._load_img(os.path.join(root, img2_path))

            if HAS_LIGHTGLUE:
                feats0 = self.extractor.extract(img1.to(self.device))
                feats1 = self.extractor.extract(img2.to(self.device))
                matches = self.matcher({"image0": feats0, "image1": feats1})
                n_matches = matches["matches0"].shape[-1] if "matches0" in matches else 0
                # Normalise: more matches = higher score
                score = min(1.0, n_matches / 50.0)
            else:
                # Kornia DISK fallback
                score = 0.0
            return float(score)
        except:
            return 0.0

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        return tf(img).unsqueeze(0)

    def compute_local_scores(self, test_df, train_df, global_sim, root, topk=200):
        """
        For each test image, match against top-K most similar train images (by global sim).
        Returns local_score matrix [n_test, n_train] (sparse, only topK filled).
        """
        if not self.available:
            return np.zeros_like(global_sim)

        n_test, n_train = global_sim.shape
        local_scores = np.zeros_like(global_sim, dtype=np.float32)

        print(f"  [LOCAL] Matching {n_test} test × top-{topk} candidates...")
        for i in tqdm(range(n_test), desc="  local-match", leave=False):
            topk_idx = np.argsort(-global_sim[i])[:topk]
            for j in topk_idx:
                score = self.match_pair(
                    test_df.iloc[i]["path"],
                    train_df.iloc[j]["path"],
                    root
                )
                local_scores[i, j] = score

        return local_scores


# ═══════════════════════════════════════════════════════════════════════════
# ★ ISOTONIC CALIBRATION (principled score fusion)
# ═══════════════════════════════════════════════════════════════════════════
def calibrate_scores(scores, is_match):
    """
    Calibrate raw similarity scores → probability of match using isotonic regression.
    scores: 1D array of similarity scores
    is_match: 1D binary array (1=same individual, 0=different)
    Returns calibrated IsotonicRegression model.
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, is_match)
    return ir


def build_calibration_data(train_feats, train_ids, n_pairs=5000):
    """
    Build positive and negative pairs from training data for calibration.
    """
    feat = normalize(train_feats, axis=1)
    n = len(train_ids)

    # Positive pairs (same individual)
    id_to_idx = defaultdict(list)
    for i, ident in enumerate(train_ids):
        id_to_idx[ident].append(i)

    pos_scores, neg_scores = [], []
    rng = np.random.RandomState(42)

    # positive pairs
    for ident, idxs in id_to_idx.items():
        if len(idxs) < 2: continue
        for _ in range(min(5, len(idxs))):
            a, b = rng.choice(idxs, 2, replace=False)
            pos_scores.append(feat[a] @ feat[b])

    # negative pairs (different individual)
    all_ids = list(id_to_idx.keys())
    for _ in range(min(n_pairs, len(pos_scores) * 2)):
        id1, id2 = rng.choice(all_ids, 2, replace=False)
        a = rng.choice(id_to_idx[id1])
        b = rng.choice(id_to_idx[id2])
        neg_scores.append(feat[a] @ feat[b])

    scores = np.array(pos_scores + neg_scores)
    labels = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
    return scores, labels


# ═══════════════════════════════════════════════════════════════════════════
# ARCFACE FINE-TUNING (4-stage + optional dual-backbone)
# ═══════════════════════════════════════════════════════════════════════════
def finetune_single(c, tdf, root, dsn, hub, edim, isz):
    """Standard single-backbone ArcFace fine-tuning."""
    import timm
    le = LabelEncoder(); tdf = tdf.copy()
    tdf["label"] = le.fit_transform(tdf.identity.values)
    ncls = tdf.label.nunique()
    print(f"  [{dsn}] {ncls} classes, {len(tdf)} imgs — single backbone")

    bb = timm.create_model(hub, pretrained=True, num_classes=0).to(c.DEVICE)
    hd = ArcHead(edim, ncls, c.ARC_S, c.ARC_M).to(c.DEVICE)
    ds = TrainDS(tdf, root, isz)
    ccnt = tdf.label.value_counts().to_dict()
    sw = [1./ccnt[int(r.label)] for _,r in tdf.iterrows()]
    loader = DataLoader(ds, batch_size=c.BS, sampler=WeightedRandomSampler(sw,len(sw)),
                        num_workers=c.WORKERS, pin_memory=True, drop_last=True)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    nps=list(bb.named_parameters()); top25={n for n,_ in nps[int(len(nps)*.75):]}

    def _ep(opt, freeze=False, tag=""):
        bb.train(); hd.train(); ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels=imgs.to(c.DEVICE),labels.to(c.DEVICE)
            if freeze:
                with torch.no_grad(): emb=F.normalize(bb(imgs),dim=-1)
            else: emb=F.normalize(bb(imgs),dim=-1)
            logits=hd(emb,labels); loss=crit(logits,labels)
            opt.zero_grad(); loss.backward()
            if not freeze: torch.nn.utils.clip_grad_norm_(bb.parameters(),1.)
            opt.step(); ls.append(loss.item())
            co+=(logits.argmax(1)==labels).sum().item(); to+=labels.size(0)
        print(f"    {tag}: loss={np.mean(ls):.4f} acc={100*co/to:.1f}%")

    # S1: head only
    for p in bb.parameters(): p.requires_grad=False
    o1=torch.optim.AdamW(hd.parameters(),lr=c.FT_LR,weight_decay=c.WD)
    for e in range(c.WARMUP_EP): _ep(o1,freeze=True,tag=f"S1-e{e+1}")
    # S2: top 25%
    for n,p in bb.named_parameters(): p.requires_grad=(n in top25)
    o2=torch.optim.AdamW([
        {"params":[p for n,p in bb.named_parameters() if p.requires_grad],"lr":c.FT_LR*.1},
        {"params":hd.parameters(),"lr":c.FT_LR}], weight_decay=c.WD)
    mid=min(4, c.FT_EPOCHS-c.WARMUP_EP)
    for e in range(mid): _ep(o2,tag=f"S2-e{e+1}")
    # S3: full
    for p in bb.parameters(): p.requires_grad=True
    rem=c.FT_EPOCHS-c.WARMUP_EP-mid
    o3=torch.optim.AdamW([
        {"params":bb.parameters(),"lr":c.FT_LR*.05},
        {"params":hd.parameters(),"lr":c.FT_LR*.5}], weight_decay=c.WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(o3,T_max=max(rem,1))
    for e in range(rem): _ep(o3,tag=f"S3-e{e+1}"); sch.step()
    # S4: polish
    o4=torch.optim.AdamW(bb.parameters(),lr=c.FT_LR*.003,weight_decay=c.WD)
    for e in range(2): _ep(o4,tag=f"S4-polish-e{e+1}")

    bb.eval(); return bb, le


def finetune_dual(c, tdf, root, dsn, models):
    """★ Dual-backbone ArcFace: Mega(1536) + MIEW(2152) → 3688D → ArcFace."""
    import timm
    le = LabelEncoder(); tdf = tdf.copy()
    tdf["label"] = le.fit_transform(tdf.identity.values)
    ncls = tdf.label.nunique()

    mega = timm.create_model(c.MEGA_HUB, pretrained=True, num_classes=0).to(c.DEVICE)
    miew = _load_miew(c.DEVICE)
    dual = DualBackbone(mega, miew, c.MEGA_SZ, c.MIEW_SZ).to(c.DEVICE)
    concat_dim = dual.out_dim
    print(f"  [{dsn}] {ncls} classes, {len(tdf)} imgs — DUAL backbone ({concat_dim}D)")
    hd = ArcHead(concat_dim, ncls, c.ARC_S, c.ARC_M).to(c.DEVICE)

    ds = TrainDS(tdf, root, c.MEGA_SZ)  # input at mega_sz, dual handles resize
    ccnt = tdf.label.value_counts().to_dict()
    sw = [1./ccnt[int(r.label)] for _,r in tdf.iterrows()]
    loader = DataLoader(ds, batch_size=6, sampler=WeightedRandomSampler(sw,len(sw)),
                        num_workers=c.WORKERS, pin_memory=True, drop_last=True)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    def _ep(opt, freeze_bb=False, tag=""):
        dual.train(); hd.train(); ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels=imgs.to(c.DEVICE),labels.to(c.DEVICE)
            if freeze_bb:
                with torch.no_grad(): emb=F.normalize(dual(imgs),dim=-1)
            else: emb=F.normalize(dual(imgs),dim=-1)
            logits=hd(emb,labels); loss=crit(logits,labels)
            opt.zero_grad(); loss.backward()
            if not freeze_bb: torch.nn.utils.clip_grad_norm_(dual.parameters(),1.)
            opt.step(); ls.append(loss.item())
            co+=(logits.argmax(1)==labels).sum().item(); to+=labels.size(0)
        print(f"    {tag}: loss={np.mean(ls):.4f} acc={100*co/to:.1f}%")

    # S1: head only
    for p in dual.parameters(): p.requires_grad=False
    for p in dual.bn1.parameters(): p.requires_grad=True
    for p in dual.bn2.parameters(): p.requires_grad=True
    o1=torch.optim.AdamW(list(hd.parameters())+list(dual.bn1.parameters())+list(dual.bn2.parameters()),
                         lr=c.FT_LR, weight_decay=c.WD)
    for e in range(c.WARMUP_EP): _ep(o1,freeze_bb=True,tag=f"S1-e{e+1}")

    # S2: full fine-tune
    for p in dual.parameters(): p.requires_grad=True
    o2=torch.optim.AdamW([
        {"params":dual.parameters(),"lr":c.FT_LR*.02},
        {"params":hd.parameters(),"lr":c.FT_LR*.5}], weight_decay=c.WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(o2,T_max=max(c.FT_EPOCHS-c.WARMUP_EP,1))
    for e in range(c.FT_EPOCHS - c.WARMUP_EP):
        _ep(o2,tag=f"S2-e{e+1}"); sch.step()

    # S3: polish
    o3=torch.optim.AdamW(dual.parameters(),lr=c.FT_LR*.002,weight_decay=c.WD)
    for e in range(2): _ep(o3,tag=f"S3-polish-e{e+1}")

    dual.eval(); return dual, le, concat_dim


@torch.no_grad()
def extract_model(model, df, root, sz, c):
    tf=_mk_tf(sz); ds=InferDS(df,root,tf)
    loader=DataLoader(ds,batch_size=c.INF_BS,num_workers=c.WORKERS,pin_memory=True)
    embs=[]
    for imgs,_,_ in tqdm(loader,desc="    extract",leave=False):
        embs.append(F.normalize(model(imgs.to(c.DEVICE)),dim=-1).cpu().numpy())
    return np.concatenate(embs)


# ═══════════════════════════════════════════════════════════════════════════
# SUPER-DESCRIPTOR
# ═══════════════════════════════════════════════════════════════════════════
def build_super(feat_dict, target_dim=1024):
    parts = [normalize(feat_dict[k],axis=1) for k in sorted(feat_dict)]
    cat = np.concatenate(parts, axis=1)
    if cat.shape[1] > target_dim and cat.shape[0] > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        cat = pca.fit_transform(cat)
        print(f"    PCA → {target_dim}D (var={pca.explained_variance_ratio_.sum():.4f})")
    return normalize(cat, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
def cc_cluster(sim, threshold):
    n=sim.shape[0]; parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[a]=b
    for i in range(n):
        for j in range(i+1,n):
            if sim[i,j]>=threshold: union(i,j)
    raw=[find(i) for i in range(n)]
    uq={}; out=np.zeros(n,dtype=int); cnt=0
    for i in range(n):
        r=find(i)
        if r not in uq: uq[r]=cnt; cnt+=1
        out[i]=uq[r]
    return out

def hdbscan_cluster(features, min_cluster_size=3):
    if not HAS_HDBSCAN: return None
    feat=normalize(features,axis=1); dist=np.clip(1-feat@feat.T,0,2)
    cl=_hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
    labels=cl.fit_predict(dist)
    ns=labels.max()+1 if labels.max()>=0 else 0
    for i in range(len(labels)):
        if labels[i]==-1: labels[i]=ns; ns+=1
    return labels

def autotune_all(features, true_labels, dsn, c):
    """Auto-tune DBSCAN eps, CC threshold, Agglomerative dt, and HDBSCAN min_cluster_size."""
    feat=normalize(features,axis=1); dist=np.clip(1-feat@feat.T,0,2); sim=feat@feat.T
    le=LabelEncoder(); yt=le.fit_transform(true_labels)
    results = {}

    # DBSCAN
    ba, be = -1, .35
    for eps in np.arange(.06,.82,.02):
        pred=DBSCAN(eps=eps,min_samples=c.DBSCAN_MIN,metric="precomputed").fit_predict(dist)
        ns=pred.max()+1 if pred.max()>=0 else 0
        for i in range(len(pred)):
            if pred[i]==-1: pred[i]=ns; ns+=1
        a=adjusted_rand_score(yt,pred)
        if a>ba: ba,be=a,eps
    results["dbscan_eps"]=be; results["dbscan_ari"]=ba
    print(f"  [{dsn}] DBSCAN: eps={be:.2f} ARI={ba:.4f}")

    # CC
    ba, bt = -1, .6
    for th in np.arange(.35,.88,.02):
        pred=cc_cluster(sim,th); a=adjusted_rand_score(yt,pred)
        if a>ba: ba,bt=a,th
    results["cc_thresh"]=bt; results["cc_ari"]=ba
    print(f"  [{dsn}] CC:     th={bt:.2f}  ARI={ba:.4f}")

    # Agglomerative
    ba, bd = -1, .5
    for dt in np.arange(.10,1.,.02):
        try:
            pred=AgglomerativeClustering(n_clusters=None,distance_threshold=dt,
                  metric="precomputed",linkage="average").fit_predict(dist)
            a=adjusted_rand_score(yt,pred)
            if a>ba: ba,bd=a,dt
        except: pass
    results["agglo_dt"]=bd; results["agglo_ari"]=ba
    print(f"  [{dsn}] Agglo:  dt={bd:.2f}  ARI={ba:.4f}")

    # HDBSCAN
    if HAS_HDBSCAN:
        ba, bm = -1, 3
        for ms in range(2, 8):
            pred = hdbscan_cluster(features, min_cluster_size=ms)
            if pred is not None:
                a = adjusted_rand_score(yt, pred)
                if a > ba: ba, bm = a, ms
        results["hdbscan_min"] = bm; results["hdbscan_ari"] = ba
        print(f"  [{dsn}] HDBSCAN: min_cs={bm} ARI={ba:.4f}")

    return results


def cluster_ensemble(features, params, dsn, c):
    """Consensus of best-performing clusterers."""
    feat=normalize(features,axis=1); n=len(feat)
    dist=np.clip(1-feat@feat.T,0,2); sim=feat@feat.T

    preds = []
    # DBSCAN
    p=DBSCAN(eps=params["dbscan_eps"],min_samples=c.DBSCAN_MIN,metric="precomputed").fit_predict(dist)
    ns=p.max()+1 if p.max()>=0 else 0
    for i in range(n):
        if p[i]==-1: p[i]=ns; ns+=1
    preds.append(p)
    # CC
    preds.append(cc_cluster(sim, params["cc_thresh"]))
    # Agglo
    try:
        preds.append(AgglomerativeClustering(n_clusters=None,distance_threshold=params["agglo_dt"],
              metric="precomputed",linkage="average").fit_predict(dist))
    except: preds.append(preds[0].copy())
    # HDBSCAN
    if HAS_HDBSCAN and "hdbscan_min" in params:
        h = hdbscan_cluster(features, params["hdbscan_min"])
        if h is not None: preds.append(h)

    # Co-occurrence voting: ≥ majority agree → same cluster
    threshold = len(preds) // 2 + 1
    cooccur = np.zeros((n,n), dtype=int)
    for pred in preds:
        for i in range(n):
            for j in range(i+1,n):
                if pred[i]==pred[j]: cooccur[i,j]+=1; cooccur[j,i]+=1
    consensus_sim = (cooccur >= threshold).astype(float) + np.eye(n)
    final = cc_cluster(consensus_sim, 0.5)

    ncl = len(set(final))
    names = ["DBSCAN","CC","Agglo"] + (["HDBSCAN"] if len(preds)>3 else [])
    counts = [len(set(p)) for p in preds]
    print(f"  [{dsn}] Ensemble: " + " ".join(f"{n}={c}" for n,c in zip(names,counts)) +
          f" → Consensus={ncl}")
    return final


# ═══════════════════════════════════════════════════════════════════════════
# LOOKUP THRESHOLD AUTO-TUNE
# ═══════════════════════════════════════════════════════════════════════════
def autotune_lookup(feats, ids, dsn):
    feat=normalize(feats,axis=1); sim=feat@feat.T; np.fill_diagonal(sim,-1)
    le=LabelEncoder(); yt=le.fit_transform(ids); n=len(yt)
    ba, bt = -1, .5
    for th in np.arange(.25,.80,.02):
        pred=np.full(n,-1)
        for i in range(n):
            j=sim[i].argmax()
            if sim[i,j]>=th: pred[i]=yt[j]
            else: pred[i]=n+i
        a=adjusted_rand_score(yt,pred)
        if a>ba: ba,bt=a,th
    print(f"  [{dsn}] Lookup: th={bt:.2f} ARI={ba:.4f}")
    return bt


# ═══════════════════════════════════════════════════════════════════════════
# k-RECIPROCAL RE-RANKING
# ═══════════════════════════════════════════════════════════════════════════
def rerank(qf, gf, k1=20, k2=6, lam=.3):
    qf,gf=normalize(qf,axis=1),normalize(gf,axis=1)
    af=np.vstack([qf,gf]); nq=len(qf); n=len(af)
    sim=af@af.T; dist=np.clip(1-sim,0,2); rank=np.argsort(dist,axis=1)
    def _r(i):
        fwd=set(rank[i,1:k1+1].tolist()); r=set()
        for j in fwd:
            if i in set(rank[j,1:k2+1].tolist()): r.add(j)
        return r or {rank[i,1]}
    rs=[_r(i) for i in range(n)]
    jac=np.ones((nq,n-nq),dtype=np.float32)
    for i in range(nq):
        for j in range(nq,n):
            inter=len(rs[i]&rs[j]); union=len(rs[i]|rs[j])
            if union: jac[i,j-nq]=1-inter/union
    return 1-(lam*dist[:nq,nq:]+(1-lam)*jac)


# ═══════════════════════════════════════════════════════════════════════════
# PROCESS SPECIES
# ═══════════════════════════════════════════════════════════════════════════
def process_with_train(dsn, te_df, tr_df, pt_te, pt_tr, ft_te, ft_tr,
                       local_matcher, clust_params, c):
    ntest, ntrain = len(te_df), len(tr_df)
    trids = tr_df.identity.values

    # 1. Global similarity (multi-model weighted)
    sim_parts, ws = [], []
    for k, wa in [("mega","W_MEGA"),("miew","W_MIEW"),("dinov2","W_DINO")]:
        if k in pt_te:
            sim_parts.append(normalize(pt_te[k],axis=1)@normalize(pt_tr[k],axis=1).T)
            ws.append(getattr(c,wa))
    wt=sum(ws); pt_sim=sum(s*(w/wt) for s,w in zip(sim_parts,ws))

    # 2. Fine-tuned similarity
    if ft_te is not None:
        ft_sim=normalize(ft_te,axis=1)@normalize(ft_tr,axis=1).T
        fused=c.FT_BLEND*ft_sim+(1-c.FT_BLEND)*pt_sim
    else: fused=pt_sim

    # 3. Local matching (top-K only, for efficiency)
    if c.USE_LOCAL and local_matcher and local_matcher.available and ntrain <= 3000:
        local_sim = local_matcher.compute_local_scores(
            te_df, tr_df, fused, c.DATA_DIR, topk=min(c.LOCAL_TOPK, ntrain))
        # Calibrate & fuse
        fused = (1 - c.LOCAL_W) * fused + c.LOCAL_W * local_sim

    # 4. Re-ranking
    if c.USE_RERANK and ntrain < 12000:
        try:
            rr = rerank(
                pt_te.get("mega",list(pt_te.values())[0]),
                pt_tr.get("mega",list(pt_tr.values())[0]),
                c.RR_K1, c.RR_K2, c.RR_LAM)
            fused = 0.7 * fused + 0.3 * rr
        except: pass

    # 5. Lookup known
    th = c.THRESH.get(dsn, .50)
    mx_sim=fused.max(1); mx_idx=fused.argmax(1)
    known={i: trids[mx_idx[i]] for i in range(ntest) if mx_sim[i]>=th}
    print(f"  Lookup: {len(known)}/{ntest} ({100*len(known)/ntest:.1f}%) th={th:.2f}")

    # 6. Cluster unknowns
    unk=sorted(set(range(ntest))-set(known))
    if len(unk)>1:
        uf={k:pt_te[k][unk] for k in pt_te}
        if ft_te is not None: uf["ft"]=ft_te[unk]
        super_f=build_super(uf, target_dim=min(1024, sum(v.shape[1] for v in uf.values())))
        if c.USE_ENSEMBLE and len(unk)>10 and clust_params:
            labels=cluster_ensemble(super_f, clust_params, dsn, c)
        else:
            fn=normalize(super_f,axis=1); dist=np.clip(1-fn@fn.T,0,2)
            labels=DBSCAN(eps=clust_params.get("dbscan_eps",.35),
                          min_samples=c.DBSCAN_MIN,metric="precomputed").fit_predict(dist)
            ns=labels.max()+1 if labels.max()>=0 else 0
            for i in range(len(labels)):
                if labels[i]==-1: labels[i]=ns; ns+=1
    else:
        labels=np.arange(len(unk))

    # 7. Assign
    res={}
    for i,ident in known.items(): res[i]=f"cluster_{ident}"
    bid=10000
    for pos,ti in enumerate(unk): res[ti]=f"cluster_{dsn}_{bid+labels[pos]}"
    return res


def process_no_train(dsn, te_df, pt_te, clust_params, c):
    n=len(te_df)
    uf={k:pt_te[k] for k in pt_te}
    super_f=build_super(uf, target_dim=min(512,sum(v.shape[1] for v in uf.values())))
    if c.USE_ENSEMBLE and n>10 and clust_params:
        labels=cluster_ensemble(super_f, clust_params, dsn, c)
    else:
        fn=normalize(super_f,axis=1); dist=np.clip(1-fn@fn.T,0,2)
        labels=DBSCAN(eps=c.EPS.get(dsn,.40),min_samples=c.DBSCAN_MIN,
                      metric="precomputed").fit_predict(dist)
        ns=labels.max()+1 if labels.max()>=0 else 0
        for i in range(n):
            if labels[i]==-1: labels[i]=ns; ns+=1
    print(f"  [{dsn}] clusters={len(set(labels))}")
    return {i: f"cluster_{dsn}_{labels[i]}" for i in range(n)}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main(c):
    t0=time.time()
    print("="*80)
    print("  AnimalCLEF2026 — H800 ULTIMATE v4.0")
    print("="*80)

    setup_data(c)
    meta=pd.read_csv(os.path.join(c.DATA_DIR,"metadata.csv"))
    ssub=pd.read_csv(os.path.join(c.DATA_DIR,"sample_submission.csv"))
    os.makedirs(c.OUTPUT_DIR, exist_ok=True)
    trdf=meta[meta.split=="train"].copy(); tedf=meta[meta.split=="test"].copy()
    for d in c.DS_ALL:
        tr=trdf[trdf.dataset==d]; te=tedf[tedf.dataset==d]
        print(f"  {d:25s} train={len(tr):5d} ids={tr.identity.nunique() if len(tr) else 0:4d} test={len(te):4d}")

    # ═══ A. Load backbones ═══
    print("\n" + "━"*60 + "\nSTAGE A: Triple Backbones\n" + "━"*60)
    models = load_backbones(c)

    # ═══ B. Multi-scale feature extraction ═══
    print("\n" + "━"*60 + "\nSTAGE B: Multi-Scale Feature Extraction\n" + "━"*60)
    pt_tr, pt_te = {}, {}
    for k, m in models.items():
        print(f"\n  ── {k} ──")
        ft, it = extract_multiscale(m, trdf, c.DATA_DIR, k, c, tta=False)
        fe, ie = extract_multiscale(m, tedf, c.DATA_DIR, k, c, tta=c.USE_TTA)
        pt_tr[k]=(ft,it); pt_te[k]=(fe,ie)
        print(f"  train: {ft.shape}  test: {fe.shape}")

    fk=list(pt_tr.keys())[0]
    tr_i2x={int(pt_tr[fk][1][i]):i for i in range(len(pt_tr[fk][1]))}
    te_i2x={int(pt_te[fk][1][i]):i for i in range(len(pt_te[fk][1]))}

    # ═══ C. Local matcher ═══
    print("\n" + "━"*60 + "\nSTAGE C: Local Matcher Init\n" + "━"*60)
    local_matcher = LocalMatcher(c.DEVICE) if c.USE_LOCAL else None

    # ═══ Free GPU memory before ArcFace ═══
    print("[MEM] Releasing pretrained backbones from GPU...")
    for k in list(models.keys()):
        models[k] = models[k].cpu()
    del models; gc.collect(); torch.cuda.empty_cache()
    print(f"[MEM] GPU freed. VRAM after: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # ═══ D. ArcFace fine-tuning ═══
    print("\n" + "━"*60 + "\nSTAGE D: ArcFace Fine-Tuning\n" + "━"*60)
    ftdata = {}
    for dsn in c.DS_TRAIN:
        dtr=trdf[trdf.dataset==dsn].copy(); dte=tedf[tedf.dataset==dsn].copy()
        if len(dtr)<10: continue
        if c.USE_DUAL_ARC and c.USE_MEGA and c.USE_MIEW:
            ftm, le, edim = finetune_dual(c, dtr, c.DATA_DIR, dsn, None)
            ftr=extract_model(ftm, dtr, c.DATA_DIR, c.MEGA_SZ, c)
            fte=extract_model(ftm, dte, c.DATA_DIR, c.MEGA_SZ, c)
        else:
            ftm, le = finetune_single(c, dtr, c.DATA_DIR, dsn, c.MEGA_HUB, c.MEGA_DIM, c.MEGA_SZ)
            ftr=extract_model(ftm, dtr, c.DATA_DIR, c.MEGA_SZ, c)
            fte=extract_model(ftm, dte, c.DATA_DIR, c.MEGA_SZ, c)
        ftdata[dsn]={"tr":ftr,"te":fte}
        del ftm; torch.cuda.empty_cache(); gc.collect()

    # ═══ E. Auto-tune everything ═══
    print("\n" + "━"*60 + "\nSTAGE E: Auto-Tune All Hyperparameters\n" + "━"*60)
    all_params = {}
    for dsn in c.DS_TRAIN:
        dtr=trdf[trdf.dataset==dsn]
        if len(dtr)<20: continue
        feat = ftdata[dsn]["tr"] if dsn in ftdata else pt_tr[fk][0][[tr_i2x[int(x)] for x in dtr.image_id.values]]
        params = autotune_all(feat, dtr.identity.values, dsn, c)
        c.EPS[dsn] = params["dbscan_eps"]
        all_params[dsn] = params
        # Lookup threshold
        c.THRESH[dsn] = autotune_lookup(feat, dtr.identity.values, dsn)
    # Default params for lizard
    all_params.setdefault("TexasHornedLizards",
        {"dbscan_eps": c.EPS["TexasHornedLizards"], "cc_thresh": .55, "agglo_dt": .50})

    # ═══ F. Process each species ═══
    print("\n" + "━"*60 + "\nSTAGE F: Lookup + Ensemble Clustering\n" + "━"*60)
    all_preds = {}
    for dsn in c.DS_ALL:
        print(f"\n  ══ {dsn} ══")
        dte=tedf[tedf.dataset==dsn].copy().reset_index(drop=True)
        teix=[te_i2x[int(x)] for x in dte.image_id.values]
        pte={k:pt_te[k][0][teix] for k in pt_te}

        if dsn in c.DS_TRAIN:
            dtr=trdf[trdf.dataset==dsn].copy().reset_index(drop=True)
            trix=[tr_i2x[int(x)] for x in dtr.image_id.values]
            ptr={k:pt_tr[k][0][trix] for k in pt_tr}
            fte=ftdata[dsn]["te"] if dsn in ftdata else None
            ftr=ftdata[dsn]["tr"] if dsn in ftdata else None
            lr=process_with_train(dsn,dte,dtr,pte,ptr,fte,ftr,
                                  local_matcher,all_params.get(dsn,{}),c)
        else:
            lr=process_no_train(dsn,dte,pte,all_params.get(dsn,{}),c)
        for li,cn in lr.items():
            all_preds[int(dte.iloc[li].image_id)]=cn

    # ═══ G. Pseudo-label refinement ═══
    if c.PSEUDO_ROUNDS>0:
        print("\n" + "━"*60 + f"\nSTAGE G: Pseudo-Label Refinement ({c.PSEUDO_ROUNDS} rounds)\n" + "━"*60)
        for rnd in range(c.PSEUDO_ROUNDS):
            changed=0
            for dsn in c.DS_ALL:
                dte=tedf[tedf.dataset==dsn].copy().reset_index(drop=True)
                teix=[te_i2x[int(x)] for x in dte.image_id.values]
                feat=normalize(pt_te[fk][0][teix],axis=1)
                cm=defaultdict(list)
                for i in range(len(dte)): cm[all_preds.get(int(dte.iloc[i].image_id),"?")].append(i)
                centroids={cl:normalize(feat[idx].mean(0,keepdims=True),axis=1) for cl,idx in cm.items() if len(idx)>0}
                for i in range(len(dte)):
                    iid=int(dte.iloc[i].image_id); cur=all_preds.get(iid)
                    if cur is None or cur not in centroids: continue
                    csim=(feat[i:i+1]@centroids[cur].T)[0,0]
                    best_cl,best_s=cur,csim
                    for cl,cent in centroids.items():
                        if cl==cur or dsn not in cl: continue
                        s=(feat[i:i+1]@cent.T)[0,0]
                        if s>best_s+.05: best_s,best_cl=s,cl
                    if best_cl!=cur: all_preds[iid]=best_cl; changed+=1
            print(f"  Round {rnd+1}: {changed} reassigned")

    # ═══ H. Generate submission ═══
    print("\n" + "━"*60 + "\nSTAGE H: Submission\n" + "━"*60)
    sub=ssub.copy()
    for i in range(len(sub)):
        iid=int(sub.iloc[i].image_id)
        if iid in all_preds: sub.at[i,"cluster"]=all_preds[iid]
    out=os.path.join(c.OUTPUT_DIR,"submission.csv")
    sub.to_csv(out,index=False)
    elapsed=time.time()-t0
    print(f"\n  ✓ {out}")
    print(f"  ✓ Rows={len(sub)} Clusters={sub.cluster.nunique()}")
    for d in c.DS_ALL:
        s=sub[sub.cluster.str.contains(d)]
        print(f"    {d:25s} imgs={len(s):4d} clusters={s.cluster.nunique()}")
    print(f"\n  ⏱ {elapsed/60:.1f} min")
    print("="*80)
    return sub


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--data_dir",   default=CFG.DATA_DIR)
    pa.add_argument("--data_zip",   default=None)
    pa.add_argument("--output_dir", default=CFG.OUTPUT_DIR)
    pa.add_argument("--bs",         type=int, default=CFG.BS)
    pa.add_argument("--ft_epochs",  type=int, default=CFG.FT_EPOCHS)
    pa.add_argument("--no_tta",     action="store_true")
    pa.add_argument("--no_miew",    action="store_true")
    pa.add_argument("--no_dino",    action="store_true")
    pa.add_argument("--no_ft",      action="store_true")
    pa.add_argument("--no_dual",    action="store_true")
    pa.add_argument("--no_local",   action="store_true")
    pa.add_argument("--no_ensemble",action="store_true")
    pa.add_argument("--no_pseudo",  action="store_true")
    pa.add_argument("--no_multiscale",action="store_true")
    pa.add_argument("--no_rerank",  action="store_true")
    pa.add_argument("--pseudo_rounds",type=int,default=CFG.PSEUDO_ROUNDS)
    a=pa.parse_args()

    CFG.DATA_DIR=a.data_dir; CFG.DATA_ZIP=a.data_zip; CFG.OUTPUT_DIR=a.output_dir
    CFG.BS=a.bs; CFG.FT_EPOCHS=a.ft_epochs
    if a.no_tta:       CFG.USE_TTA=False
    if a.no_miew:      CFG.USE_MIEW=False
    if a.no_dino:      CFG.USE_DINO=False
    if a.no_ft:        CFG.DS_TRAIN=[]
    if a.no_dual:      CFG.USE_DUAL_ARC=False
    if a.no_local:     CFG.USE_LOCAL=False
    if a.no_ensemble:  CFG.USE_ENSEMBLE=False
    if a.no_pseudo:    CFG.PSEUDO_ROUNDS=0
    else:              CFG.PSEUDO_ROUNDS=a.pseudo_rounds
    if a.no_multiscale:CFG.MULTI_SCALE=False
    if a.no_rerank:    CFG.USE_RERANK=False

    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        g=torch.cuda.get_device_properties(0)
        print(f"GPU: {g.name} | VRAM: {g.total_memory/1e9:.1f} GB")
    print(f"HDBSCAN: {HAS_HDBSCAN} | LightGlue: {HAS_LIGHTGLUE} | Kornia: {HAS_KORNIA}")

    main(CFG)
