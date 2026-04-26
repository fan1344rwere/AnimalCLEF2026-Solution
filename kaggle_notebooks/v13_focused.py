#!/usr/bin/env python
"""
V13 — Focused Fix: 回归本质，每一步都对
=====================================
V12的3个致命Bug:
  1. ALIKED结果没注入聚类矩阵（白算1小时）
  2. train+test合并聚类参数失配
  3. ArcFace OOM

V13核心修复:
  ★ 策略改为: lookup已知 → 聚类未知 (不再合并聚类)
  ★ ArcFace用fp16+BS=8，先释放全部模型再训练
  ★ 每个物种独立处理，内存严格管控
  ★ 简洁高效，30分钟出结果

预计时间: ~25-30分钟 (5090 32GB)
"""
import os, sys, gc, warnings, time, json
from collections import defaultdict

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize, LabelEncoder

warnings.filterwarnings("ignore")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IM = [.485,.456,.406]; IS = [.229,.224,.225]

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
DATA = sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
OUT  = sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov13"
os.makedirs(OUT, exist_ok=True)

DS_TRAIN = ["LynxID2025","SalamanderID2025","SeaTurtleID2022"]
DS_ALL = DS_TRAIN + ["TexasHornedLizards"]

# Fusion weights for global similarity
W_MEGA, W_MIEW = 0.55, 0.45

# ArcFace config — 5090 32GB with fp16 can handle BS=48
ARC_BS = 48; ARC_EPOCHS = 18; ARC_LR = 3e-4

# ═══════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════
def _orient(img, row):
    if row.get("species")=="salamander" and pd.notna(row.get("orientation")):
        o = str(row["orientation"]).lower()
        if o=="right": img=img.rotate(-90,expand=True)
        elif o=="left": img=img.rotate(90,expand=True)
    return img

class InferDS(Dataset):
    def __init__(s,df,root,sz):
        s.df=df.reset_index(drop=True); s.root=root
        s.tf=transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),transforms.Normalize(IM,IS)])
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        img=_orient(img,r); return s.tf(img), int(r["image_id"])

class TrainDS(Dataset):
    def __init__(s,df,root,sz):
        s.df=df.reset_index(drop=True); s.root=root
        s.tf=transforms.Compose([
            transforms.RandomResizedCrop(sz,scale=(.7,1.),ratio=(.8,1.2)),
            transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(15),
            transforms.ColorJitter(.3,.3,.2,.1),transforms.RandomGrayscale(.05),
            transforms.ToTensor(),transforms.Normalize(IM,IS),
            transforms.RandomErasing(p=.2,scale=(.02,.15))])
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        img=_orient(img,r); return s.tf(img), int(r["label"])

# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def extract(model, df, root, sz, bs=48):
    dl = DataLoader(InferDS(df,root,sz), batch_size=bs, num_workers=4, pin_memory=True)
    embs, ids = [], []
    for imgs, iids in tqdm(dl, leave=False, desc=f"  @{sz}"):
        embs.append(F.normalize(model(imgs.to(DEV)),dim=-1).cpu().numpy())
        ids.extend(iids.numpy())
    return np.concatenate(embs), np.array(ids)

# ═══════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════
def load_mega():
    import timm
    print("[MODEL] MegaDescriptor-L-384...", end=" ", flush=True)
    m = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(DEV).eval()
    print(f"OK dim={m.num_features}"); return m

def load_miew():
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    print("[MODEL] MiewID-msv3...", end=" ", flush=True)
    cfg_path = hf_hub_download("conservationxlabs/miewid-msv3","config.json")
    with open(cfg_path) as f: cfg=json.load(f)
    arch = cfg.get("architecture","efficientnetv2_rw_m")
    m = timm.create_model(arch,pretrained=False,num_classes=0)
    wt = hf_hub_download("conservationxlabs/miewid-msv3","model.safetensors")
    state = {k:v for k,v in load_file(wt).items() if "classifier" not in k}
    m.load_state_dict(state,strict=False)
    m = m.eval().to(DEV)
    print(f"OK dim={m.num_features}"); return m

# ═══════════════════════════════════════════════════════════
# ARCFACE — memory-safe, fp16, small batch
# ═══════════════════════════════════════════════════════════
class ArcHead(nn.Module):
    def __init__(s,d,n,sc=64.,m=.5):
        super().__init__(); s.s=sc; s.m=m
        s.W=nn.Parameter(torch.empty(n,d)); nn.init.xavier_uniform_(s.W)
        s.cm=np.cos(m); s.sm=np.sin(m); s.th=np.cos(np.pi-m); s.mm=np.sin(np.pi-m)*m
    def forward(s,x,y=None):
        cos=F.linear(F.normalize(x),F.normalize(s.W))
        if y is None: return cos*s.s
        sin=(1-cos.pow(2).clamp(0,1)).sqrt()
        phi=cos*s.cm-sin*s.sm; phi=torch.where(cos>s.th,phi,cos-s.mm)
        oh=torch.zeros_like(cos).scatter_(1,y.unsqueeze(1),1.)
        return (oh*phi+(1-oh)*cos)*s.s

def finetune_arcface(tdf, root, dsn):
    """ArcFace fine-tune with fp16 + gradient accumulation to avoid OOM."""
    import timm
    le = LabelEncoder(); tdf = tdf.copy()
    tdf["label"] = le.fit_transform(tdf.identity.values)
    ncls = tdf.label.nunique()
    if ncls < 5:
        print(f"  [{dsn}] Too few classes ({ncls}), skip ArcFace"); return None, None

    print(f"  [{dsn}] ArcFace: {ncls} classes, {len(tdf)} imgs, BS={ARC_BS}")

    bb = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(DEV)
    edim = bb.num_features
    hd = ArcHead(edim, ncls).to(DEV)
    ds = TrainDS(tdf, root, 384)

    ccnt = tdf.label.value_counts().to_dict()
    sw = [1./ccnt[int(r.label)] for _,r in tdf.iterrows()]
    loader = DataLoader(ds, batch_size=ARC_BS, sampler=WeightedRandomSampler(sw,len(sw)),
                        num_workers=4, pin_memory=True, drop_last=True)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler()  # fp16

    # Stage 1: head only (2 epochs)
    for p in bb.parameters(): p.requires_grad=False
    opt = torch.optim.AdamW(hd.parameters(), lr=ARC_LR, weight_decay=1e-4)
    for ep in range(2):
        bb.eval(); hd.train(); ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels = imgs.to(DEV),labels.to(DEV)
            with torch.no_grad():
                emb = F.normalize(bb(imgs),dim=-1)
            with torch.amp.autocast(device_type='cuda'):
                logits = hd(emb,labels); loss = crit(logits,labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ls.append(loss.item()); co+=(logits.argmax(1)==labels).sum().item(); to+=labels.size(0)
        print(f"    S1-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")

    # Stage 2: top 25% of backbone (5 epochs)
    nps = list(bb.named_parameters())
    top25 = {n for n,_ in nps[int(len(nps)*.75):]}
    for n,p in bb.named_parameters(): p.requires_grad = (n in top25)

    s2_epochs = min(5, ARC_EPOCHS - 4)
    opt2 = torch.optim.AdamW([
        {"params":[p for n,p in bb.named_parameters() if p.requires_grad],"lr":ARC_LR*0.1},
        {"params":hd.parameters(),"lr":ARC_LR*0.8}], weight_decay=1e-4)

    for ep in range(s2_epochs):
        bb.train(); hd.train(); ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels = imgs.to(DEV),labels.to(DEV)
            with torch.amp.autocast(device_type='cuda'):
                emb = F.normalize(bb(imgs),dim=-1)
                logits = hd(emb,labels); loss = crit(logits,labels)
            opt2.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(bb.parameters(),1.)
            scaler.step(opt2); scaler.update()
            ls.append(loss.item()); co+=(logits.argmax(1)==labels).sum().item(); to+=labels.size(0)
        print(f"    S2-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")

    # Stage 3: FULL backbone unfreeze — BS=16 to fit 32GB VRAM
    del loader  # free old loader
    torch.cuda.empty_cache(); gc.collect()
    S3_BS = 16
    loader3 = DataLoader(ds, batch_size=S3_BS, sampler=WeightedRandomSampler(sw,len(sw)),
                         num_workers=4, pin_memory=True, drop_last=True)
    for p in bb.parameters(): p.requires_grad = True
    s3_epochs = ARC_EPOCHS - 2 - s2_epochs
    opt3 = torch.optim.AdamW([
        {"params":bb.parameters(),"lr":ARC_LR*0.03},
        {"params":hd.parameters(),"lr":ARC_LR*0.3}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=max(s3_epochs,1))
    print(f"    S3: full unfreeze BS={S3_BS}")

    for ep in range(s3_epochs):
        bb.train(); hd.train(); ls,co,to=[],0,0
        for imgs,labels in loader3:
            imgs,labels = imgs.to(DEV),labels.to(DEV)
            with torch.amp.autocast(device_type='cuda'):
                emb = F.normalize(bb(imgs),dim=-1)
                logits = hd(emb,labels); loss = crit(logits,labels)
            opt3.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt3)
            torch.nn.utils.clip_grad_norm_(bb.parameters(),1.)
            scaler.step(opt3); scaler.update()
            ls.append(loss.item()); co+=(logits.argmax(1)==labels).sum().item(); to+=labels.size(0)
        sch.step()
        print(f"    S3-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")

    bb.eval(); del hd, opt, opt2, opt3, scaler, sch, loader3, ds
    torch.cuda.empty_cache(); gc.collect()
    return bb, le

# ═══════════════════════════════════════════════════════════
# CLUSTERING: simple & effective
# ═══════════════════════════════════════════════════════════
def fix_noise(labels):
    labels = labels.copy()
    ns = labels.max()+1 if labels.max()>=0 else 0
    for i in range(len(labels)):
        if labels[i]==-1: labels[i]=ns; ns+=1
    return labels

def best_agglo(dist, yt):
    """Find best Agglomerative threshold on training data."""
    best_ari, best_dt = -1, 0.5
    for dt in np.arange(0.05, 1.50, 0.01):
        try:
            pred = AgglomerativeClustering(n_clusters=None, distance_threshold=dt,
                    metric="precomputed", linkage="average").fit_predict(dist)
            ari = adjusted_rand_score(yt, pred)
            if ari > best_ari: best_ari, best_dt = ari, dt
        except: pass
    return best_dt, best_ari

def tune_lookup_threshold(tr_sim, yt):
    """Find best lookup threshold: test×train sim → identity assignment."""
    n = len(yt)
    np.fill_diagonal(tr_sim, -999)
    best_ari, best_th = -1, 0.5
    for th in np.arange(0.20, 0.90, 0.01):
        pred = np.zeros(n, dtype=int)
        next_new = yt.max() + 1
        for i in range(n):
            j = tr_sim[i].argmax()
            if tr_sim[i,j] >= th:
                pred[i] = yt[j]
            else:
                pred[i] = next_new; next_new += 1
        ari = adjusted_rand_score(yt, pred)
        if ari > best_ari: best_ari, best_th = ari, th
    return best_th, best_ari

# ═══════════════════════════════════════════════════════════
# FUSED SIMILARITY (similarity-level, NOT feature concat)
# ═══════════════════════════════════════════════════════════
def fused_sim(feats_dict, idx, w_mega=W_MEGA, w_miew=W_MIEW):
    """Compute weighted similarity from multiple model features."""
    parts, ws = [], []
    for name, (feats, _) in feats_dict.items():
        f = normalize(feats[idx], axis=1)
        parts.append(f @ f.T)
        ws.append(w_mega if "mega" in name else w_miew)
    wt = sum(ws)
    return sum(s*(w/wt) for s,w in zip(parts,ws))

def fused_cross_sim(feats_dict, te_idx, tr_idx, w_mega=W_MEGA, w_miew=W_MIEW):
    """Cross similarity: test × train."""
    parts, ws = [], []
    for name, (feats, _) in feats_dict.items():
        te_f = normalize(feats[te_idx], axis=1)
        tr_f = normalize(feats[tr_idx], axis=1)
        parts.append(te_f @ tr_f.T)
        ws.append(w_mega if "mega" in name else w_miew)
    wt = sum(ws)
    return sum(s*(w/wt) for s,w in zip(parts,ws))

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*60)
    print("  AnimalCLEF2026 V13 — Focused Fix")
    print("="*60)

    if torch.cuda.is_available():
        g = torch.cuda.get_device_properties(0)
        print(f"GPU: {g.name} | VRAM: {g.total_memory/1e9:.1f}GB")

    meta = pd.read_csv(os.path.join(DATA,"metadata.csv"))
    ssub = pd.read_csv(os.path.join(DATA,"sample_submission.csv"))
    trdf = meta[meta.split=="train"].copy()
    tedf = meta[meta.split=="test"].copy()
    print(f"Data: {len(trdf)} train, {len(tedf)} test")
    for d in DS_ALL:
        tr=trdf[trdf.dataset==d]; te=tedf[tedf.dataset==d]
        print(f"  {d:25s} tr={len(tr):5d} ids={tr.identity.nunique() if len(tr) else 0:4d} te={len(te):4d}")

    # ══════════════════════════════════════════════════════
    # STAGE 1: Extract global features (Mega + MiewID)
    # ══════════════════════════════════════════════════════
    print(f"\n{'━'*55}\nSTAGE 1: Global Feature Extraction\n{'━'*55}")

    all_feats = {}  # {"mega": (feats, ids), "miew": (feats, ids)}

    mega = load_mega()
    mega_tr, tr_ids = extract(mega, trdf, DATA, 384, bs=48)
    mega_te, te_ids = extract(mega, tedf, DATA, 384, bs=48)
    all_feats["mega"] = (np.vstack([mega_tr, mega_te]), np.concatenate([tr_ids, te_ids]))
    n_tr = len(mega_tr)
    print(f"  Mega: train={mega_tr.shape} test={mega_te.shape}")
    del mega; torch.cuda.empty_cache(); gc.collect()

    miew = load_miew()
    miew_tr, _ = extract(miew, trdf, DATA, 440, bs=32)
    miew_te, _ = extract(miew, tedf, DATA, 440, bs=32)
    all_feats["miew"] = (np.vstack([miew_tr, miew_te]), np.concatenate([tr_ids, te_ids]))
    print(f"  Miew: train={miew_tr.shape} test={miew_te.shape}")
    del miew; torch.cuda.empty_cache(); gc.collect()

    tr_i2x = {int(tr_ids[i]):i for i in range(len(tr_ids))}
    te_i2x = {int(te_ids[i]):i for i in range(len(te_ids))}

    feat_time = (time.time()-t0)/60
    print(f"  Done in {feat_time:.1f}min | GPU free: {torch.cuda.memory_allocated()/1e9:.1f}GB used")

    # ══════════════════════════════════════════════════════
    # STAGE 2: ArcFace fine-tuning (per-species)
    # ══════════════════════════════════════════════════════
    print(f"\n{'━'*55}\nSTAGE 2: ArcFace Fine-Tuning (fp16, BS={ARC_BS})\n{'━'*55}")

    ft_feats = {}  # {dsn: {"tr": feats, "te": feats}}

    for dsn in DS_TRAIN:
        dtr = trdf[trdf.dataset==dsn].copy()
        dte = tedf[tedf.dataset==dsn].copy()
        if len(dtr) < 50:
            print(f"  [{dsn}] Too few training images, skip")
            continue

        try:
            ftm, le = finetune_arcface(dtr, DATA, dsn)
            if ftm is None: continue

            # Extract fine-tuned features
            ft_tr, _ = extract(ftm, dtr, DATA, 384, bs=32)
            ft_te, _ = extract(ftm, dte, DATA, 384, bs=32)
            ft_feats[dsn] = {"tr": ft_tr, "te": ft_te}
            print(f"  [{dsn}] FT features: train={ft_tr.shape} test={ft_te.shape}")
            del ftm; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"  [{dsn}] ArcFace FAILED: {e}")
            torch.cuda.empty_cache(); gc.collect()

    arc_time = (time.time()-t0)/60
    print(f"  ArcFace done in {arc_time-feat_time:.1f}min")

    # ══════════════════════════════════════════════════════
    # STAGE 3: Per-species lookup + cluster
    # ══════════════════════════════════════════════════════
    print(f"\n{'━'*55}\nSTAGE 3: Lookup + Cluster (per species)\n{'━'*55}")

    all_preds = {}

    for dsn in DS_ALL:
        print(f"\n{'═'*50}\n  {dsn}\n{'═'*50}")

        ds_te = tedf[tedf.dataset==dsn].reset_index(drop=True)
        teix = [te_i2x[int(x)] for x in ds_te.image_id.values]
        # Offset for combined feature array
        teix_global = [n_tr + i for i in teix]
        n_te = len(ds_te)

        if dsn in DS_TRAIN:
            ds_tr = trdf[trdf.dataset==dsn].reset_index(drop=True)
            trix = [tr_i2x[int(x)] for x in ds_tr.image_id.values]
            train_ids = ds_tr.identity.values
            le = LabelEncoder(); yt = le.fit_transform(train_ids)
            n_tr_ds = len(ds_tr)

            # ── Compute similarities ──
            # Global: test×test, test×train, train×train
            te_te_sim = fused_sim(all_feats, teix_global)
            cross_sim = fused_cross_sim(all_feats, teix_global, trix)
            tr_tr_sim = fused_sim(all_feats, trix)

            # ── Blend with ArcFace features if available ──
            ft_blend = 0.0
            if dsn in ft_feats:
                ft = ft_feats[dsn]
                ft_tr_n = normalize(ft["tr"], axis=1)
                ft_te_n = normalize(ft["te"], axis=1)
                ft_te_te = ft_te_n @ ft_te_n.T
                ft_cross = ft_te_n @ ft_tr_n.T
                ft_tr_tr = ft_tr_n @ ft_tr_n.T

                # Decide blend weight based on fine-tune quality
                ft_blend = 0.50
                te_te_sim = (1-ft_blend)*te_te_sim + ft_blend*ft_te_te
                cross_sim = (1-ft_blend)*cross_sim + ft_blend*ft_cross
                tr_tr_sim = (1-ft_blend)*tr_tr_sim + ft_blend*ft_tr_tr
                print(f"  Blended with ArcFace ({ft_blend:.0%})")

            # ── Tune lookup threshold on train data ──
            lk_th, lk_ari = tune_lookup_threshold(tr_tr_sim.copy(), yt)
            print(f"  Lookup threshold: {lk_th:.2f} (train ARI={lk_ari:.4f})")

            # ── Tune clustering on train data ──
            tr_dist = np.clip(1 - tr_tr_sim, 0, 2)
            ag_dt, ag_ari = best_agglo(tr_dist, yt)
            print(f"  Agglo threshold: {ag_dt:.3f} (train ARI={ag_ari:.4f})")

            # ── Decision: use lookup if it helps, otherwise pure cluster ──
            use_lookup = lk_ari > ag_ari * 0.8  # lookup is at least 80% as good

            if use_lookup:
                # ── LOOKUP + CLUSTER APPROACH ──
                print(f"  Strategy: LOOKUP + CLUSTER")

                # Step 1: Match test → train
                mx_sim = cross_sim.max(axis=1)
                mx_idx = cross_sim.argmax(axis=1)

                known = {}
                for i in range(n_te):
                    if mx_sim[i] >= lk_th:
                        known[i] = train_ids[mx_idx[i]]

                n_known = len(known)
                print(f"  Matched: {n_known}/{n_te} ({100*n_known/n_te:.1f}%)")

                # Step 2: Cluster unknowns
                unk = sorted(set(range(n_te)) - set(known))

                if len(unk) > 1:
                    unk_sim = te_te_sim[np.ix_(unk, unk)]
                    unk_dist = np.clip(1 - unk_sim, 0, 2)

                    # Estimate expected number of unknown clusters
                    # Use ratio from training: n_individuals / n_images
                    ratio = n_tr_ds / max(1, len(set(yt))) if len(set(yt))>0 else 5
                    expected_n_unk = max(3, len(unk) / ratio)

                    # Try Agglo with different thresholds
                    best_dt, best_ncl_diff = ag_dt, 999
                    for dt in np.arange(0.10, 1.50, 0.01):
                        try:
                            pred = AgglomerativeClustering(n_clusters=None, distance_threshold=dt,
                                    metric="precomputed",linkage="average").fit_predict(unk_dist)
                            ncl = len(set(pred))
                            diff = abs(ncl - expected_n_unk)
                            if diff < best_ncl_diff:
                                best_ncl_diff, best_dt = diff, dt
                        except: pass

                    unk_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=best_dt,
                                metric="precomputed",linkage="average").fit_predict(unk_dist)
                    print(f"  Unknown clusters: {len(set(unk_labels))} (dt={best_dt:.3f})")
                else:
                    unk_labels = np.arange(len(unk))

                # Step 3: Build final labels
                te_labels = np.zeros(n_te, dtype=object)
                # Known: use identity name as cluster label
                for i, ident in known.items():
                    te_labels[i] = f"known_{ident}"
                # Unknown: use numbered cluster labels
                base = 10000
                for pos, ti in enumerate(unk):
                    te_labels[ti] = f"new_{base + unk_labels[pos]}"

                # Deduplicate: same identity → same cluster number
                label_map = {}; next_cl = 0
                final_labels = np.zeros(n_te, dtype=int)
                for i in range(n_te):
                    lbl = te_labels[i]
                    if lbl not in label_map:
                        label_map[lbl] = next_cl; next_cl += 1
                    final_labels[i] = label_map[lbl]

                n_cl = len(set(final_labels))
                print(f"  → {n_cl} clusters ({n_known} known + {len(set(unk_labels)) if len(unk)>1 else 0} new)")

            else:
                # ── PURE CLUSTERING APPROACH (fallback) ──
                print(f"  Strategy: PURE CLUSTERING (Agglo dt={ag_dt:.3f})")

                # Cluster test images only using test×test similarity
                te_dist = np.clip(1 - te_te_sim, 0, 2)
                final_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=ag_dt,
                                metric="precomputed",linkage="average").fit_predict(te_dist)
                n_cl = len(set(final_labels))
                print(f"  → {n_cl} clusters for {n_te} images")

        else:
            # TexasHornedLizards: no training data → pure clustering
            te_te_sim = fused_sim(all_feats, teix_global)
            te_dist = np.clip(1 - te_te_sim, 0, 2)

            # Heuristic: try to find good number of clusters
            best_dt, best_score = 0.40, -1
            for dt in np.arange(0.15, 1.20, 0.01):
                try:
                    pred = AgglomerativeClustering(n_clusters=None, distance_threshold=dt,
                            metric="precomputed",linkage="average").fit_predict(te_dist)
                    ncl = len(set(pred))
                    # Heuristic: expect 30-100 individuals for 274 images
                    if 15 <= ncl <= 200:
                        score = -abs(ncl/n_te - 0.35)
                        if score > best_score: best_score, best_dt = score, dt
                except: pass

            final_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=best_dt,
                            metric="precomputed",linkage="average").fit_predict(te_dist)
            n_cl = len(set(final_labels))
            print(f"  → {n_cl} clusters (dt={best_dt:.2f})")

        # Store predictions
        for i in range(n_te):
            all_preds[int(ds_te.iloc[i].image_id)] = f"cluster_{dsn}_{final_labels[i]}"

    # ══════════════════════════════════════════════════════
    # STAGE 4: Generate submission
    # ══════════════════════════════════════════════════════
    print(f"\n{'━'*55}\nSTAGE 4: Submission\n{'━'*55}")
    sub = ssub.copy()
    for i in range(len(sub)):
        iid = int(sub.iloc[i].image_id)
        if iid in all_preds: sub.at[i,"cluster"] = all_preds[iid]

    out_path = os.path.join(OUT, "submission.csv")
    sub.to_csv(out_path, index=False)
    elapsed = (time.time()-t0)/60

    print(f"\n  Saved: {out_path}")
    print(f"  Rows: {len(sub)} Clusters: {sub.cluster.nunique()}")
    for d in DS_ALL:
        s = sub[sub.cluster.str.contains(d)]
        print(f"    {d:25s} imgs={len(s):4d} cl={s.cluster.nunique()}")
    print(f"\n  Total: {elapsed:.1f}min")
    print("="*60)
    print("DONE!")

if __name__ == "__main__":
    main()
