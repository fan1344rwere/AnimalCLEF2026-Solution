#!/usr/bin/env python
"""
v10: Similarity-level fusion + species-specific strategy.
- SeaTurtle: lookup (train ARI=0.86) + cluster unknowns
- Lynx/Salamander: train+test anchored clustering (no lookup)
- All: similarity fusion of Mega+MiewID (NOT feature concat!)
- TexasHornedLizards: pure clustering with heuristic
"""
import os, sys, warnings, gc, json, time as _t
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize, LabelEncoder
from tqdm import tqdm
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
warnings.filterwarnings("ignore")
DEV="cuda"; BS=48; t0=_t.time()
IM=[.485,.456,.406]; IS=[.229,.224,.225]

class DS(Dataset):
    def __init__(s,df,root,sz):
        s.df=df.reset_index(drop=True); s.root=root
        s.tf=transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),transforms.Normalize(IM,IS)])
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        if r.get("species")=="salamander" and pd.notna(r.get("orientation")):
            o=str(r["orientation"]).lower()
            if o=="right": img=img.rotate(-90,expand=True)
            elif o=="left": img=img.rotate(90,expand=True)
        return s.tf(img), int(r["image_id"])

@torch.no_grad()
def feats(model,df,root,sz):
    dl=DataLoader(DS(df,root,sz),batch_size=BS,num_workers=4,pin_memory=True)
    e,ids=[],[]
    for imgs,iids in tqdm(dl,leave=False):
        e.append(F.normalize(model(imgs.to(DEV)),dim=-1).cpu().numpy()); ids.extend(iids.numpy())
    return np.concatenate(e), np.array(ids)

def fused_dist(mega_f, miew_f, w_mega=0.6):
    """Similarity-level fusion → distance. Key: fuse SIMILARITIES, not features."""
    mega_n = normalize(mega_f, axis=1)
    miew_n = normalize(miew_f, axis=1)
    sim = w_mega * (mega_n @ mega_n.T) + (1-w_mega) * (miew_n @ miew_n.T)
    return np.clip(1 - sim, 0, 2), sim

def fused_cross_dist(te_mega, te_miew, tr_mega, tr_miew, w_mega=0.6):
    """Cross similarity: test × train."""
    tm = normalize(te_mega, axis=1); trm = normalize(tr_mega, axis=1)
    tw = normalize(te_miew, axis=1); trw = normalize(tr_miew, axis=1)
    return w_mega * (tm @ trm.T) + (1-w_mega) * (tw @ trw.T)

def tune_clustering(dist, yt, label=""):
    """Try DBSCAN + Agglo, return best method and params."""
    best_ari, best_p, best_m = -1, 0.5, "dbscan"
    # DBSCAN sweep
    for eps in np.arange(0.05, 1.30, 0.01):
        pred = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
        ns = pred.max()+1 if pred.max()>=0 else 0
        for i in range(len(pred)):
            if pred[i]==-1: pred[i]=ns; ns+=1
        ari = adjusted_rand_score(yt, pred)
        if ari > best_ari: best_ari, best_p, best_m = ari, eps, "dbscan"
    # Agglomerative sweep
    for dt in np.arange(0.05, 1.30, 0.01):
        try:
            pred = AgglomerativeClustering(n_clusters=None, distance_threshold=dt,
                metric="precomputed", linkage="average").fit_predict(dist)
            ari = adjusted_rand_score(yt, pred)
            if ari > best_ari: best_ari, best_p, best_m = ari, dt, "agglo"
        except: pass
    print(f"  {label} best: {best_m} p={best_p:.2f} ARI={best_ari:.4f}")
    return best_m, best_p, best_ari

def apply_clustering(dist, method, param):
    if method == "dbscan":
        labels = DBSCAN(eps=param, min_samples=2, metric="precomputed").fit_predict(dist)
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=param,
            metric="precomputed", linkage="average").fit_predict(dist)
    ns = labels.max()+1 if labels.max()>=0 else 0
    for i in range(len(labels)):
        if labels[i]==-1: labels[i]=ns; ns+=1
    return labels

# ── Load models ──
print("Loading MegaDescriptor...", flush=True)
mega = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0).to(DEV).eval()
print(f"  dim={mega.num_features}")

print("Loading MiewID...", flush=True)
cfg_path = hf_hub_download("conservationxlabs/miewid-msv3", "config.json")
with open(cfg_path) as f: cfg = json.load(f)
arch = cfg.get("architecture", "efficientnetv2_rw_m")
miew = timm.create_model(arch, pretrained=False, num_classes=0)
wt_path = hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")
state = load_file(wt_path)
state = {k:v for k,v in state.items() if "classifier" not in k}
miew.load_state_dict(state, strict=False)
miew = miew.eval().to(DEV)
print(f"  dim={miew.num_features}")

DATA = sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
OUT = sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov10"
os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(os.path.join(DATA, "metadata.csv"))
ssub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
trdf = meta[meta.split=="train"].copy(); tedf = meta[meta.split=="test"].copy()

print("\nExtracting features...")
mega_tr, mega_tr_ids = feats(mega, trdf, DATA, 384)
mega_te, mega_te_ids = feats(mega, tedf, DATA, 384)
miew_tr, _ = feats(miew, trdf, DATA, 440)
miew_te, _ = feats(miew, tedf, DATA, 440)
del mega, miew; torch.cuda.empty_cache(); gc.collect()
print(f"Done in {(_t.time()-t0)/60:.1f}min")

tr_i2x = {int(mega_tr_ids[i]):i for i in range(len(mega_tr_ids))}
te_i2x = {int(mega_te_ids[i]):i for i in range(len(mega_te_ids))}

DS_TR = ["LynxID2025","SalamanderID2025","SeaTurtleID2022"]
DS_ALL = DS_TR + ["TexasHornedLizards"]
all_preds = {}

for dsn in DS_ALL:
    print(f"\n{'='*55}\n  {dsn}\n{'='*55}")
    ds_te = tedf[tedf.dataset==dsn].reset_index(drop=True)
    teix = [te_i2x[int(x)] for x in ds_te.image_id.values]
    te_m, te_w = mega_te[teix], miew_te[teix]

    if dsn in DS_TR:
        ds_tr = trdf[trdf.dataset==dsn].reset_index(drop=True)
        trix = [tr_i2x[int(x)] for x in ds_tr.image_id.values]
        tr_m, tr_w = mega_tr[trix], miew_tr[trix]
        train_ids = ds_tr.identity.values
        n_tr, n_te = len(ds_tr), len(ds_te)

        # ── Strategy: anchored clustering (train+test together) ──
        # Fused similarity distance for train+test combined
        all_mega = np.vstack([tr_m, te_m])
        all_miew = np.vstack([tr_w, te_w])
        dist, sim = fused_dist(all_mega, all_miew, w_mega=0.6)

        # Tune on train portion
        le = LabelEncoder(); yt = le.fit_transform(train_ids)

        # Extract train-only distance submatrix for tuning
        tr_dist = dist[:n_tr, :n_tr]
        method, param, train_ari = tune_clustering(tr_dist, yt, label=f"{dsn}-train-only")

        # Also try tuning on full matrix (train portion labels)
        method2, param2, train_ari2 = tune_clustering(dist,
            np.concatenate([yt, np.arange(n_te) + yt.max() + 1]),  # test gets unique labels
            label=f"{dsn}-anchored")

        # Pick whichever gives better train ARI
        if train_ari2 > train_ari:
            use_method, use_param = method2, param2
            print(f"  → Using anchored: {method2} p={param2:.2f} ARI={train_ari2:.4f}")
        else:
            use_method, use_param = method, param
            print(f"  → Using train-only: {method} p={param:.2f} ARI={train_ari:.4f}")

        # Apply to full train+test
        labels = apply_clustering(dist, use_method, use_param)
        te_labels = labels[n_tr:]
        n_cl = len(set(te_labels))
        print(f"  Test: {n_cl} clusters for {n_te} images")

        # ── For SeaTurtle (high train ARI): also try lookup approach ──
        if dsn == "SeaTurtleID2022" and train_ari > 0.5:
            cross_sim = fused_cross_dist(te_m, te_w, tr_m, tr_w, w_mega=0.6)
            # Tune lookup threshold
            tr_self_sim = fused_cross_dist(tr_m, tr_w, tr_m, tr_w, w_mega=0.6)
            np.fill_diagonal(tr_self_sim, -1)
            best_th, best_lk_ari = 0.5, -1
            for th in np.arange(0.30, 0.85, 0.01):
                pred = np.array([yt[tr_self_sim[i].argmax()] if tr_self_sim[i].max()>=th
                                else len(yt)+i for i in range(n_tr)])
                ari = adjusted_rand_score(yt, pred)
                if ari > best_lk_ari: best_lk_ari, best_th = ari, th
            print(f"  Lookup: th={best_th:.2f} train_ARI={best_lk_ari:.4f}")

            if best_lk_ari > train_ari:
                # Lookup is better! Use it.
                max_sim = cross_sim.max(axis=1)
                max_idx = cross_sim.argmax(axis=1)
                known = {i: train_ids[max_idx[i]] for i in range(n_te) if max_sim[i] >= best_th}
                print(f"  Lookup matched: {len(known)}/{n_te}")

                # Cluster unknowns
                unk = sorted(set(range(n_te)) - set(known))
                if len(unk) > 1:
                    unk_dist, _ = fused_dist(te_m[unk], te_w[unk], w_mega=0.6)
                    unk_labels = apply_clustering(unk_dist, use_method, use_param)
                else:
                    unk_labels = np.arange(len(unk))

                # Build labels: known get identity-based cluster, unknown get new clusters
                te_labels = np.zeros(n_te, dtype=int)
                # Assign known: same identity → same cluster
                id_to_cl = {}; next_cl = 0
                for i, ident in known.items():
                    if ident not in id_to_cl: id_to_cl[ident] = next_cl; next_cl += 1
                    te_labels[i] = id_to_cl[ident]
                # Assign unknown: offset from known clusters
                base = next_cl
                for pos, ti in enumerate(unk):
                    te_labels[ti] = base + unk_labels[pos]

                n_cl = len(set(te_labels))
                print(f"  Lookup+Cluster: {n_cl} clusters")

    else:
        # TexasHornedLizards: pure clustering
        dist, sim = fused_dist(te_m, te_w, w_mega=0.6)
        best_eps, best_score = 0.40, -1
        for eps in np.arange(0.15, 0.80, 0.01):
            pred = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
            n_noise = (pred==-1).sum()
            ns = pred.max()+1 if pred.max()>=0 else 0
            for i in range(len(pred)):
                if pred[i]==-1: pred[i]=ns; ns+=1
            n_cl = len(set(pred))
            if 15 <= n_cl <= 150:
                score = 1 - n_noise/len(pred) - abs(n_cl/len(pred) - 0.3)
                if score > best_score: best_score, best_eps = score, eps
        te_labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
        ns = te_labels.max()+1 if te_labels.max()>=0 else 0
        for i in range(len(te_labels)):
            if te_labels[i]==-1: te_labels[i]=ns; ns+=1
        print(f"  eps={best_eps:.2f} clusters={len(set(te_labels))}")

    for i in range(len(ds_te)):
        all_preds[int(ds_te.iloc[i].image_id)] = f"cluster_{dsn}_{te_labels[i]}"

# ── Submission ──
sub = ssub.copy()
for i in range(len(sub)):
    iid = int(sub.iloc[i].image_id)
    if iid in all_preds: sub.at[i, "cluster"] = all_preds[iid]
out = os.path.join(OUT, "submission.csv"); sub.to_csv(out, index=False)
el = (_t.time()-t0)/60
print(f"\n{'='*55}\nSaved: {out}\nRows:{len(sub)} Clusters:{sub.cluster.nunique()}")
for d in DS_ALL:
    s = sub[sub.cluster.str.contains(d)]
    print(f"  {d:25s} imgs={len(s):4d} cl={s.cluster.nunique()}")
print(f"Time: {el:.1f}min\nDONE!")
