#!/usr/bin/env python
"""
v9: Back to basics. MegaDescriptor ONLY + train anchored clustering.
Key insight: cluster train+test TOGETHER, train images anchor known individuals.
Then project cluster labels to test only.
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
warnings.filterwarnings("ignore")
DEV="cuda"; BS=64; t0=_t.time()
IM=[.485,.456,.406]; IS=[.229,.224,.225]

class DS(Dataset):
    def __init__(s,df,root):
        s.df=df.reset_index(drop=True); s.root=root
        s.tf=transforms.Compose([transforms.Resize((384,384)),transforms.ToTensor(),transforms.Normalize(IM,IS)])
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
def feats(model,df,root):
    dl=DataLoader(DS(df,root),batch_size=BS,num_workers=4,pin_memory=True)
    e,ids=[],[]
    for imgs,iids in tqdm(dl,leave=False):
        e.append(F.normalize(model(imgs.to(DEV)),dim=-1).cpu().numpy()); ids.extend(iids.numpy())
    return np.concatenate(e), np.array(ids)

print("Loading MegaDescriptor...",flush=True)
mega=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(DEV).eval()
print(f"  dim={mega.num_features}")

DATA=sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
OUT=sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov9"
os.makedirs(OUT,exist_ok=True)
meta=pd.read_csv(os.path.join(DATA,"metadata.csv"))
ssub=pd.read_csv(os.path.join(DATA,"sample_submission.csv"))
trdf=meta[meta.split=="train"].copy(); tedf=meta[meta.split=="test"].copy()

print("Extracting features...")
tr_f, tr_ids = feats(mega, trdf, DATA)
te_f, te_ids = feats(mega, tedf, DATA)
print(f"  Train:{tr_f.shape} Test:{te_f.shape}")
del mega; torch.cuda.empty_cache(); gc.collect()
print(f"  Done in {(_t.time()-t0)/60:.1f}min")

tr_i2x={int(tr_ids[i]):i for i in range(len(tr_ids))}
te_i2x={int(te_ids[i]):i for i in range(len(te_ids))}

DS_TR=["LynxID2025","SalamanderID2025","SeaTurtleID2022"]
DS_ALL=DS_TR+["TexasHornedLizards"]
all_preds={}

for dsn in DS_ALL:
    print(f"\n{'='*50}\n{dsn}\n{'='*50}")
    ds_te=tedf[tedf.dataset==dsn].reset_index(drop=True)
    teix=[te_i2x[int(x)] for x in ds_te.image_id.values]
    te_feat=normalize(te_f[teix],axis=1)

    if dsn in DS_TR:
        ds_tr=trdf[trdf.dataset==dsn].reset_index(drop=True)
        trix=[tr_i2x[int(x)] for x in ds_tr.image_id.values]
        tr_feat=normalize(tr_f[trix],axis=1)
        train_ids=ds_tr.identity.values
        n_tr=len(ds_tr); n_te=len(ds_te)

        # ── Combine train+test features for joint clustering ──
        all_feat=np.vstack([tr_feat, te_feat])
        sim=all_feat @ all_feat.T
        dist=np.clip(1-sim, 0, 2)

        # ── Tune eps: cluster train portion, check ARI ──
        le=LabelEncoder(); yt=le.fit_transform(train_ids)
        best_eps, best_ari, best_method = 0.5, -1, "dbscan"

        for eps in np.arange(0.05, 1.20, 0.01):
            labels=DBSCAN(eps=eps,min_samples=2,metric="precomputed").fit_predict(dist)
            # Extract train labels only
            tr_labels=labels[:n_tr].copy()
            ns=tr_labels.max()+1 if tr_labels.max()>=0 else 0
            for i in range(n_tr):
                if tr_labels[i]==-1: tr_labels[i]=ns; ns+=1
            ari=adjusted_rand_score(yt,tr_labels)
            if ari>best_ari: best_ari,best_eps=ari,eps

        # Also try Agglomerative (average linkage)
        for dt in np.arange(0.05, 1.20, 0.01):
            try:
                labels=AgglomerativeClustering(n_clusters=None,distance_threshold=dt,
                    metric="precomputed",linkage="average").fit_predict(dist)
                tr_labels=labels[:n_tr]
                ari=adjusted_rand_score(yt,tr_labels)
                if ari>best_ari: best_ari,best_eps,best_method=ari,dt,"agglo"
            except: pass

        print(f"  Best: {best_method} param={best_eps:.2f} train_ARI={best_ari:.4f}")

        # ── Apply best method to get final labels ──
        if best_method=="dbscan":
            labels=DBSCAN(eps=best_eps,min_samples=2,metric="precomputed").fit_predict(dist)
        else:
            labels=AgglomerativeClustering(n_clusters=None,distance_threshold=best_eps,
                metric="precomputed",linkage="average").fit_predict(dist)

        # Relabel noise
        ns=labels.max()+1 if labels.max()>=0 else 0
        for i in range(len(labels)):
            if labels[i]==-1: labels[i]=ns; ns+=1

        # Extract test portion labels
        te_labels=labels[n_tr:]
        n_cl=len(set(te_labels))
        print(f"  Test clusters: {n_cl} for {n_te} images")

    else:
        # TexasHornedLizards: no train
        dist=np.clip(1-te_feat@te_feat.T, 0, 2)
        best_eps, best_score = 0.40, -1
        for eps in np.arange(0.15, 0.80, 0.01):
            pred=DBSCAN(eps=eps,min_samples=2,metric="precomputed").fit_predict(dist)
            n_noise=(pred==-1).sum()
            ns=pred.max()+1 if pred.max()>=0 else 0
            for i in range(len(pred)):
                if pred[i]==-1: pred[i]=ns; ns+=1
            n_cl=len(set(pred))
            if 15<=n_cl<=150:
                score=1-n_noise/len(pred)-abs(n_cl/len(pred)-0.3)
                if score>best_score: best_score,best_eps=score,eps
        te_labels=DBSCAN(eps=best_eps,min_samples=2,metric="precomputed").fit_predict(dist)
        ns=te_labels.max()+1 if te_labels.max()>=0 else 0
        for i in range(len(te_labels)):
            if te_labels[i]==-1: te_labels[i]=ns; ns+=1
        print(f"  eps={best_eps:.2f} clusters={len(set(te_labels))}")

    for i in range(len(ds_te)):
        all_preds[int(ds_te.iloc[i].image_id)]=f"cluster_{dsn}_{te_labels[i]}"

sub=ssub.copy()
for i in range(len(sub)):
    iid=int(sub.iloc[i].image_id)
    if iid in all_preds: sub.at[i,"cluster"]=all_preds[iid]
out=os.path.join(OUT,"submission.csv"); sub.to_csv(out,index=False)
el=(_t.time()-t0)/60
print(f"\n{'='*60}\nSaved: {out}\nRows:{len(sub)} Clusters:{sub.cluster.nunique()}")
for d in DS_ALL:
    s=sub[sub.cluster.str.contains(d)]
    print(f"  {d:25s} imgs={len(s):4d} cl={s.cluster.nunique()}")
print(f"Time: {el:.1f}min\nDONE!")
