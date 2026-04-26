#!/usr/bin/env python
"""
V14 — v9的聚类策略(0.177) + ArcFace微调特征 + MiewID融合
=========================================================
核心原则: 不搞花活，v9的逻辑原封不动，只换更好的特征
  - v9验证过的train+test锚定Agglo聚类 → 保留
  - 加ArcFace微调特征 → 提升区分度
  - 加MiewID相似度融合 → 多模型互补
  - 不用lookup(v7/v13都证明了lookup会炸)
"""
import os, sys, gc, warnings, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize, LabelEncoder

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IM = [.485,.456,.406]; IS = [.229,.224,.225]
DATA = sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
OUT  = sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov14"
os.makedirs(OUT, exist_ok=True)
DS_TR = ["LynxID2025","SalamanderID2025","SeaTurtleID2022"]
DS_ALL = DS_TR + ["TexasHornedLizards"]

# ═══════════════════════════════════════
# Datasets
# ═══════════════════════════════════════
def _orient(img, row):
    if row.get("species")=="salamander" and pd.notna(row.get("orientation")):
        o = str(row["orientation"]).lower()
        if o=="right": img=img.rotate(-90,expand=True)
        elif o=="left": img=img.rotate(90,expand=True)
    return img

class DS(Dataset):
    def __init__(s,df,root,sz):
        s.df=df.reset_index(drop=True); s.root=root
        s.tf=transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),transforms.Normalize(IM,IS)])
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        img=_orient(img,r); return s.tf(img), int(r["image_id"])

class TrDS(Dataset):
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

@torch.no_grad()
def extract(model, df, root, sz, bs=48):
    dl=DataLoader(DS(df,root,sz),batch_size=bs,num_workers=4,pin_memory=True)
    e,ids=[],[]
    for imgs,iids in tqdm(dl,leave=False):
        e.append(F.normalize(model(imgs.to(DEV)),dim=-1).cpu().numpy()); ids.extend(iids.numpy())
    return np.concatenate(e), np.array(ids)

# ═══════════════════════════════════════
# ArcFace (proven to work in v13)
# ═══════════════════════════════════════
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

def finetune(tdf, root, dsn):
    import timm
    le=LabelEncoder(); tdf=tdf.copy(); tdf["label"]=le.fit_transform(tdf.identity.values)
    ncls=tdf.label.nunique()
    if ncls<5: return None,None
    print(f"  [{dsn}] ArcFace {ncls}cls {len(tdf)}imgs")
    bb=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(DEV)
    hd=ArcHead(bb.num_features,ncls).to(DEV)
    ds=TrDS(tdf,root,384)
    ccnt=tdf.label.value_counts().to_dict()
    sw=[1./ccnt[int(r.label)] for _,r in tdf.iterrows()]
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler=torch.amp.GradScaler()
    # S1: head only
    loader=DataLoader(ds,batch_size=48,sampler=WeightedRandomSampler(sw,len(sw)),num_workers=4,pin_memory=True,drop_last=True)
    for p in bb.parameters(): p.requires_grad=False
    opt=torch.optim.AdamW(hd.parameters(),lr=3e-4)
    for ep in range(2):
        bb.eval();hd.train();ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels=imgs.to(DEV),labels.to(DEV)
            with torch.no_grad(): emb=F.normalize(bb(imgs),dim=-1)
            with torch.amp.autocast('cuda'): logits=hd(emb,labels);loss=crit(logits,labels)
            opt.zero_grad();scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            ls.append(loss.item());co+=(logits.argmax(1)==labels).sum().item();to+=labels.size(0)
        print(f"    S1-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")
    # S2: top25%
    nps=list(bb.named_parameters()); top25={n for n,_ in nps[int(len(nps)*.75):]}
    for n,p in bb.named_parameters(): p.requires_grad=(n in top25)
    opt2=torch.optim.AdamW([{"params":[p for n,p in bb.named_parameters() if p.requires_grad],"lr":3e-5},
                             {"params":hd.parameters(),"lr":2.4e-4}],weight_decay=1e-4)
    for ep in range(5):
        bb.train();hd.train();ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels=imgs.to(DEV),labels.to(DEV)
            with torch.amp.autocast('cuda'): emb=F.normalize(bb(imgs),dim=-1);logits=hd(emb,labels);loss=crit(logits,labels)
            opt2.zero_grad();scaler.scale(loss).backward();scaler.unscale_(opt2)
            torch.nn.utils.clip_grad_norm_(bb.parameters(),1.);scaler.step(opt2);scaler.update()
            ls.append(loss.item());co+=(logits.argmax(1)==labels).sum().item();to+=labels.size(0)
        print(f"    S2-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")
    # S3: full unfreeze BS=16
    del loader; torch.cuda.empty_cache(); gc.collect()
    loader3=DataLoader(ds,batch_size=16,sampler=WeightedRandomSampler(sw,len(sw)),num_workers=4,pin_memory=True,drop_last=True)
    for p in bb.parameters(): p.requires_grad=True
    opt3=torch.optim.AdamW([{"params":bb.parameters(),"lr":9e-6},{"params":hd.parameters(),"lr":9e-5}],weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt3,T_max=11)
    for ep in range(11):
        bb.train();hd.train();ls,co,to=[],0,0
        for imgs,labels in loader3:
            imgs,labels=imgs.to(DEV),labels.to(DEV)
            with torch.amp.autocast('cuda'): emb=F.normalize(bb(imgs),dim=-1);logits=hd(emb,labels);loss=crit(logits,labels)
            opt3.zero_grad();scaler.scale(loss).backward();scaler.unscale_(opt3)
            torch.nn.utils.clip_grad_norm_(bb.parameters(),1.);scaler.step(opt3);scaler.update()
            ls.append(loss.item());co+=(logits.argmax(1)==labels).sum().item();to+=labels.size(0)
        sch.step()
        print(f"    S3-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")
    bb.eval(); del hd,opt,opt2,opt3,scaler,sch,loader3,ds; torch.cuda.empty_cache();gc.collect()
    return bb,le

# ═══════════════════════════════════════
# v9原版聚类逻辑(验证过0.177)
# ═══════════════════════════════════════
def v9_cluster(sim, n_tr, yt, dsn):
    """v9的核心: train+test合并聚类, train部分调参, 提取test部分标签"""
    dist = np.clip(1 - sim, 0, 2)
    n = len(sim)

    # DBSCAN sweep
    best_eps, best_ari_db = 0.5, -1
    for eps in np.arange(0.05, 1.20, 0.01):
        labels = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
        tr_labels = labels[:n_tr].copy()
        ns = tr_labels.max()+1 if tr_labels.max()>=0 else 0
        for i in range(n_tr):
            if tr_labels[i]==-1: tr_labels[i]=ns; ns+=1
        ari = adjusted_rand_score(yt, tr_labels)
        if ari > best_ari_db: best_ari_db, best_eps = ari, eps

    # Agglo sweep
    best_dt, best_ari_ag = 0.5, -1
    for dt in np.arange(0.05, 1.20, 0.01):
        try:
            labels = AgglomerativeClustering(n_clusters=None, distance_threshold=dt,
                        metric="precomputed", linkage="average").fit_predict(dist)
            ari = adjusted_rand_score(yt, labels[:n_tr])
            if ari > best_ari_ag: best_ari_ag, best_dt = ari, dt
        except: pass

    # Pick best
    if best_ari_ag >= best_ari_db:
        print(f"  [{dsn}] Agglo dt={best_dt:.2f} train_ARI={best_ari_ag:.4f}")
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=best_dt,
                    metric="precomputed", linkage="average").fit_predict(dist)
    else:
        print(f"  [{dsn}] DBSCAN eps={best_eps:.2f} train_ARI={best_ari_db:.4f}")
        labels = DBSCAN(eps=best_eps, min_samples=2, metric="precomputed").fit_predict(dist)
        ns = labels.max()+1 if labels.max()>=0 else 0
        for i in range(len(labels)):
            if labels[i]==-1: labels[i]=ns; ns+=1

    te_labels = labels[n_tr:]
    print(f"  [{dsn}] clusters={len(set(te_labels))} for {n-n_tr} test imgs")
    return te_labels

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    t0=time.time()
    print("="*55)
    print("  V14: v9聚类 + ArcFace特征 + MiewID融合")
    print("="*55)
    if torch.cuda.is_available():
        g=torch.cuda.get_device_properties(0)
        print(f"GPU: {g.name} VRAM: {g.total_memory/1e9:.1f}GB")

    meta=pd.read_csv(os.path.join(DATA,"metadata.csv"))
    ssub=pd.read_csv(os.path.join(DATA,"sample_submission.csv"))
    trdf=meta[meta.split=="train"].copy(); tedf=meta[meta.split=="test"].copy()
    for d in DS_ALL:
        tr=trdf[trdf.dataset==d];te=tedf[tedf.dataset==d]
        print(f"  {d:25s} tr={len(tr):5d} te={len(te):4d}")

    # ── STAGE 1: 全局特征 ──
    print(f"\n{'━'*50}\nSTAGE 1: Global Features\n{'━'*50}")
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    mega=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(DEV).eval()
    print(f"[MEGA] dim={mega.num_features}")
    mega_tr,tr_ids=extract(mega,trdf,DATA,384); mega_te,te_ids=extract(mega,tedf,DATA,384)
    del mega; torch.cuda.empty_cache(); gc.collect()

    cfg_path=hf_hub_download("conservationxlabs/miewid-msv3","config.json")
    with open(cfg_path) as f: cfg=json.load(f)
    miew=timm.create_model(cfg.get("architecture","efficientnetv2_rw_m"),pretrained=False,num_classes=0)
    wt=hf_hub_download("conservationxlabs/miewid-msv3","model.safetensors")
    state={k:v for k,v in load_file(wt).items() if "classifier" not in k}
    miew.load_state_dict(state,strict=False); miew=miew.eval().to(DEV)
    print(f"[MIEW] dim={miew.num_features}")
    miew_tr,_=extract(miew,trdf,DATA,440,bs=32); miew_te,_=extract(miew,tedf,DATA,440,bs=32)
    del miew; torch.cuda.empty_cache(); gc.collect()
    print(f"  Features done: {(time.time()-t0)/60:.1f}min")

    tr_i2x={int(tr_ids[i]):i for i in range(len(tr_ids))}
    te_i2x={int(te_ids[i]):i for i in range(len(te_ids))}

    # ── STAGE 2: ArcFace ──
    print(f"\n{'━'*50}\nSTAGE 2: ArcFace Fine-Tuning\n{'━'*50}")
    ft_tr_feats, ft_te_feats = {}, {}
    for dsn in DS_TR:
        dtr=trdf[trdf.dataset==dsn].copy(); dte=tedf[tedf.dataset==dsn].copy()
        if len(dtr)<50: continue
        try:
            ftm,le=finetune(dtr,DATA,dsn)
            if ftm is None: continue
            ftr,_=extract(ftm,dtr,DATA,384,bs=32); fte,_=extract(ftm,dte,DATA,384,bs=32)
            ft_tr_feats[dsn]=ftr; ft_te_feats[dsn]=fte
            print(f"  [{dsn}] FT done: tr={ftr.shape} te={fte.shape}")
            del ftm; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"  [{dsn}] FAILED: {e}"); torch.cuda.empty_cache(); gc.collect()

    # ── STAGE 3: v9聚类 ──
    print(f"\n{'━'*50}\nSTAGE 3: v9-style Clustering\n{'━'*50}")
    all_preds={}

    for dsn in DS_ALL:
        print(f"\n{'═'*45}\n  {dsn}\n{'═'*45}")
        ds_te=tedf[tedf.dataset==dsn].reset_index(drop=True)
        teix=[te_i2x[int(x)] for x in ds_te.image_id.values]
        n_te=len(ds_te)

        if dsn in DS_TR:
            ds_tr=trdf[trdf.dataset==dsn].reset_index(drop=True)
            trix=[tr_i2x[int(x)] for x in ds_tr.image_id.values]
            train_ids=ds_tr.identity.values
            le=LabelEncoder(); yt=le.fit_transform(train_ids)
            n_tr_ds=len(ds_tr)

            # 合并 train+test 特征, 算相似度 (v9原版逻辑)
            # Mega similarity
            all_mega = np.vstack([normalize(mega_tr[trix],axis=1), normalize(mega_te[teix],axis=1)])
            sim_mega = all_mega @ all_mega.T

            # MiewID similarity
            all_miew = np.vstack([normalize(miew_tr[trix],axis=1), normalize(miew_te[teix],axis=1)])
            sim_miew = all_miew @ all_miew.T

            # 相似度融合 (不拼接特征!)
            sim = 0.55 * sim_mega + 0.45 * sim_miew

            # 如果有ArcFace微调特征, 再融合
            if dsn in ft_tr_feats:
                all_ft = np.vstack([normalize(ft_tr_feats[dsn],axis=1), normalize(ft_te_feats[dsn],axis=1)])
                sim_ft = all_ft @ all_ft.T
                # 三路融合: 全局55% + ArcFace45%
                sim = 0.55 * sim + 0.45 * sim_ft
                print(f"  Fused: global(55%) + ArcFace(45%)")
            else:
                print(f"  Global only: Mega(55%) + MiewID(45%)")

            # v9聚类
            te_labels = v9_cluster(sim, n_tr_ds, yt, dsn)

        else:
            # TexasHornedLizards: 纯聚类
            te_mega = normalize(mega_te[teix],axis=1)
            te_miew = normalize(miew_te[teix],axis=1)
            sim = 0.55*(te_mega@te_mega.T) + 0.45*(te_miew@te_miew.T)
            dist = np.clip(1-sim, 0, 2)

            best_eps, best_score = 0.40, -1
            for eps in np.arange(0.15, 0.80, 0.01):
                pred = DBSCAN(eps=eps,min_samples=2,metric="precomputed").fit_predict(dist)
                n_noise=(pred==-1).sum()
                ns=pred.max()+1 if pred.max()>=0 else 0
                for i in range(len(pred)):
                    if pred[i]==-1: pred[i]=ns;ns+=1
                n_cl=len(set(pred))
                if 15<=n_cl<=150:
                    score=1-n_noise/len(pred)-abs(n_cl/len(pred)-0.3)
                    if score>best_score: best_score,best_eps=score,eps
            te_labels=DBSCAN(eps=best_eps,min_samples=2,metric="precomputed").fit_predict(dist)
            ns=te_labels.max()+1 if te_labels.max()>=0 else 0
            for i in range(len(te_labels)):
                if te_labels[i]==-1: te_labels[i]=ns;ns+=1
            print(f"  eps={best_eps:.2f} clusters={len(set(te_labels))}")

        for i in range(n_te):
            all_preds[int(ds_te.iloc[i].image_id)]=f"cluster_{dsn}_{te_labels[i]}"

    # ── Submit ──
    sub=ssub.copy()
    for i in range(len(sub)):
        iid=int(sub.iloc[i].image_id)
        if iid in all_preds: sub.at[i,"cluster"]=all_preds[iid]
    out=os.path.join(OUT,"submission.csv"); sub.to_csv(out,index=False)
    print(f"\n{'━'*50}\nSaved: {out}")
    print(f"Rows:{len(sub)} Clusters:{sub.cluster.nunique()}")
    for d in DS_ALL:
        s=sub[sub.cluster.str.contains(d)]
        print(f"  {d:25s} imgs={len(s):4d} cl={s.cluster.nunique()}")
    print(f"Time: {(time.time()-t0)/60:.1f}min\nDONE!")

if __name__=="__main__":
    main()
