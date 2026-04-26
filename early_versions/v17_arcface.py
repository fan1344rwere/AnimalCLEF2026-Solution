#!/usr/bin/env python3
"""
V17 — ArcFace Fine-tuning + Calibrated Fusion
Target: 0.53+ (Top5)
V16=0.231的瓶颈: Lynx/Salamander特征区分度不够(train ARI~0.15)
V17核心: 对每个有训练集的物种做ArcFace微调，让模型学会区分个体
"""
import os,sys,gc,time,json,warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT","https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"]="false"

import numpy as np,pandas as pd
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize,LabelEncoder

D=torch.device("cuda" if torch.cuda.is_available() else "cpu")
M,S=[.485,.456,.406],[.229,.224,.225]
DATA=sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
OUT=sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov17"
os.makedirs(OUT,exist_ok=True)
SP_TR=["LynxID2025","SalamanderID2025","SeaTurtleID2022"]
SP_ALL=SP_TR+["TexasHornedLizards"]

# ═══════════════════════════════════════════
class DS(Dataset):
    def __init__(s,df,root,sz,flip=False):
        s.df,s.root,s.flip=df.reset_index(drop=True),root,flip
        s.tf=transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),transforms.Normalize(M,S)])
    def __len__(s):return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try:img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except:img=Image.new("RGB",(384,384),(128,128,128))
        if s.flip:img=img.transpose(Image.FLIP_LEFT_RIGHT)
        return s.tf(img),int(r["image_id"])

class TrainDS(Dataset):
    def __init__(s,df,root,sz):
        s.df,s.root=df.reset_index(drop=True),root
        s.tf=transforms.Compose([
            transforms.RandomResizedCrop(sz,scale=(.65,1.),ratio=(.8,1.2)),
            transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(20),
            transforms.ColorJitter(.4,.4,.2,.1),transforms.RandomGrayscale(.05),
            transforms.ToTensor(),transforms.Normalize(M,S),
            transforms.RandomErasing(p=.25,scale=(.02,.2))])
    def __len__(s):return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try:img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except:img=Image.new("RGB",(384,384),(128,128,128))
        return s.tf(img),int(r["label"])

@torch.no_grad()
def extract(model,df,root,sz,bs=48):
    dl=DataLoader(DS(df,root,sz),batch_size=bs,num_workers=4,pin_memory=True)
    e,ids=[],[]
    for imgs,iids in tqdm(dl,desc=f"  @{sz}",leave=False):
        e.append(model(imgs.to(D)).cpu());ids.extend(iids.numpy())
    o=torch.cat(e)
    dl2=DataLoader(DS(df,root,sz,flip=True),batch_size=bs,num_workers=4,pin_memory=True)
    f=[]
    for imgs,_ in tqdm(dl2,desc=f"  tta@{sz}",leave=False):
        f.append(model(imgs.to(D)).cpu())
    return F.normalize((o+torch.cat(f))/2,dim=-1).numpy(),np.array(ids)

def load_mega():
    import timm
    m=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(D).eval()
    print(f"[M] Mega OK dim={m.num_features}");return m,384

def load_miew():
    import timm;from huggingface_hub import hf_hub_download;from safetensors.torch import load_file
    c=json.load(open(hf_hub_download("conservationxlabs/miewid-msv3","config.json")))
    m=timm.create_model(c.get("architecture","efficientnetv2_rw_m"),pretrained=False,num_classes=0)
    s={k:v for k,v in load_file(hf_hub_download("conservationxlabs/miewid-msv3","model.safetensors")).items() if "classifier" not in k}
    m.load_state_dict(s,strict=False);m=m.to(D).eval()
    print(f"[M] Miew OK dim={m.num_features}");return m,440

# ═══════════════════════════════════════════
# ARCFACE
# ═══════════════════════════════════════════
class ArcHead(nn.Module):
    def __init__(s,d,n,sc=30.,m=.5):
        super().__init__();s.s=sc;s.m=m
        s.W=nn.Parameter(torch.empty(n,d));nn.init.xavier_uniform_(s.W)
        s.cm=np.cos(m);s.sm=np.sin(m);s.th=np.cos(np.pi-m);s.mm=np.sin(np.pi-m)*m
    def forward(s,x,y=None):
        cos=F.linear(F.normalize(x),F.normalize(s.W))
        if y is None:return cos*s.s
        sin=(1-cos.pow(2).clamp(0,1)).sqrt()
        phi=cos*s.cm-sin*s.sm;phi=torch.where(cos>s.th,phi,cos-s.mm)
        oh=torch.zeros_like(cos).scatter_(1,y.unsqueeze(1),1.)
        return(oh*phi+(1-oh)*cos)*s.s

def finetune(sp_train,root,dsname):
    """ArcFace fine-tune MegaDescriptor for a specific species. Returns fine-tuned model."""
    import timm
    df=sp_train.copy()
    le=LabelEncoder();df["label"]=le.fit_transform(df.identity.values)
    nc=df.label.nunique()
    if nc<5:print(f"  [{dsname}] <5 classes, skip");return None

    print(f"  [{dsname}] ArcFace: {nc} classes, {len(df)} imgs")
    bb=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(D)
    edim=bb.num_features
    hd=ArcHead(edim,nc).to(D)
    ds=TrainDS(df,root,384)

    # Weighted sampler for class balance
    cc=df.label.value_counts().to_dict()
    sw=[1./cc[int(r.label)] for _,r in df.iterrows()]
    scaler=torch.amp.GradScaler()

    # Stage 1: head only (3 ep)
    for p in bb.parameters():p.requires_grad=False
    opt=torch.optim.AdamW(hd.parameters(),lr=3e-4,weight_decay=1e-4)
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)
    loader=DataLoader(ds,batch_size=48,sampler=WeightedRandomSampler(sw,len(sw)),num_workers=4,pin_memory=True,drop_last=True)

    for ep in range(3):
        bb.eval();hd.train();ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels=imgs.to(D),labels.to(D)
            with torch.no_grad():emb=F.normalize(bb(imgs),dim=-1)
            with torch.amp.autocast(device_type='cuda'):
                logits=hd(emb,labels);loss=crit(logits,labels)
            opt.zero_grad();scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            ls.append(loss.item());co+=(logits.argmax(1)==labels).sum().item();to+=labels.size(0)
        print(f"    H-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")

    # Stage 2: unfreeze top 30% backbone (8 ep)
    nps=list(bb.named_parameters())
    top30={n for n,_ in nps[int(len(nps)*.7):]}
    for n,p in bb.named_parameters():p.requires_grad=(n in top30)
    opt2=torch.optim.AdamW([
        {"params":[p for n,p in bb.named_parameters() if p.requires_grad],"lr":5e-5},
        {"params":hd.parameters(),"lr":2e-4}],weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=8)

    for ep in range(8):
        bb.train();hd.train();ls,co,to=[],0,0
        for imgs,labels in loader:
            imgs,labels=imgs.to(D),labels.to(D)
            with torch.amp.autocast(device_type='cuda'):
                emb=F.normalize(bb(imgs),dim=-1);logits=hd(emb,labels);loss=crit(logits,labels)
            opt2.zero_grad();scaler.scale(loss).backward()
            scaler.unscale_(opt2);torch.nn.utils.clip_grad_norm_(bb.parameters(),1.)
            scaler.step(opt2);scaler.update()
            ls.append(loss.item());co+=(logits.argmax(1)==labels).sum().item();to+=labels.size(0)
        sch.step()
        print(f"    B-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")

    # Stage 3: full unfreeze (5 ep, smaller BS)
    del loader;torch.cuda.empty_cache()
    for p in bb.parameters():p.requires_grad=True
    loader3=DataLoader(ds,batch_size=24,sampler=WeightedRandomSampler(sw,len(sw)),num_workers=4,pin_memory=True,drop_last=True)
    opt3=torch.optim.AdamW([
        {"params":bb.parameters(),"lr":1e-5},
        {"params":hd.parameters(),"lr":5e-5}],weight_decay=1e-4)
    sch3=torch.optim.lr_scheduler.CosineAnnealingLR(opt3,T_max=5)

    for ep in range(5):
        bb.train();hd.train();ls,co,to=[],0,0
        for imgs,labels in loader3:
            imgs,labels=imgs.to(D),labels.to(D)
            with torch.amp.autocast(device_type='cuda'):
                emb=F.normalize(bb(imgs),dim=-1);logits=hd(emb,labels);loss=crit(logits,labels)
            opt3.zero_grad();scaler.scale(loss).backward()
            scaler.unscale_(opt3);torch.nn.utils.clip_grad_norm_(bb.parameters(),1.)
            scaler.step(opt3);scaler.update()
            ls.append(loss.item());co+=(logits.argmax(1)==labels).sum().item();to+=labels.size(0)
        sch3.step()
        print(f"    F-e{ep+1}: loss={np.mean(ls):.3f} acc={100*co/to:.1f}%")

    bb.eval()
    del hd,opt,opt2,opt3,scaler,sch,sch3,loader3,ds;torch.cuda.empty_cache();gc.collect()
    return bb

# ═══════════════════════════════════════════
def best_threshold(sim,labels):
    dist=np.clip(1-sim,0,2);np.fill_diagonal(dist,0)
    ba,bt,bn=-1,0.5,-1
    for th in np.arange(0.05,1.50,0.01):
        try:
            p=AgglomerativeClustering(n_clusters=None,distance_threshold=th,metric="precomputed",linkage="average").fit_predict(dist)
            a=adjusted_rand_score(labels,p)
            if a>ba:ba,bt,bn=a,th,len(set(p))
        except:pass
    return bt,ba,bn

def cluster(sim,th):
    dist=np.clip(1-sim,0,2);np.fill_diagonal(dist,0)
    return AgglomerativeClustering(n_clusters=None,distance_threshold=th,metric="precomputed",linkage="average").fit_predict(dist)

def search_blend(sim_pre,sim_ft,labels):
    """Search optimal blend weight for pretrained+finetuned features."""
    ba,bw,bth=-1,0.5,0.5
    for w in np.arange(0,1.01,0.05):
        sim=w*sim_ft+(1-w)*sim_pre
        th,ari,_=best_threshold(sim,labels) if False else (0,0,0)
        # Fast search
        dist=np.clip(1-sim,0,2);np.fill_diagonal(dist,0)
        for th in np.arange(0.05,1.50,0.02):
            try:
                p=AgglomerativeClustering(n_clusters=None,distance_threshold=th,metric="precomputed",linkage="average").fit_predict(dist)
                a=adjusted_rand_score(labels,p)
                if a>ba:ba,bw,bth=a,w,th
            except:pass
    return bw,bth,ba

# ═══════════════════════════════════════════
def main():
    t0=time.time()
    print("="*60)
    print("  V17 — ArcFace Fine-tuning + Fusion")
    print("="*60)
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | {p.total_memory/1e9:.0f}GB")

    meta=pd.read_csv(os.path.join(DATA,"metadata.csv"))
    ssub=pd.read_csv(os.path.join(DATA,"sample_submission.csv"))
    trdf=meta[meta.split=="train"];tedf=meta[meta.split=="test"]
    alldf=pd.concat([trdf,tedf],ignore_index=True)

    print(f"  {len(trdf)} train, {len(tedf)} test")

    # ── STAGE 1: Pretrained features ──
    print(f"\n{'━'*55}\nSTAGE 1: Pretrained Features\n{'━'*55}")
    fd={}
    m,sz=load_mega()
    fd["mega"],_=extract(m,alldf,DATA,sz,48)
    del m;torch.cuda.empty_cache();gc.collect()
    m,sz=load_miew()
    fd["miew"],_=extract(m,alldf,DATA,sz,32)
    del m;torch.cuda.empty_cache();gc.collect()

    all_ids=alldf.image_id.values
    id2idx={int(all_ids[i]):i for i in range(len(all_ids))}
    ft1=(time.time()-t0)/60
    print(f"  Done: {ft1:.1f}min")

    # ── STAGE 2: ArcFace per species ──
    print(f"\n{'━'*55}\nSTAGE 2: ArcFace Fine-tuning\n{'━'*55}")
    ft_feats={}  # {species: features_array}

    for sp in SP_TR:
        print(f"\n  {'─'*40} {sp}")
        sp_tr=trdf[trdf.dataset==sp].reset_index(drop=True)
        sp_te=tedf[tedf.dataset==sp].reset_index(drop=True)
        sp_all=pd.concat([sp_tr,sp_te],ignore_index=True)

        ftm=finetune(sp_tr,DATA,sp)
        if ftm is not None:
            ft_f,_=extract(ftm,sp_all,DATA,384,32)
            ft_feats[sp]=ft_f
            print(f"  [{sp}] FT features: {ft_f.shape}")
            del ftm;torch.cuda.empty_cache();gc.collect()

    ft2=(time.time()-t0)/60
    print(f"  ArcFace done: {ft2-ft1:.1f}min")

    # ── STAGE 3: Per-species clustering ──
    print(f"\n{'━'*55}\nSTAGE 3: Clustering\n{'━'*55}")
    preds={}

    for sp in SP_ALL:
        print(f"\n  {'═'*40} {sp}")
        te=tedf[tedf.dataset==sp].reset_index(drop=True)
        te_gidx=np.array([id2idx[int(x)] for x in te.image_id])
        nte=len(te)

        if sp in SP_TR:
            tr=trdf[trdf.dataset==sp].reset_index(drop=True)
            tr_gidx=np.array([id2idx[int(x)] for x in tr.image_id])
            tr_labels=tr.identity.values
            ntr=len(tr)

            # Pretrained similarity (best of mega/miew from V16 insight)
            # V16 found: Lynx→mega=0.9, Sal→dino=0.8(no dino here, try miew), Turtle→mega=0.5+miew=0.2
            pre_w={"LynxID2025":{"mega":0.85,"miew":0.15},
                   "SalamanderID2025":{"mega":0.6,"miew":0.4},
                   "SeaTurtleID2022":{"mega":0.55,"miew":0.45}}

            w=pre_w.get(sp,{"mega":0.5,"miew":0.5})

            # Sample train for threshold search
            if ntr>1200:
                rng=np.random.RandomState(42)
                si=rng.choice(ntr,1200,replace=False)
                s_gidx,s_lab=tr_gidx[si],tr_labels[si]
            else:
                s_gidx,s_lab=tr_gidx,tr_labels

            # Pretrained sim on train sample
            pre_sim=None
            for name,wt in w.items():
                f=normalize(fd[name][s_gidx],axis=1);s=f@f.T
                pre_sim=s*wt if pre_sim is None else pre_sim+s*wt

            # If fine-tuned features available, blend
            if sp in ft_feats:
                sp_tr_local=pd.concat([trdf[trdf.dataset==sp],tedf[tedf.dataset==sp]],ignore_index=True)
                ntr_local=len(trdf[trdf.dataset==sp])

                # Map sample indices to local ft_feats indices
                # ft_feats[sp] has shape [ntr_local + nte_local, dim]
                # s_gidx are global indices, need local train indices
                tr_local_map={int(trdf[trdf.dataset==sp].iloc[i].image_id):i for i in range(ntr_local)}

                if ntr>1200:
                    s_local_idx=np.array([tr_local_map[int(trdf[trdf.dataset==sp].iloc[si[j]].image_id)] for j in range(len(si))])
                else:
                    s_local_idx=np.arange(ntr_local)

                ft_f=normalize(ft_feats[sp][s_local_idx],axis=1)
                ft_sim=ft_f@ft_f.T

                # Search optimal blend
                blend_w,blend_th,blend_ari=search_blend(pre_sim,ft_sim,s_lab)
                print(f"    Blend: ft_w={blend_w:.2f} th={blend_th:.3f} ARI={blend_ari:.4f}")

                # Also check pretrained-only
                _,pre_ari,_=best_threshold(pre_sim,s_lab)
                print(f"    Pretrained-only ARI={pre_ari:.4f}")

                if blend_ari>pre_ari:
                    # Apply to test
                    te_pre=None
                    for name,wt in w.items():
                        f=normalize(fd[name][te_gidx],axis=1);s=f@f.T
                        te_pre=s*wt if te_pre is None else te_pre+s*wt

                    te_ft_idx=np.arange(ntr_local,ntr_local+nte)
                    ft_te=normalize(ft_feats[sp][te_ft_idx],axis=1)
                    te_ft_sim=ft_te@ft_te.T

                    te_sim=blend_w*te_ft_sim+(1-blend_w)*te_pre
                    final_th=blend_th
                    print(f"    Using blended (ft_w={blend_w:.2f})")
                else:
                    te_sim=None
                    for name,wt in w.items():
                        f=normalize(fd[name][te_gidx],axis=1);s=f@f.T
                        te_sim=s*wt if te_sim is None else te_sim+s*wt
                    final_th=blend_th  # use best threshold anyway
                    print(f"    FT didn't help, using pretrained only")
            else:
                te_sim=None
                for name,wt in w.items():
                    f=normalize(fd[name][te_gidx],axis=1);s=f@f.T
                    te_sim=s*wt if te_sim is None else te_sim+s*wt
                _,final_ari,_=best_threshold(pre_sim,s_lab)
                final_th=_  # oops wrong var
                th,ari,ncl=best_threshold(te_sim if te_sim is not None else pre_sim, s_lab)
                # Actually search on train
                th_train,ari_train,ncl_train=best_threshold(pre_sim,s_lab)
                final_th=th_train
                print(f"    Pretrained: th={final_th:.3f} ARI={ari_train:.4f}")

            labels=cluster(te_sim,final_th)
            print(f"    → {len(set(labels))} clusters for {nte} images")

        else:
            # TexasHornedLizards
            print(f"    No training data, {nte} images")
            te_sim=None
            w={"mega":0.5,"miew":0.5}
            for name,wt in w.items():
                f=normalize(fd[name][te_gidx],axis=1);s=f@f.T
                te_sim=s*wt if te_sim is None else te_sim+s*wt

            # Heuristic threshold search
            best_th,best_sc=0.40,-1
            dist=np.clip(1-te_sim,0,2);np.fill_diagonal(dist,0)
            for th in np.arange(0.10,1.50,0.01):
                try:
                    p=AgglomerativeClustering(n_clusters=None,distance_threshold=th,metric="precomputed",linkage="average").fit_predict(dist)
                    ratio=len(set(p))/nte
                    if .15<=ratio<=.55:
                        sc=-abs(ratio-.35)
                        if sc>best_sc:best_sc,best_th=sc,th
                except:pass
            labels=cluster(te_sim,best_th)
            print(f"    → {len(set(labels))} clusters (th={best_th:.3f})")

        for i in range(nte):
            preds[int(te.iloc[i].image_id)]=f"cluster_{sp}_{labels[i]}"

    # ── STAGE 4 ──
    print(f"\n{'━'*55}\nSTAGE 4: Submission\n{'━'*55}")
    sub=ssub.copy()
    for i in range(len(sub)):
        iid=int(sub.iloc[i].image_id)
        if iid in preds:sub.at[i,"cluster"]=preds[iid]
    out=os.path.join(OUT,"submission.csv")
    sub.to_csv(out,index=False)
    print(f"  {out}")
    print(f"  Rows={len(sub)} Clusters={sub.cluster.nunique()}")
    for sp in SP_ALL:
        s=sub[sub.cluster.str.contains(sp)]
        print(f"    {sp:25s} {len(s):4d} imgs, {s.cluster.nunique()} cl")
    print(f"  Time: {(time.time()-t0)/60:.1f}min")
    print("DONE!")

if __name__=="__main__":
    main()
