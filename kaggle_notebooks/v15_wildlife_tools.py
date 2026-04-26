#!/usr/bin/env python
"""
V15 — 站在巨人肩膀上: wildlife-tools官方库 + v9聚类
====================================================
核心改变: 不再手写相似度计算，全交给wildlife-tools
  1. wildlife-tools DeepFeatures 提取 MegaDescriptor + MiewID
  2. wildlife-tools ALIKED+LightGlue 局部匹配 (gluefactory)
  3. wildlife-tools 校准融合 (WildFusion风格)
  4. ArcFace微调 (v13验证过: Lynx 32.5%, SeaTurtle 57.5%)
  5. v9聚类策略 (验证过: 0.177, 不用lookup)

安装:
  pip install git+https://github.com/WildlifeDatasets/wildlife-tools
  pip install timm tqdm scikit-learn pandas numpy Pillow safetensors huggingface_hub hdbscan
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
IM=[.485,.456,.406]; IS=[.229,.224,.225]
DATA = sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
OUT  = sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov15"
os.makedirs(OUT, exist_ok=True)

DS_TR=["LynxID2025","SalamanderID2025","SeaTurtleID2022"]
DS_ALL=DS_TR+["TexasHornedLizards"]

# ═══════════════════════════════════════
# 1. wildlife-tools 特征提取
# ═══════════════════════════════════════
def extract_with_wt(model_name, df, root, sz, bs=32):
    """用wildlife-tools的标准方式提取特征"""
    import timm
    from wildlife_tools.features import DeepFeatures
    from wildlife_tools.data import ImageDataset

    tf = transforms.Compose([
        transforms.Resize([sz, sz]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IM, std=IS)])

    # 创建ImageDataset (wildlife-tools标准格式)
    dataset = ImageDataset(df.reset_index(drop=True), root, transform=tf)

    # 用DeepFeatures提取
    model = timm.create_model(model_name, num_classes=0, pretrained=True)
    extractor = DeepFeatures(model, device=DEV, batch_size=bs)
    features = extractor(dataset)

    # 清理GPU
    del model, extractor
    torch.cuda.empty_cache(); gc.collect()

    return features  # numpy array [N, D]

def extract_miew(df, root, bs=32):
    """MiewID需要特殊加载"""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from wildlife_tools.features import DeepFeatures
    from wildlife_tools.data import ImageDataset

    cfg_path = hf_hub_download("conservationxlabs/miewid-msv3", "config.json")
    with open(cfg_path) as f: cfg = json.load(f)
    arch = cfg.get("architecture", "efficientnetv2_rw_m")
    model = timm.create_model(arch, pretrained=False, num_classes=0)
    wt_path = hf_hub_download("conservationxlabs/miewid-msv3", "model.safetensors")
    state = {k:v for k,v in load_file(wt_path).items() if "classifier" not in k}
    model.load_state_dict(state, strict=False)

    tf = transforms.Compose([
        transforms.Resize([440, 440]),
        transforms.ToTensor(),
        transforms.Normalize(mean=IM, std=IS)])
    dataset = ImageDataset(df.reset_index(drop=True), root, transform=tf)

    extractor = DeepFeatures(model, device=DEV, batch_size=bs)
    features = extractor(dataset)

    del model, extractor
    torch.cuda.empty_cache(); gc.collect()
    return features

# ═══════════════════════════════════════
# 2. ALIKED局部匹配 (wildlife-tools集成)
# ═══════════════════════════════════════
def compute_aliked_similarity(df, root, global_sim, topk=20):
    """用wildlife-tools/gluefactory的ALIKED+LightGlue计算局部匹配相似度"""
    try:
        from wildlife_tools.similarity import MatchSimilarity
        from wildlife_tools.data import ImageDataset

        tf = transforms.Compose([transforms.Resize([512, 512]), transforms.ToTensor()])
        dataset = ImageDataset(df.reset_index(drop=True), root, transform=tf)

        # MatchSimilarity uses ALIKED+LightGlue via gluefactory
        matcher = MatchSimilarity(
            matcher_name="aliked-lightglue",
            device=DEV,
            batch_size=1
        )

        n = len(df)
        local_sim = np.zeros((n, n), dtype=np.float32)

        print(f"  [ALIKED] {n} images × top-{topk}...", flush=True)
        for i in tqdm(range(n), desc="  ALIKED", leave=False):
            # 只对top-K全局候选做局部匹配
            topk_idx = np.argsort(-global_sim[i])[:topk+1]
            for j in topk_idx:
                if j <= i: continue  # 对称，只算一半
                if local_sim[i,j] > 0: continue
                try:
                    img_i = dataset[i][0].unsqueeze(0)
                    img_j = dataset[j][0].unsqueeze(0)
                    score = matcher.match_pair(img_i, img_j)
                    local_sim[i,j] = score
                    local_sim[j,i] = score
                except:
                    pass

        print(f"  [ALIKED] Done. Non-zero: {(local_sim>0).sum()}")
        return local_sim
    except Exception as e:
        print(f"  [ALIKED] Failed: {e}")
        print(f"  [ALIKED] Falling back to global-only similarity")
        return None

# ═══════════════════════════════════════
# 3. WildFusion校准融合
# ═══════════════════════════════════════
def calibrated_fusion(global_sim, local_sim, train_mask, yt, w_local=0.3):
    """
    WildFusion风格: 用isotonic regression校准后融合
    train_mask: 训练集索引, 用于拟合校准器
    """
    from sklearn.isotonic import IsotonicRegression

    if local_sim is None:
        return global_sim

    n_tr = len(train_mask)

    # 从训练数据构建正负样本对用于校准
    pos_g, pos_l, neg_g, neg_l = [], [], [], []
    rng = np.random.RandomState(42)

    id_to_idx = {}
    for i, y in enumerate(yt):
        id_to_idx.setdefault(y, []).append(i)

    # 正样本对（同一个体）
    for idxs in id_to_idx.values():
        if len(idxs) < 2: continue
        for _ in range(min(10, len(idxs)*(len(idxs)-1)//2)):
            a, b = rng.choice(idxs, 2, replace=False)
            ti, tj = train_mask[a], train_mask[b]
            pos_g.append(global_sim[ti, tj])
            pos_l.append(local_sim[ti, tj])

    # 负样本对
    all_ids = list(id_to_idx.keys())
    for _ in range(min(5000, len(pos_g)*3)):
        id1, id2 = rng.choice(all_ids, 2, replace=False)
        a = rng.choice(id_to_idx[id1])
        b = rng.choice(id_to_idx[id2])
        ti, tj = train_mask[a], train_mask[b]
        neg_g.append(global_sim[ti, tj])
        neg_l.append(local_sim[ti, tj])

    if len(pos_g) < 10:
        # 校准数据不够，直接简单融合
        return (1-w_local) * global_sim + w_local * local_sim

    # 校准全局分数
    scores_g = np.array(pos_g + neg_g)
    labels = np.array([1]*len(pos_g) + [0]*len(neg_g))
    ir_g = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    ir_g.fit(scores_g, labels)

    # 校准局部分数
    scores_l = np.array(pos_l + neg_l)
    ir_l = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
    ir_l.fit(scores_l, labels)

    # 应用校准
    cal_g = ir_g.predict(global_sim.ravel()).reshape(global_sim.shape)
    cal_l = ir_l.predict(local_sim.ravel()).reshape(local_sim.shape)

    # 融合
    fused = (1-w_local) * cal_g + w_local * cal_l
    print(f"  [WildFusion] Calibrated fusion: global({1-w_local:.0%}) + local({w_local:.0%})")
    return fused

# ═══════════════════════════════════════
# 4. ArcFace微调 (v13验证过的方案)
# ═══════════════════════════════════════
class ArcHead(nn.Module):
    def __init__(s,d,n,sc=64.,m=.5):
        super().__init__(); s.s=sc; s.m=m
        s.W=nn.Parameter(torch.empty(n,d)); nn.init.xavier_uniform_(s.W)
        s.cm=np.cos(m);s.sm=np.sin(m);s.th=np.cos(np.pi-m);s.mm=np.sin(np.pi-m)*m
    def forward(s,x,y=None):
        cos=F.linear(F.normalize(x),F.normalize(s.W))
        if y is None: return cos*s.s
        sin=(1-cos.pow(2).clamp(0,1)).sqrt()
        phi=cos*s.cm-sin*s.sm;phi=torch.where(cos>s.th,phi,cos-s.mm)
        oh=torch.zeros_like(cos).scatter_(1,y.unsqueeze(1),1.)
        return (oh*phi+(1-oh)*cos)*s.s

def _orient(img, row):
    if row.get("species")=="salamander" and pd.notna(row.get("orientation")):
        o=str(row["orientation"]).lower()
        if o=="right": img=img.rotate(-90,expand=True)
        elif o=="left": img=img.rotate(90,expand=True)
    return img

class TrDS(Dataset):
    def __init__(s,df,root,sz):
        s.df=df.reset_index(drop=True);s.root=root
        s.tf=transforms.Compose([transforms.RandomResizedCrop(sz,scale=(.7,1.)),
            transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(15),
            transforms.ColorJitter(.3,.3,.2,.1),transforms.ToTensor(),
            transforms.Normalize(IM,IS),transforms.RandomErasing(p=.2)])
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        img=_orient(img,r); return s.tf(img),int(r["label"])

class InfDS(Dataset):
    def __init__(s,df,root,sz):
        s.df=df.reset_index(drop=True);s.root=root
        s.tf=transforms.Compose([transforms.Resize((sz,sz)),transforms.ToTensor(),transforms.Normalize(IM,IS)])
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(384,384),(128,128,128))
        img=_orient(img,r); return s.tf(img),int(r["image_id"])

@torch.no_grad()
def extract_ft(model, df, root, sz=384, bs=32):
    dl=DataLoader(InfDS(df,root,sz),batch_size=bs,num_workers=4,pin_memory=True)
    e=[]
    for imgs,_ in tqdm(dl,leave=False):
        e.append(F.normalize(model(imgs.to(DEV)),dim=-1).cpu().numpy())
    return np.concatenate(e)

def finetune(tdf, root, dsn):
    import timm
    le=LabelEncoder();tdf=tdf.copy();tdf["label"]=le.fit_transform(tdf.identity.values)
    ncls=tdf.label.nunique()
    if ncls<5: return None
    print(f"  [{dsn}] ArcFace {ncls}cls {len(tdf)}imgs")
    bb=timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384",pretrained=True,num_classes=0).to(DEV)
    hd=ArcHead(bb.num_features,ncls).to(DEV)
    ds=TrDS(tdf,root,384)
    ccnt=tdf.label.value_counts().to_dict()
    sw=[1./ccnt[int(r.label)] for _,r in tdf.iterrows()]
    crit=nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler=torch.amp.GradScaler()
    # S1: head only BS=48
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
    # S2: top25% BS=48
    nps=list(bb.named_parameters());top25={n for n,_ in nps[int(len(nps)*.75):]}
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
    del loader;torch.cuda.empty_cache();gc.collect()
    loader3=DataLoader(ds,batch_size=16,sampler=WeightedRandomSampler(sw,len(sw)),num_workers=4,pin_memory=True,drop_last=True)
    for p in bb.parameters(): p.requires_grad=True
    opt3=torch.optim.AdamW([{"params":bb.parameters(),"lr":9e-6},{"params":hd.parameters(),"lr":9e-5}],weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt3,T_max=11)
    print(f"    S3: full unfreeze BS=16")
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
    bb.eval();del hd,opt,opt2,opt3,scaler,sch,loader3,ds;torch.cuda.empty_cache();gc.collect()
    return bb

# ═══════════════════════════════════════
# 5. v9聚类 (验证过0.177)
# ═══════════════════════════════════════
def v9_cluster(sim, n_tr, yt, dsn):
    dist=np.clip(1-sim,0,2)
    best_eps,best_ari_db=0.5,-1
    for eps in np.arange(0.05,1.20,0.01):
        labels=DBSCAN(eps=eps,min_samples=2,metric="precomputed").fit_predict(dist)
        tr_labels=labels[:n_tr].copy()
        ns=tr_labels.max()+1 if tr_labels.max()>=0 else 0
        for i in range(n_tr):
            if tr_labels[i]==-1: tr_labels[i]=ns;ns+=1
        ari=adjusted_rand_score(yt,tr_labels)
        if ari>best_ari_db: best_ari_db,best_eps=ari,eps
    best_dt,best_ari_ag=0.5,-1
    for dt in np.arange(0.05,1.20,0.01):
        try:
            labels=AgglomerativeClustering(n_clusters=None,distance_threshold=dt,
                    metric="precomputed",linkage="average").fit_predict(dist)
            ari=adjusted_rand_score(yt,labels[:n_tr])
            if ari>best_ari_ag: best_ari_ag,best_dt=ari,dt
        except: pass
    if best_ari_ag>=best_ari_db:
        print(f"  [{dsn}] Agglo dt={best_dt:.2f} ARI={best_ari_ag:.4f}")
        labels=AgglomerativeClustering(n_clusters=None,distance_threshold=best_dt,
                metric="precomputed",linkage="average").fit_predict(dist)
    else:
        print(f"  [{dsn}] DBSCAN eps={best_eps:.2f} ARI={best_ari_db:.4f}")
        labels=DBSCAN(eps=best_eps,min_samples=2,metric="precomputed").fit_predict(dist)
        ns=labels.max()+1 if labels.max()>=0 else 0
        for i in range(len(labels)):
            if labels[i]==-1: labels[i]=ns;ns+=1
    te_labels=labels[n_tr:]
    print(f"  [{dsn}] clusters={len(set(te_labels))} for {len(te_labels)} test")
    return te_labels

# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    t0=time.time()
    print("="*55)
    print("  V15: wildlife-tools + ArcFace + v9聚类")
    print("="*55)
    if torch.cuda.is_available():
        g=torch.cuda.get_device_properties(0)
        print(f"GPU: {g.name} VRAM: {g.total_memory/1e9:.1f}GB")

    meta=pd.read_csv(os.path.join(DATA,"metadata.csv"))
    ssub=pd.read_csv(os.path.join(DATA,"sample_submission.csv"))
    trdf=meta[meta.split=="train"].copy()
    tedf=meta[meta.split=="test"].copy()

    # ── STAGE 1: 用wildlife-tools提取全局特征 ──
    print(f"\n{'━'*50}\nSTAGE 1: Feature Extraction (wildlife-tools)\n{'━'*50}")

    print("[MEGA] MegaDescriptor-L-384...")
    mega_tr = extract_with_wt("hf-hub:BVRA/MegaDescriptor-L-384", trdf, DATA, 384, bs=48)
    mega_te = extract_with_wt("hf-hub:BVRA/MegaDescriptor-L-384", tedf, DATA, 384, bs=48)
    print(f"  train={mega_tr.shape} test={mega_te.shape}")

    print("[MIEW] MiewID-msv3...")
    miew_tr = extract_miew(trdf, DATA, bs=32)
    miew_te = extract_miew(tedf, DATA, bs=32)
    print(f"  train={miew_tr.shape} test={miew_te.shape}")

    tr_ids_arr = trdf.image_id.values
    te_ids_arr = tedf.image_id.values
    tr_i2x = {int(tr_ids_arr[i]):i for i in range(len(tr_ids_arr))}
    te_i2x = {int(te_ids_arr[i]):i for i in range(len(te_ids_arr))}

    print(f"  Features: {(time.time()-t0)/60:.1f}min")

    # ── STAGE 2: ArcFace微调 ──
    print(f"\n{'━'*50}\nSTAGE 2: ArcFace Fine-Tuning\n{'━'*50}")
    ft_feats = {}
    for dsn in DS_TR:
        dtr=trdf[trdf.dataset==dsn].copy()
        dte=tedf[tedf.dataset==dsn].copy()
        if len(dtr)<50: continue
        try:
            ftm=finetune(dtr,DATA,dsn)
            if ftm is None: continue
            ftr=extract_ft(ftm,dtr,DATA); fte=extract_ft(ftm,dte,DATA)
            ft_feats[dsn]={"tr":ftr,"te":fte}
            print(f"  [{dsn}] FT: tr={ftr.shape} te={fte.shape}")
            del ftm;torch.cuda.empty_cache();gc.collect()
        except Exception as e:
            print(f"  [{dsn}] FAILED: {e}");torch.cuda.empty_cache();gc.collect()

    # ── STAGE 3: 分物种处理 ──
    print(f"\n{'━'*50}\nSTAGE 3: Per-species Clustering\n{'━'*50}")
    all_preds={}

    for dsn in DS_ALL:
        print(f"\n{'═'*45}\n  {dsn}\n{'═'*45}")
        ds_te=tedf[tedf.dataset==dsn].reset_index(drop=True)
        teix=[te_i2x[int(x)] for x in ds_te.image_id.values]
        n_te=len(ds_te)

        if dsn in DS_TR:
            ds_tr=trdf[trdf.dataset==dsn].reset_index(drop=True)
            trix=[tr_i2x[int(x)] for x in ds_tr.image_id.values]
            le=LabelEncoder(); yt=le.fit_transform(ds_tr.identity.values)
            n_tr_ds=len(ds_tr)

            # 合并train+test, 算多模型相似度融合
            all_mega=np.vstack([normalize(mega_tr[trix],axis=1),normalize(mega_te[teix],axis=1)])
            all_miew=np.vstack([normalize(miew_tr[trix],axis=1),normalize(miew_te[teix],axis=1)])
            sim_mega=all_mega@all_mega.T
            sim_miew=all_miew@all_miew.T
            global_sim = 0.55*sim_mega + 0.45*sim_miew

            # ArcFace特征融合
            if dsn in ft_feats:
                all_ft=np.vstack([normalize(ft_feats[dsn]["tr"],axis=1),normalize(ft_feats[dsn]["te"],axis=1)])
                sim_ft=all_ft@all_ft.T
                global_sim = 0.50*global_sim + 0.50*sim_ft
                print(f"  Fused: global(50%) + ArcFace(50%)")

            # ALIKED局部匹配 (只对Lynx和Salamander，它们最需要)
            local_sim = None
            if dsn in ["LynxID2025","SalamanderID2025"]:
                combined_df = pd.concat([ds_tr, ds_te], ignore_index=True)
                local_sim = compute_aliked_similarity(combined_df, DATA, global_sim, topk=20)
                if local_sim is not None:
                    # 校准融合
                    train_mask = list(range(n_tr_ds))
                    global_sim = calibrated_fusion(global_sim, local_sim, train_mask, yt, w_local=0.3)

            # v9聚类
            te_labels = v9_cluster(global_sim, n_tr_ds, yt, dsn)

        else:
            # TexasHornedLizards
            te_mega=normalize(mega_te[teix],axis=1)
            te_miew=normalize(miew_te[teix],axis=1)
            sim=0.55*(te_mega@te_mega.T)+0.45*(te_miew@te_miew.T)
            dist=np.clip(1-sim,0,2)
            best_eps,best_score=0.40,-1
            for eps in np.arange(0.15,0.80,0.01):
                pred=DBSCAN(eps=eps,min_samples=2,metric="precomputed").fit_predict(dist)
                ns=pred.max()+1 if pred.max()>=0 else 0
                for i in range(len(pred)):
                    if pred[i]==-1: pred[i]=ns;ns+=1
                n_cl=len(set(pred))
                if 15<=n_cl<=150:
                    score=1-(pred==-1).sum()/len(pred)-abs(n_cl/len(pred)-0.3)
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
    out=os.path.join(OUT,"submission.csv");sub.to_csv(out,index=False)
    print(f"\n{'━'*50}\nSaved: {out}")
    print(f"Rows:{len(sub)} Clusters:{sub.cluster.nunique()}")
    for d in DS_ALL:
        s=sub[sub.cluster.str.contains(d)]
        print(f"  {d:25s} imgs={len(s):4d} cl={s.cluster.nunique()}")
    print(f"Time: {(time.time()-t0)/60:.1f}min\nDONE!")

if __name__=="__main__":
    main()
