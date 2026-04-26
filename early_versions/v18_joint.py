#!/usr/bin/env python3
"""
V18 — Joint Train+Test Clustering (训练数据当锚点)
===================================================
V14-V17的根本问题：聚类时完全不用训练数据，只用它调阈值。
Top队伍做法：train+test一起聚类，训练数据是锚点。

V18核心改变：
  ★ 把train和test图片放在一起建相似度矩阵
  ★ 同一个体的训练图片之间：sim=1（must-link约束）
  ★ 不同个体的训练图片之间：sim降低（cannot-link约束）
  ★ 聚类整体，然后提取test图片的cluster标签
  ★ 训练图片当"引力中心"，把相似的test图片拉进正确的cluster
"""
import os,sys,gc,time,json,warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT","https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"]="false"

import numpy as np,pandas as pd
import torch,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize,LabelEncoder

D=torch.device("cuda" if torch.cuda.is_available() else "cpu")
M,S=[.485,.456,.406],[.229,.224,.225]
DATA=sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
OUT=sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov18"
os.makedirs(OUT,exist_ok=True)
SP_TR=["LynxID2025","SalamanderID2025","SeaTurtleID2022"]
SP_ALL=SP_TR+["TexasHornedLizards"]

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
    print(f"[M] Mega dim={m.num_features}");return m,384

def load_miew():
    import timm;from huggingface_hub import hf_hub_download;from safetensors.torch import load_file
    c=json.load(open(hf_hub_download("conservationxlabs/miewid-msv3","config.json")))
    m=timm.create_model(c.get("architecture","efficientnetv2_rw_m"),pretrained=False,num_classes=0)
    s={k:v for k,v in load_file(hf_hub_download("conservationxlabs/miewid-msv3","model.safetensors")).items() if "classifier" not in k}
    m.load_state_dict(s,strict=False);m=m.to(D).eval()
    print(f"[M] Miew dim={m.num_features}");return m,440

def load_dino():
    try:
        m=torch.hub.load("facebookresearch/dinov2","dinov2_vitl14_reg",trust_repo=True).to(D).eval()
        print("[M] DINOv2-L-reg dim=1024");return m,518
    except:
        try:
            m=torch.hub.load("facebookresearch/dinov2","dinov2_vitl14",trust_repo=True).to(D).eval()
            print("[M] DINOv2-L dim=1024");return m,518
        except Exception as e:
            print(f"[M] DINOv2 fail: {e}");return None,0

def cluster(sim,th):
    dist=np.clip(1-sim,0,2);np.fill_diagonal(dist,0)
    return AgglomerativeClustering(n_clusters=None,distance_threshold=th,metric="precomputed",linkage="average").fit_predict(dist)

# ═══════════════════════════════════════════
# CORE: Joint Train+Test Clustering
# ═══════════════════════════════════════════
def joint_cluster(feat_dict, tr_idx, te_idx, tr_labels, weights, constraint_strength=0.3):
    """
    Cluster train+test together with training constraints.

    1. Build combined similarity matrix [train+test × train+test]
    2. Inject must-link constraints for same-identity training pairs
    3. Inject cannot-link constraints for different-identity training pairs
    4. Search optimal threshold on training portion (known labels)
    5. Cluster everything, extract test labels
    """
    all_idx = np.concatenate([tr_idx, te_idx])
    n_tr = len(tr_idx)
    n_te = len(te_idx)
    n_all = n_tr + n_te

    # Build combined similarity
    sim = np.zeros((n_all, n_all))
    for name, w in weights.items():
        f = normalize(feat_dict[name][all_idx], axis=1)
        sim += w * (f @ f.T)

    # Inject training constraints
    le = LabelEncoder()
    tr_encoded = le.fit_transform(tr_labels)

    for i in range(n_tr):
        for j in range(i+1, n_tr):
            if tr_encoded[i] == tr_encoded[j]:
                # Must-link: boost similarity toward 1
                sim[i,j] = sim[i,j] * (1 - constraint_strength) + 1.0 * constraint_strength
                sim[j,i] = sim[i,j]
            else:
                # Cannot-link: reduce similarity toward 0
                sim[i,j] = sim[i,j] * (1 - constraint_strength * 0.5)
                sim[j,i] = sim[i,j]

    # Search optimal threshold using training portion
    # We evaluate: cluster everything, check ARI on training images only
    best_ari, best_th = -1, 0.5
    for th in np.arange(0.05, 1.50, 0.01):
        try:
            labels = cluster(sim, th)
            # Evaluate ARI on training portion only
            tr_pred = labels[:n_tr]
            ari = adjusted_rand_score(tr_labels, tr_pred)
            if ari > best_ari:
                best_ari, best_th = ari, th
        except:
            continue

    # Final clustering with best threshold
    final_labels = cluster(sim, best_th)
    te_labels = final_labels[n_tr:]  # extract test portion

    return te_labels, best_th, best_ari


def joint_cluster_sampled(feat_dict, tr_idx, te_idx, tr_labels, weights,
                          constraint_strength=0.3, max_train=800):
    """
    Same as joint_cluster but samples training data if too large.
    Keeps ALL test images, samples training images proportionally per identity.
    """
    n_tr = len(tr_idx)
    n_te = len(te_idx)

    if n_tr > max_train:
        # Sample training images, keeping proportional representation
        rng = np.random.RandomState(42)
        unique_ids = np.unique(tr_labels)
        sampled = []
        per_id = max(2, max_train // len(unique_ids))
        for uid in unique_ids:
            mask = tr_labels == uid
            idx_for_id = np.where(mask)[0]
            if len(idx_for_id) > per_id:
                chosen = rng.choice(idx_for_id, per_id, replace=False)
            else:
                chosen = idx_for_id
            sampled.extend(chosen)
        sampled = np.array(sampled)
        s_tr_idx = tr_idx[sampled]
        s_tr_labels = tr_labels[sampled]
        print(f"      Sampled {len(s_tr_idx)}/{n_tr} train images")
    else:
        s_tr_idx = tr_idx
        s_tr_labels = tr_labels

    return joint_cluster(feat_dict, s_tr_idx, te_idx, s_tr_labels, weights, constraint_strength)


# ═══════════════════════════════════════════
def main():
    t0=time.time()
    print("="*60)
    print("  V18 — Joint Train+Test Clustering")
    print("="*60)
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name} | {p.total_memory/1e9:.0f}GB")

    meta=pd.read_csv(os.path.join(DATA,"metadata.csv"))
    ssub=pd.read_csv(os.path.join(DATA,"sample_submission.csv"))
    trdf=meta[meta.split=="train"];tedf=meta[meta.split=="test"]
    alldf=pd.concat([trdf,tedf],ignore_index=True)
    print(f"  {len(trdf)} train, {len(tedf)} test")

    # ── Features ──
    print(f"\n{'━'*55}\nSTAGE 1: Features\n{'━'*55}")
    fd={}
    m,sz=load_mega();fd["mega"],_=extract(m,alldf,DATA,sz,48);del m;torch.cuda.empty_cache();gc.collect()
    m,sz=load_miew();fd["miew"],_=extract(m,alldf,DATA,sz,32);del m;torch.cuda.empty_cache();gc.collect()
    dm,dsz=load_dino()
    if dm:fd["dino"],_=extract(dm,alldf,DATA,dsz,16);del dm;torch.cuda.empty_cache();gc.collect()

    all_ids=alldf.image_id.values
    id2idx={int(all_ids[i]):i for i in range(len(all_ids))}
    print(f"  Models: {list(fd.keys())} | {(time.time()-t0)/60:.1f}min")

    # V16 best weights per species
    SP_WEIGHTS={
        "LynxID2025":{"mega":0.9,"miew":0.09,"dino":0.01} if "dino" in fd else {"mega":0.85,"miew":0.15},
        "SalamanderID2025":{"mega":0.2,"miew":0.0,"dino":0.8} if "dino" in fd else {"mega":0.6,"miew":0.4},
        "SeaTurtleID2022":{"mega":0.5,"miew":0.2,"dino":0.3} if "dino" in fd else {"mega":0.55,"miew":0.45},
        "TexasHornedLizards":{k:1./len(fd) for k in fd},
    }

    # ── Joint Clustering ──
    print(f"\n{'━'*55}\nSTAGE 2: Joint Train+Test Clustering\n{'━'*55}")
    preds={}

    for sp in SP_ALL:
        print(f"\n  {'═'*40} {sp}")
        te=tedf[tedf.dataset==sp].reset_index(drop=True)
        te_idx=np.array([id2idx[int(x)] for x in te.image_id])
        nte=len(te)
        w=SP_WEIGHTS[sp]
        # Remove zero-weight models
        w={k:v for k,v in w.items() if v>0}
        ws=sum(w.values())
        w={k:v/ws for k,v in w.items()}

        if sp in SP_TR:
            tr=trdf[trdf.dataset==sp].reset_index(drop=True)
            tr_idx=np.array([id2idx[int(x)] for x in tr.image_id])
            tr_labels=tr.identity.values
            ntr=len(tr)
            nids=len(set(tr_labels))
            print(f"    {ntr} train ({nids} ids) + {nte} test")

            # Search over constraint strengths
            best_ari_overall = -1
            best_result = None

            for cs in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:
                te_labels, th, ari = joint_cluster_sampled(
                    fd, tr_idx, te_idx, tr_labels, w,
                    constraint_strength=cs, max_train=800
                )
                print(f"    cs={cs:.1f}: th={th:.3f} train_ARI={ari:.4f} → {len(set(te_labels))} test clusters")
                if ari > best_ari_overall:
                    best_ari_overall = ari
                    best_result = (te_labels, th, ari, cs)

            te_labels, th, ari, cs = best_result
            print(f"    ★ Best: cs={cs:.1f} th={th:.3f} ARI={ari:.4f} → {len(set(te_labels))} clusters")
            labels = te_labels

        else:
            print(f"    {nte} test (no training data)")
            # Pure test clustering
            sim=np.zeros((nte,nte))
            for name,wt in w.items():
                f=normalize(fd[name][te_idx],axis=1);sim+=wt*(f@f.T)

            best_th,best_sc=0.40,-1
            dist=np.clip(1-sim,0,2);np.fill_diagonal(dist,0)
            for th in np.arange(0.10,1.50,0.01):
                try:
                    p=AgglomerativeClustering(n_clusters=None,distance_threshold=th,metric="precomputed",linkage="average").fit_predict(dist)
                    ratio=len(set(p))/nte
                    if .15<=ratio<=.55:
                        sc=-abs(ratio-.35)
                        if sc>best_sc:best_sc,best_th=sc,th
                except:pass
            labels=cluster(sim,best_th)
            print(f"    → {len(set(labels))} clusters (th={best_th:.3f})")

        for i in range(nte):
            preds[int(te.iloc[i].image_id)]=f"cluster_{sp}_{labels[i]}"

    # ── Submit ──
    print(f"\n{'━'*55}\nSTAGE 3: Submission\n{'━'*55}")
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
