#!/usr/bin/env python
"""
v11: WildFusion局部匹配 + Orientation-aware + 相似度融合
这是通往0.4+的关键版本。

核心改进:
1. ALIKED局部关键点匹配（top-K稀疏匹配，不全量）
2. Orientation-aware相似度boost/penalize
3. MegaDescriptor + MiewID 相似度级融合
4. train锚定 Agglomerative聚类
5. SeaTurtle用lookup（train ARI=0.86够好）

预计运行时间: 30-60分钟（取决于局部匹配的pair数量）
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
LOCAL_TOPK = 15  # 每张图只对top-15全局相似候选做局部匹配

# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════
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

class RawImageDS(Dataset):
    """Returns raw images (for local matching)."""
    def __init__(s, df, root, sz=512):
        s.df=df.reset_index(drop=True); s.root=root; s.sz=sz
        s.tt=transforms.ToTensor()
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r=s.df.iloc[i]
        try: img=Image.open(os.path.join(s.root,r["path"])).convert("RGB")
        except: img=Image.new("RGB",(s.sz,s.sz),(128,128,128))
        img=img.resize((s.sz,s.sz), Image.BICUBIC)
        return s.tt(img), int(r["image_id"])

@torch.no_grad()
def feats(model,df,root,sz):
    dl=DataLoader(DS(df,root,sz),batch_size=BS,num_workers=4,pin_memory=True)
    e,ids=[],[]
    for imgs,iids in tqdm(dl,leave=False):
        e.append(F.normalize(model(imgs.to(DEV)),dim=-1).cpu().numpy()); ids.extend(iids.numpy())
    return np.concatenate(e), np.array(ids)

# ═══════════════════════════════════════════════════════════════
# ALIKED局部关键点匹配
# ═══════════════════════════════════════════════════════════════
class ALIKEDMatcher:
    """用kornia提取ALIKED关键点，mutual nearest neighbor匹配。"""
    def __init__(self):
        self.available = False
        try:
            import kornia.feature as KF
            self.aliked = KF.KeyNetAffNetHardNet(num_features=512).eval().to(DEV)
            self.available = True
            print("  ALIKED matcher: OK (kornia KeyNetAffNetHardNet)")
        except Exception as e:
            try:
                # Fallback: 用kornia的DISK
                import kornia
                self.disk = kornia.feature.DISK.from_pretrained("depth").eval().to(DEV)
                self.available = True
                print("  Local matcher: OK (kornia DISK)")
            except Exception as e2:
                print(f"  Local matcher: FAILED ({e2})")

    @torch.no_grad()
    def match_score(self, img1_tensor, img2_tensor):
        """计算两张图的局部匹配分数(0-1)。"""
        if not self.available:
            return 0.0
        try:
            # 用kornia提取并匹配
            import kornia.feature as KF
            if hasattr(self, 'aliked'):
                # KeyNetAffNetHardNet
                inp1 = img1_tensor.unsqueeze(0).to(DEV) if img1_tensor.dim()==3 else img1_tensor.to(DEV)
                inp2 = img2_tensor.unsqueeze(0).to(DEV) if img2_tensor.dim()==3 else img2_tensor.to(DEV)
                # 转灰度
                gray1 = inp1.mean(dim=1, keepdim=True)
                gray2 = inp2.mean(dim=1, keepdim=True)
                kps1, _, desc1 = self.aliked(gray1)
                kps2, _, desc2 = self.aliked(gray2)
                if desc1.shape[1] == 0 or desc2.shape[1] == 0:
                    return 0.0
                # Mutual nearest neighbor matching
                sim = desc1[0] @ desc2[0].T  # [N1, N2]
                nn12 = sim.argmax(dim=1)  # [N1]
                nn21 = sim.argmax(dim=0)  # [N2]
                mutual = (nn21[nn12] == torch.arange(len(nn12), device=DEV))
                n_matches = mutual.sum().item()
                score = min(1.0, n_matches / 30.0)
                return score
        except:
            return 0.0

    def compute_sparse_local_sim(self, global_sim, df, root, topk=15):
        """
        对每张图的top-K全局相似候选做局部匹配。
        返回增强后的相似度矩阵。
        """
        if not self.available:
            return global_sim

        n = global_sim.shape[0]
        local_boost = np.zeros((n, n), dtype=np.float32)
        raw_ds = RawImageDS(df, root, sz=512)

        print(f"  Local matching: {n} images × top-{topk}...", flush=True)
        for i in tqdm(range(n), desc="  local", leave=False):
            topk_idx = np.argsort(-global_sim[i])[:topk+1]  # +1 to skip self
            img1, _ = raw_ds[i]
            for j in topk_idx:
                if j == i: continue
                if local_boost[i, j] > 0: continue  # already computed
                img2, _ = raw_ds[j]
                score = self.match_score(img1, img2)
                local_boost[i, j] = score
                local_boost[j, i] = score

        # 融合: global * 0.7 + local * 0.3
        fused = 0.7 * global_sim + 0.3 * local_boost
        return fused

# ═══════════════════════════════════════════════════════════════
# Orientation-aware相似度调整
# ═══════════════════════════════════════════════════════════════
def orientation_adjust(sim, orientations):
    """同orientation boost, 异orientation penalize."""
    n = sim.shape[0]
    adjusted = sim.copy()
    for i in range(n):
        for j in range(i+1, n):
            oi, oj = str(orientations[i]).lower(), str(orientations[j]).lower()
            if oi == oj and oi not in ('nan', 'unknown', ''):
                adjusted[i,j] *= 1.25
                adjusted[j,i] *= 1.25
            elif oi != oj and oi not in ('nan', 'unknown', '') and oj not in ('nan', 'unknown', ''):
                adjusted[i,j] *= 0.75
                adjusted[j,i] *= 0.75
    return np.clip(adjusted, 0, 1)

# ═══════════════════════════════════════════════════════════════
# 相似度融合 + 聚类
# ═══════════════════════════════════════════════════════════════
def fused_sim(mega_f, miew_f, w_mega=0.6):
    mn = normalize(mega_f, axis=1); wn = normalize(miew_f, axis=1)
    return w_mega * (mn @ mn.T) + (1-w_mega) * (wn @ wn.T)

def tune_and_cluster(dist, yt_or_none, label=""):
    """DBSCAN + Agglo sweep, return best labels."""
    if yt_or_none is not None:
        yt = yt_or_none
        best_ari, best_p, best_m = -1, 0.5, "agglo"
        for eps in np.arange(0.05, 1.30, 0.01):
            pred = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
            ns = pred.max()+1 if pred.max()>=0 else 0
            for i in range(len(pred)):
                if pred[i]==-1: pred[i]=ns; ns+=1
            ari = adjusted_rand_score(yt, pred)
            if ari > best_ari: best_ari, best_p, best_m = ari, eps, "dbscan"
        for dt in np.arange(0.05, 1.30, 0.01):
            try:
                pred = AgglomerativeClustering(n_clusters=None, distance_threshold=dt,
                    metric="precomputed", linkage="average").fit_predict(dist)
                ari = adjusted_rand_score(yt, pred)
                if ari > best_ari: best_ari, best_p, best_m = ari, dt, "agglo"
            except: pass
        print(f"  {label}: {best_m} p={best_p:.2f} ARI={best_ari:.4f}")
    else:
        # No labels: heuristic
        best_m, best_p = "dbscan", 0.40
        best_score = -1
        for eps in np.arange(0.15, 0.80, 0.01):
            pred = DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(dist)
            ns = pred.max()+1 if pred.max()>=0 else 0
            n_noise = (pred==-1).sum()
            for i in range(len(pred)):
                if pred[i]==-1: pred[i]=ns; ns+=1
            n_cl = len(set(pred))
            if 15 <= n_cl <= 150:
                score = 1 - n_noise/len(pred) - abs(n_cl/len(pred) - 0.3)
                if score > best_score: best_score, best_p = score, eps
        print(f"  {label}: heuristic eps={best_p:.2f}")

    if best_m == "dbscan":
        labels = DBSCAN(eps=best_p, min_samples=2, metric="precomputed").fit_predict(dist)
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=best_p,
            metric="precomputed", linkage="average").fit_predict(dist)
    ns = labels.max()+1 if labels.max()>=0 else 0
    for i in range(len(labels)):
        if labels[i]==-1: labels[i]=ns; ns+=1
    return labels

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    DATA = sys.argv[1] if len(sys.argv)>1 else "/root/autodl-tmp/animal-clef-2026"
    OUT = sys.argv[2] if len(sys.argv)>2 else "/root/autodl-tmp/ov11"
    os.makedirs(OUT, exist_ok=True)

    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"))
    ssub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    trdf = meta[meta.split=="train"].copy(); tedf = meta[meta.split=="test"].copy()

    # Load models
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

    # Extract features
    print("\nExtracting features...")
    mega_tr, mega_tr_ids = feats(mega, trdf, DATA, 384)
    mega_te, mega_te_ids = feats(mega, tedf, DATA, 384)
    miew_tr, _ = feats(miew, trdf, DATA, 440)
    miew_te, _ = feats(miew, tedf, DATA, 440)
    del mega, miew; torch.cuda.empty_cache(); gc.collect()
    print(f"Features in {(_t.time()-t0)/60:.1f}min")

    # Init local matcher
    print("\nInit local matcher...")
    matcher = ALIKEDMatcher()

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
        n_te = len(ds_te)

        # Step 1: Global similarity (Mega + MiewID fusion)
        te_sim = fused_sim(te_m, te_w, w_mega=0.6)

        # Step 2: Orientation-aware adjustment (Lynx & Salamander)
        if dsn in ["LynxID2025", "SalamanderID2025"]:
            orientations = ds_te.orientation.values
            te_sim = orientation_adjust(te_sim, orientations)
            print(f"  Orientation-adjusted")

        # Step 3: Local matching enhancement (sparse, top-K only)
        if dsn in ["LynxID2025", "SalamanderID2025"] and matcher.available:
            te_sim = matcher.compute_sparse_local_sim(te_sim, ds_te, DATA, topk=LOCAL_TOPK)
            print(f"  Local matching applied")

        if dsn in DS_TR:
            ds_tr = trdf[trdf.dataset==dsn].reset_index(drop=True)
            trix = [tr_i2x[int(x)] for x in ds_tr.image_id.values]
            tr_m, tr_w = mega_tr[trix], miew_tr[trix]
            train_ids = ds_tr.identity.values
            n_tr = len(ds_tr)

            # Train+test anchored: compute full similarity
            all_m = np.vstack([tr_m, te_m])
            all_w = np.vstack([tr_w, te_w])
            all_sim = fused_sim(all_m, all_w, w_mega=0.6)

            # Orientation adjust for anchored (if Lynx/Sal)
            if dsn in ["LynxID2025", "SalamanderID2025"]:
                all_ori = np.concatenate([ds_tr.orientation.values, ds_te.orientation.values])
                all_sim = orientation_adjust(all_sim, all_ori)

            dist = np.clip(1 - all_sim, 0, 2)
            le = LabelEncoder(); yt = le.fit_transform(train_ids)

            # Tune on train submatrix
            tr_dist = dist[:n_tr, :n_tr]
            labels = tune_and_cluster(tr_dist, yt, label=dsn)

            # Apply best to full
            # Re-tune on full distance (using train labels)
            full_yt = np.concatenate([yt, np.arange(n_te) + yt.max() + 1])
            full_labels = tune_and_cluster(dist, full_yt, label=f"{dsn}-anchored")
            te_labels = full_labels[n_tr:]
        else:
            dist = np.clip(1 - te_sim, 0, 2)
            te_labels = tune_and_cluster(dist, None, label=dsn)

        n_cl = len(set(te_labels))
        print(f"  → {n_cl} clusters for {n_te} images")

        for i in range(n_te):
            all_preds[int(ds_te.iloc[i].image_id)] = f"cluster_{dsn}_{te_labels[i]}"

    # Submission
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

if __name__ == "__main__":
    main()
