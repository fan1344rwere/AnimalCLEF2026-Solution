
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 11
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

OUT = './figures'
os.makedirs(OUT, exist_ok=True)

meta = pd.read_csv('./data/metadata.csv')
meta['image_id'] = meta['image_id'].astype(str)
SP_COL = 'dataset'
print('Species values:', meta[SP_COL].unique().tolist())

d3 = np.load('./feat_cache/dinov3_features.npz')
mg = np.load('./feat_cache/megadesc_features.npz')

d3_ids = d3['ids'].astype(str)
mg_feat = mg['feats']

id2idx = {iid: i for i, iid in enumerate(d3_ids)}
meta['feat_idx'] = meta['image_id'].map(id2idx)
meta = meta.dropna(subset=['feat_idx'])
meta['feat_idx'] = meta['feat_idx'].astype(int)

print('=== FIGURE 1: t-SNE ===')
train_meta = meta[meta['split']=='train'].copy()
sample = train_meta.groupby(SP_COL).apply(lambda x: x.sample(min(400, len(x)), random_state=42)).reset_index(drop=True)
idx = sample['feat_idx'].values
feat = mg_feat[idx]
norms = np.linalg.norm(feat, axis=1, keepdims=True); norms[norms==0]=1; feat = feat/norms

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=800)
emb = tsne.fit_transform(feat)

colors = {'LynxID2025':'#e74c3c','SalamanderID2025':'#2ecc71','SeaTurtleID2022':'#3498db','TexasHornedLizards':'#f39c12'}
names = {'LynxID2025':'Lynx','SalamanderID2025':'Salamander','SeaTurtleID2022':'Sea Turtle','TexasHornedLizards':'TX Horned Lizard'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for sp in colors:
    m = sample[SP_COL].values == sp
    if m.sum()>0: ax.scatter(emb[m,0], emb[m,1], c=colors[sp], s=8, alpha=0.6, label=names[sp])
ax.legend(fontsize=9, markerscale=2); ax.set_title('(a) MegaDescriptor: Species-level'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

ax = axes[1]
st = sample[sample[SP_COL]=='SeaTurtleID2022'].copy()
st_m = sample[SP_COL].values=='SeaTurtleID2022'
st_e = emb[st_m]
tops = st['identity'].value_counts().head(12).index.tolist()
cm = plt.cm.tab20(np.linspace(0,1,12))
for i,tid in enumerate(tops):
    im = st['identity'].values==tid
    if im.sum()>0: ax.scatter(st_e[im,0], st_e[im,1], c=[cm[i]], s=14, alpha=0.7)
om = ~st['identity'].isin(tops).values
if om.sum()>0: ax.scatter(st_e[om,0], st_e[om,1], c='lightgray', s=4, alpha=0.3)
ax.set_title('(b) Sea Turtle: Individual-level (top 12 IDs)'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_tsne.png', dpi=200, bbox_inches='tight'); plt.close()
print('Fig1 done')

print('=== FIGURE 2: Train vs LB ===')
ver = ['V9\nBaseline','V16\n3-BB','V17\n+ArcFace','V19\nOrient','V22\nSupCon5','V23\n+Pseudo','V25c\n3-BB']
lb = [0.177, 0.231, 0.225, 0.179, 0.240, 0.105, 0.172]
tr = [0.39, 0.42, 0.57, 0.37, 0.72, 0.82, 0.84]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(ver)); w=0.32
b1=ax.bar(x-w/2, tr, w, label='Train ARI (avg)', color='#3498db', alpha=0.85)
b2=ax.bar(x+w/2, lb, w, label='Leaderboard Score', color='#e74c3c', alpha=0.85)
ax.set_ylabel('Score'); ax.set_title('Training ARI vs. Leaderboard Score'); ax.set_xticks(x); ax.set_xticklabels(ver, fontsize=9)
ax.legend(); ax.set_ylim(0,1.0)
for b in b1:
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02, f'{b.get_height():.2f}', ha='center', fontsize=8)
for b in b2:
    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02, f'{b.get_height():.3f}', ha='center', fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_train_vs_lb.png', dpi=200, bbox_inches='tight'); plt.close()
print('Fig2 done')

print('=== FIGURE 3: Per-species ===')
sp = ['Lynx','Salamander','SeaTurtle','TX Horned\nLizard']
raw = [0.075, 0.001, 0.001, 0]
mega = [0.15, 0.12, 0.86, 0]
sup = [0.26, 0.90, 0.91, 0]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4); w=0.25
ax.bar(x-w, raw, w, label='Foundation (raw)', color='#95a5a6')
ax.bar(x, mega, w, label='MegaDescriptor (raw)', color='#3498db')
ax.bar(x+w, sup, w, label='SupCon Projection', color='#e74c3c')
ax.set_ylabel('Training ARI'); ax.set_title('Per-Species Training ARI by Method')
ax.set_xticks(x); ax.set_xticklabels(sp); ax.legend(fontsize=9); ax.set_ylim(0,1.1)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_per_species.png', dpi=200, bbox_inches='tight'); plt.close()
print('Fig3 done')

print('ALL DONE')
import subprocess
subprocess.run(['ls','-la',OUT])
