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

mg = np.load('./feat_cache/megadesc_features.npz')
mg_ids = mg['ids'].astype(str)
mg_feat = mg['feats']

id2idx = dict(zip(mg_ids, range(len(mg_ids))))
meta['feat_idx'] = meta['image_id'].map(id2idx)
meta = meta.dropna(subset=['feat_idx'])
meta['feat_idx'] = meta['feat_idx'].astype(int)

# Only sample from species with train data for t-SNE
train3 = meta[(meta['split']=='train') & (meta['dataset']!='TexasHornedLizards')].copy()
samples = []
for sp in ['LynxID2025','SalamanderID2025','SeaTurtleID2022']:
    sp_data = train3[train3['dataset']==sp]
    n = min(400, len(sp_data))
    samples.append(sp_data.sample(n, random_state=42))
sample = pd.concat(samples, ignore_index=True)

idx = sample['feat_idx'].values
feat = mg_feat[idx]
norms = np.linalg.norm(feat, axis=1, keepdims=True)
norms[norms==0] = 1
feat = feat / norms

print(f'Running t-SNE on {len(feat)} samples...')
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=800)
emb = tsne.fit_transform(feat)
print('t-SNE done')

colors = {'LynxID2025':'#e74c3c','SalamanderID2025':'#2ecc71','SeaTurtleID2022':'#3498db'}
names = {'LynxID2025':'Lynx','SalamanderID2025':'Salamander','SeaTurtleID2022':'Sea Turtle'}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for sp, c in colors.items():
    m = sample['dataset'].values == sp
    if m.sum() > 0:
        ax.scatter(emb[m,0], emb[m,1], c=c, s=8, alpha=0.6, label=names[sp])
ax.legend(fontsize=9, markerscale=2)
ax.set_title('(a) MegaDescriptor: Species-level separation')
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

ax = axes[1]
st = sample[sample['dataset']=='SeaTurtleID2022'].copy()
st_mask = sample['dataset'].values == 'SeaTurtleID2022'
st_emb = emb[st_mask]
tops = st['identity'].value_counts().head(12).index.tolist()
cm = plt.cm.tab20(np.linspace(0, 1, 12))
for i, tid in enumerate(tops):
    im = st['identity'].values == tid
    if im.sum() > 0:
        ax.scatter(st_emb[im,0], st_emb[im,1], c=[cm[i]], s=14, alpha=0.7)
other = ~st['identity'].isin(tops).values
if other.sum() > 0:
    ax.scatter(st_emb[other,0], st_emb[other,1], c='lightgray', s=4, alpha=0.3)
ax.set_title('(b) Sea Turtle: Individual-level (top 12 IDs)')
ax.set_xlabel('t-SNE dim 1'); ax.set_ylabel('t-SNE dim 2')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT}/fig1_tsne.png', dpi=300, bbox_inches='tight')
print(f'Saved {OUT}/fig1_tsne.png')
