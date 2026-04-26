#!/usr/bin/env python3
"""
AnimalCLEF2026 V23 — Semi-Supervised + t-SNE + HDBSCAN
========================================================
Key improvements over V22:
  1. BioCLIP 2.5 ViT-H/14 as new backbone (biology-specific, fine-grained)
  2. t-SNE dimensionality reduction before clustering (+26-38% from paper)
  3. HDBSCAN instead of HAC (proven better in zero-shot clustering paper)
  4. Iterative pseudo-labeling: SupCon → cluster → pseudo-label → retrain
  5. Strong regularization to prevent V22-style overfitting
  6. MegaDescriptor brought back (has Re-ID signal: 0.86 ARI on SeaTurtle)

Backbones:
  - BioCLIP 2.5 ViT-H/14 (1280 dim) — biology fine-grained
  - DINOv3-7B (8192 dim) — best self-supervised features
  - SigLIP2-Giant (1536 dim) — strong vision-language
  - MegaDescriptor-L-384 (1536 dim) — animal Re-ID specific

Pipeline:
  0. Extract features from all 4 backbones
  1. Per-species: SupCon projection + iterative pseudo-labeling
  2. t-SNE reduction to intermediate dim (32D)
  3. HDBSCAN clustering
  4. Post-processing → submission.csv

Target: Top 5 (ARI > 0.528)
"""

import os, sys, gc, time, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
from PIL import Image
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "/root/autodl-tmp/animal-clef-2026"
MODEL_DIR = "/root/autodl-tmp/models"
OUTPUT_DIR = "/root/autodl-tmp/ov23"
FEAT_CACHE = "/root/autodl-tmp/feat_cache_v23"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEAT_CACHE, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory/1024**3:.1f}GB")

SPECIES_ORDER = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"]

# Backbone configs
BACKBONES = {
    "bioclip25": {
        "type": "open_clip",
        "model_path": os.path.join(MODEL_DIR, "bioclip25-vith14"),
        "dim": 1280,
        "img_size": 224,
    },
    "dinov3": {
        "type": "dinov3",
        "model_path": os.path.join(MODEL_DIR, "dinov3-vit7b"),
        "dim": 4096,  # CLS token dim for 7B; will concat with patch_mean for 8192
        "img_size": 518,
    },
    "siglip2": {
        "type": "siglip2",
        "model_path": os.path.join(MODEL_DIR, "siglip2-giant"),
        "dim": 1536,
        "img_size": 384,
    },
    "megadesc": {
        "type": "megadesc",
        "model_name": "hf-hub:BVRA/MegaDescriptor-L-384",
        "dim": 1536,
        "img_size": 384,
    },
}

# SupCon training hyperparams (more conservative than V22 to prevent overfitting)
PROJ_HIDDEN = 1024
PROJ_OUT = 256
SUPCON_TEMP = 0.1  # higher temp = softer probabilities = less overfitting
SUPCON_LR = 2e-4   # lower LR than V22 (was 5e-4)
SUPCON_WD = 1e-3    # weight decay for regularization
SUPCON_EPOCHS_SUPERVISED = 30  # fewer epochs (V22 used 50)
SUPCON_EPOCHS_PSEUDO = 15      # even fewer for pseudo-label rounds
SUPCON_BATCH = 256
DROPOUT_RATE = 0.3  # dropout in projection head

# HDBSCAN params (from paper recommendations)
HDBSCAN_MIN_CLUSTER = {
    "LynxID2025": 10,        # ~77 known individuals, expect more in test
    "SalamanderID2025": 3,   # many singletons, small clusters
    "SeaTurtleID2022": 5,    # moderate clusters
    "TexasHornedLizards": 5, # unknown, moderate
}
HDBSCAN_MIN_SAMPLES = 3

# t-SNE params
TSNE_DIM = 32         # intermediate dimensionality (paper suggests >2D is better)
TSNE_PERPLEXITY = 30  # standard default

# Pseudo-labeling
N_PSEUDO_ROUNDS = 2
PSEUDO_CONFIDENCE_THRESHOLD = 0.7  # only use high-confidence pseudo-labels

# ============================================================
# UTILITIES
# ============================================================
def cosine_sim_matrix(A, B=None):
    """Compute cosine similarity matrix between A and B (or A and A)."""
    A = F.normalize(torch.from_numpy(A).float(), dim=1)
    if B is None:
        B = A
    else:
        B = F.normalize(torch.from_numpy(B).float(), dim=1)
    return (A @ B.T).numpy()


def l2_normalize(feats):
    """L2 normalize features."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return feats / norms


# ============================================================
# FEATURE EXTRACTION
# ============================================================
class ImageListDataset(Dataset):
    """Simple dataset for image paths."""
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), idx
        except Exception:
            # Return a black image on error
            return self.transform(Image.new("RGB", (224, 224))), idx


def extract_bioclip25(image_paths, batch_size=32):
    """Extract BioCLIP 2.5 ViT-H/14 features using open_clip."""
    import open_clip
    cfg = BACKBONES["bioclip25"]
    model_path = cfg["model_path"]
    print(f"  Loading BioCLIP 2.5 from {model_path}...")

    # Try loading from local directory
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:imageomics/bioclip-2.5-vith14",
            pretrained=model_path,
        )
    except Exception as e1:
        print(f"    Direct load failed ({e1}), trying alternative...")
        # Try loading model config + weights separately
        import glob, json
        config_path = os.path.join(model_path, "open_clip_config.json")
        weight_files = glob.glob(os.path.join(model_path, "open_clip_pytorch_model*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(os.path.join(model_path, "open_clip_pytorch_model*.bin"))
        if not weight_files:
            raise RuntimeError(f"No model weights found in {model_path}")

        with open(config_path) as f:
            clip_cfg = json.load(f)
        model_cfg = clip_cfg.get("model_cfg", {})
        model_name = model_cfg.get("model_name", "ViT-H-14")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=weight_files[0],
        )

    model = model.to(DEVICE).eval()
    if DEVICE.type == "cuda":
        model = model.half()

    dataset = ImageListDataset(image_paths, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_feats = np.zeros((len(image_paths), cfg["dim"]), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(DEVICE)
            if DEVICE.type == "cuda":
                imgs = imgs.half()
            feats = model.encode_image(imgs)
            feats = F.normalize(feats.float(), dim=1)
            all_feats[idxs.numpy()] = feats.cpu().numpy()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return all_feats


def extract_dinov3(image_paths, batch_size=2):
    """Extract DINOv3-7B features (CLS + patch_mean = 8192 dim).
    NOTE: DINOv3-7B from ModelScope may need special handling.
    Falls back to loading with torch if transformers fails.
    """
    from torchvision import transforms
    cfg = BACKBONES["dinov3"]
    model_path = cfg["model_path"]
    print(f"  Loading DINOv3-7B from {model_path}...")

    # Try loading with transformers first
    try:
        from transformers import AutoModel, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        model = model.to(DEVICE).eval()
        use_processor = True
        print("    Loaded via AutoModel")
    except Exception as e:
        print(f"    AutoModel failed: {e}")
        # Fallback: load with safetensors directly
        print("    Trying direct safetensors load...")
        from safetensors.torch import load_file
        import json, glob

        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            hidden_size = config.get("hidden_size", 4096)
        else:
            hidden_size = 4096

        # Load model state dict from safetensors shards
        shard_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        state_dict = {}
        for sf in shard_files:
            state_dict.update(load_file(sf))

        # Build a simple wrapper that does forward through the transformer
        # This is a fallback - we just use the features directly
        use_processor = False
        model = None  # Will handle differently below

    if use_processor:
        # Standard transformers path
        dataset = ImageListDataset(image_paths, lambda img: img)

        dim = 4096  # DINOv3-7B hidden dim
        all_feats = np.zeros((len(image_paths), dim * 2), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                imgs = []
                idxs = []
                for j, p in enumerate(batch_paths):
                    try:
                        img = Image.open(p).convert("RGB")
                        imgs.append(img)
                        idxs.append(i + j)
                    except Exception:
                        imgs.append(Image.new("RGB", (518, 518)))
                        idxs.append(i + j)

                if not imgs:
                    continue

                inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
                outputs = model(**inputs)

                cls_feats = outputs.last_hidden_state[:, 0, :]
                patch_feats = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
                combined = torch.cat([cls_feats, patch_feats], dim=1)
                combined = F.normalize(combined.float(), dim=1)

                for k, idx in enumerate(idxs):
                    all_feats[idx] = combined[k].cpu().numpy()

                if (i // batch_size) % 100 == 0:
                    print(f"    DINOv3: {i}/{len(image_paths)}")

        del model
    else:
        # Fallback: if model can't be loaded via transformers,
        # use a simpler approach or raise
        raise RuntimeError(
            "DINOv3-7B could not be loaded via transformers. "
            "Please verify the model files in " + model_path
        )

    gc.collect()
    torch.cuda.empty_cache()
    return all_feats


def extract_siglip2(image_paths, batch_size=16):
    """Extract SigLIP2-Giant features."""
    from transformers import AutoModel, AutoProcessor
    cfg = BACKBONES["siglip2"]
    print(f"  Loading SigLIP2-Giant from {cfg['model_path']}...")

    processor = AutoProcessor.from_pretrained(cfg["model_path"])
    model = AutoModel.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    )
    # SigLIP2 has vision_model
    vision_model = model.vision_model.to(DEVICE).eval()

    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"]),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageListDataset(image_paths, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_feats = np.zeros((len(image_paths), cfg["dim"]), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(DEVICE)
            if DEVICE.type == "cuda":
                imgs = imgs.half()
            out = vision_model(imgs)
            # Pool: use CLS or mean of last_hidden_state
            feats = out.last_hidden_state[:, 0, :]  # CLS token
            feats = F.normalize(feats.float(), dim=1)
            all_feats[idxs.numpy()] = feats.cpu().numpy()

    del model, vision_model
    gc.collect()
    torch.cuda.empty_cache()
    return all_feats


def extract_megadesc(image_paths, batch_size=32):
    """Extract MegaDescriptor-L-384 features."""
    import timm
    from torchvision import transforms

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"

    print(f"  Loading MegaDescriptor-L-384...")
    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
    model = model.to(DEVICE).eval()
    if DEVICE.type == "cuda":
        model = model.half()

    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageListDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    cfg = BACKBONES["megadesc"]
    all_feats = np.zeros((len(image_paths), cfg["dim"]), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(DEVICE)
            if DEVICE.type == "cuda":
                imgs = imgs.half()
            feats = model(imgs)
            feats = F.normalize(feats.float(), dim=1)
            all_feats[idxs.numpy()] = feats.cpu().numpy()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return all_feats


def check_bioclip_ready():
    """Check if BioCLIP 2.5 model weights are available."""
    model_dir = BACKBONES["bioclip25"]["model_path"]
    if not os.path.isdir(model_dir):
        return False
    # Check for model weight files
    import glob
    safetensors = glob.glob(os.path.join(model_dir, "*.safetensors"))
    bins = glob.glob(os.path.join(model_dir, "*.bin"))
    return len(safetensors) > 0 or len(bins) > 0


def extract_all_features(meta, force_reextract=False):
    """Extract or load cached features from all available backbones."""
    all_paths = [os.path.join(DATA_DIR, p) for p in meta['path'].values]
    all_ids = [str(x) for x in meta['image_id'].values]
    n = len(all_paths)

    features = {}

    # Build extractor list — BioCLIP is optional
    extractors = {}
    if check_bioclip_ready():
        extractors["bioclip25"] = extract_bioclip25
        print("  [OK] BioCLIP 2.5 model available")
    else:
        print("  [SKIP] BioCLIP 2.5 not ready (still downloading?), skipping")
    extractors["dinov3"] = extract_dinov3
    extractors["siglip2"] = extract_siglip2
    extractors["megadesc"] = extract_megadesc

    for name, extractor in extractors.items():
        cache_path = os.path.join(FEAT_CACHE, f"{name}_features.npz")
        if os.path.exists(cache_path) and not force_reextract:
            print(f"  Loading cached {name} features...")
            data = np.load(cache_path, allow_pickle=True)
            features[name] = np.array(data['feats'], dtype=np.float32)
            cached_ids = [str(x) for x in data['ids']]
            assert cached_ids == all_ids, f"ID mismatch in {name} cache!"
            print(f"    {name}: {features[name].shape}")
        else:
            print(f"  Extracting {name} features...")
            t0 = time.time()
            try:
                feats = extractor(all_paths)
                features[name] = feats
                print(f"    {name}: {feats.shape}, took {time.time()-t0:.0f}s")
                np.savez_compressed(cache_path, ids=all_ids, feats=feats)
                print(f"    Saved to {cache_path}")
            except Exception as e:
                print(f"    ERROR extracting {name}: {e}")
                print(f"    Skipping {name}")

    if len(features) == 0:
        raise RuntimeError("No features extracted! Cannot proceed.")
    print(f"\n  Successfully loaded {len(features)} backbones: {list(features.keys())}")
    return features, all_ids


# ============================================================
# PROJECTION HEAD WITH STRONG REGULARIZATION
# ============================================================
class ProjectionHead(nn.Module):
    """2-layer MLP with dropout and layer norm for better generalization."""
    def __init__(self, in_dim, hidden_dim=1024, out_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020)."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [N, D] L2-normalized
        labels: [N] integer labels
        """
        device = features.device
        N = features.shape[0]

        # Similarity matrix
        sim = features @ features.T / self.temperature  # [N, N]

        # Mask: same label = positive
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()  # [N, N]
        self_mask = 1.0 - torch.eye(N, device=device)
        pos_mask = pos_mask * self_mask  # exclude self

        # Number of positives per sample
        n_pos = pos_mask.sum(dim=1)  # [N]

        # For stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Log-sum-exp of all non-self similarities
        exp_sim = torch.exp(sim) * self_mask
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of positive log-probs
        log_prob = sim - log_sum_exp  # [N, N]
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)

        # Only compute loss for samples with at least 1 positive
        valid = n_pos > 0
        loss = -mean_log_prob_pos[valid].mean()
        return loss


# ============================================================
# PK SAMPLER FOR BALANCED BATCHES
# ============================================================
class PKSampler:
    """Sample P identities, K images each, per batch."""
    def __init__(self, labels, P=16, K=4):
        self.P = P
        self.K = K
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        # Only use labels with >= 2 images (need at least 1 positive)
        self.valid_labels = [l for l, idxs in self.label_to_indices.items() if len(idxs) >= 2]
        if len(self.valid_labels) == 0:
            self.valid_labels = list(self.label_to_indices.keys())

    def sample(self):
        """Return indices for one batch."""
        selected_labels = np.random.choice(self.valid_labels,
                                           size=min(self.P, len(self.valid_labels)),
                                           replace=False)
        indices = []
        for label in selected_labels:
            pool = self.label_to_indices[label]
            if len(pool) >= self.K:
                chosen = np.random.choice(pool, size=self.K, replace=False)
            else:
                chosen = np.random.choice(pool, size=self.K, replace=True)
            indices.extend(chosen)
        return indices


# ============================================================
# TRAINING: SUPCON + PSEUDO-LABELING
# ============================================================
def train_supcon_projector(
    features, labels, in_dim,
    n_epochs=30, lr=2e-4, wd=1e-3, temp=0.1,
    P=16, K=4, dropout=0.3, verbose=True
):
    """
    Train a projection head with SupCon loss.
    Returns the trained projection head.
    """
    proj = ProjectionHead(in_dim, PROJ_HIDDEN, PROJ_OUT, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = SupConLoss(temperature=temp)
    sampler = PKSampler(labels, P=P, K=K)

    features_t = torch.from_numpy(features).float().to(DEVICE)
    labels_t = torch.from_numpy(labels).long().to(DEVICE)

    best_loss = float('inf')
    for epoch in range(n_epochs):
        proj.train()
        epoch_losses = []

        n_batches = max(len(features) // (P * K), 10)
        for _ in range(n_batches):
            indices = sampler.sample()
            batch_feats = features_t[indices]
            batch_labels = labels_t[indices]

            z = proj(batch_feats)
            loss = criterion(z, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss

    proj.eval()
    return proj


def project_features(proj, features, batch_size=1024):
    """Project features through trained projection head."""
    proj.eval()
    features_t = torch.from_numpy(features).float()
    all_projected = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features_t[i:i+batch_size].to(DEVICE)
            z = proj(batch)
            all_projected.append(z.cpu().numpy())
    return np.vstack(all_projected)


# ============================================================
# t-SNE DIMENSIONALITY REDUCTION
# ============================================================
def tsne_reduce(features, n_components=32, perplexity=30, verbose=True):
    """
    Reduce features via t-SNE.
    For large N, use PCA pre-reduction then t-SNE.
    Paper proved: t-SNE improves clustering by 26-38% over raw features.
    """
    n = features.shape[0]
    if verbose:
        print(f"  t-SNE: {features.shape} → {n_components}D (perplexity={perplexity})")

    # PCA pre-reduction if dim > 256
    if features.shape[1] > 256:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(256, features.shape[1], n - 1))
        features = pca.fit_transform(features)
        if verbose:
            print(f"    PCA pre-reduction → {features.shape[1]}D (var={pca.explained_variance_ratio_.sum():.3f})")

    # Adjust perplexity for small datasets
    effective_perplexity = min(perplexity, max(5, n // 4))

    tsne = TSNE(
        n_components=n_components,
        perplexity=effective_perplexity,
        learning_rate='auto',
        init='pca' if n > 50 else 'random',
        max_iter=1000,
        random_state=42,
        method='barnes_hut' if n_components <= 3 else 'exact',
    )

    # t-SNE only supports 2D or 3D with barnes_hut; for higher dims use exact
    if n_components > 3:
        tsne.method = 'exact'
        if n > 3000:
            # exact t-SNE is O(N²), too slow for large datasets
            # Use UMAP instead as fallback
            try:
                import umap
                if verbose:
                    print(f"    Using UMAP (N={n} too large for exact t-SNE at {n_components}D)")
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42,
                )
                return reducer.fit_transform(features)
            except ImportError:
                if verbose:
                    print("    UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components)
                return pca.fit_transform(features)

    return tsne.fit_transform(features)


def tsne_reduce_2d(features, perplexity=30):
    """t-SNE to 2D (for HDBSCAN, which works best in low dimensions)."""
    n = features.shape[0]

    # PCA pre-reduction
    if features.shape[1] > 64:
        from sklearn.decomposition import PCA
        pca_dim = min(64, n - 1)
        pca = PCA(n_components=pca_dim)
        features = pca.fit_transform(features)
        print(f"    PCA→{pca_dim}D (var={pca.explained_variance_ratio_.sum():.3f})")

    effective_perplexity = min(perplexity, max(5, n // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate='auto',
        init='pca' if n > 50 else 'random',
        max_iter=1500,
        random_state=42,
    )
    return tsne.fit_transform(features)


# ============================================================
# HDBSCAN CLUSTERING
# ============================================================
def hdbscan_cluster(features_2d, min_cluster_size=10, min_samples=3):
    """
    HDBSCAN clustering on 2D features.
    Paper showed: HDBSCAN >> DBSCAN for automatic cluster discovery.
    """
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(features_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"    HDBSCAN: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(labels)*100:.1f}%)")
    return labels, clusterer


def assign_noise_points(features_2d, labels):
    """Assign noise points (-1) to nearest cluster."""
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels

    labels = labels.copy()
    cluster_ids = set(labels[~noise_mask])
    if len(cluster_ids) == 0:
        # All points are noise, assign to single cluster
        return np.zeros_like(labels)

    # Compute cluster centroids
    centroids = {}
    for c in cluster_ids:
        centroids[c] = features_2d[labels == c].mean(axis=0)

    centroid_ids = list(centroids.keys())
    centroid_feats = np.array([centroids[c] for c in centroid_ids])

    # Assign each noise point to nearest centroid
    noise_indices = np.where(noise_mask)[0]
    for idx in noise_indices:
        dists = np.linalg.norm(centroid_feats - features_2d[idx], axis=1)
        labels[idx] = centroid_ids[np.argmin(dists)]

    return labels


# ============================================================
# PSEUDO-LABELING
# ============================================================
def generate_pseudo_labels(proj, features, confidence_threshold=0.7):
    """
    Project features → cosine similarity → simple clustering → confidence filtering.
    Returns: pseudo_labels (int array, -1 for uncertain), confidence scores
    """
    projected = project_features(proj, features)

    # t-SNE → 2D for clustering
    reduced = tsne_reduce_2d(projected, perplexity=min(30, len(projected) // 4))

    # HDBSCAN
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(reduced)

    # Confidence = HDBSCAN probability
    probs = clusterer.probabilities_

    # Mark low-confidence as uncertain
    pseudo = labels.copy()
    pseudo[probs < confidence_threshold] = -1

    return pseudo, probs


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 70)
    print("AnimalCLEF2026 V23: Semi-Supervised + t-SNE + HDBSCAN")
    print("=" * 70)
    t_start = time.time()

    # ---- Load metadata ----
    meta = pd.read_csv(f"{DATA_DIR}/metadata.csv")
    sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
    print(f"Metadata: {len(meta)} rows, Submission: {len(sample_sub)} rows")

    # Build index
    all_ids = [str(x) for x in meta['image_id'].values]
    id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}

    # ---- Stage 0: Feature Extraction ----
    print("\n" + "=" * 70)
    print("STAGE 0: Feature Extraction (4 Backbones)")
    print("=" * 70)

    features, _ = extract_all_features(meta)

    # Concatenate all backbone features
    backbone_names = sorted(features.keys())
    print(f"\nConcatenating features from: {backbone_names}")
    concat_feats = np.concatenate([features[name] for name in backbone_names], axis=1)
    print(f"Concatenated feature dim: {concat_feats.shape[1]}")
    total_dim = concat_feats.shape[1]

    # ---- Stage 1: Per-Species Processing ----
    print("\n" + "=" * 70)
    print("STAGE 1: Per-Species Semi-Supervised Learning")
    print("=" * 70)

    submission_dict = {}

    for species in SPECIES_ORDER:
        print(f"\n{'='*50}")
        print(f"Species: {species}")
        print(f"{'='*50}")

        sp_mask = meta['dataset'] == species
        sp_meta = meta[sp_mask].reset_index(drop=True)
        sp_indices = np.where(sp_mask.values)[0]
        sp_feats = concat_feats[sp_indices]

        train_mask = sp_meta['split'] == 'train'
        test_mask = sp_meta['split'] == 'test'
        train_indices = np.where(train_mask.values)[0]
        test_indices = np.where(test_mask.values)[0]

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        print(f"  Train: {n_train}, Test: {n_test}")

        if n_train > 0:
            # --- Species with training data ---
            train_feats = sp_feats[train_indices]
            test_feats = sp_feats[test_indices]

            # Encode identity labels
            train_identities = sp_meta.loc[train_mask, 'identity'].values
            unique_ids = sorted(set(train_identities))
            id_to_label = {id_: i for i, id_ in enumerate(unique_ids)}
            train_labels = np.array([id_to_label[x] for x in train_identities])
            n_identities = len(unique_ids)
            print(f"  Known identities: {n_identities}")

            # ---- Phase A: Initial SupCon on training data ----
            print(f"\n  Phase A: SupCon on labeled training data")
            proj = train_supcon_projector(
                train_feats, train_labels, total_dim,
                n_epochs=SUPCON_EPOCHS_SUPERVISED,
                lr=SUPCON_LR, wd=SUPCON_WD, temp=SUPCON_TEMP,
                P=min(16, n_identities), K=4, dropout=DROPOUT_RATE,
            )

            # Evaluate on training set
            train_projected = project_features(proj, train_feats)
            train_2d = tsne_reduce_2d(train_projected)
            import hdbscan as hdb
            train_clusterer = hdb.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER[species], min_samples=HDBSCAN_MIN_SAMPLES)
            train_pred = train_clusterer.fit_predict(train_2d)
            train_pred = assign_noise_points(train_2d, train_pred)
            train_ari = adjusted_rand_score(train_labels, train_pred)
            print(f"  Train ARI after Phase A: {train_ari:.4f}")

            # ---- Phase B & C: Iterative Pseudo-Labeling ----
            all_sp_feats = sp_feats  # train + test together
            for round_idx in range(N_PSEUDO_ROUNDS):
                print(f"\n  Phase B/C Round {round_idx+1}/{N_PSEUDO_ROUNDS}: Pseudo-labeling")

                # Project all images
                all_projected = project_features(proj, all_sp_feats)

                # t-SNE → 2D
                all_2d = tsne_reduce_2d(all_projected, perplexity=min(30, len(all_sp_feats) // 4))

                # HDBSCAN on all images
                clusterer = hdb.HDBSCAN(
                    min_cluster_size=HDBSCAN_MIN_CLUSTER[species],
                    min_samples=HDBSCAN_MIN_SAMPLES
                )
                all_pseudo = clusterer.fit_predict(all_2d)
                all_probs = clusterer.probabilities_

                # Assign noise points
                all_pseudo = assign_noise_points(all_2d, all_pseudo)

                # Compute number of test clusters
                test_pseudo = all_pseudo[len(train_indices):]  # test part
                n_test_clusters = len(set(test_pseudo))
                print(f"    Pseudo clusters (test): {n_test_clusters}, singletons: {(np.bincount(test_pseudo + (1 if test_pseudo.min() < 0 else 0)) == 1).sum() if len(test_pseudo) > 0 else 0}")

                # Create combined labels for retraining:
                # - Real labels for training data
                # - High-confidence pseudo-labels for test data
                combined_labels = np.full(len(all_sp_feats), -1, dtype=np.int64)
                combined_labels[:len(train_indices)] = train_labels

                # Map test pseudo-labels to continue from max train label
                max_train_label = train_labels.max()
                test_offset = max_train_label + 1
                test_pseudo_shifted = test_pseudo + test_offset

                # Only use high-confidence pseudo-labels
                test_probs = all_probs[len(train_indices):]
                high_conf = test_probs >= PSEUDO_CONFIDENCE_THRESHOLD
                combined_labels[len(train_indices):] = np.where(
                    high_conf, test_pseudo_shifted, -1
                )

                # Filter out -1 (uncertain) for training
                valid = combined_labels >= 0
                n_valid = valid.sum()
                print(f"    Valid samples for retraining: {n_valid}/{len(all_sp_feats)} "
                      f"({n_valid/len(all_sp_feats)*100:.1f}%)")

                if n_valid > SUPCON_BATCH:
                    # Retrain with combined real + pseudo labels
                    proj = train_supcon_projector(
                        all_sp_feats[valid], combined_labels[valid], total_dim,
                        n_epochs=SUPCON_EPOCHS_PSEUDO,
                        lr=SUPCON_LR * 0.5,  # lower LR for refinement
                        wd=SUPCON_WD * 2,    # stronger regularization
                        temp=SUPCON_TEMP,
                        P=min(16, len(set(combined_labels[valid]))),
                        K=4, dropout=DROPOUT_RATE,
                    )

                    # Re-evaluate on training set
                    train_projected = project_features(proj, train_feats)
                    train_2d = tsne_reduce_2d(train_projected)
                    train_pred2 = hdb.HDBSCAN(
                        min_cluster_size=HDBSCAN_MIN_CLUSTER[species],
                        min_samples=HDBSCAN_MIN_SAMPLES
                    ).fit_predict(train_2d)
                    train_pred2 = assign_noise_points(train_2d, train_pred2)
                    train_ari2 = adjusted_rand_score(train_labels, train_pred2)
                    print(f"    Train ARI after Round {round_idx+1}: {train_ari2:.4f}")

            # ---- Final clustering for test data ----
            print(f"\n  Final clustering for {species}")
            all_projected = project_features(proj, all_sp_feats)
            all_2d = tsne_reduce_2d(all_projected, perplexity=min(30, len(all_sp_feats) // 4))

            all_labels = hdb.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER[species],
                min_samples=HDBSCAN_MIN_SAMPLES
            ).fit_predict(all_2d)
            all_labels = assign_noise_points(all_2d, all_labels)

            # Extract test labels
            test_cluster_labels = all_labels[len(train_indices):]

        else:
            # --- TexasHornedLizards: no training data ---
            print(f"  Zero-shot clustering (no training data)")
            test_feats = sp_feats  # all are test

            # Use DINOv3 features primarily (best for zero-shot from paper)
            dinov3_feats = features["dinov3"][sp_indices] if "dinov3" in features else test_feats

            # t-SNE → 2D
            print(f"  t-SNE reduction for zero-shot clustering")
            test_2d = tsne_reduce_2d(l2_normalize(dinov3_feats))

            # HDBSCAN
            import hdbscan as hdb
            test_cluster_labels = hdb.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER[species],
                min_samples=HDBSCAN_MIN_SAMPLES
            ).fit_predict(test_2d)
            test_cluster_labels = assign_noise_points(test_2d, test_cluster_labels)

        # ---- Map to submission format ----
        test_image_ids = sp_meta.loc[test_mask if n_train > 0 else sp_meta.index, 'image_id'].values
        unique_clusters = sorted(set(test_cluster_labels))
        cluster_map = {c: f"cluster_{species}_{i}" for i, c in enumerate(unique_clusters)}

        for img_id, cl in zip(test_image_ids, test_cluster_labels):
            submission_dict[str(img_id)] = cluster_map[cl]

        # Stats
        n_clusters = len(unique_clusters)
        counts = Counter(test_cluster_labels)
        singletons = sum(1 for c in counts.values() if c == 1)
        max_cluster = max(counts.values()) if counts else 0
        print(f"\n  {species} Final: {len(test_image_ids)} images → {n_clusters} clusters")
        print(f"    Singletons: {singletons} ({singletons/n_clusters*100:.0f}%)")
        print(f"    Largest cluster: {max_cluster}")

    # ---- Stage 4: Generate Submission ----
    print("\n" + "=" * 70)
    print("STAGE 4: Generating Submission")
    print("=" * 70)

    # Build submission from sample_submission template
    sub_ids = [str(x) for x in sample_sub['image_id'].values]
    sub_clusters = []
    missing = 0
    for img_id in sub_ids:
        if img_id in submission_dict:
            sub_clusters.append(submission_dict[img_id])
        else:
            sub_clusters.append(f"cluster_unknown_{missing}")
            missing += 1

    submission = pd.DataFrame({
        'image_id': sample_sub['image_id'].values,
        'cluster': sub_clusters,
    })

    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    print(f"Total: {len(submission)} images, {submission['cluster'].nunique()} unique clusters")
    if missing > 0:
        print(f"WARNING: {missing} images not found in predictions!")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("Done!")


if __name__ == "__main__":
    main()
