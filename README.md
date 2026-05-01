# AnimalCLEF2026-Solution

> **#1 Open-Source Solution for AnimalCLEF26 @ CVPR & CLEF 2026**
>
> **Private Leaderboard ARI: 0.31974** | Public LB: 0.24012
>
> Highest-scoring publicly available solution for the AnimalCLEF 2026 Kaggle competition
>
> **Achieved WITHOUT using the official 26GB animal training dataset**

---

## Final Competition Results (Private Leaderboard)

| Solution | Private Score | Public Score | Source |
| --- | --- | --- | --- |
| **Ours (This Repo)** | **0.31974** | **0.24012** | [GitHub](https://github.com/fan1344rwere/AnimalCLEF2026-Solution) |
| 2nd Open-Source (Fruit19/Lakhindar fork) | 0.26604 | 0.22475 | Kaggle Notebook |
| 3rd Open-Source (Lakhindar Pal) | 0.22760 | 0.23044 | Kaggle Notebook |
| Official Starter Baseline | ~0.19 | 0.19401 | Kaggle Notebook |

> We outperform the 2nd-best open-source solution by **+5.4 ARI points (20% relative improvement)** on the private leaderboard.

---

## Highlights

- **#1 among all publicly shared solutions** on both public and private leaderboard
- **No official 26GB dataset used** 鈥?purely foundation model features + lightweight SupCon projection
- **Single RTX 5090**, ~2 weeks development, full code & logs open-sourced
- **Private LB rose from 0.24 to 0.32** 鈥?proving the learned metric space generalizes well beyond the public test subset

---

## Visual Results

### Feature Space Visualization (t-SNE)

_Left: MegaDescriptor features clearly separate **species** but cannot distinguish **individuals**. Right: Within Sea Turtle, top-12 individual IDs are completely mixed 鈥?raw features have near-zero individual discriminability. This motivated our SupCon projection approach._

### Zero-Shot Challenge: Texas Horned Lizards

_Texas Horned Lizards has **zero** training images. These montages show the test set 鈥?clustering must rely entirely on transfer learning from other species._

---

## Competition Overview

**AnimalCLEF26 @ CVPR & CLEF 2026** (Kaggle) challenges participants to cluster test images by individual animal identity across 4 wildlife datasets. The evaluation metric is **Adjusted Rand Index (ARI)**.

- **Competition URL**: https://www.kaggle.com/competitions/animal-clef-2026
- **Task**: Discovery and Re-Identification of Individual Animals
- **Metric**: Adjusted Rand Index (ARI)
- **Deadline**: April 30, 2026
- **835 Entrants, 283 Participants, 230 Teams, 3,491 Submissions**

| Species | Train Images | Train IDs | Test Images | Key Challenge |
| --- | --- | --- | --- | --- |
| LynxID2025 | 2,957 | 77 | 946 | Orientation variation (left/right/front/back) |
| SalamanderID2025 | 1,388 | 587 | 689 | Extreme sparsity (median 1 img/identity) |
| SeaTurtleID2022 | 8,729 | 438 | 500 | Underwater quality variation |
| TexasHornedLizards | 0 | 0 | 274 | **Zero** training data |

---

## Best Solution: V22 SupCon (Private ARI = 0.31974)

Our best private-LB submission is **V22**: Supervised Contrastive Projection on 5-backbone foundation model features.

### Core Architecture

```
Input Image
    |
    v
[SAM2.1 Segmentation] --> Animal Mask
    |
    v
[5 Foundation Model Backbones (ALL FROZEN, no fine-tuning)]
    |-- DINOv3-ViT-7B (Meta, 2025)      --> 8192-dim
    |-- InternViT-6B (OpenGVLab)         --> 6400-dim
    |-- SigLIP2-Giant (Google)           --> 1536-dim
    |-- EVA02-CLIP-E+ (BAAI)            --> 1024-dim
    |-- MegaDescriptor-L-384 (Re-ID)    --> 1536-dim
    v
[Concatenation] --> 18,688-dim
    |
    v
[Per-Species SupCon Projection Head]
    |-- 2-layer MLP: 18688 -> 1024 -> 256
    |-- Supervised Contrastive Loss (tau=0.07)
    |-- 50 epochs, lr=3e-4
    v
[256-dim Projected Features]
    |
    v
[Cosine Similarity -> HAC Clustering]
    |
    v
Final Predictions (cluster_id per image)
```

### Why It Works

**Key Insight**: Raw foundation model features (even 7B-parameter models) have **near-zero individual discriminability**. A simple 2-layer SupCon projection head transforms these features into a highly discriminative metric space:

| Species | Before SupCon (Train ARI) | After SupCon (Train ARI) |
| --- | --- | --- |
| SeaTurtleID2022 | ~0.00 | **0.91** |
| SalamanderID2025 | ~0.00 | **0.91** |
| LynxID2025 | ~0.00 | **0.85** |

### V21: Foundation Model Ensemble + k-Reciprocal Re-ranking

**Pipeline:** `final_solutions/v21_foundation_ensemble.py`

1. **SAM2.1 Segmentation** 鈥?Animal masks for cleaner feature extraction
2. **4-Backbone Feature Extraction** (all frozen, no fine-tuning):
   - DINOv3-ViT-7B (Meta, 2025) 鈥?8192-dim
   - InternViT-6B (OpenGVLab) 鈥?6400-dim
   - SigLIP2-Giant (Google) 鈥?1536-dim
   - EVA02-CLIP-E+ (BAAI) 鈥?1024-dim
3. **Per-species weight optimization** via grid search on training set
4. **k-Reciprocal Re-ranking** (k1=20, k2=6) for refined similarity
5. **HAC (Agglomerative Clustering)** with per-species threshold
6. **4-pass post-processing**: split-big, merge-small, transitivity, anchor

### V22: Supervised Contrastive Projection Heads

**Pipeline:** `final_solutions/v22_supcon.py`

1. **Load V21 cached features** (DINOv3/InternViT/SigLIP2/EVA02)
2. **Add MegaDescriptor** as 5th backbone (has Re-ID inductive bias)
3. **Per-species SupCon projection head** on concatenated 5-backbone features (18,688-dim)
   - 2-layer MLP: input -> 1024 hidden -> 256 output
   - Supervised Contrastive Loss (temperature=0.07)
   - 50 epochs, lr=3e-4
4. **Cosine similarity on projected features -> HAC clustering**

---

## Repository Structure

```
.
|-- final_solutions/          # Best solutions (V21-V24)
|   |-- v21_foundation_ensemble.py   # V21: 4-backbone + SAM2 + k-Reciprocal
|   |-- v22_supcon.py                # V22: SupCon projection heads (BEST)
|   |-- v23_supcon.py                # V23: SupCon without InternViT
|   `-- v24_perbbone.py              # V24: Per-backbone SupCon
|
|-- early_versions/           # Evolution of approaches (V14-V30)
|   |-- v14_clean.py                 # First baseline (ARI=0.205)
|   |-- v15_local_matching.py        # + LightGlue local matching
|   |-- v16_calibrated.py            # 3-backbone calibrated (ARI=0.231)
|   |-- v17_arcface.py               # + ArcFace fine-tuning
|   |-- v18_joint.py                 # Joint train+test clustering
|   |-- v19_orientation_breakthrough.py  # Orientation-aware matching
|   |-- v20_paradigm_shift.py        # Paradigm shift
|   |-- v25_hybrid.py                # Hybrid approaches
|   `-- v30_wildlife_supcon.py       # Wildlife SupCon experiments
|
|-- logs/
|   `-- run_logs/             # Real training logs from GPU runs
|       |-- run_v14.log ~ run_v30b.log  # Full stdout from each version
|       `-- v21w_log.txt ~ v24_log.txt  # Final solution run logs
|
|-- figures/                  # Visualization results
|   |-- fig1_tsne.png               # t-SNE: species vs individual features
|   `-- texas_sheet_*.jpg           # Texas Horned Lizards montages
|
|-- scripts/                  # Analysis & visualization scripts
|   |-- gen_tsne.py                  # Generate t-SNE figure
|   |-- gen_figures.py               # Generate comparison figures
|   `-- analysis.py                  # Feature analysis utilities
|
|-- kaggle_notebooks/         # Kaggle notebook versions (V4-V15)
|-- pipeline_main.py          # Main Kaggle notebook pipeline
|-- pipeline_full.py          # Full pipeline (4-backbone ensemble)
|-- pipeline_offline.py       # Offline pipeline
`-- docs/                     # Development notes and records
```

---

## Version History & Score Progression

| Version | Public ARI | Private ARI | Method | Key Insight |
| --- | --- | --- | --- | --- |
| V14 | 0.205 | 鈥?| Mega+Miew, pure test clustering | First time beating baseline |
| V16 | 0.231 | 鈥?| 3-backbone weighted fusion | Per-species weight optimization |
| V17 | 0.225 | 鈥?| + ArcFace fine-tuning | Overfitting on Salamander |
| V18 | 0.117 | 鈥?| Joint train+test clustering | Disaster |
| **V22** | **0.24012** | **0.31974** | **5-backbone SupCon projection** | **#1 open-source, +33% on private** |

---

## Development Timeline

Developed over ~2 weeks of intensive iteration on a single RTX 5090 GPU. Full run logs in `logs/run_logs/`.

| Date | Version | What Happened | Result |
| --- | --- | --- | --- |
| **Apr 11** | V14 | First baseline: MegaDescriptor + MiewID | ARI=0.205 |
| **Apr 11** | V16 | 3-backbone weighted fusion + calibrated similarity | ARI=0.231 |
| **Apr 11** | V17 | ArcFace fine-tuning | ARI=0.225 (Salamander 0%) |
| **Apr 12** | V20 | Paradigm shift to DINOv3-7B + InternViT-6B + SigLIP2 + EVA02 | Foundation loaded |
| **Apr 12** | V21 | Foundation ensemble + SAM2 + k-Reciprocal | Structure solid |
| **Apr 12** | V22 | **SupCon projection on 5-backbone 18,688-dim** | **Public 0.24, Private 0.32** |
| **Apr 13** | V23-25 | Hybrid experiments | None beat V22 |
| **Apr 24** | V30 | WildlifeReID-10k pretraining | Lynx improved locally |

### Training Log Highlight (V22)

```
SeaTurtleID2022:
    Epoch 10/50: loss=0.7046
    Epoch 30/50: loss=0.2597
    Epoch 50/50: loss=0.1109
    SupCon projected train ARI: 0.9060   <-- from near-zero!

SalamanderID2025:
    Epoch 50/50: loss=0.0294
    SupCon projected train ARI: 0.9115   <-- 587 identities learned
```

---

## Key Findings

1. **Raw foundation features fail at individual re-ID** 鈥?Even 7B models (DINOv3, InternViT) cannot distinguish individuals. Train ARI near 0 with raw cosine similarity.

2. **SupCon projection is the breakthrough** 鈥?A 2-layer MLP with SupCon loss transforms useless features into discriminative ones. This is the single most important contribution.

3. **Multi-backbone diversity is critical** 鈥?5 backbones capture complementary information. No single backbone achieves the ensemble result.

4. **Official 26GB dataset NOT required** 鈥?Our approach uses only identity labels from the competition-provided metadata, not the raw 26GB image dataset. Foundation models provide sufficient visual features.

5. **Private LB validates generalization** 鈥?Score rising from 0.24 (public) to 0.32 (private) proves the SupCon metric space genuinely generalizes, not overfitting.

---

## Technical Stack

- **GPU**: NVIDIA RTX 5090 (32GB GDDR7)
- **Framework**: PyTorch 2.8.0 + CUDA 12.8
- **Key Libraries**: timm, open_clip_torch, wildlife-datasets, wildlife-tools, hdbscan, kornia
- **Models**: DINOv3-7B, InternViT-6B, SigLIP2-Giant, EVA02-CLIP-E+, MegaDescriptor-L-384, MegaDescriptor-DINOv2-518, MiewID, SAM2.1

---

## How to Reproduce

### Prerequisites

```bash
pip install torch torchvision timm open_clip_torch safetensors hdbscan
pip install wildlife-datasets wildlife-tools
pip install kornia  # for SAM2 segmentation
```

### Running

1. Download competition data from Kaggle
2. Download model weights (DINOv3, SigLIP2, EVA02, MegaDescriptor, etc.)
3. Run V21 for feature extraction: `python final_solutions/v21_foundation_ensemble.py`
4. Run V22 for SupCon training: `python final_solutions/v22_supcon.py`
5. Combine per-species best results from V21 and V22 submissions

> **Note**: Paths in the scripts are configured for AutoDL server. Modify `BASE`, `DATA_DIR`, `MODEL_DIR` etc. for your environment.

---

## Citation

If you find this solution helpful, please star this repository and cite:

```
@misc{animalclef2026_top1_opensource,
  title={#1 Open-Source Solution for AnimalCLEF26 @ CVPR & CLEF 2026},
  author={fan1344rwere},
  year={2026},
  url={https://github.com/fan1344rwere/AnimalCLEF2026-Solution},
  note={Private LB ARI=0.31974, achieved without official 26GB dataset}
}
```

---

## References

- MegaDescriptor (WACV 2024 Best Paper)
- WildFusion: Multi-score calibration
- DINOv3 (Meta, 2025)
- WildlifeReID-10k Dataset
- AnimalCLEF2025 Top Solutions

## License

MIT License

## Acknowledgments

- Competition organizers: Wildlife Datasets team
- Compute: AutoDL Cloud GPU (RTX 5090)

---

**Keywords**: AnimalCLEF2026, AnimalCLEF26, CVPR 2026, CLEF 2026, Kaggle competition solution, animal re-identification, wildlife re-ID, individual animal identification, supervised contrastive learning, SupCon, foundation model ensemble, DINOv3, InternViT, SigLIP2, EVA02, MegaDescriptor, clustering, Adjusted Rand Index, ARI, open-source solution, top solution, best score, private leaderboard, FGVC13
