# AnimalCLEF2026 比赛攻坚记录

## 比赛信息
- **比赛**: AnimalCLEF26 @ CVPR & CLEF
- **任务**: 个体动物发现与重新识别（聚类任务）
- **评价指标**: Adjusted Rand Index (ARI)
- **截止**: 2026年4月30日
- **当前排名Top5门槛**: 0.528
- **我们最高分**: 0.231 (V16)

## 数据概况

| 物种 | 训练图 | 训练个体 | 中位imgs/id | 测试图 | 关键问题 |
|------|--------|---------|------------|--------|---------|
| LynxID2025 | 2957 | 77 | 17 | 946 | orientation混合(left/right/front/back) |
| SalamanderID2025 | 1388 | 587 | **1** | 689 | **310个体只1张图！** 87%聚类为singleton |
| SeaTurtleID2022 | 8729 | 438 | 13 | 500 | 数据最充足，效果最好 |
| TexasHornedLizards | 0 | 0 | N/A | 274 | 零训练数据 |

### Orientation分布
- **Lynx训练**: left=1258, right=1175, front=163, back=131, unknown=230
- **Lynx测试**: right=440, left=418, front=37, back=36
- **Salamander训练**: top=784, right=600
- **Salamander测试**: top=391, right=296

## 提交记录

| 版本 | 分数 | 方法 | 问题诊断 |
|------|------|------|---------|
| V14 | 0.205 | Mega+Miew, 纯test聚类, 每物种阈值grid search | 首次超baseline(0.194)，纯聚类思路正确 |
| V15 | 0.068 | +DINOv2+ALIKED+LightGlue局部匹配 | local score直接覆盖similarity matrix破坏分布 |
| V16 | **0.231** | 3backbone(Mega+Miew+DINOv2), 每物种权重搜索, 校准局部匹配 | **最高分**。Salamander仍87% singleton |
| V17 | 0.225 | +ArcFace微调 | Salamander 0%acc过拟合，SeaTurtle train ARI 0.91但没泛化 |
| V18 | 0.117 | 联合train+test聚类+训练约束注入 | 约束太强，train ARI=0.99但test崩溃 |

## V16每物种最优模型权重（训练集grid search）

| 物种 | MegaDescriptor | MiewID | DINOv2 | Train ARI |
|------|---------------|--------|--------|-----------|
| Lynx | **0.90** | 0.09 | 0.01 | 0.152 |
| Salamander | 0.20 | 0.00 | **0.80** | 0.118 |
| SeaTurtle | 0.50 | 0.20 | 0.30 | 0.860 |

## V16聚类输出分析

```
Lynx:       946 imgs → 47 cl, max=343! singletons=32%  → 严重under-clustering
Salamander: 689 imgs → 573 cl, max=8,  singletons=87%  → 几乎全是singleton
SeaTurtle:  500 imgs → 261 cl, max=8,  singletons=57%  → 相对合理
TexasHorned:274 imgs → 93 cl,  max=15, singletons=44%  → 相对合理
```

## 关键技术栈（服务器已安装）

- **GPU**: RTX 5090 32GB GDDR7
- **Python**: 3.12.3 (miniconda3)
- **PyTorch**: 2.8.0+cu128
- **已安装**: timm, hdbscan, kornia, lightglue, wildlife-datasets, wildlife-tools, open_clip_torch, safetensors
- **HF镜像**: hf-mirror.com
- **服务器**: AutoDL Cloud GPU

## 前沿方法研究

### 2025年AnimalCLEF冠军方案
- **DataBoom (Top1, 0.713)**: 多种局部特征匹配+MiewID预筛选，不用MegaDescriptor做最终分数
- **webmaking (Top2, 0.675)**: WildFusion+XGBoost(MegaDescriptor+MiewID特征), 双流meta-algorithm
- **共同点**: 全局-局部融合是标配, score calibration(isotonic regression)是关键

### 核心工具
- **MegaDescriptor-L-384**: Swin Transformer, ArcFace训练, 动物Re-ID专用 (WACV 2024 Best Paper)
- **MiewID-MSV3**: EfficientNetV2, 另一个强Re-ID模型
- **WildFusion**: 校准融合deep+local scores, mean acc 84% on 17 datasets
- **DINOv3**: Meta 2025, ViT-7B teacher→distilled ViT-S/B/L/H+, instance retrieval比DINOv2强+10.9 GAP

### 公开Notebook最高分
- 0.2848 (LB=0.2848 研究数据集)
- 说明0.28左右是纯全局特征+标准聚类的天花板

## 深度诊断：0.23→0.53的差距

### 根因分析
1. **Orientation完全被忽略** — 同一只动物left和right拍的照片完全不同，混在一起算similarity必然失败
2. **局部特征匹配用错了** — 作为supplement(β很小)而非primary。2025冠军是local matching为主
3. **Salamander数据极端稀疏** — 中位1张/个体，任何模型级方法都无效，只能靠pattern matching
4. **聚类策略有缺陷** — 纯test聚类没有锚点；联合聚类约束太强过拟合

### 突破路径

**1. Orientation-aware matching (预计+0.10~0.15)**
- 只比较同orientation: left↔left, right↔right
- 跨orientation设为低相似度
- 减少50%+错误匹配

**2. 局部特征匹配作为PRIMARY (预计+0.10~0.15)**
- 先全局prefilter(top-100 within same orientation)
- 再LightGlue算最终相似度
- Salamander斑纹/Lynx毛皮pattern靠这个

**3. Representative matching + 残差聚类 (预计+0.05~0.10)**
- 每个训练个体→representative embedding(同orientation的mean)
- 测试图匹配最近representative
- 高置信度→分配；低置信度→组内自聚类

## 明日V19计划

```python
For each species with training data:
    1. 按orientation分组
    2. 对每组:
        a. 计算training representative per identity
        b. Test → nearest representative (same orientation)
        c. sim > threshold → assign
        d. else → "new"
    3. 跨orientation合并已分配的
    4. "new"在组内聚类
    5. Ambiguous pairs做LightGlue验证

For TexasHornedLizards:
    Local matching + clustering
```

**5次提交分配**:
1. V19a: orientation-aware + representative matching (不含local)
2. V19b: +local matching verification
3. V19c: 调参
4-5. 迭代

## 文件位置
- 代码: `v14_clean.py` ~ `v18_joint.py`
- 比赛数据: AnimalCLEF2026 Kaggle dataset

## 排行榜参考 (2026/4/11)

| 排名 | 队伍 | ARI | 提交数 |
|------|------|-----|--------|
| 1 | Ram Lab | 0.720 | 74 |
| 2 | M.I.A+6 | 0.707 | 198 |
| 3 | evnsnclr | 0.665 | 12 |
| 4 | Roberto Alcaraz | 0.586 | 8 |
| 5 | Dominick Tan | 0.529 | 9 |
| 6 | Sreevaatsav Bavana | 0.517 | 49 |
| 7 | VIPL-VSU | 0.497 | 48 |
| ... | ... | ... | ... |
| ~30+ | **我们** | **0.231** | 5 |
