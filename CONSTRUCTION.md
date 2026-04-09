# DualTreeVLA — 架构与设计文档

> **「双树」命名由来**：DualTreeVLA 同时运行两棵树——
> - **视觉树**（Visual Tree）：SGMTS 每帧在 patch 特征图上动态构建语义加权最大生成树，沿 BFS 序进行 Mamba 扫描，目标是 **< 5 ms / 帧**（224×224 输入，RTX 4090）
> - **记忆树**（Memory Tree）：HMT 跨帧在线增量维护层级语义记忆，分支/合并操作均为 O(depth) 复杂度，目标是 **< 1 ms / 帧** 增量更新
>
> 两棵树并行：视觉树提供当前帧的空间语义特征，记忆树提供跨时间尺度的任务上下文，共同驱动动作头完成长时程操作。

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构框图](#2-整体架构框图)
3. [架构组件详解](#3-架构组件详解)
   - 3.1 [JumpAwareHead（纯动作跳变感知头）](#31-jumpawarehead纯动作跳变感知头)
   - 3.2 [SGMTS（语义引导 Mamba 树扫描编码器）](#32-sgmts语义引导-mamba-树扫描编码器)
   - 3.3 [HierarchicalMemoryTree（层级记忆树）](#33-hierarchicalmemory-tree层级记忆树)
   - 3.4 [CrossModalFusion（跨模态融合）](#34-crossmodalfusion跨模态融合)
   - 3.5 [FlowMatchingActionHead（流匹配动作头）](#35-flowmatchingactionhead流匹配动作头)
4. [两阶段训练流程](#4-两阶段训练流程)
5. [核心设计原则：损失函数完全分离](#5-核心设计原则损失函数完全分离)
6. [损失函数详解](#6-损失函数详解)
   - 6.1 [预训练损失](#61-预训练损失)
   - 6.2 [Phase 1/2 损失](#62-phase-12-损失)
7. [模型 Forward API](#7-模型-forward-api)
8. [项目文件结构](#8-项目文件结构)
9. [快速开始](#9-快速开始)

---

## 1. 项目概述

DualTreeVLA 是一个面向长时程机器人操作的视觉-语言-动作模型（VLA），其名称中的**「双树」**明确指代同时维护的两棵树结构：

| 树 | 全称 | 作用 | 延迟目标 |
|----|------|------|----------|
| **视觉树** | Semantic-Guided Visual Tree（SGMTS） | 每帧在 patch 空间动态构建语义加权 MST，Mamba BFS 扫描输出空间语义特征 | **< 5 ms / 帧** |
| **记忆树** | Hierarchical Memory Tree（HMT） | 跨帧增量维护层级语义记忆，O(depth) 分支/合并，记录子任务边界与历史上下文 | **< 1 ms / 帧**（增量更新） |

两棵树以**最低推理延迟**并行执行：视觉树负责当前帧的空间语义编码，记忆树负责全局任务记忆读取，共同输入动作头完成连续动作生成。整体推理延迟目标：**单帧端到端 < 50 ms**（不含 LLM prefill）。

核心设计原则：

- **双树并行**：Visual Tree（每帧重建）与 Memory Tree（增量更新）解耦执行，互不阻塞
- **轻量视觉树扫描**：SGMTS 避免 Transformer 注意力的 $O(P^2)$ 复杂度，改用线性 Tree-SSM $O(P)$ 扫描；MST 构建采用 CPU 上 Kruskal 算法与 GPU 特征提取流水线并行
- **O(depth) 记忆树更新**：HMT 每帧仅更新活跃叶子的 Welford 均值（Merge），分支时仅沿路径更新祖先链，节点数控制在 $\leq 4$ 层
- 用 JumpAwareHead 检测动作突变点（= 分支点），自动分割子任务边界
- 用 FlowMatching 动作头生成连续动作序列

---

## 2. 整体架构框图

### 2.1 双树并行结构（低延迟核心）

```
 ╔══════════════════════════════════════════════════════════╗
 ║              DualTreeVLA — 双树并行架构                   ║
 ║                                                          ║
 ║  ┌─────────────────────────────────────┐                 ║
 ║  │  🌿 视觉树（Visual Tree）— SGMTS    │  目标: <5ms/帧  ║
 ║  │  patch MST + BFS + Mamba Tree-SSM  │  O(P) 线性复杂度 ║
 ║  └─────────────────────────────────────┘                 ║
 ║                   ↕ 语义引导（s_top）                     ║
 ║  ┌─────────────────────────────────────┐                 ║
 ║  │  🌳 记忆树（Memory Tree）— HMT     │  目标: <1ms/帧  ║
 ║  │  增量 Merge/Branch，O(depth) 更新  │  max depth=4     ║
 ║  └─────────────────────────────────────┘                 ║
 ╚══════════════════════════════════════════════════════════╝
```

### 2.2 单帧 Forward 数据流

```
  ┌──────────────────────────┐              ┌──────────────────────┐
  │      任务描述 text          │              │    RGB 帧 I_t         │
  │      (List[str])           │              │    (B, C, H, W)      │
  └──┬───────────────────┬────┘              └──────────┬───────────┘
     │ g_task            │ task_tokens                  │
     ▼                   │                    ▼ [视觉树 <5ms]
  ┌──────────────────────────────────┐    ┌──────────────────────────────────────┐
  │  🌿 SGMTS（视觉树 / Visual Tree）│◄───┤  🌳 HMT（记忆树 / Memory Tree）      │
  │   语义加权MST + BFS + Tree-SSM   │s_top│  Merge(<1ms): Welford 更新 z_v      │
  │   O(P) Mamba 扫描，无 O(P²) Attn│    │  Branch+Elevate: O(depth) 路径更新  │
  └──────────────┬────────────────────┘    │    MLPElevation → s_abs              │
                 │ Z_v (B, N_v, d)          └──────────────▲───────────┬────────────┘
                 │◄──── task_tokens                        │ p_jump    │ s_top
                 │◄──── s_top (← HMT) ────────────────────┘            │
                 ▼                                                       │
  ┌──────────────────────────────────┐                                   │
  │           LLM (Qwen2.5)           │◄──────────────────────────────────┘
  │   [task_tokens ; Z_v ; s_top]     │   (s_top 作为高层语义 token)
  │   → H (B, N, d_llm)              │
  └──────────────┬────────────────────┘
                 │ H                   q_t (B, d_q)
                 └──────────────────────────┐
                               [H ; q_t]    ▼
                 ┌──────────────────────────────────┐
                 │         FlowMatchingHead           │
                 │     v_θ(x_t, t, [H ; q_t])        │
                 │    → â_{1..Ha} (B, Ha, d_a)       │
                 └──────────────┬───────────────────┘
                                │ â_1 (B, d_a)
                                ▼
                 ┌──────────────────────────────────┐
                 │           JumpAwareHead            │
                 │         动作 Mamba SSM             │──── p_jump ────────────────► HMT
                 │     → p_jump (B,) ∈ [0,1]        │
                 └──────────────────────────────────┘
```



---

## 3. 架构组件详解

### 3.1 JumpAwareHead（纯动作跳变感知头）

**文件**: `dual_tree_vla/model/semantic_jump_head.py`

**设计原则**：只消费动作序列，完全不接触语义特征。

```
输入:
  A_act  (B, L, d_a)  — 当前活跃节点的动作历史（max_len=64，尾部截断）
  a_new  (B, d_a)     — 当前时刻动作

前向流程:
  [A_act; a_new]  → Linear(d_a→d_inner)  → seq (B, L+1, d_inner)
  seq             → 一层 Mamba-SSM（ZOH离散化，选择性扫描）
  末尾隐状态 h_ctx (B, d_inner)
  h_ctx           → Linear(d_inner→1)
  
输出:
  p_jump (B,)  ∈ [0,1]   — 跳变概率（≥0.5 创建分支节点）
  logit  (B,)            — 原始 logit（供 BCEWithLogitsLoss）
```

**参数量**: 约 0.05M

#### Mamba SSM 数学公式（ZOH 离散化）

输入嵌入：

$$x_t = W_\text{embed}\, a_t \in \mathbb{R}^{d_\text{inner}}$$

输入依赖参数（选择性机制）：

$$\Delta_t = \text{softplus}(W_\Delta\, x_t) \in \mathbb{R}^{d_\text{inner}}, \quad B_t = W_B\, x_t \in \mathbb{R}^{d_\text{state}}, \quad C_t = W_C\, x_t \in \mathbb{R}^{d_\text{state}}$$

ZOH 离散化（$A$ 为固定 S4D-real 对角矩阵，$A_{nd} = -n,\ n\in[1,d_\text{state}]$）：

$$\bar{A}_t = \exp(\Delta_t \cdot A) \in \mathbb{R}^{d_\text{inner} \times d_\text{state}}, \qquad \bar{B}_t = \Delta_t \cdot B_t$$

状态递推与输出：

$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot x_t, \qquad y_t = C_t^\top h_t + D \odot x_t$$

分类头（取末尾时刻隐状态）：

$$\text{logit} = W_\text{cls}\;\text{LayerNorm}(h_L) \in \mathbb{R}$$

**边界标签生成（自监督）**：

$$y_t = \mathbf{1}\bigl[\|a_t - \bar{a}_\text{act}\| > \gamma\,\sigma_\text{act}\bigr], \quad \gamma = 2.0$$

其中 $\bar{a}_\text{act}$、$\sigma_\text{act}$ 为活跃节点动作历史的均值与标准差。

### 3.2 SGMTS（语义引导 Mamba 树扫描编码器）— 视觉树

**文件**: `dual_tree_vla/model/sgmts/sgmts.py`

> **视觉树（Visual Tree）**：SGMTS 是双树中的第一棵树。每帧独立在图像 patch 特征图上构建一棵语义加权最大生成树（MST），以 Mamba Tree-SSM 沿 BFS 序进行线性时间扫描，输出语义增强的逐 patch 特征。
>
> **延迟目标 < 5 ms / 帧**（224×224 输入，batch=1，RTX 4090）：
> - MST 构建（Kruskal）：$O(E \log E)$，$E = O(P)$（4-邻域），CPU 上约 **0.3–0.8 ms**
> - Tree-SSM 扫描：$O(P \cdot d_f \cdot d_\text{state})$，纯 GPU 向量化，约 **1–2 ms**
> - CLIP patch 提取（冻结，adapter only）：约 **2–3 ms**
> - **合计 < 5 ms**，远低于 Self-Attention 的 $O(P^2)$ ViT 方案（同尺寸 ~18 ms）

SGMTS 在 GrootV 的 Mamba 树扫描（`tree_scanning.py: MinimumSpanningTree + BFS + Tree_SSM`）基础上引入层级语义引导，将任务描述和 HMT 高层抽象节点的语义同时注入扫描过程。整体架构如下：

```
输入图像
    ↓
[CLIP Vision Encoder] ──┬──→ 视觉特征 [B, H, W, C] ───┐
    │                   │                             │
    │                   └──→ 多尺度特征图 ────────────┤
    │                                                 ▼
[CLIP Text Encoder] ←── 文本/类别提示 ───────→ [语义引导树构建器]
                                                    │
                                            语义重要性图 + 树拓扑
                                                    │
                                            [MambaTree扫描层]
                                                    │
                                            语义增强视觉特征
                                                    │
                                            [下游任务头]
```

---

#### 视觉骨干：CLIP 双编码器（无需 mini-imagenet 预训练）

> **核心思路**：CLIP 同时提供 Vision Encoder（patch 级视觉特征）和 Text Encoder（文本语义向量），两路输出共同驱动语义引导树构建器。Vision Encoder 抽取多尺度特征图，Text Encoder 将任务描述/类别提示编码为跨模态对齐的语义向量，二者在语义引导树构建器中融合，生成**语义重要性图**和**树拓扑**，再由 MambaTree 扫描层完成序列化编码。

```
输入图像 (B, C, H, W)
      ↓
[CLIP Vision Encoder]  ← 冻结，直接使用预训练权重
  ├─→ 视觉特征 [B, H, W, C]          # 全分辨率空间特征
  └─→ 多尺度特征图 (1/8, 1/16, 1/32) # 用于细粒度语义定位

文本/类别提示 (str / token_ids)
      ↓
[CLIP Text Encoder]   ← 冻结，跨模态语义对齐
      ↓
文本语义向量 g_text ∈ R^{d_lang}

[视觉特征] + [文本语义向量]
      ↓
[语义引导树构建器]
  ├─→ 语义重要性图 σ_i = cos(p_i, g_sem) ∈ R^P
  └─→ 树拓扑（语义加权 MST + 动态语义根 r*）
      ↓
[MambaTree 扫描层]   # Tree-SSM 沿 BFS 序递推
      ↓
语义增强视觉特征 Z_v ∈ R^{P × d_visual}
      ↓
[下游任务头]         # 动作解码 / 融合 / 分类
```

- `CLIPPatchExtractor`：封装 `transformers.CLIPVisionModel`，冻结所有 CLIP 参数，仅 Adapter（Linear + LayerNorm）参与训练
- 文本编码路径：通过 `lang_gate` 线性层将 `g_task`（LLM 均值池化嵌入）投影到视觉空间，对齐两路特征维度
- **优势**：Vision Encoder 特征和 Text Encoder 特征已在同一语义空间对齐，语义重要性图可直接反映图像区域与任务描述的跨模态相关度

---

#### 步骤 1：语义引导向量构造（分层）

CLIP Text Encoder 将文本/类别提示编码为 $g_\text{text}$，再与 HMT 子任务嵌入 $\bar{s}_\text{top}$ 混合：

$$g_\text{sem} = \begin{cases} g_\text{task} & \text{HMT 尚未建立（第0帧）} \\ \displaystyle\frac{1}{|S_\text{top}|}\sum_{k \in S_\text{top}} s_k & \text{HMT 已有 ≥2 层非叶子节点} \end{cases}$$

其中 $S_\text{top}$ 为 HMT **最顶两层非叶子节点**（BFS 深度 0~1）的语义嵌入集合；$g_\text{task}$ 为 LLM 对任务描述的均值池化嵌入。二者混合：

$$g_\text{sem} = \beta\, g_\text{task} + (1-\beta)\,\bar{s}_\text{top}, \quad \beta \in [0,1] \text{（随树深度线性衰减）}$$

> **直觉**：树刚建立时以总任务为中心扫描；随子任务抽象节点出现，逐渐以当前活跃子任务语义为中心。

---

#### 步骤 2：语义引导树构建器

语义引导树构建器接收**视觉特征**（Vision Encoder 输出的空间特征图）和**文本语义向量**（$g_\text{sem}$），输出两个核心结构：

**① 语义重要性图**：对每个 patch 计算与 $g_\text{sem}$ 的跨模态相似度：

$$\sigma_i = \cos(p_i,\; W_g\, g_\text{sem}) \in [-1, 1], \quad i = 1, \ldots, N_p$$

$\sigma_i$ 构成空间语义热图，高值区域为当前任务焦点。

**② 树拓扑（语义根节点选择）**：GrootV 中 BFS 根隐式由 MST 结构决定，等效于从某固定角落出发。SGMTS 改为**显式语义根**：

$$r^* = \arg\max_{i} \sigma_i$$

$r^*$ 是语义最相关的 patch，作为 BFS 起始根，确保 SSM 最先编码任务焦点区域。

---

#### 步骤 3：语义加权 MST 构建

相邻 patch 的边权（取代 GrootV 纯 Cosine），结合语义重要性图 $\sigma$：

$$w_{ij} = \underbrace{\cos(p_i,\, p_j)}_{\text{视觉相似度}} + \alpha\;\underbrace{\sigma_i \cdot \sigma_j}_{\text{任务/子任务相关度偏置}}$$

$\alpha = 0.5$（默认）。Kruskal 最大生成树在此新边权上建 MST，再从 $r^*$ 出发 BFS 得扫描序列。

> **效果**：高权边连接两个都与当前子任务相关的 patch，MST 优先将它们串联；BFS 从语义中心向外展开，SSM 隐态在遍历到非相关区域前已编码充分的任务上下文。

---

#### 步骤 4：MambaTree 扫描层（语义增强输入）

MambaTree 扫描层接收语义引导树构建器的输出（树拓扑 + 语义重要性图），以语义增强的输入 $X_i$ 替代原始 patch 特征：

$$X_i = p_i + \sigma_i \cdot W_{g'}\, g_\text{sem}$$

即在每个 patch 特征上叠加按其语义重要性 $\sigma_i$ 加权的任务向量，使 SSM 在扫描时同时感知视觉内容和语义显著程度。

Tree-SSM 沿 BFS 序递推（同 GrootV `tree_scanning_core`）：

$$h_i = \bar{A}_i \odot h_{\text{parent}(i)} + \bar{B}_i \odot X_i, \qquad z_i = C_i^\top h_i + D \odot X_i$$

父子关系由树拓扑决定（树形依赖，非线性链）。每个 patch 的输出 $z_i$ 携带从根到该节点路径上的完整语义上下文。

**输出**：语义增强视觉特征，按原始 patch 位置重排后送入下游任务头：

$$Z_v \in \mathbb{R}^{P \times d_\text{visual}}$$

---

#### 创新点总结（相对 GrootV）

| 机制 | GrootV（原版） | SGMTS（本工作） |
|------|-------------|--------------|
| 视觉特征来源 | 随机初始化骨干 | CLIP Vision Encoder（多尺度特征图） |
| 文本/语义输入 | 无 | CLIP Text Encoder（文本/类别提示 → 跨模态对齐向量） |
| 语义重要性图 | 无 | $\sigma_i = \cos(p_i, W_g g_\text{sem})$ 空间热图 |
| BFS 根 | 固定（隐式左上角） | 动态语义根 $r^* = \arg\max \sigma_i$ |
| 边权 | 纯视觉 Cosine | 视觉 Cosine + $\sigma_i \cdot \sigma_j$ 偏置 |
| SSM 输入 | 原始 patch 特征 | 语义增强输入 $X_i = p_i + \sigma_i \cdot W_{g'} g_\text{sem}$ |
| 语义引导来源 | 无 | CLIP Text Encoder + HMT 顶层抽象节点（自适应切换） |

**三条简单但有效的创新**：
1. **CLIP 双编码器驱动**：Vision Encoder 提供多尺度视觉特征，Text Encoder 提供跨模态语义向量，二者在语义引导树构建器中联合生成语义重要性图，无需额外跨模态对齐训练。
2. **语义根漂移**（Semantic Root Drift）：每帧根据 $\sigma$ 热图重选 $r^*$，根随任务焦点自然漂移，无任何额外参数。
3. **$\beta$ 调度**（Task-to-Subtask Schedule）：$\beta$ 从 1.0 线性衰减到 0.3，树越深越信任子任务语义而非总任务，无需任何监督信号。

**低延迟设计要点**：
- MST 仅在 4-邻域（或 8-邻域）稀疏图上运行，边数 $E \leq 4P$，Kruskal 排序规模小，CPU 计算约 **0.3–0.8 ms**
- Tree-SSM 递推在 BFS 序上展开为向量化矩阵乘，无 Python 循环热路径，GPU 单帧约 **1–2 ms**
- 视觉树每帧**完全重建**（无状态），与记忆树增量更新解耦，两者可流水线并行执行，不引入额外同步开销

### 3.3 HierarchicalMemoryTree（层级记忆树）— 记忆树

**文件**: `dual_tree_vla/model/memory_tree/`

> **记忆树（Memory Tree）**：HMT 是双树中的第二棵树。跨帧在线增量维护，以树形层级组织任务轨迹的语义记忆，供视觉树（SGMTS）提取当前子任务的引导向量 $s_\text{top}$，并向动作头提供历史上下文。
>
> **延迟目标 < 1 ms / 帧**（增量更新，RTX 4090）：
> - Merge 路径（大多数帧）：仅 Welford 均值更新，**O(1)**，< **0.05 ms**
> - Branch+Elevate 路径（子任务边界帧）：沿祖先链 MLPElevation，**O(depth × MLP)**，depth ≤ 4，< **0.8 ms**
> - 树节点数上界由 `max_tree_depth=4` 保证，不随轨迹长度增长

HMT 是在线构建的树形记忆结构。**叶子节点与抽象节点存储完全不同的字段**：

```
叶子节点（is_leaf() == True）
────────────────────────────
  z_v    : (d,)       视觉嵌入（Welford 在线均值）
  a_hist : List[d_a]  动作历史（供 JumpAwareHead 使用）
  w      : float      节点权重

抽象节点（由 MLPElevation 创建，is_leaf() == False）
────────────────────────────────────────────────────
  s      : (d,)   语义嵌入（MLPElevation 的输出）
  w      : float  节点权重
```

**树初始化**（第一帧）：第一帧无论 JumpAwareHead 输出何值，均视为跳变点，直接建立两层结构：

```
[abs_root (s = s_cur_frame1)]  ← 抽象根（语义探针）
    └── [leaf_0]               ← 第一帧叶子，active_id
```

`elevation_pending_parent = abs_root_id`，`propagate_elevation_to_root` 随即用 `leaf_0.z_v` 更新 `abs_root.s`。此后树中始终存在至少一层抽象节点，`_branch_split` 无需处理零抽象层特殊情况。

**三种决策**（每帧 forward 时执行）：

| 决策 | 触发条件 | 操作 |
|------|---------|------|
| 合并（Merge） | `force_branch=False`（JumpAwareHead 判定） | Welford 更新活跃叶子的 `z_v`，追加 `a_hist` |
| 分支（Branch） | `force_branch=True`（JumpAwareHead 判定） | 语义爬升找挂载点 → 创建新叶子；**立即触发 Elevate** |
| 提升（Elevate） | Branch 后立即执行（无数量阈值） | 对挂载点的叶子子节点加权池化 `z_v`，MLPElevation → 插入抽象父节点 `s_abs` |

---

#### Branch 事件完整流程（语义感知分支挂载 — 3种情况）

当 JumpAwareHead 判定当前帧为语义跳变时（`force_branch=True`），分支操作分为 **4个步骤** 顺序执行：

---

##### 步骤 1：提取当前帧的语义探针

对当前帧视觉嵌入 $z_v^\text{cur}$ 通过 MLPElevation 提取一个**临时语义探针**：

$$s_\text{cur} = \text{MLPElevation}(z_v^\text{cur}) \in \mathbb{R}^d$$

$s_\text{cur}$ 是单帧的粗略语义估计，**仅用于本次挂载位置决策，不存入任何节点**（以 `detach().cpu()` 处理，不参与梯度）。

---

##### 步骤 2：语义爬升 + 分类（`tree._classify_mount`）

从活跃叶子的**第一抽象祖先**（`active_leaf.parent_id`，最低语义节点）开始，沿父链向上逐节点计算：

$$d_k = 1 - \cos(s_\text{cur},\, s_k)$$

根据爬升结果，分为 **3种情况（Case A / B / C）**：

| 情况 | 触发条件 | 描述 |
|------|---------|------|
| **Case A**（最低满足） | 第一个抽象节点即 $d_\text{first} \geq \tau_\text{mount}$ | s_cur 与该子任务差异已足够大，不需插入中间层 |
| **Case B**（中间满足） | $d_\text{first} < \tau_\text{mount}$，爬升中某节点 $v_k$ 满足 $d_k \geq \tau_\text{mount}$ | s_cur 代表更高粒度的语义，需新建中间抽象层 |
| **Case C**（超出根节点） | 爬升至根仍无 $d_k \geq \tau_\text{mount}$ | s_cur 语义超出了整棵树，需创建新的顶层节点 |

> **核心直觉**：
> - **Case A**："第一抽象层就能区分" → 说明新帧在当前子任务下只是另一个普通分支，挂在已有抽象节点下即可，**不需要新建中间抽象层**（避免频繁提升）
> - **Case B**："需要爬升才能区分" → 说明新帧属于更高层次的语义切换，需要在 $v_k$ 下**新建一个语义探针抽象节点**来表示这个新子任务分支
> - **Case C**："超出整棵树" → 说明新帧的语义高于现有所有层级，需要**创建超级根**来容纳它

---

##### 步骤 3：挂载（3种路径）

**Case A — 直接挂叶子（1层新增）**：

```
[first_abstract (已有)]
    ├── [active_leaf (已有)]
    └── [new_leaf]  ← NEW，active_id 切换至此
```

`elevation_pending_parent = first_abstract_id`

---

**Case B — 插入语义探针层，再挂叶子（2层新增）**：

在 $v_k$ 下先插入**语义探针抽象节点**（$s = s_\text{cur}$），再在其下挂叶子：

```
[v_k (已有)]
    ├── ... (已有子树，含 active_leaf 分支)
    └── [probe_abstract (s = s_cur)]  ← NEW，代表新子任务分支
             └── [new_leaf]           ← NEW，active_id 切换至此
```

`elevation_pending_parent = probe_abstract_id`

---

**Case C — 创建超级根 + 语义探针层（2层新增，共3层结构变化）**：

创建超级根，旧根挂其下；再在超级根下新建**语义探针抽象节点**（$s = s_\text{cur}$），新叶挂探针下：

```
[super_root (s ← propagate 更新)]  ← NEW，成为新 root_id
    ├── [old_root (原整棵树)]
    └── [probe_abstract (s = s_cur)]  ← NEW，代表新高层语义分支
             └── [new_leaf]           ← NEW，active_id 切换至此
```

`elevation_pending_parent = probe_abstract_id`

> 超级根的 $s$ 在 `propagate_elevation_to_root` 时由 `MLPElevation(pool(old_root.s, probe.s))` 自动得到，相当于将 $s_\text{cur}$ 与旧根语义"融合"出更高层次的概括。与 Case B 同理：探针节点初始持有 $s_\text{cur}$，由 propagate 使用已更新的子层嵌入重算覆盖。

---

##### 步骤 4：全路径语义更新（`propagate_elevation_to_root`）

**只要发生挂载操作**，就从 `elevation_pending_parent` 开始，沿父链向上逐节点更新每个抽象节点的 $s$，**直至根节点**：

$$s_{\text{node}} \leftarrow \text{MLPElevation}\!\left(\frac{\displaystyle\sum_{i \in \text{直接子节点}} w_i\, e_i}{\displaystyle\sum_i w_i}\right)$$

其中子节点嵌入 $e_i$ 按类型取值：
- 叶子子节点 → 用 $z_v$
- 抽象子节点 → 用其 $s$（已被本轮下层更新）

**自下而上**确保每层使用最新子层语义。这样每次分支后整棵树所有祖先的语义概括都保持最新，下次 Case B 爬升时的比较结果不会过期。

> **提升频率设计**：只有分支（Branch）时才触发全路径更新，合并（Merge）时不更新。Merge 阶段叶子的 $z_v$ 通过 Welford 在线均值逐渐稳定；当下次分支发生时，`propagate` 才用这些稳定后的 $z_v$ 更新祖先 $s$。这避免了每帧都做 MLPElevation 的高昂代价，同时保证分支决策时语义是最新的。

---

##### 步骤 5：深度剪枝（`tree._prune_to_max_depth(max_depth=4)`）

全路径更新完成后，立即检查树的深度。若存在深度 $> 4$ 的节点（根节点深度 = 0），从最深层向上依次删除：

$$\text{有效树深度} \leq 4 \quad (\text{支持最多 4 层语义粒度})$$

> 受 Case B 和 Case C 的影响，树会在纵向上增长（Case B 在中间插入一层，Case C 增加一层超级根）。树倾向于**左侧（旧记忆）积累更大深度**，剪枝相当于遗忘久远的历史——这与人类工作记忆的衰减规律一致。剪枝后若 active_id 被删除，切换至最右叶子（最新记忆）。

---

**超参数**：

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `mount_tau` | 0.4 | 余弦距离阈值；大 → 更难触发 Case A → 树倾向 Case B/C（更深）；小 → 树更扁平 |
| `max_tree_depth` | 4 | 最大树深度；超出则剪枝旧记忆节点 |

均在 `configs/default.yaml` 中配置。

**TreeSSMReadout**：沿 HMT BFS 顺序仅对**语义节点**（`node.s is not None`）执行 SSM 扫描，原始帧叶子节点（仅有 `z_v`/`a_hist`，无 `s`）不参与 Readout：

> **设计动机**：叶子节点存储的是原始视觉嵌入 $z_v$ 和动作历史 $a_\text{hist}$，其语义信息已在 Elevate 阶段由 MLPElevation 归纳压缩进父语义节点的 $s$ 中。对原始帧节点再做 SSM 扫描是冗余的，且会引入噪声帧级细节。
>
> **注意**：剪枝后，语义节点可能因子节点全部被删而变为 `is_leaf()==True`，但它仍持有语义嵌入 $s$，代表已完成的子任务记忆，**仍需参与 Readout**。因此判断标准为 `node.s is not None`，而非 `not is_leaf()`。

$$m_\text{ctx} = \text{SSM}_\text{Tree}\bigl(\{x_i\}_{i \in \text{BFS-abstract}}\bigr)[-1] \in \mathbb{R}^{d_\text{ssm}}$$

$$x_i = W_\text{abs}\bigl[s^{(i)};\ \log w_i\bigr], \quad i \in \{\text{抽象节点}\}$$

抽象节点的父节点也必然是抽象节点（或 None=根），父子链在抽象层级内自洽，SSM 递推 $(h_i = \bar{A}_i \odot h_{\text{par}(i)} + \bar{B}_i \odot x_i)$ 直接在抽象节点间进行。

**MLPElevation**：只接收子叶子节点 `z_v` 加权池化，输出抽象节点语义嵌入：

$$z_\text{pool} = \frac{\sum_i w_i\, z_v^{(i)}}{\sum_i w_i}, \qquad s_\text{abs} = \text{MLP}(z_\text{pool}) \in \mathbb{R}^d$$

### 3.4 CrossModalFusion（跨模态融合）

**文件**: `dual_tree_vla/model/fusion.py`

将视觉特征 $z_v$、记忆上下文 $m_\text{ctx}$、本体感知状态 $q$ 和语言嵌入 $g_\text{lang}$ 融合。

#### 门控融合公式

$$g = \sigma\!\bigl(W_g\,[z_v;\, m_\text{ctx};\, q;\, g_\text{lang}] + b_g\bigr) \in [0,1]^d$$

$$f_\text{fused} = g \odot W_1[z_v;\, m_\text{ctx}] + (1-g) \odot W_2[q;\, g_\text{lang}] \in \mathbb{R}^d$$

Phase 1 开始时该模块处于随机初始化状态，Phase 1 专门训练它。

### 3.5 FlowMatchingActionHead（流匹配动作头）

**文件**: `dual_tree_vla/model/action_head/flow_matching.py`

基于连续归一化流（Flow Matching）的动作生成头，输入 $f_\text{fused}$ 生成 $H_a$ 步动作序列。

#### Flow Matching 训练目标

定义线性插值轨迹（$x_0 \sim \mathcal{N}(0,I)$，$x_1 = a_\text{gt}$）：

$$x_t = (1-t)\,x_0 + t\,x_1, \quad t \sim \mathcal{U}[0,1]$$

目标速度场（与路径无关的常数场）：

$$u_t(x_t) = x_1 - x_0$$

模型预测速度 $v_\theta(x_t,\, t,\, f_\text{fused})$，训练损失：

$$\mathcal{L}_\text{flow} = \mathbb{E}_{t,\,x_0,\,x_1}\;\bigl\|v_\theta(x_t,\, t,\, f_\text{fused}) - (x_1 - x_0)\bigr\|_2^2$$

推理时以 $x_0 \sim \mathcal{N}(0,I)$ 为起点，用 Euler 法积分 `n_ode` 步：

$$x_{t+\Delta t} = x_t + \Delta t\cdot v_\theta(x_t,\, t,\, f_\text{fused}), \qquad \hat{a} = x_1$$

---

## 4. 两阶段训练流程

```
        预训练阶段                           LIBERO                LIBERO
   （RoboCerebra 全模型）        →      Phase 1         →     Phase 2
─────────────────────────────────     ─────────────         ──────────────
RoboCerebra                           冻结: LLM +           冻结: 无
  · 冻结: LLM, Fusion, FlowHead             全部预训练模块
  · 可训练: SGMTS(adapter), s_proj,                        可训练: 全部
    JumpAwareHead,                     可训练: Fusion,            (LLM: 0.1× LR)
    TreeSSM, MLPElevation                      FlowHead
  · 损失: L_boundary                   损失: L_flow          损失: L_flow
    + L_sem + L_elev                      （仅此一项）            （仅此一项）

脚本: scripts/pretrain.py             脚本: scripts/train.py --phase 1/2
配置: configs/pretrain.yaml           配置: configs/train_phase1/2.yaml
```

### 阶段说明

#### 全模型预训练（RoboCerebra）

目标：让模型学会"什么是子任务边界"以及"边界处的语义应当与子任务描述对齐"。

- 视觉骨干（CLIP ViT）完全冻结，仅训练 SGMTS 中的轻量 adapter（Linear + LayerNorm）
- `JumpAwareHead` 学习从动作序列中识别动作突变点（纯动作信号，不依赖语义）
- `SGMTS` adapter 学习将 CLIP 特征映射到本项目的内部维度 `d_f`，使 MST 边权适配任务语义
- `MLPElevation` 学习将子节点语义归纳为抽象父节点语义
- `L_sem` 验证"分支时刻的视觉嵌入是否接近子任务文本描述"（此时 LLM 冻结，语义空间稳定）

#### Phase 1（LIBERO，LLM 冻结）

目标：让 FlowMatchingActionHead 和 CrossModalFusion 学会利用已有的记忆树来预测动作。

- 从 `pretrain_best.pt` 加载预训练权重，冻结全部预训练模块
- 只优化 Fusion + FlowHead，损失仅 L_flow
- 避免预训练阶段建立的语义结构被破坏

#### Phase 2（LIBERO，全量微调）

目标：全模型端到端微调，进一步提升动作质量。

- 解冻全部参数，LLM 使用 0.1× 学习率（防止灾难性遗忘）
- 损失仅 L_flow
- 继续不引入语义损失（语义结构已在预训练确立）

---

## 5. 核心设计原则：损失函数完全分离

> **最重要的设计约束**：语义相关损失与 FlowMatching 损失不能混合在同一训练阶段。

```
全模型预训练（RoboCerebra）   →  L_boundary + L_sem + L_elev
第一阶段训练（LIBERO）        →  L_flow  ← 仅此一项
第二阶段训练（LIBERO）        →  L_flow  ← 仅此一项
```

> **关于 CLIP 预训练**：CLIP Vision Encoder 的权重在整个训练流程中保持冻结，无需任何分类损失（已弃用的 L_cls 已从项目中移除）。CLIP 的语义能力来自其自身的预训练，与本项目的三个损失函数完全解耦。
---

## 6. 损失函数详解

**文件**: `dual_tree_vla/losses/tree_losses.py`

### 6.1 预训练损失

#### `l_boundary` — 动作突变边界 BCE

$$\mathcal{L}_\text{boundary} = -\sum_t \Bigl[r\,y_t\log\sigma(\ell_t) + (1-y_t)\log(1-\sigma(\ell_t))\Bigr]$$

其中 $r = n_\text{neg}/n_\text{pos}$ 为正样本重加权因子，$\ell_t$ 为 JumpAwareHead 输出 logit，自监督标签：

$$y_t = \mathbf{1}\bigl[\|a_t - \bar{a}_\text{act}\| > \gamma\,\sigma_\text{act}\bigr]$$

#### `l_sem` — 分支点语义对齐 InfoNCE

$$\mathcal{L}_\text{sem} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(s_i^\top s_i^\text{text} / \tau)}{\sum_{j=1}^N \exp(s_i^\top s_j^\text{text} / \tau)}$$

- $s_i \in \mathbb{R}^d$：分支点时刻的视觉语义嵌入（SGMTS 输出）
- $s_i^\text{text} \in \mathbb{R}^d$：对应真实子任务描述的语言嵌入（LLM 编码，冻结）
- $\tau = 0.07$：InfoNCE 温度

#### `l_elev` — 语义提升一致性 MSE

$$\mathcal{L}_\text{elev} = \frac{1}{M}\sum_{k=1}^M \left\|s_k^\text{abs} - \frac{\sum_i w_i\,s_i^{(k)}}{\sum_i w_i}\right\|_2^2$$

- $s_k^\text{abs}$：第 $k$ 个抽象节点的语义嵌入
- $s_i^{(k)},\, w_i$：其子节点语义嵌入及权重

#### 预训练总损失

$$\mathcal{L}_\text{pretrain} = w_b\,\mathcal{L}_\text{boundary} + w_s\,\mathcal{L}_\text{sem} + w_e\,\mathcal{L}_\text{elev}$$

默认权重：$w_b = 1.0,\; w_s = 0.5,\; w_e = 0.2$

### 6.2 Phase 1/2 损失

Phase 1 和 Phase 2 **只使用 $\mathcal{L}_\text{flow}$**，由 `FlowMatchingActionHead.forward()` 内部计算并返回：

$$\mathcal{L}_\text{flow} = \mathbb{E}_{t,x_0,x_1}\;\bigl\|v_\theta(x_t,\, t,\, f_\text{fused}) - (x_1 - x_0)\bigr\|_2^2$$

```python
{"L_flow": <tensor>, "total": <tensor>}   # total == L_flow
```

没有任何语义或树结构相关损失混入。

---

## 7. 模型 Forward API

```python
losses = model(
    images,        # (B, T, C, H, W)  — 原始 RGB 帧序列
    instructions,  # List[str] len=B   — 任务自然语言描述
    states,        # (B, T, d_q)       — 本体感知状态序列（关节角等）
    actions,       # (B, T, d_a)       — 专家动作序列（训练时提供）
    subtask_ids,   # (B, T) optional   — 子任务 ID（预训练时需要）
    mode,          # 'pretrain' | 'phase1' | 'phase2'
)
```

### 返回值约定

| `mode` | 返回 dict 的键 | 实际计算的损失 |
|--------|--------------|--------------|
| `'pretrain'` | `L_boundary`, `L_sem`, `total` | $\mathcal{L}_b + \mathcal{L}_s + \mathcal{L}_e$ |
| `'phase1'` | `L_flow`, `total` | $\mathcal{L}_\text{flow}$ only |
| `'phase2'` | `L_flow`, `total` | $\mathcal{L}_\text{flow}$ only |

---

## 8. 项目文件结构

```
DualTreeVLA/
├── CONSTRUCTION.md              ← 本文档
├── README.md
├── requirements.txt
│
├── configs/
│   ├── pretrain.yaml            ← 预训练配置（L_boundary+L_sem+L_elev，无 L_flow）
│   ├── train_phase1.yaml        ← Phase 1 配置（仅 L_flow，冻结 LLM）
│   ├── train_phase2.yaml        ← Phase 2 配置（仅 L_flow，全量微调）
│   ├── ds_zero2.json            ← DeepSpeed ZeRO-2（Phase 1 推荐）
│   └── ds_zero3.json            ← DeepSpeed ZeRO-3（Phase 2 推荐）
│
├── scripts/
│   ├── pretrain.py              ← 预训练脚本（RoboCerebra 全模型）
│   ├── train.py                 ← Phase 1/2 训练脚本（LIBERO, mode='phase1'|'phase2'）
│   ├── eval.py                  ← 评测脚本
│   ├── pretrain.sh              ← 单机启动脚本
│   ├── train_phase1.sh
│   └── train_phase2.sh
│
├── dual_tree_vla/
│   ├── model/
│   │   ├── dual_tree_vla.py   ← 主模型（DualTreeVLA, 两阶段 forward）
│   │   ├── semantic_jump_head.py  ← JumpAwareHead（纯动作 Mamba SSM）
│   │   ├── fusion.py            ← CrossModalFusion
│   │   ├── attn.py              ← 注意力辅助模块
│   │   ├── action_head/
│   │   │   └── flow_matching.py ← FlowMatchingActionHead（L_flow 内部计算）
│   │   ├── memory_tree/
│   │   │   ├── node.py          ← MemoryNode（6-元组）
│   │   │   ├── tree.py          ← HierarchicalMemoryTree（在线构建）
│   │   │   ├── operations.py    ← merge / branch / prune / reinforce
│   │   │   ├── tree_ssm.py      ← TreeSSMReadout（记忆上下文向量）
│   │   │   └── __init__.py
│   │   └── sgmts/
│   │       ├── sgmts.py         ← SGMTS（语义引导 Mamba 树扫描）
│   │       │                       CLIPPatchExtractor：冻结 CLIP ViT + adapter
│   │       │                       PatchCNN：轻量 fallback
│   │       └── __init__.py
│   ├── losses/
│   │   ├── tree_losses.py       ← l_boundary, l_sem, l_elev（预训练）/ stubs
│   │   └── __init__.py
│   └── dataset/
│       ├── robocerebra.py       ← RoboCerebra 预训练数据集
│       ├── robocerebra_bench.py ← RoboCerebra Bench 评测
│       └── libero.py            ← LIBERO LeRobot 格式数据集
│
├── checkpoints/
│   ├── Qwen2.5-0.5B/            ← 预训练 LLM（Phase 1/2 默认）
│   └── Qwen2.5-1.5B-Instruct/  ← 较强 LLM（Phase 2 可选升级）
│
└── dataset/
    ├── RoboCerebra/             ← 预训练数据集
    │   ├── RoboCerebra_trainset/
    │   └── RoboCerebraBench/
    └── LIBERO/                  ← Phase 1/2 数据集
        ├── libero_object/
        └── libero_spatial/
```

---



## 参考项目

| 项目 | 贡献 |
|------|------|
| [Evo-1](https://github.com/SakanaAI/evo) | 训练脚本风格（AdamW param groups, cosine LR, Accelerate, W&B） |
| [MemoryVLA](https://github.com/PKU-RL/MemoryVLA) | 记忆树三视角训练范式（ICLR 2026） |
| [GrootV / MambaTree](https://github.com/GROOT-V) | SGMTS 语义引导 Mamba 树扫描设计 |
| [Mamba](https://github.com/state-spaces/mamba) | SSM 核心实现（JumpAwareHead 中使用） |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | FlowMatchingActionHead 理论基础 |
| [CLIP](https://openai.com/research/clip) | CLIP ViT 视觉骨干预训练权重（CLIPPatchExtractor 中冻结使用） |

