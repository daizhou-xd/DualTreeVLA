
# MemoryTreeVLA: 基于记忆树的视觉-语言-动作模型

## 1. 概述

MemoryTreeVLA 是一种新型的长时程机器人操控架构，通过显式的**层次化任务树（Hierarchical Task Tree）**记忆机制，结合双大语言模型（LLM）协同工作，实现复杂多步骤任务的规划、执行与回溯。该架构主体借鉴了Evo-1轻量化VLA模型，并利用 BT-TL-DMPs  中行为树的任务分解思想，以及 RoboCerebra  的长时程子任务标注方法和GrootVL - 树状扫描状态空间模型，构建了一个具备长程推理能力的 VLA 系统。

---

## 2. 核心架构

### 2.1 修正后的整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MemoryTreeVLA Architecture (Updated)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input Layer                                                               │
│   ┌──────────────────┐         ┌──────────────────┐                         │
│   │   Visual Input   │         │    Tree.json     │                         │
│   │  (Camera Image)  │         │  (Subtask State) │                         │
│   └────────┬─────────┘         └────────┬─────────┘                         │
│            │                            │                                   │
│   Modality-Specific Encoders                                                │
│            ▼                            ▼                                   │
│   ┌──────────────────┐         ┌──────────────────┐                         │
│   │  Vision Mamba    │         │   Tree Mamba     │                         │
│   │  (Tree Scan)     │         │    (Tree Scan)   │                         │
│   │  Output: Z_v     │         │  Output: Z_t     │                         │
│   └────────┬─────────┘         └────────┬─────────┘                         │
│            │                            │                                   │
│   Multimodal Fusion                                                         │
│            └────────────┬───────────────┘                                   │
│                         ▼                                                   │
│              ┌────────────────────┐                                         │
│              │  Multimodal Mamba   │                                        │
│              │  (Cross-Modal Fuse) │                                        │
│              │  Input: [Z_v, Z_t]  │                                        │
│              │  Output: Z_fused    │                                        │
│              └──────────┬──────────┘                                        │
│                         │                                                   │
│   Action Generation                                                         │
│                         ▼                                                   │
│   ┌──────────────────────────────────────────────────┐                      │
│   │              Action LLM (Frozen/Trainable)       │                      │
│   │  Input: Z_fused (Fused Multimodal Features)      │                      │
│   │  Output: Token Sequence + State Token            │                      │
│   └────────────────────────┬─────────────────────────┘                      │
│                            │                                                │
│                            ▼                                                │
│   ┌──────────────────────────────────────────────────┐                      │
│   │              Action Head (Diffusion/MLP)         │                      │
│   │  Input: [Token Seq + State Token]                │                      │
│   │  Output: Action Sequence (End-effector Pose)     │                      │
│   └──────────────────────────────────────────────────┘                      │
│                                                                             │
│   ═══════════════════════════════════════════════════════════════════       │
│                                                                             │
│   Tree Management                                                           │
│   ┌──────────────────────────────────────────────────┐                      │
│   │              Tree LLM (Trainable)                │                      │
│   │  Input: Z_v (Visual Features) + Current Tree     │                      │
│   │  Output: Updated Tree.json                       │                      │
│   │  Function: Subtask completion detection          │                      │
│   └──────────────────────────────────────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 双 LLM 分工与多模态融合设计

> **架构说明**：视觉编码完全由 Vision Mamba 负责（GrootVL 树状扫描），**不使用 ViT 等视觉骨干**。Action LLM 和 Tree LLM 均为纯语言模型（LLM backbone），接收 Mamba 编码后的特征序列作为输入。

| 组件 | 功能定位 | 输入 | 输出 | 架构细节 |
|------|----------|------|------|----------|
| **Vision Mamba** | 视觉时序编码（替代 ViT） | 图像序列 $I_t$ | 视觉特征 $Z_v \in \mathbb{R}^{T \times D}$ | GrootVL 树状扫描 SSM，无需预训练视觉骨干 |
| **Tree Mamba** | 结构化任务编码 | Tree.json | 树特征 $Z_t \in \mathbb{R}^{N \times D}$ | 将层次化树结构线性化为序列，保留父子关系 |
| **Multimodal Mamba** | **跨模态融合** | $[Z_v; Z_t]$ | 融合特征 $Z_{fused} \in \mathbb{R}^{L \times D}$ | **核心创新**：通过选择性 SSM 实现视觉-任务动态对齐 |
| **Action LLM** | 动作生成 | $Z_{fused}$（投影后作为 prefix tokens） | Token 序列 + State Token | **Qwen2.5-0.5B**（base）；高频推理，速度优先，无视觉编码器 |
| **Tree LLM** | 任务管理 | $Z_v$（投影）+ Tree 文本描述 | 更新后的 Tree.json | **Qwen2.5-1.5B-Instruct**；低频调用，Instruct 版保证 JSON 格式稳定，推理能力更强 |

**关键数据流**：

| 路径 | 流向 |
|------|------|
| 视觉模态 | `Image` →(Vision Mamba)→ `Z_v` →(Concat)→ `Multimodal Mamba` → `Z_fused` |
| 树模态 | `Tree.json` →(Tree Mamba)→ `Z_t` →(Concat)→ `Multimodal Mamba` → `Z_fused` |
| 融合投影 | `Z_fused` →(Linear Projector)→ `prefix tokens` → Action LLM |
| Tree LLM 输入 | `Z_v` →(Linear Projector)→ `v_tokens` + Tree 文本序列化 → Tree LLM |

---

## 3. 关键技术：Multimodal Mamba 设计

### 3.1 架构动机
传统的多模态融合（如简单的 Concat + Linear 或 Cross-Attention）难以处理视觉时序与任务树结构之间的**动态跨模态依赖**。Multimodal Mamba 利用状态空间模型（SSM）的选择性机制，实现：
1. **时序对齐**：视觉动态与子任务进度的时序同步
2. **选择性关注**：根据当前子任务类型，动态关注视觉特征的相关区域
3. **长程依赖**：跨越多子任务的长时程记忆保持

### 3.2 融合策略细节

**跨树选择性机制**：以任务树特征的全局均值作为条件信号，对视觉特征进行残差门控，动态调节视觉与树信息的融合比例；再将门控后的视觉 tokens 与树 tokens 拼接为统一序列，经多层 Mamba SSM 处理后输出 `Z_fused`。

**统一树状扫描**：视觉树与任务树作为同一超图的子树，构建基于语义相似度的跨树连接矩阵，采用交替扫描（intra-tree → inter-tree → intra-tree）的方式统一处理两棵树的结构信息。

---

## 4. 训练策略（三阶段修正版）

基于 RoboCerebraBench  的高质量子任务标注数据：

### 第一阶段：Tree LLM + Multimodal Mamba 基础训练
- **目标**：建立视觉-树-动作的基础关联
- **训练对象**：Tree LLM 独立训练；Multimodal Mamba 使用对比学习预训练（对齐视觉与树特征空间）
- **数据**：RoboCerebraBench 子任务标注（平均 9.1 步/任务）

### 第二阶段：动作头与融合模块联合训练
- **目标**：优化跨模态融合到动作的映射
- **设置**：
  - 冻结：Action LLM, Vision Mamba, Tree Mamba
  - 训练：**Multimodal Mamba** + Action Head
- **损失函数**：动作重建损失 + 融合特征对齐损失

### 第三阶段：端到端全量微调
- **目标**：整体架构适应端到端任务执行
- **设置**：
  - 解冻：Action LLM, Multimodal Mamba（部分层）
  - 冻结：Vision Mamba（保留通用视觉特征）
  - 学习率分层：Action LLM (1e-5) > Multimodal Mamba (5e-5) > Action Head (1e-4)

---

## 5. 推理流程（修正版）

### 5.1 标准执行流程（含多模态融合）

```
1. 初始化阶段
   └─> Tree LLM 根据初始视觉图像初始化 Tree.json
       └─> 生成完整子任务序列与初始状态

2. 执行循环（每个时间步 t）
   ├─> 视觉编码：Image_t → Vision Mamba → Z_v (Visual Tokens)
   ├─> 树编码：Tree.json → Tree Mamba → Z_t (Tree Tokens)  
   ├─> 关键步骤：Multimodal Mamba 融合
   │   └─> Input: Concat[Z_v, Z_t]
   │   └─> Process: 跨模态选择性 SSM 处理
   │   └─> Output: Z_fused (Fused Representation)
   ├─> Action VLM：Z_fused → Token Sequence + State Token
   ├─> 动作头：Token Seq → Action Sequence (Δt 时域动作)
   ├─> 执行动作并观察状态变化
   └─> 子任务完成检测？
       ├─> 是：将当前 Token Sequence 存入 Tree.json 对应子任务节点
       │       Tree VLM 更新子任务状态，激活下一子任务
       └─> 否：继续当前子任务执行

3. 任务完成：所有子任务状态为 completed，输出 success
```

### 5.2 回溯机制（利用融合上下文）

```
回溯触发条件：
暂未定义明确触发条件，计划在后续版本中基于失败子任务的特征（如连续失败次数、视觉-树特征不匹配度）设计动态触发机制

回溯流程：
1. 读取 Tree.json 中的 backtrack_pointer
2. **关键**：从 Tree.json 加载上一成功子任务的 token_sequence
3. 重新构建 Z_t（上一子任务状态）并与当前视觉 Z_v 送入 Multimodal Mamba
4. 生成融合特征 Z_fused^backtrack，包含"回溯上下文"
5. Action VLM 基于 Z_fused^backtrack 生成修正动作
6. Tree VLM 重置当前子任务状态，重置 failure_count
```

**回溯中的多模态优势**：Multimodal Mamba 在回溯时能动态调节视觉与历史任务记忆的权重，例如当重新执行 "reach" 时，更关注当前视觉中的物体位置而非历史树状态。

---

## 6. 技术优势分析

### 6.1 Multimodal Mamba 的核心价值

1. **动态跨模态注意力**：相比 Transformer 的静态 Cross-Attention，SSM 的选择性机制允许根据当前子任务动态调整视觉与树信息的融合比例

2. **计算效率**：Mamba 的线性复杂度 $O(L)$ 使长时程任务（RoboCerebra 中长达 20 步）的融合计算可行，而 Transformer 的 $O(L^2)$ 在 $L = T + N$ 较大时开销过高

3. **时序-结构统一建模**：视觉时序（连续）与任务树（离散层次）在 SSM 的统一状态转移框架下融合，避免了模态间语义鸿沟

### 6.2 与现有方案对比

| 方案 | 长时程记忆 | 任务回溯 | 多模态融合策略 | 计算复杂度 | VLA集成 |
|------|-----------|---------|--------------|-----------|--------|
| **MemoryTreeVLA（本文）** | ✅ 显式树状记忆 | ✅ 指针式回溯 | Tree-guided SSM 门控融合 | $O(L)$ | ✅ Evo-1轻量化VLM主干 |
| RoboCerebra (System 2) | ✅ 内存库(Memory Bank) | ❌ 无结构化回溯 | VLM轮询（离散切换） | $O(L^2)$ | ✅ 外部VLM + VLA控制器 |
| BT-TL-DMPs | ✅ 行为树状态机 | ✅ BT反应性恢复 | 符号推理（无神经融合） | 符号求解 | ❌ 基于DMP，非端到端 |
| GrootVL | ❌ 无任务记忆 | ❌ | 树状SSM（单模态） | $O(L)$ | ❌ 视觉/文本骨干，非VLA |
| Evo-1 | ❌ 无长时程机制 | ❌ | 集成模块 + 扩散Action Head | $O(L^2)$ | ✅ 轻量化0.77B VLA |
| OpenVLA / $\pi_0$ | ❌ | ❌ | Token拼接 | $O(L^2)$ | ✅ 大模型VLA |

**核心差异化优势**：
1. **结构化记忆 vs 隐式记忆**：RoboCerebra 的 Memory Bank 是扁平化 key-value 存储，MemoryTreeVLA 的 Tree.json 保留了子任务的**层次依赖关系**，可以做有约束的回溯（仅回到父节点，而非任意跳转）。
2. **神经-符号融合**：BT-TL-DMPs 在符号层（STL→BT）做任务结构，而 MemoryTreeVLA 将树结构**嵌入神经特征空间**，通过 Tree Mamba 和 Multimodal Mamba 实现端到端可微分优化。
3. **轻量化设计**：基于 Evo-1 的 InternVL3-1B 主干（0.77B参数），相比 OpenVLA（7B）计算成本降低约 9×，同时通过两阶段训练保留 VLM 预训练语义表示。

---

## 7. 实验与评估建议

### 7.1 基准测试选择

对标 RoboCerebra 的评估协议，建议在以下基准上验证 MemoryTreeVLA：

| 基准 | 特性 | 适配理由 |
|------|------|----------|
| **RoboCereBraBench** | 长时程（平均 9.1 步/任务），含 Memory-Exploration / Memory-Execution / Random-Disturbance 模式 | 直接验证树状记忆的长时程规划能力与回溯机制 |
| **LIBERO**（Spatial / Object / Goal / Long） | 四类任务套组，涵盖空间关系、物体操作、目标导向与长时程序列 | 验证跨任务类型的子任务切换准确性与动作头泛化能力 |
| **RoboMME** | 多模态机器人操控评估基准，含丰富的场景与语义多样性 | 验证 Tree VLM 对多样化任务描述的树结构初始化与执行鲁棒性 |

### 7.2 消融实验设计

建议围绕三个核心设计进行消融：

```
A. 记忆树结构消融
   - MemoryTreeVLA-full：完整树记忆 + 回溯
   - MemoryTreeVLA-flat：将树退化为扁平列表（等价于 RoboCerebra Memory Bank）
   - MemoryTreeVLA-no-memory：无历史记忆（纯 reactive VLA）
   → 验证树结构相对扁平记忆的优势

B. 多模态融合模块消融
   - w/ Multimodal Mamba（本文）
   - w/ Cross-Attention 替换 Multimodal Mamba
   - w/ Concat + Linear 简单融合
   → 验证树状 SSM 融合的效果与计算效率权衡

C. 回溯机制消融
   - w/ backtrack（指针式回溯）
   - w/ restart（从头重新执行，BT-TL-DMPs 风格）
   - w/o backtrack（不触发回溯）
   → 验证指针式回溯相对全局重置的效率优势
```

### 7.3 评估指标

- **成功率（SR）**：任务完整完成（所有叶子节点 `completed`）
- **子任务精度（Sub-SR）**：每个子任务的独立完成率，衡量树状态的准确性
- **回溯利用率**：成功案例中触发回溯后最终成功的比例
- **推理延迟**：每步动作生成时间（目标参考 Evo-1 实测 ≥ 15 Hz）
- **树状态准确率（Tree-Acc）**：Tree VLM 判断子任务完成状态的 F1 分数（对比人工标注）

### 7.4 与 RoboCerebra 基线的对比设置

RoboCerebra 提出的 System 1–System 2 交互框架以 VLM 轮询方式检测子目标完成，属于**离散式监控**。MemoryTreeVLA 的 Tree VLM 以**每步视觉观测**为输入做连续检测，可以在 RoboCerebra 的三种任务模式（Memory Exploration、Memory Execution、Random Disturbance）下分别对比：
- **Memory-Exploration**：测试树结构对未见区域的预测性覆盖能力
- **Memory-Execution**：核心测试场景，验证树导引下的精准执行
- **Random-Disturbance**：验证回溯机制在随机干扰下的鲁棒恢复能力

---

## 8. Tree.json 数据结构规范

任务树以 JSON 格式存储，每个节点代表一个子任务，包含执行状态、视觉证据和历史 token 序列：

```json
{
  "task_id": "prepare_breakfast_001",
  "task_description": "Prepare breakfast with toast and coffee",
  "root": {
    "id": "root",
    "type": "sequence",
    "status": "in_progress",
    "children": [
      {
        "id": "subtask_0",
        "description": "pick up bread slice",
        "type": "primitive",
        "status": "completed",
        "token_sequence": "<stored_token_ids>",
        "completion_evidence": {
          "frame_idx": 142,
          "confidence": 0.94
        },
        "failure_count": 0,
        "backtrack_pointer": null
      },
      {
        "id": "subtask_1",
        "description": "place bread in toaster",
        "type": "primitive",
        "status": "in_progress",
        "token_sequence": null,
        "completion_evidence": null,
        "failure_count": 1,
        "backtrack_pointer": "subtask_0"
      },
      {
        "id": "subtask_2",
        "description": "make coffee",
        "type": "sequence",
        "status": "pending",
        "children": [
          {"id": "subtask_2_0", "description": "pour water", "status": "pending"},
          {"id": "subtask_2_1", "description": "press brew button", "status": "pending"}
        ]
      }
    ]
  }
}
```

**状态迁移规则**（对应 BT-TL-DMPs 中的行为树节点状态）：
- `pending` → `in_progress`：父节点激活，前序兄弟节点全部 `completed`
- `in_progress` → `completed`：Tree VLM 检测到完成条件（置信度 > 阈值）
- `in_progress` → `failed`：`failure_count` 超过最大重试次数（默认 3 次）
- `failed` → `in_progress`：回溯指针激活，重置并重新执行

---

## 9. 总结

MemoryTreeVLA 通过以下四个核心贡献构建了一个面向长时程机器人操控的完整框架：

1. **层次化任务树记忆（Tree.json）**：借鉴 BT-TL-DMPs 的行为树模块化思想，将复杂任务分解为层次化子任务节点，每个节点存储执行状态、视觉证据及历史 action token，支持精确的指针式回溯，解决了现有 VLA 模型无法有效处理长时程任务的问题。

2. **Multimodal Mamba 跨模态融合**：基于 GrootVL 的树状扫描 SSM，将视觉时序特征 $Z_v$ 与任务树特征 $Z_t$ 进行动态门控融合，以 $O(L)$ 线性复杂度实现比 Cross-Attention 更高效的跨模态依赖建模，直接支持 RoboCerebra 数量级（2,972 步/任务）的长时程序列。

3. **双 LLM 协同分工（Mamba 视觉编码）**：视觉特征完全由 Vision Mamba 提取，Action LLM 采用 **Qwen2.5-0.5B**（高频推理，速度优先），Tree LLM 采用 **Qwen2.5-1.5B-Instruct**（低频调用，精度优先；Instruct 版保证结构化 JSON 输出的稳定性），两者均无视觉骨干，通过 Tree.json 解耦，推理延迟可满足 ≥15 Hz 的实时控制需求。

4. **三阶段渐进训练**：受 Evo-1 两阶段训练范式启发，扩展为三阶段训练（Tree VLM 预训练 → 融合模块与动作头联合训练 → 端到端微调），在保留 VLM 预训练语义表示的同时，逐步建立视觉-树-动作的联合分布。

**未来方向**：
- 回溯触发条件的自动化设计（基于视觉-树特征分布偏移检测）
- 引入 STL（Signal Temporal Logic，来自 BT-TL-DMPs）作为树节点的形式化约束，提升任务规范的精确性
- 在 RoboCerebra 1,000 条长时程轨迹上进行完整实验验证
- 探索在线树结构更新（任务执行过程中 Tree VLM 动态新增/删除子任务节点）
