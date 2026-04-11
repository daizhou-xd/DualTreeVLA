# DualTreeVLA

> **「双树」**：Visual Tree（视觉树）+ Memory Tree（记忆树），两棵树并行驱动长时程机器人操作。

**DualTreeVLA** 同时维护两棵树结构，以最低推理延迟完成视觉编码与跨帧记忆读取：

| 树 | 全称 | 延迟目标 | 职责 |
|----|------|----------|------|
| 🌿 **视觉树** | SGMTS（语义引导 Mamba 树扫描编码器） | **< 5 ms / 帧** | 每帧在 CLIP patch 特征上构建语义加权 MST，Tree-SSM O(P) 扫描 |
| 🌳 **记忆树** | HMT（层级记忆树） | **< 1 ms / 帧** | 跨帧增量维护层级语义记忆，O(depth) 分支/合并 |

两棵树的输出共同输入 Qwen2.5 LLM + Flow Matching 动作头，完成连续动作预测。

---

## 目录

1. [项目结构](#项目结构)
2. [云服务器环境配置](#云服务器环境配置)
   - [硬件要求](#硬件要求)
   - [系统依赖](#系统依赖)
   - [Python 环境](#python-环境)
   - [PyTorch 安装](#pytorch-安装)
   - [Flash Attention 安装](#flash-attention-安装)
   - [DeepSpeed 安装](#deepspeed-安装)
   - [LIBERO 仿真环境（仿真评估专用）](#libero-仿真环境仿真评估专用)
3. [数据准备](#数据准备)
   - [RoboCerebra 训练集](#robocerebra-训练集)
   - [RoboCerebraBench](#robocerebrabench)
   - [LIBERO](#libero)
4. [模型权重下载](#模型权重下载)
5. [配置文件说明](#配置文件说明)
6. [训练](#训练)
   - [3 阶段训练流程总览](#3-阶段训练流程总览)
   - [全模型预训练（RoboCerebra）](#全模型预训练robocerebra)
   - [Phase 1 — FlowMatching 热身（LIBERO）](#phase-1--flowmatching-热身libero)
   - [Phase 2 — 全量微调（LIBERO）](#phase-2--全量微调libero)
   - [断点续训](#断点续训)
   - [Weights & Biases 可视化](#weights--biases-可视化)
7. [评估](#评估)
   - [评估指标](#评估指标)
   - [RoboCerebraBench 评估](#robocerebrabench-评估)
   - [RoboCerebra 训练集评估](#robocerebra-训练集评估)
   - [LIBERO 离线评估](#libero-离线评估)
   - [LIBERO 仿真成功率评估（单进程）](#libero-仿真成功率评估)
   - [LIBERO 仿真成功率评估（服务端+客户端）](#libero-仿真成功率评估服务端客户端)
   - [结果解读](#结果解读)
8. [常见问题](#常见问题)

---

## 项目结构

```
DualTreeVLA/
├── configs/
│   ├── default.yaml          # 评估 / 单阶段默认超参
│   ├── pretrain.yaml         # 全模型预训练（RoboCerebra）配置
│   ├── train_phase1.yaml     # Phase 1 LIBERO FlowMatching 热身
│   ├── train_phase2.yaml     # Phase 2 LIBERO 全量微调
│   ├── ds_zero2.json         # DeepSpeed ZeRO-2（Phase 1 推荐）
│   └── ds_zero3.json         # DeepSpeed ZeRO-3 + CPU offload（Phase 2 推荐）
├── dataset/
│   ├── RoboCerebra/
│   │   ├── RoboCerebra_trainset/   # 三场景训练集
│   │   └── RoboCerebraBench/       # 六子集评测集
│   └── LIBERO/               # LeRobot v2 parquet 格式数据集
│       └── libero_10/
├── dual_tree_vla/
│   ├── dataset/
│   │   ├── libero.py             # LIBERO LeRobot 数据加载器
│   │   ├── robocerebra.py        # RoboCerebra 训练集加载器
│   │   └── robocerebra_bench.py  # RoboCerebraBench 六子集评测加载器
│   ├── losses/
│   │   └── tree_losses.py        # l_boundary / l_sem
│   └── model/
│       ├── attn.py               # FlashMHA（自动选择 Flash Attn 2 / SDPA）
│       ├── fusion.py             # CrossModalFusion（门控融合）
│       ├── dual_tree_vla.py      # DualTreeVLA 主模型
│       ├── semantic_jump_head.py # JumpAwareHead（Mamba 动作突变检测）
│       ├── action_head/
│       │   └── flow_matching.py  # FlowMatchingActionHead（ODE 推理）
│       ├── memory_tree/          # 🌳 记忆树
│       │   ├── node.py           # MemoryNode（leaf / abstract 双类型）
│       │   ├── tree.py           # HierarchicalMemoryTree（在线构建）
│       │   ├── operations.py     # MLPElevation / semantic_elevation
│       │   └── tree_ssm.py       # TreeSSMReadout（< 1 ms，批量预计算优化）
│       └── sgmts/                # 🌿 视觉树
│           └── sgmts.py          # SGMTSEncoder（网格边缓存 + 层级并行 Tree-SSM，< 5 ms）
├── scripts/
│   ├── pretrain.py           # 全模型预训练主入口
│   ├── pretrain.sh           # 多卡启动脚本
│   ├── train.py              # Phase 1 / 2 训练主入口
│   ├── train_phase1.sh       # Phase 1 多卡启动脚本
│   ├── train_phase2.sh       # Phase 2 多卡启动脚本（ZeRO-3）
│   ├── eval.py               # 离线轨迹评估（L1/L2，teacher-forcing）
│   ├── eval_libero_sim.py    # LIBERO 仿真成功率评估（单进程）
│   ├── eval_server.py        # LIBERO 评估：WebSocket 推理服务端（GPU 节点）
│   └── eval_client.py        # LIBERO 评估：WebSocket 仿真客户端
├── checkpoints/
│   ├── Qwen2.5-0.5B/         # LLM 权重（已预置）
│   └── Qwen2.5-1.5B-Instruct/
├── requirements.txt
└── CONSTRUCTION.md           # 双树架构设计详细文档
```

---

## 云服务器环境配置

### 硬件要求

| 组件 | 最低 | 推荐（本项目） |
|---|---|---|
| GPU | 1× A100 40G | **8× RTX A6000 48G** |
| CPU | 16 核 | 64 核 |
| 内存 | 128 GB | 256 GB |
| 存储 | 500 GB SSD | 2 TB NVMe |
| 互联 | PCIe | **NVLink / NVSwitch** |

> RTX A6000 为 Ampere 架构（sm_86），完整支持 Flash Attention 2、BFloat16 和 DeepSpeed ZeRO。

---

### 系统依赖

以 **Ubuntu 22.04 / 20.04** 为例：

```bash
# 无 sudo 权限时，用 conda 安装等价依赖（推荐）
conda install -c conda-forge ninja cmake compilers openmpi -y

# 查看当前 CUDA 驱动版本
nvidia-smi

# 配置环境变量（添加到 ~/.bashrc）
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version && nvidia-smi
```

---

### Python 环境

```bash
conda create -n dualtree python=3.10 -y
conda activate dualtree

git clone <YOUR_REPO_URL> DualTreeVLA
cd DualTreeVLA
```

---

### PyTorch 安装

**必须先安装 PyTorch，再安装 flash-attn**（flash-attn 编译时需要与 torch 版本匹配）。

```bash
# PyTorch 2.2 + CUDA 12.1（推荐组合）
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 验证
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('BF16 support:', torch.cuda.is_bf16_supported())
"
```

---

### Flash Attention 安装

```bash
export CUDA_HOME=/usr/local/cuda
pip install flash-attn --no-build-isolation

# 验证
python -c "from flash_attn import flash_attn_func; print('flash-attn OK')"
```

> 若编译失败可跳过，项目自动回退到 PyTorch 内置 SDPA，性能损失极小。

---

### DeepSpeed 安装

```bash
# 安装项目其余依赖
pip install -r requirements.txt

# 安装 DeepSpeed
DS_BUILD_OPS=1 pip install deepspeed --no-build-isolation

# 验证
python -c "import deepspeed; print('DeepSpeed', deepspeed.__version__)"
ds_report
```

> 若编译过慢，可使用 `pip install deepspeed` 跳过预编译，算子将在首次使用时 JIT 编译。

---

### Weights & Biases 安装（可选）

```bash
pip install wandb
wandb login   # 令牌保存在 ~/.netrc，只需执行一次
```

---

### LIBERO 仿真环境（仿真评估专用）

> **仅运行 `eval_libero_sim.py`（MuJoCo 在线推理成功率评估）时才需要安装**。离线 `eval.py` 不依赖这些包。

**版本严格要求**：`robosuite` 必须使用 `1.4.0`，更新版本会导致 `single_arm_env` 导入失败。

```bash
# 1. 安装 LIBERO 仿真依赖（版本钉死）
pip install robosuite==1.4.0
pip install bddl==1.0.1
pip install easydict einops thop cloudpickle gym==0.25.2 future matplotlib

# 2. 安装 LIBERO Python 包（从子目录）
pip install -e dataset/LIBERO

# 3. 验证
python -c "
import robosuite; print('robosuite', robosuite.__version__)
import bddl;       print('bddl OK')
import libero;     print('libero OK')
"
```

**无头渲染（服务器 / 无显示器）**：脚本默认设置 `MUJOCO_GL=osmesa`。若 osmesa 未安装：

```bash
# 方式一：conda 安装 osmesa（推荐，无 sudo）
conda install -c conda-forge mesalib -y

# 方式二：切换为 EGL（需要 NVIDIA EGL 支持）
export MUJOCO_GL=egl

# 方式三：apt 安装（需要 sudo）
sudo apt-get install -y libosmesa6-dev
```

---

## 数据准备

> 本项目**不需要 mini-ImageNet**。视觉树（SGMTS）使用冻结的 CLIP ViT 作为 patch 特征提取器，其视觉语义能力直接来自 CLIP 的预训练，无需额外的分类预训练阶段。

### RoboCerebra 训练集

```
dataset/RoboCerebra/RoboCerebra_trainset/
├── coffee_table/
│   ├── case1/
│   │   ├── demo.hdf5              # 动作(T,7) + 状态(T,84)
│   │   ├── case1.mp4              # RGB 视频
│   │   └── task_description.json  # 子任务标注
│   └── case2/ ...
├── kitchen_table/
└── study_table/
```

`task_description.json` 格式示例：

```json
{
  "high_level_instruction": "Pick the mug and place it on the tray.",
  "steps": [
    {
      "step_number": 1,
      "subtask_description": "Reach toward the mug",
      "timestep": {"start": 0, "end": 120},
      "related_objects": ["mug"]
    }
  ]
}
```

下载：

```bash
pip install -U huggingface_hub
huggingface-cli download qiukingballball/RoboCerebra \
    --repo-type dataset \
    --include "RoboCerebra_trainset/**" \
    --local-dir dataset/RoboCerebra
```

验证加载：

```bash
python -c "
from dual_tree_vla.dataset import RoboCerebraDataset
ds = RoboCerebraDataset('dataset/RoboCerebra/RoboCerebra_trainset', subsample=4)
print(f'Trajectories: {len(ds)}')
s = ds[0]
print('frames:', s['frames'].shape, '  actions:', s['actions'].shape)
"
```

---

### RoboCerebraBench

官方基准集，包含六种任务类型各 10 个 case：

```
dataset/RoboCerebra/RoboCerebraBench/
├── Ideal/
├── Memory_Execution/
├── Memory_Exploration/
├── Mix/
├── Observation_Mismatching/
└── Random_Disturbance/
    └── case1/
        ├── demo.hdf5
        ├── case1.mp4
        └── task_description.txt
```

下载：

```bash
huggingface-cli download qiukingballball/RoboCerebra \
    --repo-type dataset \
    --include "RoboCerebraBench/**" \
    --local-dir dataset/RoboCerebra
```

---

### LIBERO

用于 Phase 1 / 2 训练（LeRobot v2 parquet 格式）：

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lerobot/libero_10_image',
    repo_type='dataset',
    local_dir='dataset/LIBERO/libero_10',
)
print('Done')
"
```

> 如需下载 libero_spatial / libero_object / libero_goal，将 `repo_id` 中的 `libero_10_image` 替换为对应名称。

---

## 模型权重下载

项目已预置 `checkpoints/Qwen2.5-0.5B/` 和 `checkpoints/Qwen2.5-1.5B-Instruct/`。若权重缺失：

```bash
# 方式一：ModelScope（国内推荐）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='checkpoints')
"

# 方式二：HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir checkpoints/Qwen2.5-1.5B-Instruct
```

> CLIP 视觉骨干（`openai/clip-vit-base-patch16`）在首次运行时由 `transformers` 自动下载到 HuggingFace 缓存，无需手动管理。如需离线使用，设置 `clip_model_name` 为本地路径。

---

## 配置文件说明

| 配置文件 | 用途 | 主要可训练模块 |
|---|---|---|
| `configs/pretrain.yaml` | 全模型预训练（RoboCerebra） | SGMTS adapter, sem_proj, JumpAwareHead, TreeSSM, MLPElevation |
| `configs/train_phase1.yaml` | Phase 1：LIBERO FlowMatching 热身 | CrossModalFusion, FlowMatchingActionHead |
| `configs/train_phase2.yaml` | Phase 2：LIBERO 全量微调 | 全部模块 |
| `configs/default.yaml` | 评估 / 单阶段默认 | — |

关键共享参数（`configs/default.yaml`）：

```yaml
model:
  llm_path:        "checkpoints/Qwen2.5-0.5B"
  clip_model_name: null   # CLIP 视觉骨干 HuggingFace ID（null = PatchCNN fallback）
                          # 推荐填入 "openai/clip-vit-base-patch16"
  d:               256    # 统一嵌入维度
  H_a:             16     # 动作预测步长
  theta_fuse:      0.35   # 记忆树合并阈值（预训练后 0.35，未对齐时 0.65）
  mount_tau:       0.4    # 挂载点搜索阈值（余弦距离）
  max_tree_depth:  4      # 记忆树最大深度（超出则剪枝旧记忆）
```

> **关于 CLIP**：`clip_model_name` 留空时退回轻量 `PatchCNN`（适合无网络调试）。生产训练推荐填入 CLIP 模型 ID，视觉树可利用预训练语义特征，**无需任何分类预训练**。

---

## 训练

### 3 阶段训练流程总览

```
全模型预训练（RoboCerebra）  ──→  Phase 1（LIBERO）  ──→  Phase 2（LIBERO）
  L_boundary + L_sem                 L_flow only              L_flow only
  SGMTS adapter + JumpAwareHead      CrossModalFusion          全部模块
  TreeSSM + MLPElevation             FlowMatchingHead          LLM: 0.1× LR
```

| 阶段 | 数据集 | 可训练模块 | 损失函数 | 脚本 |
|---|---|---|---|---|
| **全模型预训练** | RoboCerebra | SGMTS adapter, JumpAwareHead, TreeSSM, MLPElevation | $L_\text{boundary}+L_\text{sem}$ | `pretrain.sh` |
| **Phase 1** | LIBERO | CrossModalFusion, FlowMatchingActionHead | $L_\text{flow}$ | `train_phase1.sh` |
| **Phase 2** | LIBERO | 全部模块 | $L_\text{flow}$ | `train_phase2.sh` |

**冻结说明**：
- CLIP ViT 在整个训练流程中**始终冻结**，只有轻量 Adapter（Linear + LayerNorm）参与训练
- LLM backbone 在预训练 / Phase 1 阶段冻结；Phase 2 以 0.1× 学习率微调

---

### 全模型预训练（RoboCerebra）

目标：让视觉树学会语义引导的 patch 扫描，让记忆树学会子任务边界检测与语义提升。

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

# 单卡调试
python scripts/pretrain.py --config configs/pretrain.yaml

# 多卡（推荐，8 GPU）
bash scripts/pretrain.sh

# 指定 GPU 数量
bash scripts/pretrain.sh 4
```

Checkpoint 保存至 `checkpoints/runs/pretrain/`，最优模型为 `pretrain_best.pt`。

---

### Phase 1 — FlowMatching 热身（LIBERO）

冻结双树与 LLM，仅训练 CrossModalFusion 和 FlowMatchingActionHead。
`train.init_from` 默认指向预训练输出 `checkpoints/runs/pretrain/pretrain_best.pt`，请确保已完成预训练。

```bash
cd /data/wyx/zd/DualTreeVLA
conda activate dualtree

# 8 GPU（推荐，ZeRO-2）
bash scripts/train_phase1.sh

# 指定 GPU 数量（如 4 卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_phase1.sh 4

# 单卡调试
python scripts/train.py --config configs/train_phase1.yaml --phase 1
```

Checkpoint 保存至 `checkpoints/runs/phase1/`，最优模型为 `phase1_best.pt`。
训练过程可视化视频保存至 `results/viz/phase1/phase1_ep001.mp4`，…（详见[训练可视化](#训练可视化gt-vs-预测对比视频)）。

---

### Phase 2 — 全量微调（LIBERO）

解冻全部模块（LLM 以 0.1× 学习率），使用 DeepSpeed ZeRO-3：

```bash
# 多卡（ZeRO-3，自动加载 configs/ds_zero3.json）
bash scripts/train_phase2.sh

# 指定 GPU 数量
bash scripts/train_phase2.sh 4
```

配置文件 `configs/train_phase2.yaml` 中 `train.init_from` 指向 Phase 1 最优 checkpoint。

---

### 断点续训

```bash
# 预训练断点续训
python scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --resume checkpoints/runs/pretrain/pretrain_ep015.pt

# Phase 1 断点续训
python scripts/train.py \
    --config configs/train_phase1.yaml \
    --phase 1 \
    --resume checkpoints/runs/phase1/phase1_ep010.pt
```

Checkpoint 结构：

```
checkpoints/runs/
├── pretrain/
│   ├── pretrain_ep010.pt
│   └── pretrain_best.pt
├── phase1/
│   ├── phase1_ep005.pt
│   └── phase1_best.pt
└── phase2/
    ├── phase2_ep005/        # DeepSpeed ZeRO-3 格式（目录）
    └── phase2_best.pt
```

---

### Weights & Biases 可视化

```bash
# 国内网络推荐 offline 模式
WANDB_MODE=offline python scripts/pretrain.py \
    --config configs/pretrain.yaml

# 手动同步
wandb sync wandb/offline-run-*
```

各阶段 wandb 项目名通过对应 yaml 中的 `wandb_project` 字段配置。

---

### 训练可视化（GT vs 预测对比视频）

`train.py` 每个 epoch 结束后自动用数据集中一个固定 episode 生成对比视频，
无需仿真环境，直接观察 flow matching 收敛情况。

视频保存位置：`results/viz/phase{1,2}/phase{1,2}_ep{epoch:03d}.mp4`

每帧内容：
```
┌────────────────────────────────────┐
│  观测图像（agentview, 224×224）     │
├────────────────────────────────────┤
│ ep0 t=42  pick up the bowl ...     │  ← 灰色：任务描述
│ GT   xyz:+0.12 +0.03 -0.04  g:+1  │  ← 绿色：归一化空间 GT 动作
│ Pred xyz:+0.48 +0.25 +0.15  g:+0  │  ← 蓝色：模型预测动作
│ MAE norm: 0.312  (mean: 0.287)     │  ← 黄色：逐步 / 全集平均 MAE
└────────────────────────────────────┘
```

**收敛参考**：
- MAE > 0.3：flow head 尚未收敛，仿真评估意义不大
- MAE < 0.1：GT 与 Pred 方向基本一致，可开始仿真评估

可在 yaml 的 `train:` 块中调整可视化参数：

```yaml
train:
  viz_every:       1       # 每 N epoch 保存一次；0 = 禁用
  viz_dir:         "results/viz"
  viz_episode:     0       # 使用数据集第几个 episode
  viz_max_frames:  120     # 最多截取帧数（viz_fps=10 → 12 秒）
  viz_fps:         10
```

> **依赖**：仅需 `cv2`（`pip install opencv-python`），无需 `imageio` 或仿真环境。

---

## 评估

### 评估指标

`eval.py` 支持对以下三个基准进行**离线轨迹评估**，使用 teacher-forcing 方式逐帧喂入模型：

| 指标 | 说明 | 适用数据集 |
|---|---|---|
| `action_l1` | 每步 $\|a_{\text{pred}}[0] - a_{\text{gt}}\|_1$ | 全部 |
| `action_l2` | 每步 $\|a_{\text{pred}}[0] - a_{\text{gt}}\|_2$ | 全部 |
| `tree_nodes` | 轨迹末尾记忆树节点数（均值） | 全部 |
| `tree_depth` | 轨迹末尾记忆树最大深度（均值） | 全部 |
| `tree_branches` | 每条轨迹分支创建次数 | 全部 |
| `tree_elevations` | 每条轨迹语义提升次数 | 全部 |
| `subtask_boundary_f1` | 分支事件与 GT 子任务边界的 F1（±`boundary_tol` 步容忍） | RoboCerebra / Bench |
| `subtask_sr` | GT 子任务边界中被成功检测的比例 | RoboCerebra / Bench |
| `prog_monotone_rate` | 树中祖先-后代对语义进度单调的比例 | RoboCerebra / Bench |

---

### RoboCerebraBench 评估

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

# 评估全部 6 种任务类型（推荐）
python scripts/eval.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --config configs/default.yaml \
    --dataset robocerebra_bench \
    --bench_root dataset/RoboCerebra/RoboCerebraBench \
    --out results/bench_eval.json

# 只评估特定子集
python scripts/eval.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --dataset robocerebra_bench \
    --bench_root dataset/RoboCerebra/RoboCerebraBench \
    --task_types Ideal Random_Disturbance \
    --out results/bench_partial.json

# 快速调试（限制轨迹数）
python scripts/eval.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --dataset robocerebra_bench \
    --bench_root dataset/RoboCerebra/RoboCerebraBench \
    --max_traj 6 \
    --device cpu
```

**典型输出**：

```
PER TASK-TYPE RESULTS  (RoboCerebraBench)
================================================================================
  Ideal                    n=10  L1=0.0812  L2=0.1534  F1=0.7241  SR=0.7833
  Memory_Execution         n=10  L1=0.0891  L2=0.1672  F1=0.6823  SR=0.7500
  Memory_Exploration       n=10  L1=0.0934  L2=0.1758  F1=0.6512  SR=0.7167
  Mix                      n=10  L1=0.1023  L2=0.1921  F1=0.6102  SR=0.6833
  Observation_Mismatching  n=10  L1=0.0967  L2=0.1813  F1=0.6321  SR=0.7000
  Random_Disturbance       n=10  L1=0.0988  L2=0.1856  F1=0.6234  SR=0.6917

OVERALL  action_l1=0.0936  action_l2=0.1759  F1=0.6539  SR=0.7208  mono=0.8612
```

---

### RoboCerebra 训练集评估

```bash
python scripts/eval.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --config configs/default.yaml \
    --dataset robocerebra \
    --data_root dataset/RoboCerebra/RoboCerebra_trainset \
    --out results/robocerebra_train_eval.json

# 只评估部分场景
python scripts/eval.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --dataset robocerebra \
    --data_root dataset/RoboCerebra/RoboCerebra_trainset \
    --scenes coffee_table kitchen_table \
    --max_traj 50
```

---

### LIBERO 离线评估

```bash
# LIBERO-10（主要测试集）
python scripts/eval.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --config configs/default.yaml \
    --dataset libero \
    --data_root dataset/LIBERO \
    --libero_split long \
    --out results/libero_long_eval.json

# 所有子集
for SPLIT in spatial object goal long; do
    python scripts/eval.py \
        --ckpt  checkpoints/runs/phase2/phase2_best.pt \
        --dataset libero \
        --data_root dataset/LIBERO \
        --libero_split ${SPLIT} \
        --out results/libero_${SPLIT}_eval.json
done
```

---

### LIBERO 仿真成功率评估

`eval_libero_sim.py` 直接运行 MuJoCo 仿真，统计每个任务的**真实成功率（SR）**，不依赖预录轨迹。

> 需要先按 [LIBERO 仿真环境](#libero-仿真环境仿真评估专用) 一节完成依赖安装。

**单进程模式**：模型和仿真在同一进程中运行。

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

# LIBERO-10（10 个任务，每任务 20 rollout）
python scripts/eval_libero_sim.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --config configs/train_phase2.yaml \
    --suite libero_10 \
    --num_episodes 20 \
    --out results/libero10_sim_eval.json

# 快速验证（每任务 3 rollout）
python scripts/eval_libero_sim.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --config configs/train_phase2.yaml \
    --suite libero_10 \
    --num_episodes 3 \
    --out results/libero10_sim_quick.json

# 其他子集
python scripts/eval_libero_sim.py \
    --ckpt  checkpoints/runs/phase2/phase2_best.pt \
    --suite libero_spatial \
    --num_episodes 20
```

---

### LIBERO 仿真成功率评估（服务端+客户端）

模仿 Evo-1 的 Server/Client 架构：GPU 推理节点与 MuJoCo 仿真节点用 WebSocket 分离，
也可在同一台机器上两个终端分别运行。

```
eval_server.py  ←── WebSocket ──→  eval_client.py
(GPU 节点，模型)                    (CPU/GPU，MuJoCo 仿真)
```

**步骤一：启动推理服务端**（GPU 节点）

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

python scripts/eval_server.py \
    --ckpt   checkpoints/runs/phase2/phase2_best.pt \
    --config configs/train_phase2.yaml \
    --port   9000
# 输出：[Server] Listening on ws://0.0.0.0:9000
```

**步骤二：启动仿真客户端**（另开终端或远程节点）

```bash
conda activate dualtree
cd /path/to/DualTreeVLA

# 评估所有 4 个 suite
python scripts/eval_client.py \
    --server  ws://127.0.0.1:9000 \
    --suites  libero_spatial libero_object libero_goal libero_10 \
    --num_episodes 10 \
    --out results/all_suites_eval.json

# 只评估 libero_10
python scripts/eval_client.py \
    --server ws://127.0.0.1:9000 \
    --suite  libero_10 \
    --num_episodes 10 \
    --save_video

# 快速确认（2 任务 × 3 rollout）
python scripts/eval_client.py \
    --server ws://127.0.0.1:9000 \
    --suite  libero_10 \
    --num_episodes 3 \
    --max_task_id 2
```

**客户端常用参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server` | `ws://127.0.0.1:9000` | 服务端 WebSocket URL |
| `--suite` / `--suites` | `libero_10` | 单个或多个 suite（空格分隔） |
| `--num_episodes` | `10` | 每任务 rollout 次数 |
| `--max_steps` | 自动 | 每 rollout 最大步数 |
| `--horizon` | `16` | 每次推理后执行的动作步数 |
| `--save_video` | 关闭 | 保存每 episode MP4 至 `--video_dir` |
| `--gripper_thresh` | `0.0` | 夹爪二值化阈值（> thresh → 开 +1） |
| `--out` | 不保存 | 结果保存为 JSON |

**服务端常用参数**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | 必填 | Phase 2 checkpoint 路径 |
| `--config` | `configs/train_phase2.yaml` | 模型配置 YAML |
| `--port` | `9000` | 监听端口 |
| `--host` | `0.0.0.0` | 绑定地址（`0.0.0.0` = 全部网卡） |
| `--stats` | 自动检测 | stats.json 路径 |

> **依赖**：`pip install "websockets>=12.0"`（已加入 `requirements.txt`）

---

### 常用评估参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | 必填 | `.pt` 模型文件或 DeepSpeed checkpoint 目录 |
| `--config` | `configs/default.yaml` | 与训练时一致的 YAML |
| `--dataset` | 必填 | `robocerebra_bench` / `robocerebra` / `libero` |
| `--bench_root` | `dataset/RoboCerebra/RoboCerebraBench` | RoboCerebraBench 目录 |
| `--task_types` | 全 6 类 | 只评估指定子集，空格分隔 |
| `--data_root` | — | robocerebra / libero 数据目录 |
| `--libero_split` | `long` | `spatial`, `object`, `goal`, `long` |
| `--max_traj` | 全部 | 限制轨迹数（调试用） |
| `--boundary_tol` | `5` | 子任务边界 F1 容忍窗口（步数） |
| `--print_tree` | 关闭 | 打印每条轨迹结束时的记忆树 ASCII 结构 |
| `--out` | 不保存 | 结果保存为 JSON |

---

### 结果解读

| 指标 | 良好范围 | 含义 |
|---|---|---|
| `action_l1 / l2` | 越小越好 | 预测动作与真值的平均偏差 |
| `tree_nodes / depth` | 与任务复杂度匹配 | 过大 → `theta_fuse` 偏小；过小 → 偏大 |
| `subtask_boundary_f1` | 接近 1.0 | 模型感知子任务切换的准确度 |
| `subtask_sr` | 接近 1.0 | GT 边界被成功检测的比例 |
| `prog_monotone_rate` | 接近 1.0 | 树中时序语义层次的清晰程度 |

---

## 常见问题

### 1. `NCCL` 通信超时

```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0    # 替换为实际网卡名（ip link 查看）
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
```

### 2. Flash Attention 编译失败

```bash
# 检查 CUDA 与 torch 版本是否匹配
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# 清理缓存后重新编译
pip uninstall flash-attn -y && pip cache purge
pip install flash-attn --no-build-isolation
```

### 3. DeepSpeed 算子未编译

```bash
# 强制预编译
DS_BUILD_OPS=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed --no-build-isolation
```

### 4. 显存不足（OOM）

```bash
# 1. 减小 batch_size（对应 yaml 的 train.batch_size: 2）
# 2. 增大 grad_accum（对应 yaml 的 grad_accum: 4）
# 3. Phase 2 已默认使用 ZeRO-3，若仍 OOM 可开启 CPU offload（ds_zero3.json）
# 4. 使用 0.5B 替代 1.5B LLM
model:
  llm_path: "checkpoints/Qwen2.5-0.5B"
```

### 5. CLIP 权重下载失败（国内网络）

```bash
# 方式一：设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
# 方式二：手动下载后指定本地路径
model:
  clip_model_name: "/path/to/local/clip-vit-base-patch16"
```

### 6. 数据加载慢

```bash
# 增加 DataLoader workers（yaml 的 train.num_workers: 8）
```

### 7. `robosuite.environments.manipulation.single_arm_env` 导入失败

```bash
# robosuite 版本必须为 1.4.0，新版本已移除该模块路径
pip uninstall robosuite -y
pip install robosuite==1.4.0
```

### 8. LIBERO 仿真黑屏 / `No such display` / osmesa 报错

```bash
# 服务器无显示器时，必须使用软件渲染
export MUJOCO_GL=osmesa   # 或 egl（需 NVIDIA EGL）

# 验证 osmesa 可用
python -c "import mujoco; import os; os.environ['MUJOCO_GL']='osmesa'; print('osmesa OK')"

# 若 osmesa 缺失，conda 无 sudo 安装
conda install -c conda-forge mesalib -y
```

### 9. `No module named 'bddl'` / `easydict'` / `gym'`

```bash
# 一次性安装 LIBERO 仿真全部依赖
pip install bddl==1.0.1 easydict einops thop cloudpickle gym==0.25.2 future matplotlib
```

### 10. LIBERO 仿真初始化极慢（首次 > 5 min）

robosuite 首次运行会下载并编译 MuJoCo asset，属于正常现象，后续运行会使用缓存。若服务器无外网，提前将 `~/.mujoco` 目录从有网机器复制过去。

---

### 11. `websockets` 未安装（eval_server / eval_client）

```bash
pip install "websockets>=12.0"
```

---

## 引用

如果本项目对您的研究有帮助，请考虑引用相关工作。

