# MemoryTreeVLA

层次记忆树视觉-语言-动作模型（Hierarchical Memory Tree Vision-Language-Action Model）

**MemoryTreeVLA** 将在线构建的层次记忆树（HMT）与语义图最大生成树扫描（SGMTS）视觉编码器、Qwen2.5 大语言模型以及基于流匹配的动作预测头融合，用于长时序机器人操作任务。

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
3. [数据准备](#数据准备)
   - [mini-ImageNet](#mini-imagenet)
   - [RoboCerebra 训练集](#robocerebra-训练集)
   - [RoboCerebraBench](#robocerebrabench)
   - [LIBERO](#libero)
4. [模型权重下载](#模型权重下载)
5. [配置文件说明](#配置文件说明)
6. [训练](#训练)
   - [4 阶段训练流程总览](#4-阶段训练流程总览)
   - [阶段 a — SGMTS Backbone 预训练（mini-ImageNet）](#阶段-a--sgmts-backbone-预训练mini-imagenet)
   - [阶段 b — 全模型预训练（RoboCerebra）](#阶段-b--全模型预训练robocerebra)
   - [Phase 1 — FlowMatching 热身（LIBERO）](#phase-1--flowmatching-热身libero)
   - [Phase 2 — 全量微调（LIBERO）](#phase-2--全量微调libero)
   - [断点续训](#断点续训)
   - [Weights & Biases 可视化](#weights--biases-可视化)
7. [评估](#评估)
   - [评估指标](#评估指标)
   - [RoboCerebraBench 评估](#robocerebrabench-评估)
   - [RoboCerebra 训练集评估](#robocerebra-训练集评估)
   - [LIBERO 评估](#libero-评估)
   - [结果解读](#结果解读)
8. [常见问题](#常见问题)

---

## 项目结构

```
MemoryTreeVLA/
├── configs/
│   ├── default.yaml          # 评估 / 单阶段默认超参
│   ├── pretrain.yaml         # 阶段 b RoboCerebra 预训练配置
│   ├── train_phase1.yaml     # Phase 1 LIBERO FlowMatching 热身
│   ├── train_phase2.yaml     # Phase 2 LIBERO 全量微调
│   ├── ds_zero2.json         # DeepSpeed ZeRO-2（Phase 1 推荐）
│   └── ds_zero3.json         # DeepSpeed ZeRO-3 + CPU offload（Phase 2 推荐）
├── dataset/
│   ├── mini-imagenet/        # SGMTS backbone 预训练数据
│   ├── RoboCerebra/
│   │   ├── RoboCerebra_trainset/   # 三场景训练集
│   │   └── RoboCerebraBench/       # 六子集评测集
│   └── LIBERO/               # LeRobot v2 parquet 格式数据集
│       └── libero_10/
├── memory_tree_vla/
│   ├── dataset/
│   │   ├── libero.py             # LIBERO LeRobot 数据加载器
│   │   ├── robocerebra.py        # RoboCerebra 训练集加载器
│   │   └── robocerebra_bench.py  # RoboCerebraBench 六子集评测加载器
│   ├── losses/
│   │   └── tree_losses.py        # l_boundary / l_sem / l_elev / l_recon
│   └── model/
│       ├── attn.py               # FlashMHA（自动选择 Flash Attn 2 / SDPA）
│       ├── fusion.py             # CrossModalFusion（三路门控融合）
│       ├── memory_tree_vla.py    # MemoryTreeVLA 主模型
│       ├── semantic_jump_head.py # JumpAwareHead（Mamba 动作突变检测）
│       ├── action_head/
│       │   └── flow_matching.py  # FlowMatchingActionHead（DiT + ODE）
│       ├── memory_tree/
│       │   ├── node.py           # MemoryNode（leaf / abstract 双类型）
│       │   ├── tree.py           # HierarchicalMemoryTree（insert / merge / branch）
│       │   ├── operations.py     # MLPElevation / semantic_elevation
│       │   └── tree_ssm.py       # TreeSSMReadout（权重自适应 Mamba 树递推）
│       └── sgmts/
│           └── sgmts.py          # SGMTSEncoder（MST + 语义 Tree-SSM 扫描）
├── scripts/
│   ├── pretrain.py           # 阶段 b 训练主入口
│   ├── pretrain.sh           # 阶段 b 多卡启动脚本
│   ├── train.py              # Phase 1 / 2 训练主入口
│   ├── train_phase1.sh       # Phase 1 多卡启动脚本
│   ├── train_phase2.sh       # Phase 2 多卡启动脚本（ZeRO-3）
│   └── eval.py               # 离线评估脚本
├── checkpoints/
│   ├── Qwen2.5-0.5B/         # LLM 权重（已预置）
│   └── Qwen2.5-1.5B-Instruct/
├── requirements.txt
└── CONSTRUCTION.md           # 架构设计详细文档
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

以 **Ubuntu 22.04 / 20.04** 为例（CentOS/Rocky 类似）：

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
# 推荐 conda 管理环境（Python 3.10）
conda create -n memorytree python=3.10 -y
conda activate memorytree

# 克隆项目
git clone <YOUR_REPO_URL> MemoryTreeVLA
cd MemoryTreeVLA
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

## 数据准备

### mini-ImageNet

用于阶段 a 的 **SGMTS backbone 预训练**（图像分类监督），需提前下载。

```
dataset/mini-imagenet/
└── data/
    ├── train/
    │   ├── n01532829/   # 每类约 600 张 JPEG
    │   └── ...          # 共 64 类
    ├── val/             # 16 类
    └── test/            # 20 类
```

> 从 [Kaggle mini-ImageNet](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) 或官方来源下载后解压到 `dataset/mini-imagenet/data/`。

---

### RoboCerebra 训练集

```
dataset/RoboCerebra/RoboCerebra_trainset/
├── coffee_table/
│   ├── case1/
│   │   ├── demo.hdf5              # 动作(T,7) + 状态(T,84)
│   │   ├── case1.mp4              # RGB 视频
│   │   └── task_description.json  # 子任务标注
│   └── case2/
│       └── ...
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

重新下载：

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
from memory_tree_vla.dataset import RoboCerebraDataset
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
        ├── demo.hdf5              # 动作(T,7) + 状态(T,71)
        ├── case1.mp4
        └── task_description.txt
```

重新下载：

```bash
huggingface-cli download qiukingballball/RoboCerebra \
    --repo-type dataset \
    --include "RoboCerebraBench/**" \
    --local-dir dataset/RoboCerebra
```

---

### LIBERO

用于 Phase 1 / 2 训练（LeRobot v2 parquet 格式），已通过脚本下载至 `dataset/LIBERO/libero_10/`：

```bash
# 重新下载（若目录缺失）
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

---

## 配置文件说明

| 配置文件 | 用途 | 主要可训练模块 |
|---|---|---|
| `configs/pretrain.yaml` | 阶段 b：RoboCerebra 预训练 | SGMTS, sem_proj, JumpAwareHead, TreeSSM, MLPElevation |
| `configs/train_phase1.yaml` | Phase 1：LIBERO FlowMatching 热身 | CrossModalFusion, FlowMatchingActionHead |
| `configs/train_phase2.yaml` | Phase 2：LIBERO 全量微调 | 全部模块 |
| `configs/default.yaml` | 评估 / 单阶段默认 | — |

关键共享参数（`configs/default.yaml`）：

```yaml
model:
  llm_path:    "checkpoints/Qwen2.5-0.5B"
  sgmts_ckpt:  null        # 阶段 a 训练后的 SGMTS 权重路径
  d:           256         # 统一嵌入维度
  H_a:         16          # 动作预测步长
  theta_fuse:  0.35        # 记忆树合并阈值（预训练后 0.35，未对齐时 0.65）
  K_elev:      4           # 触发语义提升的子节点数阈值
```

---

## 训练

### 4 阶段训练流程总览

```
[a] SGMTS Backbone 预训练  ──→  [b] 全模型预训练  ──→  Phase 1  ──→  Phase 2
    mini-ImageNet (L_cls)        RoboCerebra             LIBERO         LIBERO
                                 (L_boundary+L_sem+L_elev) (L_flow)      (L_flow)
```

| 阶段 | 数据集 | 可训练模块 | 损失函数 | 脚本 |
|---|---|---|---|---|
| **[a] SGMTS 预训练** | mini-ImageNet | SGMTSEncoder (backbone only) | $L_\text{cls}$ (分类交叉熵) | `pretrain.py --stage sgmts` |
| **[b] 全模型预训练** | RoboCerebra | SGMTS, sem_proj, JumpAwareHead, TreeSSM, MLPElevation | $L_\text{boundary}+L_\text{sem}+L_\text{elev}$ | `pretrain.sh` |
| **Phase 1** | LIBERO | CrossModalFusion, FlowMatchingActionHead | $L_\text{flow}$ | `train_phase1.sh` |
| **Phase 2** | LIBERO | 全部模块 | $L_\text{flow}$ | `train_phase2.sh` |

> **冻结说明**：LLM backbone 在 [b] / Phase 1 阶段始终冻结。Phase 2 中以 0.1× 学习率微调 LLM。

---

### 阶段 a — SGMTS Backbone 预训练（mini-ImageNet）

单独训练 SGMTS 视觉编码器的 Backbone，在 mini-ImageNet 上做图像分类，获得有效的语义补丁特征：

```bash
conda activate memorytree
cd /path/to/MemoryTreeVLA

# 单卡调试
python scripts/pretrain.py --config configs/default.yaml --stage sgmts

# 多卡（推荐）
accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    scripts/pretrain.py \
        --config configs/default.yaml \
        --stage sgmts \
        --data_root dataset/mini-imagenet/data \
        --epochs 100 \
        --ckpt_dir checkpoints/runs/sgmts
```

训练完成后将权重路径填入 `configs/default.yaml` 的 `model.sgmts_ckpt`：

```yaml
model:
  sgmts_ckpt: "checkpoints/runs/sgmts/sgmts_best.pt"
```

---

### 阶段 b — 全模型预训练（RoboCerebra）

基于阶段 a 的 SGMTS，训练 JumpAwareHead、TreeSSM、MLPElevation 等语义结构学习模块：

```bash
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

冻结语义模块，仅训练 CrossModalFusion 和 FlowMatchingActionHead：

```bash
# 单卡调试（需先完成阶段 b）
python scripts/train.py --config configs/train_phase1.yaml --phase 1

# 多卡
bash scripts/train_phase1.sh

# 指定 GPU 数量
bash scripts/train_phase1.sh 4
```

配置文件 `configs/train_phase1.yaml` 中 `model.pretrain_ckpt` 指向阶段 b 的输出：

```yaml
model:
  pretrain_ckpt: "checkpoints/runs/pretrain/pretrain_best.pt"
```

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
# 阶段 b 断点续训（指定 .pt 文件）
python scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --resume checkpoints/runs/pretrain/pretrain_ep015.pt

# Phase 1 断点续训（修改 yaml 中的 resume_from 字段即可）
# 或命令行指定（train.py 支持 --resume）
python scripts/train.py \
    --config configs/train_phase1.yaml \
    --phase 1 \
    --resume checkpoints/runs/phase1/phase1_ep010.pt
```

Checkpoint 结构：

```
checkpoints/runs/
├── sgmts/
│   ├── sgmts_ep050.pt
│   └── sgmts_best.pt
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
# 训练时写入本地（国内网络推荐 offline 模式）
WANDB_MODE=offline python scripts/pretrain.py \
    --config configs/pretrain.yaml

# 手动同步
wandb sync wandb/offline-run-*
```

各阶段 wandb 项目名在对应 yaml 中通过 `wandb_project` 字段配置。

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
conda activate memorytree
cd /path/to/MemoryTreeVLA

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

### LIBERO 评估

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

### 5. 数据加载慢

```bash
# 增加 DataLoader workers（yaml 的 train.num_workers: 8）
```

### 6. mini-ImageNet 找不到类目录

阶段 a 预训练期望目录结构为 `dataset/mini-imagenet/data/train/<class>/`。若下载来源不同，可通过 `--data_root` 指定实际路径。

---

## 引用

如果本项目对您的研究有帮助，请考虑引用相关工作。