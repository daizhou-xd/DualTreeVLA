# MemoryTreeVLA

层次记忆树视觉-语言-动作模型（Hierarchical Memory Tree Vision-Language-Action Model）

**MemoryTreeVLA** 将在线构建的层次记忆树（HMT）与语义图最小生成树扫描（SGMTS）视觉编码器、Qwen2.5 大语言模型以及基于流匹配的动作预测头融合，用于长时序机器人操作任务。

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
4. [模型权重下载](#模型权重下载)
5. [配置文件说明](#配置文件说明)
6. [训练](#训练)
   - [单卡调试](#单卡调试)
   - [8 卡 DeepSpeed 训练](#8-卡-deepspeed-训练)
  - [3 阶段训练流程](#3-阶段训练流程)
   - [断点续训](#断点续训)
   - [Weights & Biases 可视化](#weights--biases-可视化)
7. [评估](#评估)
   - [评估指标](#评估指标)
   - [RoboCerebra 评估](#robocerebra-评估)
   - [LIBERO 评估](#libero-评估)
   - [结果解读](#结果解读)
8. [常见问题](#常见问题)

---

## 项目结构

```
MemoryTreeVLA/
├── configs/
│   ├── default.yaml          # 主训练超参数配置
│   ├── ds_zero2.json         # DeepSpeed ZeRO-2（Phase 1-2 推荐）
│   └── ds_zero3.json         # DeepSpeed ZeRO-3 + CPU offload（Phase 3）
├── dataset/
│   ├── __init__.py
│   └── robocerebra.py        # RoboCerebra 数据集加载器
├── losses/
│   ├── __init__.py
│   └── tree_losses.py        # L_recon / L_sem / L_prog / L_elev
├── models/
│   ├── __init__.py
│   ├── attn.py               # FlashMHA（自动选择 Flash Attn / SDPA 后端）
│   ├── fusion.py             # CrossModalFusion（三路融合）
│   ├── memory_tree_vla.py    # MemoryTreeVLA 主模型
│   ├── action_head/
│   │   └── flow_matching.py  # Flow Matching 动作预测头（DiT + ODE）
│   ├── memory_tree/
│   │   ├── node.py           # MemoryNode 六元组
│   │   ├── tree.py           # HierarchicalMemoryTree（两路插入决策）
│   │   ├── operations.py     # reinforce / semantic_elevation / prune
│   │   └── tree_ssm.py       # TreeSSMReadout（权重自适应 Mamba 树递推）
│   └── sgmts/
│       └── sgmts.py          # SGMTS 编码器（MST + 语义 Tree-SSM 扫描）
├── scripts/
│   └── train_8gpu.sh         # 8 卡 DeepSpeed 启动脚本
├── checkpoints/              # 预训练权重（Qwen2.5）
├── dataset/RoboCerebra/      # 训练数据集
├── requirements.txt
└── train.py                  # 训练主入口
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
# 若有 sudo 权限，安装系统级编译依赖（可选，有 sudo 才能执行）
# sudo apt-get update && sudo apt-get install -y build-essential libaio-dev libopenmpi-dev openmpi-bin

# 无 sudo 权限时，用 conda 安装等价依赖（推荐）
conda install -c conda-forge ninja cmake compilers openmpi -y

# ninja / cmake 也可以用 pip 安装（已写入 requirements.txt）
pip install ninja cmake

# CUDA Toolkit（若服务器未预装，安装对应驱动版本的 CUDA）
# 查看当前驱动版本
nvidia-smi

# CUDA 12.1 示例安装（根据驱动版本选择匹配的 CUDA）
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# 配置环境变量（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
nvidia-smi
```

---

### Python 环境

```bash
# 推荐使用 conda 管理环境
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh

# 创建并激活环境（Python 3.10 与主流深度学习框架最兼容）
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
# PyTorch 2.2 + CUDA 12.1（推荐组合，A6000 完全支持）
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 验证 GPU 可见且 BFloat16/SDPA 可用
python -c "
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('GPU 0:', torch.cuda.get_device_name(0))
print('BF16 support:', torch.cuda.is_bf16_supported())
print('SDPA available:', hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
"
```

期望输出：
```
CUDA: True
GPU count: 8
GPU 0: NVIDIA RTX A6000
BF16 support: True
SDPA available: True
```

---

### Flash Attention 安装

Flash Attention 2 需要在线编译，耗时约 10-30 分钟：

```bash
# 确保环境变量正确
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# 安装（--no-build-isolation 使用当前环境的 torch）
pip install flash-attn --no-build-isolation

# 验证
python -c "from flash_attn import flash_attn_func; print('flash-attn OK')"
```

> **注意**：若编译失败，可跳过此步骤。项目会自动回退到 PyTorch 内置的 SDPA 后端，在 A6000（sm_86）上仍可调用高效的 Flash Attention 内核，性能损失极小。

---

### DeepSpeed 安装

```bash
# 安装其余依赖
pip install -r requirements.txt

# 安装 DeepSpeed（包含 CUDA 算子编译）
DS_BUILD_OPS=1 pip install deepspeed --no-build-isolation

# 验证
python -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"
ds_report   # 打印各算子编译状态
```

> 若 `DS_BUILD_OPS=1` 编译过慢，可使用 `pip install deepspeed` 跳过预编译，算子将在首次使用时 JIT 编译。

---

### Weights & Biases 安装

wandb 为可选依赖，不安装不影响训练（graceful fallback）：

```bash
pip install wandb

# 登录（只需执行一次，令牌保存在 ~/.netrc）
wandb login
# 国内访问不稳定时，可使用代理或 offline 模式（见下方说明）
```

---

## 数据准备

### RoboCerebra 数据集

将数据集解压至项目目录，目录结构如下：

```
dataset/
└── RoboCerebra/
    └── RoboCerebra_trainset/
        ├── coffee_table/
        │   ├── case1/
        │   │   ├── demo.hdf5           # 动作(T,7) + 状态(T,84)
        │   │   ├── case1.mp4           # RGB 视频
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

验证数据加载：

```bash
python -c "
from dataset import RoboCerebraDataset
ds = RoboCerebraDataset('dataset/RoboCerebra/RoboCerebra_trainset', subsample=4)
print(f'Trajectories: {len(ds)}')
sample = ds[0]
print('frames:', sample['frames'].shape)
print('actions:', sample['actions'].shape)
print('states:', sample['states'].shape)
print('instruction:', sample['instruction'])
"
```

---

## 模型权重下载

项目使用 **Qwen2.5-1.5B-Instruct**（Phase 1-2）或 **Qwen2.5-0.5B**（快速调试）：

```bash
# 方式一：从 ModelScope 下载（国内推荐）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='checkpoints')
"

# 方式二：从 HuggingFace 下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir checkpoints/Qwen2.5-1.5B-Instruct

# 验证权重文件
ls checkpoints/Qwen2.5-1.5B-Instruct/
# 应包含: config.json  model.safetensors  tokenizer.json  tokenizer_config.json
```

> 项目中已预置 `checkpoints/Qwen2.5-1.5B-Instruct/` 目录，若权重已存在可跳过此步。

---

## 配置文件说明

主配置文件 `configs/default.yaml`，关键参数：

```yaml
model:
  llm_path:   "checkpoints/Qwen2.5-1.5B-Instruct"
  d:          256       # 统一嵌入维度
  H_a:        16        # 动作预测步长
  theta_fuse: 0.35      # 记忆树合并阈值（越小越易合并）
  K_elev:     4         # 触发语义提升的子节点数阈值

train:
  batch_size: 4         # 每 GPU micro-batch 大小
  lr:         1.0e-4

deepspeed:
  enabled: false                    # 命令行 --deepspeed 覆盖
  config:  "configs/ds_zero2.json"  # Phase 1-2；Phase 3 用 ds_zero3.json
```

DeepSpeed 配置对应关系：

| Phase | 推荐 DeepSpeed 配置 | ZeRO 级别 | 说明 |
|---|---|---|---|
| 1-2 | `configs/ds_zero2.json` | ZeRO-2 | 切分优化器状态+梯度，参数整体保留 |
| 3   | `configs/ds_zero3.json` | ZeRO-3 + CPU offload | 联合微调，显存压力最小 |

---

## 训练

### 单卡调试

在正式多卡训练前，建议先在单卡验证数据流：

```bash
conda activate memorytree
cd /path/to/MemoryTreeVLA

python train.py \
    --config configs/default.yaml \
    --phase 1
```

### 8 卡 DeepSpeed 训练

使用提供的启动脚本：

```bash
# 给脚本添加执行权限
chmod +x scripts/train_8gpu.sh

# Phase 1 — 视觉主干预热（约 20 epochs）
bash scripts/train_8gpu.sh 1

# Phase 2 — 动作预测头训练（约 30 epochs，基于 Phase 1 最优 checkpoint）
bash scripts/train_8gpu.sh 2 --resume checkpoints/runs/phase1_best

# Phase 3 — 全参数联合微调（自动切换 ZeRO-3，约 10 epochs，基于 Phase 2 最优 checkpoint）
bash scripts/train_8gpu.sh 3 --resume checkpoints/runs/phase2_best
```

也可手动调用 `deepspeed`：

```bash
deepspeed \
    --num_gpus 8 \
    --master_port 29500 \
    train.py \
    --deepspeed \
    --deepspeed_config configs/ds_zero2.json \
    --config configs/default.yaml \
    --phase 1
```

### 3 阶段训练流程

| 阶段 | 训练模块 | 损失函数 | 建议 epochs | 建议 lr |
|---|---|---|---|---|
| **Phase 1** 视觉预热 | SGMTS + s_proj + TreeSSM | $L_\text{recon}$ | 20 | 1e-4 |
| **Phase 2** 动作头 | + CrossModalFusion + prog_head + FlowMatchingActionHead | $L_\text{flow} + L_\text{prog}$ | 30 | 1e-4 |
| **Phase 3** 联合微调 | + LLM（全参数） | $L_\text{flow} + L_\text{recon} + L_\text{prog}$ | 10 | 1e-5 |

每阶段完成后修改 `configs/default.yaml` 中的 `train.lr` 和 `train.epochs`，并通过 `--resume` 从上一阶段最佳 checkpoint 继续训练下一阶段。训练会自动维护每阶段的 `phaseX_best`（按最小 `L_total`）。

### 断点续训

```bash
# 8 卡 DeepSpeed：从 Phase 1 最优 checkpoint 继续训练 Phase 2
bash scripts/train_8gpu.sh 2 configs/default.yaml \
  --resume checkpoints/runs/phase1_best

# 单卡（.pt 格式 checkpoint）
python train.py \
    --config configs/default.yaml \
    --phase 3 \
    --resume checkpoints/runs/phase3_epoch0010.pt

# 8 卡 DeepSpeed（目录格式 checkpoint）
bash scripts/train_8gpu.sh 3 configs/default.yaml \
    --resume checkpoints/runs/phase3_epoch0010
```

Checkpoint 默认保存在 `checkpoints/runs/`，每 `save_every`（默认 5）个 epoch 保存一次：

```
checkpoints/runs/
├── phase1_epoch0005/        # DeepSpeed 格式（目录）
│   ├── zero_pp_rank_0_mp_rank_00_model_states.pt
│   └── ...
└── phase3_epoch0010.pt      # 单卡格式（文件）
```

---

### Weights & Biases 可视化

#### 启用 wandb

在任何训练命令后加 `--wandb` 即可启用：

```bash
# 单卡调试 + wandb
python train.py \
    --config configs/default.yaml \
    --phase 1 \
    --wandb \
    --wandb_project MemoryTreeVLA \
    --wandb_name "phase1_debug"

# 8 卡 DeepSpeed + wandb（--wandb 会通过 EXTRA_ARGS 透传）
bash scripts/train_8gpu.sh 1 configs/default.yaml \
    --wandb --wandb_project MemoryTreeVLA --wandb_tags phase1 a6000

# 全阶段训练脚本（带 wandb）
for PHASE in 1 2 3; do
    bash scripts/train_8gpu.sh ${PHASE} configs/default.yaml \
        --wandb --wandb_project MemoryTreeVLA --wandb_tags "phase${PHASE}"
done
```

#### wandb 上可查看的指标

| 面板 | 指标 | 说明 |
|---|---|---|
| **train/** | `step_loss` | 每 `log_every` 步的总损失 |
| **train/** | `step_flow` | Flow Matching 速度场损失 |
| **train/** | `step_recon` | 节点视觉重建损失 |
| **train/** | `step_prog` | 进度单调损失 |
| **train/** | `epoch_loss/flow/recon/prog` | 每 epoch 均值损失曲线 |
| **train/** | `lr` | 当前学习率（余弦衰减可视化） |
| **train/** | `epoch_time_s` | 每 epoch 耗时 |
| **train/** | `grad_norm_max/mean` | 梯度范数（每 `save_every` epoch 记录） |

#### 国内网络离线模式

若服务器访问 wandb.ai 不稳定，可先用 offline 模式再上传：

```bash
# 训练时写入本地
WANDB_MODE=offline python train.py --config configs/default.yaml --phase 1 --wandb

# 训练完毕后手动同步
wandb sync wandb/offline-run-*
```

---

## 评估

### 评估指标

`eval.py` 支持对以下两个基准进行**离线轨迹评估**（offline trajectory evaluation），使用 teacher-forcing 方式逐帧喂入模型，对比预测动作与真值动作：

| 指标 | 说明 | 适用数据集 |
|---|---|---|
| `action_l1` | 每步 $\|a_{\text{pred}}[0] - a_{\text{gt}}\|_1$（预测动作块第一步 vs 真值） | 全部 |
| `action_l2` | 每步 $\|a_{\text{pred}}[0] - a_{\text{gt}}\|_2$ | 全部 |
| `tree_nodes` | 轨迹末尾记忆树节点数（均值） | 全部 |
| `tree_depth` | 轨迹末尾记忆树最大深度（均值） | 全部 |
| `tree_branches` | 每条轨迹分支创建次数（语义跳变次数） | 全部 |
| `tree_elevations` | 每条轨迹语义提升（elevation）次数 | 全部 |
| `subtask_boundary_f1` | 分支创建事件与 GT 子任务边界的 F1（±5 步容忍） | RoboCerebra |
| `subtask_sr` | GT 子任务边界中被正确检测到的比例 | RoboCerebra |
| `prog_monotone_rate` | 树中祖先-后代对语义进度单调的比例 | RoboCerebra |

---

### RoboCerebra 评估

RoboCerebra 包含子任务边界标注，可计算全部指标：

```bash
conda activate memorytree
cd /path/to/MemoryTreeVLA

# 评估 Phase 2（动作头）checkpoint
python eval.py \
  --ckpt  checkpoints/runs/phase2_epoch0030.pt \
    --config configs/default.yaml \
    --dataset robocerebra \
    --data_root dataset/RoboCerebra/RoboCerebra_trainset \
    --out results/robocerebra_eval.json

# 只评估部分场景（快速验证）
python eval.py \
  --ckpt  checkpoints/runs/phase2_epoch0030.pt \
    --config configs/default.yaml \
    --dataset robocerebra \
    --data_root dataset/RoboCerebra/RoboCerebra_trainset \
    --scenes coffee_table kitchen_table \
    --max_traj 50 \
    --out results/partial_eval.json

# 使用 Phase 3（联合微调）checkpoint，关闭时序下采样
python eval.py \
  --ckpt  checkpoints/runs/phase3_epoch0010.pt \
    --config configs/default.yaml \
    --dataset robocerebra \
    --data_root dataset/RoboCerebra/RoboCerebra_trainset \
    --subsample 1 \
  --out results/robocerebra_phase3_eval.json
```

---

### LIBERO 评估

LIBERO 数据集目录结构如下，请提前准备好：

```
dataset/LIBERO/
    libero_spatial/
        LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_demo.hdf5
        ...   (每个子集 10 个 task，每个 task 约 50 条 demo)
    libero_object/
    libero_goal/
    libero_long/
```

> **下载**：从 [LIBERO 官网](https://libero-project.github.io/) 或 [Hugging Face](https://huggingface.co/datasets/LIBERO-project/LIBERO) 下载数据集，放置到 `dataset/LIBERO/`。

```bash
# 评估 LIBERO-LONG（主要长程测试集，对应 Phase 3）
python eval.py \
  --ckpt  checkpoints/runs/phase3_epoch0010.pt \
    --config configs/default.yaml \
    --dataset libero \
    --data_root dataset/LIBERO \
    --libero_split long \
    --out results/libero_long_eval.json

# 评估 LIBERO-SPATIAL（空间关系泛化）
python eval.py \
  --ckpt  checkpoints/runs/phase3_epoch0010.pt \
    --config configs/default.yaml \
    --dataset libero \
    --data_root dataset/LIBERO \
    --libero_split spatial \
    --out results/libero_spatial_eval.json

# 评估所有子集（循环脚本）
for SPLIT in spatial object goal long; do
    python eval.py \
    --ckpt  checkpoints/runs/phase3_epoch0010.pt \
        --config configs/default.yaml \
        --dataset libero \
        --data_root dataset/LIBERO \
        --libero_split ${SPLIT} \
        --out results/libero_${SPLIT}_eval.json
done

# 快速调试（每个子集只评估 20 条轨迹）
python eval.py \
  --ckpt  checkpoints/runs/phase3_epoch0010.pt \
    --config configs/default.yaml \
    --dataset libero \
    --data_root dataset/LIBERO \
    --libero_split long \
    --max_traj 20 \
    --device cpu
```

---

### 常用参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | 必填 | `.pt` 模型文件（单卡）或 DeepSpeed checkpoint 目录 |
| `--config` | `configs/default.yaml` | 与训练时使用的 YAML 配置文件 |
| `--dataset` | 必填 | `robocerebra` 或 `libero` |
| `--data_root` | 必填 | 数据集根目录 |
| `--libero_split` | `long` | LIBERO 子集：`spatial`, `object`, `goal`, `long` |
| `--subsample` | 来自 config | 时序下采样（1 = 每帧，4 = 每 4 帧） |
| `--max_traj` | 全部 | 限制评估轨迹数（调试用） |
| `--max_seqlen` | 来自 config | 截断长轨迹到该步数 |
| `--device` | `cuda` | `cuda`, `cuda:0`, `cpu` 等 |
| `--boundary_tol` | `5` | 子任务边界 F1 的时步容忍窗口 |
| `--out` | 不保存 | 结果保存为 JSON 文件路径 |

---

### 结果解读

**典型输出示例**（RoboCerebra）：

```
============================================================
Metric                              Value
------------------------------------------------------------
  action_l1                        0.0823
  action_l1_std                    0.0341
  action_l2                        0.1547
  action_l2_std                    0.0612
  tree_nodes                       8.3000
  tree_depth                       2.4000
  tree_branches                    5.7000
  tree_elevations                  1.2000
  subtask_boundary_f1              0.7134
  subtask_sr                       0.7812
  prog_monotone_rate               0.8923
============================================================
Results saved to results/robocerebra_eval.json
```

**指标解读**：

- `action_l1 / l2`：越小越好，表示预测动作与真值的平均偏差
- `tree_nodes / depth`：反映记忆树规模，过大说明阈值偏小，过小说明阈值偏大
- `subtask_boundary_f1`：越接近 1.0 说明模型越准确地感知到子任务切换时机
- `prog_monotone_rate`：越接近 1.0 说明树结构中时序语义层次越清晰

---

## 常见问题

### 1. `NCCL` 通信超时

多卡训练前设置 NCCL 超时和网络接口：

```bash
export NCCL_IB_DISABLE=0          # 启用 InfiniBand（若有）
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0    # 替换为实际网卡名（ip link 查看）
export NCCL_DEBUG=INFO             # 调试时开启
export NCCL_TIMEOUT=1800           # 超时时间（秒）
```

### 2. Flash Attention 编译失败

```bash
# 检查 CUDA 版本与 torch 版本匹配
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# 若版本不一致，重装对应 CUDA 版本的 torch
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 清理缓存后重新编译
pip uninstall flash-attn -y
pip cache purge
pip install flash-attn --no-build-isolation
```

### 3. DeepSpeed `ds_report` 报算子未编译

```bash
# 手动预编译所有 DeepSpeed CUDA 算子
python -c "
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.transformer import DeepSpeedTransformerLayer
print('DeepSpeed ops OK')
"
# 或强制构建
DS_BUILD_OPS=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed --no-build-isolation
```

### 4. 显存不足（OOM）

按以下顺序逐步降低显存占用：

```bash
# 1. 减小 batch_size（configs/default.yaml）
train:
  batch_size: 2   # 从 4 降到 2

# 2. 增大 gradient_accumulation_steps（configs/ds_zero2.json）
"gradient_accumulation_steps": 4

# 3. 切换 ZeRO-3（Phase 3 默认使用）
bash scripts/train_8gpu.sh 3 configs/default.yaml \
    --deepspeed_config configs/ds_zero3.json

# 4. 使用更小的 LLM（0.5B 替代 1.5B）
model:
  llm_path: "checkpoints/Qwen2.5-0.5B"
```

### 5. 数据加载慢

```bash
# 增加 DataLoader workers（configs/default.yaml）
train:
  num_workers: 8   # 默认 4，根据 CPU 核心数调整

# 预解压视频到帧图片（可选，绕过 cv2 实时解码）
# 修改 dataset/robocerebra.py 中的帧读取逻辑即可
```

### 6. 查看训练日志

```bash
# 日志保存在 logs/ 目录（scripts/train_8gpu.sh 自动创建）
tail -f logs/phase1_20260331_120000/train.log

# 仅查看损失
grep "Epoch" logs/phase1_*/train.log
```

---

## 引用

如果本项目对您的研究有帮助，请考虑引用相关工作。
