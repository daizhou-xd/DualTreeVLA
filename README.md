# MemoryTreeVLA

**MemoryTreeVLA** is a Vision-Language-Action model augmented with a hierarchical memory tree for long-horizon robot manipulation.

---

## Project Structure

```
MemoryTreeVLA/
├── MTVLA/                        # Core model code
│   ├── models/
│   │   ├── mtvla_model.py        # MemoryTreeVLA full architecture
│   │   ├── memory_tree.py        # Hierarchical memory tree (Tree.json spec)
│   │   ├── action_condition.py   # LLM token + robot state → action condition
│   │   ├── action_head/          # Flow-matching action head (Evo-1 style)
│   │   └── tree_scan/            # Vision Mamba + Tree Mamba (GrootVL)
│   ├── configs/
│   │   ├── config.py             # Config dataclasses & YAML loader
│   │   ├── default.yaml          # Default training config
│   │   └── ds_config.json        # DeepSpeed ZeRO-2 config
│   ├── utils/
│   │   ├── metrics.py            # Evaluation metrics (success rate, etc.)
│   │   └── logger.py             # Logging utility
│   └── train.py                  # Training entry point
│
├── LIBERO_evaluation/            # LIBERO benchmark evaluation
│   ├── libero_evaluator.py
│   ├── libero_client_4tasks.py
│   ├── eval_libero.py
│   └── configs/libero_eval.yaml
│
├── CALVIN_evaluation/            # CALVIN benchmark evaluation
│   ├── calvin_evaluator.py
│   ├── eval_calvin.py
│   └── configs/calvin_eval.yaml
│
├── ROBOMME_evaluation/           # RoboMME benchmark evaluation
│   ├── robomme_evaluator.py
│   ├── eval_robomme.py
│   └── configs/robomme_eval.yaml
│
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Environment Setup

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1 × RTX 3090 (24 GB) | 4 × A100 80 GB |
| CPU RAM | 32 GB | 64 GB |
| Disk | 50 GB (checkpoints + dataset) | 200 GB |
| CUDA | 11.8 | 12.1 |

### 1. Create Conda Environment

```bash
# Clone the repo
git clone https://github.com/your-org/MemoryTreeVLA.git
cd MemoryTreeVLA

# Create a new conda env (Python 3.11 recommended)
conda create -n mtvla python=3.11 -y
conda activate mtvla
```

### 2. Install PyTorch

Choose the command that matches your CUDA version.

**CUDA 12.1 (recommended):**
```bash
conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
```

**CUDA 11.8:**
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y
```

**CPU-only (for debugging without GPU):**
```bash
conda install pytorch==2.3.0 torchvision==0.18.0 cpuonly -c pytorch -y
```

Verify the installation:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install DeepSpeed (multi-GPU training)

DeepSpeed requires a compatible C++ compiler. On most Linux servers:

```bash
# Install build tools if not present
conda install -c conda-forge gxx_linux-64 -y

# Install DeepSpeed (builds from source if needed)
DS_BUILD_OPS=0 pip install deepspeed>=0.13.0

# Verify
ds_report
```

> **Tip:** On clusters managed by SLURM, load the required modules first:
> ```bash
> module load cuda/12.1 gcc/11.3
> ```

### 5. Install Evaluation Benchmarks (optional)

**LIBERO:**
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
```

**CALVIN:**
```bash
cd /tmp
git clone --recurse-submodules https://github.com/mees/calvin.git
cd calvin && pip install -e . && cd -
```

### 6. Verify Full Installation

```bash
python -c "
import sys; sys.path.insert(0, '.')
from MTVLA.models.mtvla_model import MemoryTreeVLA
from MTVLA.models.memory_tree import MemoryTree
print('Import OK')
"
```

---

## Model Preparation

MemoryTreeVLA uses two lightweight Qwen2.5 language models as its LLM backbones. Visual features are extracted entirely by Vision Mamba — **no ViT or visual encoder download is needed**.

### Download Action LLM & Tree LLM

| Role | Model | Notes |
|------|-------|-------|
| `ACTION_LLM` | **Qwen2.5-0.5B** | High-frequency action generation; speed-first |
| `TREE_LLM` | **Qwen2.5-1.5B-Instruct** | Low-frequency subtask state reasoning; Instruct variant ensures stable JSON output |

**Option 1: huggingface-cli (recommended)**
```bash
pip install huggingface_hub

# Action LLM
huggingface-cli download Qwen/Qwen2.5-0.5B \
    --local-dir checkpoints/Qwen2.5-0.5B

# Tree LLM
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir checkpoints/Qwen2.5-1.5B-Instruct
```

**Option 2: HF mirror (if HuggingFace is inaccessible)**
```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Qwen/Qwen2.5-0.5B \
    --local-dir checkpoints/Qwen2.5-0.5B

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir checkpoints/Qwen2.5-1.5B-Instruct
```

**Option 3: git lfs**
```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B checkpoints/Qwen2.5-0.5B
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct checkpoints/Qwen2.5-1.5B-Instruct
```

The paths above match the defaults already set in `MTVLA/configs/default.yaml`:
```yaml
model:
  action_llm_path: checkpoints/Qwen2.5-0.5B
  tree_llm_path:   checkpoints/Qwen2.5-1.5B-Instruct
```
Modify these values in `default.yaml` if you prefer a different location.

---

## Quick Start

**Single-GPU training:**
```bash
conda activate mtvla
python MTVLA/train.py --config MTVLA/configs/default.yaml
```

**Multi-GPU training with DeepSpeed (ZeRO-2):**
```bash
conda activate mtvla
deepspeed --num_gpus=4 MTVLA/train.py \
    --config MTVLA/configs/default.yaml \
    --deepspeed MTVLA/configs/ds_config.json
```

**SLURM job script example (`scripts/train_slurm.sh`):**
```bash
#!/bin/bash
#SBATCH --job-name=mtvla_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

module load cuda/12.1 gcc/11.3
conda activate mtvla

deepspeed --num_gpus=4 MTVLA/train.py \
    --config MTVLA/configs/default.yaml \
    --deepspeed MTVLA/configs/ds_config.json
```

**LIBERO Evaluation:**
```bash
conda activate mtvla
python LIBERO_evaluation/eval_libero.py --model_ckpt outputs/checkpoint.pth --suite all
```

**CALVIN Evaluation:**
```bash
conda activate mtvla
python CALVIN_evaluation/eval_calvin.py --model_ckpt outputs/checkpoint.pth --split D->D
```

**RoboMME Evaluation:**
```bash
conda activate mtvla
python ROBOMME_evaluation/eval_robomme.py --model_ckpt outputs/checkpoint.pth --category all
```

