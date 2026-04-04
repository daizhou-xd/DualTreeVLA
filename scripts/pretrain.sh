#!/bin/bash
# Pre-training: RoboCerebra → SemanticJumpHead + SGMTS + s_proj
# Requires: accelerate, 8× GPU (A100/80G recommended or 4× for small batch)
# ────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/pretrain.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/pretrain.sh   # 4 GPU
# ────────────────────────────────────────────────────────────────────────
set -e

NUM_GPUS=${1:-8}
CONFIG=configs/pretrain.yaml

echo "[pretrain.sh] GPUs=$NUM_GPUS  config=$CONFIG"

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/pretrain.py \
        --config "$CONFIG"
