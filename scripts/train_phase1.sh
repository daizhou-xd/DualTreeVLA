#!/bin/bash
# Phase 1 training: LIBERO → FlowMatching warm-up (semantic modules frozen)
# Requires: accelerate, DeepSpeed, 8× GPU
# Init from: checkpoints/runs/pretrain/pretrain_best.pt
# ────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/train_phase1.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_phase1.sh 4
# ────────────────────────────────────────────────────────────────────────
set -e

NUM_GPUS=${1:-8}
CONFIG=configs/train_phase1.yaml

echo "[train_phase1.sh] GPUs=$NUM_GPUS  config=$CONFIG"

# Verify init checkpoint exists
INIT_CKPT=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['train'].get('init_from',''))")
if [ -n "$INIT_CKPT" ] && [ ! -f "$INIT_CKPT" ]; then
    echo "ERROR: init_from checkpoint not found: $INIT_CKPT"
    echo "       Run pretrain.sh first."
    exit 1
fi

accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --dynamo_backend no \
    scripts/train.py \
        --config "$CONFIG" \
        --phase 1
