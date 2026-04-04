#!/bin/bash
# Phase 2 training: LIBERO → Full fine-tuning (all non-LLM modules)
# Requires: accelerate, DeepSpeed ZeRO-3, 8× GPU
# Init from: checkpoints/runs/phase1/phase1_best.pt
# ────────────────────────────────────────────────────────────────────────
# Usage:
#   bash scripts/train_phase2.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_phase2.sh 4
# ────────────────────────────────────────────────────────────────────────
set -e

NUM_GPUS=${1:-8}
CONFIG=configs/train_phase2.yaml

echo "[train_phase2.sh] GPUs=$NUM_GPUS  config=$CONFIG"

# Verify init checkpoint
INIT_CKPT=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['train'].get('init_from',''))")
if [ -n "$INIT_CKPT" ] && [ ! -f "$INIT_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $INIT_CKPT"
    echo "       Run train_phase1.sh first."
    exit 1
fi

# ZeRO-3 for larger model state
accelerate launch \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --deepspeed_config_file configs/ds_zero3.json \
    scripts/train.py \
        --config "$CONFIG" \
        --phase 2
