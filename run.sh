#!/bin/bash

ENV="dmc_vision"
PRETRAIN_STEPS=1000000
FINETUNE_STEPS=500000

# Generate unique timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export MUJOCO_GL=egl

# === Stage 1: Unsupervised Pre-training ===
echo "=== Stage 1: Pre-training ==="
python dreamsac/main.py \
    --configs dreamsac_e3nn pretrain ${ENV} \
    --logdir logdir/dreamsac_pretrain_${ENV}/${TIMESTAMP} \
    --agent.enc.typ savi \
    --agent.dec.typ savi \
    --agent.viewpoint_temperature 0.07 \
    --agent.use_viewpoint_robustness True \
    --run.steps ${PRETRAIN_STEPS} \
    --agent.intrinsic_reward_scale 1.0

# === Stage 2: Task-specific Fine-tuning ===
echo "=== Stage 2: Fine-tuning ==="

# Automatically find the latest pretrain checkpoint
PRETRAIN_CKPT_DIR="logdir/dreamsac_pretrain_${ENV}/${TIMESTAMP}/ckpt"
if [ ! -f "${PRETRAIN_CKPT_DIR}/latest" ]; then
    echo "Error: No checkpoint found in ${PRETRAIN_CKPT_DIR}"
    exit 1
fi
LATEST_CKPT=$(cat ${PRETRAIN_CKPT_DIR}/latest)
CHECKPOINT_PATH="${PRETRAIN_CKPT_DIR}/${LATEST_CKPT}"
echo "Loading pretrain checkpoint: ${CHECKPOINT_PATH}"

python dreamsac/main.py \
    --configs dreamsac_e3nn finetune ${ENV} \
    --logdir logdir/dreamsac_finetune_${ENV}/${TIMESTAMP} \
    --agent.enc.typ savi \
    --agent.dec.typ savi \
    --run.from_checkpoint "${CHECKPOINT_PATH}" \
    --run.steps ${FINETUNE_STEPS}

echo "=== Training Complete ==="