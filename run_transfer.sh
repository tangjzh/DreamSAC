#!/bin/bash

# ============================================================================
# Specific Parameter Transfer Learning Experiment
# ============================================================================
# This script tests transfer learning on specific physics parameter changes:
# 1. Pre-training on standard parameters
# 2. Fine-tuning on modified parameters (e.g., strong gravity, high friction)
# 3. Compare with DreamerV3 baseline trained from scratch
# ============================================================================

# Parse command line arguments
PARAM_TYPE=${1:-"friction"}  # gravity, friction, mass, damping
CHANGE_FACTOR=${2:-"2.0"}
ENV_NAME=${3:-"cheetah_run"}
# Training parameters
PRETRAIN_STEPS=1000000    # 1M steps for pre-training on standard params
FINETUNE_STEPS=500000     # 500K steps for fine-tuning on modified params
EVAL_EPISODES=100         # Episodes for evaluation

# Generate unique timestamp for this experiment
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PARAM_NAME="${PARAM_TYPE}_${CHANGE_FACTOR}x"
EXPERIMENT_NAME="specific_${ENV_NAME}_${PARAM_NAME}_${TIMESTAMP}"

export MUJOCO_GL=egl

echo "========================================"
echo "Specific Transfer Learning Experiment"
echo "========================================"
echo "Environment: ${ENV_NAME}"
echo "Parameter: ${PARAM_TYPE}"
echo "Change Factor: ${CHANGE_FACTOR}x"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "========================================"

# ============================================================================
# Step 0: Generate specific parameter sets
# ============================================================================
echo ""
echo "=== Step 0: Generating parameter sets ==="
echo "Generating ${PARAM_TYPE} ${CHANGE_FACTOR}x parameter set..."
python experiment_config_specific.py

# ============================================================================
# Stage 1: Pre-training on STANDARD parameters
# ============================================================================
echo ""
echo "=== Stage 1: Pre-training on STANDARD parameters ==="
echo "Using intrinsic reward from Hamiltonian dynamics"
echo "Steps: ${PRETRAIN_STEPS}"
echo ""

TASK="dmc_${ENV_NAME}"
PARAM_FILE="param_sets/${ENV_NAME}_${PARAM_NAME}_params.pkl"

python dreamerv3/main.py \
    --configs dreamsac_e3nn pretrain dmc_vision \
    --task ${TASK} \
    --logdir logdir/${EXPERIMENT_NAME}/pretrain_standard \
    --run.steps ${PRETRAIN_STEPS} \
    --agent.intrinsic_reward_scale 1.0 \
    --env.dmc.use_randomized True \
    --env.dmc.param_file "${PARAM_FILE}" \
    --env.dmc.use_train_set True

echo "Pre-training completed."

# ============================================================================
# Stage 2: Fine-tuning on MODIFIED parameters
# ============================================================================
echo ""
echo "=== Stage 2: Fine-tuning on MODIFIED parameters (${PARAM_TYPE} ${CHANGE_FACTOR}x) ==="
echo "Using task reward (external reward)"
echo "Steps: ${FINETUNE_STEPS}"
echo ""

# Find pretrain checkpoint
PRETRAIN_CKPT_DIR="logdir/${EXPERIMENT_NAME}/pretrain_standard/ckpt"
if [ ! -f "${PRETRAIN_CKPT_DIR}/latest" ]; then
    echo "Error: No checkpoint found in ${PRETRAIN_CKPT_DIR}"
    exit 1
fi
LATEST_CKPT=$(cat ${PRETRAIN_CKPT_DIR}/latest)
PRETRAIN_CHECKPOINT_PATH="${PRETRAIN_CKPT_DIR}/${LATEST_CKPT}"
echo "Loading pretrain checkpoint: ${PRETRAIN_CHECKPOINT_PATH}"

python dreamerv3/main.py \
    --configs dreamsac_e3nn finetune dmc_vision \
    --task ${TASK} \
    --logdir logdir/${EXPERIMENT_NAME}/finetune_modified \
    --run.from_checkpoint "${PRETRAIN_CHECKPOINT_PATH}" \
    --run.steps ${FINETUNE_STEPS} \
    --agent.intrinsic_reward_scale 0.0 \
    --env.dmc.use_randomized True \
    --env.dmc.param_file "${PARAM_FILE}" \
    --env.dmc.use_train_set False

echo "Fine-tuning completed."

# ============================================================================
# Stage 3: Zero-shot evaluation on MODIFIED parameters
# ============================================================================
echo ""
echo "=== Stage 3: Zero-shot evaluation on MODIFIED parameters ==="
echo "Evaluating pre-trained model WITHOUT fine-tuning"
echo "Episodes: ${EVAL_EPISODES}"
echo ""

python dreamerv3/main.py \
    --configs dreamsac_e3nn pretrain dmc_vision \
    --task ${TASK} \
    --script eval_only \
    --logdir logdir/${EXPERIMENT_NAME}/zeroshot_modified \
    --run.from_checkpoint "${PRETRAIN_CHECKPOINT_PATH}" \
    --run.eval_eps ${EVAL_EPISODES} \
    --env.dmc.use_randomized True \
    --env.dmc.param_file "${PARAM_FILE}" \
    --env.dmc.use_train_set False

echo "Zero-shot evaluation completed."

# ============================================================================
# Baseline: DreamerV3 from scratch on MODIFIED parameters
# ============================================================================
echo ""
echo "=== Baseline: DreamerV3 from scratch on MODIFIED parameters ==="
echo "For comparison with fine-tuning results"
echo "Steps: ${FINETUNE_STEPS}"
echo ""

python dreamerv3/main.py \
    --configs dmc_vision \
    --task ${TASK} \
    --logdir logdir/${EXPERIMENT_NAME}/baseline_dreamerv3 \
    --run.steps ${FINETUNE_STEPS} \
    --env.dmc.use_randomized True \
    --env.dmc.param_file "${PARAM_FILE}" \
    --env.dmc.use_train_set False

echo "Baseline training completed."

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================"
echo "Experiment Completed!"
echo "========================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Parameter: ${PARAM_TYPE} changed by ${CHANGE_FACTOR}x"
echo "Results saved in: logdir/${EXPERIMENT_NAME}/"
echo ""
echo "Directories:"
echo "  - pretrain_standard:   Pre-training on standard params"
echo "  - finetune_modified:   Fine-tuning on modified params (${PARAM_TYPE} ${CHANGE_FACTOR}x)"
echo "  - zeroshot_modified:   Zero-shot eval on modified params"
echo "  - baseline_dreamerv3:  DreamerV3 baseline from scratch"
echo ""
echo "To compare results, run:"
echo "  python plot_specific_transfer_results.py --experiment ${EXPERIMENT_NAME}"
echo "========================================"

