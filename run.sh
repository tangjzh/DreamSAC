#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python}"
TASK="${TASK:-dmc_cheetah_run}"
SEED="${SEED:-0}"
MODEL_CONFIG="${MODEL_CONFIG:-dreamsac_full}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-1000000}"
FINETUNE_STEPS="${FINETUNE_STEPS:-500000}"
USE_SAVI="${USE_SAVI:-False}"
USE_VIEWPOINT="${USE_VIEWPOINT:-False}"
REWARD_FREE="${REWARD_FREE:-False}"
RUN_ROOT="${RUN_ROOT:-logdir/${TASK}/seed-${SEED}/$(date +%Y%m%d_%H%M%S)}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

encoder=simple
case "${USE_SAVI,,}" in true|1) encoder=savi ;; false|0) ;; *) echo 'USE_SAVI must be True or False' >&2; exit 2 ;; esac
case "${USE_VIEWPOINT,,}" in true|1) USE_VIEWPOINT=True ;; false|0) USE_VIEWPOINT=False ;; *) echo 'USE_VIEWPOINT must be True or False' >&2; exit 2 ;; esac
extra=()
case "${REWARD_FREE,,}" in true|1) extra+=(--agent.loss_scales.repval 0.0) ;; false|0) ;; *) echo 'REWARD_FREE must be True or False' >&2; exit 2 ;; esac
common=(--logger.outputs jsonl --task "$TASK" --seed "$SEED" --agent.enc.typ "$encoder" --agent.dec.typ "$encoder")

# Keep policy/value training active; optionally disable task-reward replay value.
"$PYTHON" dreamsac/main.py --configs "$MODEL_CONFIG" pretrain dmc_vision \
  "${common[@]}" --agent.use_viewpoint_robustness "$USE_VIEWPOINT" \
  --logdir "$RUN_ROOT/pretrain" --run.steps "$PRETRAIN_STEPS" "${extra[@]}"

latest_file="$RUN_ROOT/pretrain/ckpt/latest"
[[ -f "$latest_file" ]] || { echo "Missing checkpoint: $latest_file" >&2; exit 1; }
IFS= read -r latest < "$latest_file" || [[ -n "$latest" ]]
checkpoint="$RUN_ROOT/pretrain/ckpt/$latest"
[[ -f "$checkpoint/done" ]] || { echo "Incomplete checkpoint: $checkpoint" >&2; exit 1; }

# Reset optimizer state when removing the RND module; keep learned model and
# policy/value weights for task-reward fine-tuning.
"$PYTHON" dreamsac/main.py --configs "$MODEL_CONFIG" finetune dmc_vision \
  "${common[@]}" --agent.use_viewpoint_robustness False \
  --logdir "$RUN_ROOT/finetune" --run.steps "$FINETUNE_STEPS" \
  --run.from_checkpoint "$checkpoint" \
  --run.from_checkpoint_verify True \
  --run.from_checkpoint_regex '^(enc|dyn|dec|rew|con|pol|val|slowval)/'
