# DreamSAC

**DreamSAC** is a model-based reinforcement learning agent that integrates structured world models with physics-inspired dynamics for robust visual control and zero-shot domain transfer.

![](assets/fig.png)

## Abstract

Learned world models excel at interpolative generalization but fail at extrapolative generalization to novel physical properties. This limitation arises because they learn statistical correlations rather than the environment's underlying generative rules, such as physical invariances and conservation laws. We argue that learning these invariances is key to robust extrapolation. To achieve this, we first introduce **Symmetry Exploration**, an unsupervised exploration strategy where an agent is intrinsically motivated by a Hamiltonian-based curiosity bonus to actively probe and challenge its understanding of conservation laws, thereby collecting physically informative data. Second, we design a Hamiltonian-based world model that learns from the collected data, using a novel self-supervised contrastive objective to identify the invariant physical state from raw, view-dependent pixel observations. Our framework, **DreamSAC**, trained on this actively curated data, significantly outperforms state-of-the-art baselines in 3D physics simulations on tasks requiring extrapolation.

## Installation

```bash
pip install -r requirements.txt
# Optional: for SE(3)-equivariant networks
pip install e3nn-jax
```

> Requires JAX with CUDA 12. See [JAX installation guide](https://github.com/google/jax#installation).

## Usage

### Two-Stage Training (Pretrain → Finetune)

```bash
bash run.sh
```

### Custom Run

```bash
# Stage 1: Unsupervised pre-training
python dreamsac/main.py \
    --configs dreamsac_e3nn pretrain dmc_vision \
    --logdir logdir/pretrain \
    --run.steps 1000000

# Stage 2: Fine-tuning
python dreamsac/main.py \
    --configs dreamsac_e3nn finetune dmc_vision \
    --logdir logdir/finetune \
    --run.from_checkpoint logdir/pretrain/ckpt/<step> \
    --run.steps 500000
```

### Domain Randomization

```bash
bash run_transfer.sh
```

## Environments

Evaluated on [DeepMind Control Suite](https://github.com/deepmind/dm_control) (visual observations):

- `walker_walk`, `cheetah_run`, `hopper_hop`, `reacher_hard`, `humanoid_walk`

## Citation

Coming soon ...

