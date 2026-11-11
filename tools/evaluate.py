"""Evaluate frozen checkpoints on complete ID episodes at a separate seed."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamsac.runtime import ensure_cuda_paths
ensure_cuda_paths()

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from functools import partial
import elements
import embodied
import numpy as np
from ruamel.yaml import YAML
from dreamsac.main import make_agent, make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    if not (args.checkpoint / 'done').is_file():
        parser.error('--checkpoint must point to a completed checkpoint directory')
    config = elements.Config(YAML(typ='safe').load(args.config.read_text()))
    training_seed = config.seed
    config = elements.Config({**config.flat, 'jax.precompile': False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = config.update({'seed': args.seed, 'agent.report': False,
                            'logdir': str(args.output.parent)})
    agent = make_agent(config)
    elements.checkpoint.load(str(args.checkpoint), {'agent': agent.load})
    # Pair evaluation policy random streams as well as environment seeds.
    with agent.n_actions.lock:
        agent.n_actions.value = 0
    digest = hashlib.sha256()
    for path in sorted([*(ROOT / 'dreamsac').rglob('*.py'), *(ROOT / 'embodied').rglob('*.py')]):
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0' + path.read_bytes())
    result = dict(task=config.task, checkpoint=str(args.checkpoint),
                  training_seed=training_seed, evaluation_seed=args.seed,
                  environment=dict(config.env.dmc),
                  render_backend=os.environ.get('MUJOCO_GL', 'egl'),
                  model_source_sha256=digest.hexdigest(),
                  python=sys.version.split()[0],
                  dependencies={name: version(name) for name in (
                      'jax', 'jaxlib', 'ninjax', 'numpy', 'dm-control', 'mujoco')},
                  policy='sampled, frozen, no updates', returns=[], lengths=[])
    for index in range(args.episodes):
        driver = embodied.Driver([partial(make_env, config, index)], parallel=True)
        total = {'score': 0., 'length': 0}
        def record(obs, worker):
            total['score'] += float(obs['reward'])
            total['length'] += int(not obs['is_first'])
        driver.on_step(record)
        try:
            driver.reset(agent.init_policy)
            driver(partial(agent.policy, mode='eval'), episodes=1)
        finally:
            driver.close()
        result['returns'].append(total['score'])
        result['lengths'].append(total['length'])
        result.update(mean=float(np.mean(result['returns'])),
                      std_episodes=float(np.std(result['returns'])),
                      complete=index + 1 == args.episodes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print('EVAL', index + 1, total, flush=True)


if __name__ == '__main__':
    main()
