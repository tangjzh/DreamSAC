"""Reduced real-DMC two-stage checks; not a task-performance experiment."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import elements
from ruamel.yaml import YAML


def config(task, stage, case, directory):
    definitions = YAML(typ='safe').load((ROOT / 'dreamsac/configs.yaml').read_text())
    cfg = elements.Config(definitions['defaults'])
    for key in ('dreamsac_full', stage, 'dmc_vision'):
        cfg = cfg.update(definitions[key])
    cfg = cfg.update({
        'task': 'dmc_' + task, 'logdir': str(directory), 'seed': 0,
        'batch_size': 2, 'batch_length': 4, 'report_length': 4,
        'replay_context': 1, 'run.envs': 1, 'run.debug': False,
        'run.steps': 32, 'run.train_ratio': 2, 'run.report_every': 0,
        'run.log_every': 2, 'run.save_every': 300, 'run.usage.nvsmi': False,
        'logger.outputs': ['jsonl'], 'jax.prealloc': False,
        'agent.report': False, 'agent.opt.warmup': 0,
        'agent.imag_length': 2,
        'agent.enc.typ': 'savi' if case == 'savi_viewpoint' else 'simple',
        'agent.dec.typ': 'savi' if case == 'savi_viewpoint' else 'simple',
        'agent.use_viewpoint_robustness': case == 'savi_viewpoint' and stage == 'pretrain',
        'agent.dyn.hamiltonian.deter': 16, 'agent.dyn.hamiltonian.hidden': 8,
        'agent.dyn.hamiltonian.stoch': 4, 'agent.dyn.hamiltonian.classes': 4,
        'agent.dyn.hamiltonian.blocks': 4, 'agent.dyn.hamiltonian.ham_layers': 1,
        'agent.dyn.hamiltonian.ham_units': 8,
        'agent.dyn.hamiltonian.input_matrix_layers': 1,
        'agent.dyn.hamiltonian.input_matrix_units': 8,
        'agent.enc.simple.depth': 2, 'agent.dec.simple.depth': 2,
        'agent.enc.simple.units': 8, 'agent.dec.simple.units': 8,
        'agent.enc.simple.layers': 1, 'agent.dec.simple.layers': 1,
        'agent.dec.simple.bspace': 4,
        'agent.enc.savi.depth': 2, 'agent.dec.savi.depth': 2,
        'agent.enc.savi.slot_dim': 8, 'agent.dec.savi.slot_dim': 8,
        'agent.enc.savi.feature_dim': 8,
        'agent.enc.savi.num_slots': 2, 'agent.dec.savi.num_slots': 2,
        'agent.enc.savi.num_iterations': 1,
        'agent.enc.savi.num_initial_iterations': 1,
        'agent.policy.units': 8, 'agent.policy.layers': 1,
        'agent.value.units': 8, 'agent.value.layers': 1,
        'agent.rewhead.units': 8, 'agent.conhead.units': 8,
        'agent.rnd.units': 8, 'agent.rnd.layers': 1, 'agent.rnd.output_dim': 8,
    })
    return cfg


def arguments(cfg):
    args = []
    for key, value in cfg.flat.items():
        if isinstance(value, (tuple, list)):
            value = ','.join(str(x) for x in value)
        args += ['--' + key, str(value)]
    return args


def latest(directory):
    root = directory / 'ckpt'
    path = root / (root / 'latest').read_text().strip()
    assert (path / 'done').exists(), path
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', default=['cheetah_run', 'quadruped_run'])
    p.add_argument('--cases', nargs='+', choices=['plain', 'savi_viewpoint'], default=['plain', 'savi_viewpoint'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    results = []
    for task in args.tasks:
        for case in args.cases:
            base = args.output.resolve() / task / case
            for stage in ('pretrain', 'finetune'):
                directory = base / stage
                directory.mkdir(parents=True, exist_ok=True)
                cfg = config(task, stage, case, directory)
                if stage == 'finetune':
                    cfg = cfg.update({'run.from_checkpoint': str(latest(base / 'pretrain')),
                        'run.from_checkpoint_verify': True,
                        'run.from_checkpoint_regex': '^(enc|dyn|dec|rew|con|pol|val|slowval)/'})
                (directory / 'requested-config.json').write_text(json.dumps(dict(cfg.flat), indent=2) + '\n')
                print('Checking', task, case, stage, flush=True)
                with (directory / 'console.log').open('w') as log:
                    subprocess.run([sys.executable, '-u', str(ROOT / 'dreamsac/main.py'), *arguments(cfg)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                rows = [json.loads(x) for x in (directory / 'metrics.jsonl').read_text().splitlines()]
                train = [x for x in rows if 'train/opt/loss' in x]
                assert train, 'No gradient update was logged'
                assert all(math.isfinite(x[k]) for x in train for k in ('train/opt/loss', 'train/opt/grad_norm'))
                if stage == 'finetune':
                    audit = json.loads((directory / 'transfer.json').read_text())
                    assert audit['all_arrays_exactly_equal'] and audit['arrays'] > 0
                results.append({'task': task, 'case': case, 'stage': stage,
                    'step': rows[-1]['step'], 'loss': train[-1]['train/opt/loss'],
                    'checkpoint': str(latest(directory)), 'finite': True})
                (args.output / ('validation-' + '-'.join(args.tasks) + '.json')).write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
