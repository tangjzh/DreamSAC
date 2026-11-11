"""Run a saved full YAML configuration with explicit path/budget overrides."""
import argparse
from pathlib import Path
import subprocess
import sys

import elements
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--logdir', type=Path, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--from-checkpoint', type=Path)
    args = parser.parse_args()
    cfg = elements.Config(YAML(typ='safe').load(args.config.read_text()))
    cfg = cfg.update(logdir=str(args.logdir.resolve()))
    if args.steps is not None:
        if args.steps < 1:
            parser.error('--steps must be positive')
        cfg = cfg.update({'run.steps': args.steps})
    if args.from_checkpoint:
        checkpoint = args.from_checkpoint.resolve()
        if not (checkpoint / 'done').is_file():
            parser.error('--from-checkpoint must point to a completed checkpoint')
        cfg = cfg.update({'run.from_checkpoint': str(checkpoint)})
    flags = []
    for key, value in cfg.flat.items():
        if isinstance(value, (list, tuple)):
            value = ','.join(str(x) for x in value)
        flags.extend(['--' + key, str(value)])
    subprocess.run([sys.executable, '-u', str(ROOT / 'dreamsac/main.py'), *flags],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
