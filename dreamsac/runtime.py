"""Locate pip CUDA libraries when using a system-site-packages venv."""
import glob
import os
import site
import sys


def ensure_cuda_paths():
    # The dynamic loader reads LD_LIBRARY_PATH at process startup. Merely
    # assigning it after importing JAX is insufficient, so restart once when
    # necessary. No system files or other Python environments are modified.
    additions = {
        'LD_LIBRARY_PATH': [p for root in site.getsitepackages()
                            for p in glob.glob(root + '/nvidia/*/lib')],
        'PATH': [p for root in site.getsitepackages()
                 for p in glob.glob(root + '/nvidia/*/bin')],
    }
    environment = dict(os.environ)
    changed = False
    if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in environment:
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        changed = True
    if '--xla_gpu_cuda_data_dir=' not in environment.get('XLA_FLAGS', ''):
        for root in site.getsitepackages():
            toolkit = root + '/nvidia/cuda_nvcc'
            if os.path.isfile(toolkit + '/nvvm/libdevice/libdevice.10.bc'):
                environment['XLA_FLAGS'] = (environment.get('XLA_FLAGS', '') +
                    ' --xla_gpu_cuda_data_dir=' + toolkit).strip()
                changed = True
                break
    for name, paths in additions.items():
        existing = [p for p in environment.get(name, '').split(':') if p]
        missing = [p for p in paths if p not in existing]
        if missing:
            environment[name] = ':'.join(missing + existing)
            changed = True
    if changed:
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
