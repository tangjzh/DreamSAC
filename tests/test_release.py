import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import numpy as np
import jax
import jax.numpy as jnp
import ninjax as nj
import elements
from pathlib import Path
from ruamel.yaml import YAML
from dreamsac.savi_encoder import SAViEncoder
from embodied.jax.opt import OptimizerState


def test_large_finite_gradient_norm_and_clipping():
    from embodied.jax.opt import stable_global_norm, clip_by_agc
    import optax
    gradient = jnp.array([3e20, 4e20], jnp.float32)
    # The old norm overflows and AGC incorrectly erases this finite gradient.
    assert not np.isfinite(optax.global_norm(gradient))
    np.testing.assert_allclose(stable_global_norm({'g': gradient}), 5e20, rtol=1e-6)
    params = jnp.ones(2)
    clip = clip_by_agc(.3)
    clipped, _ = clip.update(gradient, clip.init(params), params)
    np.testing.assert_allclose(clipped, np.array([.6, .8]) * .3 * np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(stable_global_norm({'g': jnp.zeros(2)}), 0)
    for bad in (jnp.inf, jnp.nan):
        assert not np.isfinite(stable_global_norm(jnp.array([bad, 1.])))

def test_stable_norm_matches_float64_reference():
    from embodied.jax.opt import stable_global_norm
    rng = np.random.default_rng(3)
    leaves = [rng.normal(size=(5, 7)).astype(np.float32), np.zeros(3, np.float32)]
    for scale in (1e-20, 1., 1e20):
        tree = {'a': jnp.array(leaves[0] * scale), 'b': jnp.array(leaves[1])}
        expected = np.sqrt(sum(np.square(np.asarray(x, np.float64)).sum() for x in tree.values()))
        np.testing.assert_allclose(jax.jit(stable_global_norm)(tree), expected, rtol=2e-6)

def test_optimizer_leaf_order_with_underscores():
    tree = {'dec_pos': jnp.ones((3, 4)), 'deconv': jnp.ones((2,))}
    state = OptimizerState(lambda _: tree, {}, name='state')
    params, read = nj.pure(state.read)({}, seed=0, create=True)
    assert jax.tree.structure(read) == jax.tree.structure(tree)
    for key in tree:
        np.testing.assert_array_equal(read[key], tree[key])
    _, read_again = nj.pure(state.read)(params, seed=1)
    for key in tree:
        np.testing.assert_array_equal(read_again[key], tree[key])

def test_savi_sequence_matches_causal_single_steps():
    from embodied.jax import nets
    nets.COMPUTE_DTYPE = jnp.float32
    space = {'image': elements.Space(np.uint8, (16, 16, 3))}
    enc = SAViEncoder(space, {}, depth=2, num_slots=2, slot_dim=8,
        feature_dim=8, num_iterations=1, num_initial_iterations=1, name='enc')
    images = np.random.default_rng(1).integers(0, 255, (2, 3, 16, 16, 3), dtype=np.uint8)
    first = jnp.array([[True, False, False], [False, False, True]])
    fn = nj.pure(lambda obs, reset: enc(enc.initial(2), obs, reset, True))
    params, (_, entries, tokens) = fn({}, {'image': images}, first, seed=1, create=True)
    assert entries['prev_slots'].shape == (2, 3, 2, 8)
    assert tokens.shape == (2, 3, 16)
    changed = images.copy()
    changed[:, -1] = 0
    _, (_, _, changed_tokens) = fn(params, {'image': changed}, first, seed=2)
    _, (_, _, base_tokens) = fn(params, {'image': images}, first, seed=2)
    np.testing.assert_allclose(changed_tokens[:, :-1], base_tokens[:, :-1], atol=1e-6)
    assert not np.allclose(changed_tokens[:, -1], base_tokens[:, -1])

def test_savi_restores_replay_to_compute_dtype(monkeypatch):
    from embodied.jax import nets
    monkeypatch.setattr(nets, 'COMPUTE_DTYPE', jnp.bfloat16)
    space = {'image': elements.Space(np.uint8, (16, 16, 3))}
    enc = SAViEncoder(space, {}, num_slots=2, slot_dim=8, name='enc')
    restored = enc.truncate({'prev_slots': jnp.ones((1, 1, 2, 8), jnp.float32)})
    assert restored['prev_slots'].dtype == jnp.bfloat16
    assert restored['prev_slots'].shape == (1, 2, 8)

def test_reference_pretrain_objectives_and_explicit_reward_free_variant():
    definitions=YAML(typ='safe').load((Path(__file__).resolve().parents[1]/'dreamsac/configs.yaml').read_text())
    cfg=elements.Config(definitions['defaults'])
    for key in ('dreamsac_full','pretrain','dmc_vision'):
        cfg=cfg.update(definitions[key])
    assert cfg.agent.loss_scales.policy == 1
    assert cfg.agent.loss_scales.value == 1
    assert cfg.agent.loss_scales.repval == .3
    assert cfg.agent.loss_scales.rew == 0
    assert cfg.agent.enc.typ == 'simple'
    assert not cfg.agent.use_viewpoint_robustness
    free=cfg.update(definitions['pretrain_reward_free'])
    assert free.agent.loss_scales.repval == 0
    assert free.agent.loss_scales.policy == free.agent.loss_scales.value == 1



def test_launcher_transfer_and_fail_fast(tmp_path):
    import json
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    stub = tmp_path / 'fake python'
    stub.write_text('#!' + sys.executable + '\n' + '''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
p=Path(os.environ['CALL_LOG'])
with p.open('a') as f:f.write(json.dumps(args)+'\\n')
logdir=Path(args[args.index('--logdir')+1])
if os.environ.get('FAIL_STAGE')==logdir.name:sys.exit(7)
if logdir.name=='pretrain':
 (logdir/'ckpt/test').mkdir(parents=True)
 (logdir/'ckpt/test/done').touch()
 (logdir/'ckpt/latest').write_text('test')
''')
    stub.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = {**os.environ, 'PYTHON': str(stub), 'CALL_LOG': str(log),
           'RUN_ROOT': str(tmp_path / 'run with spaces'), 'TASK': 'dmc_quadruped_run',
           'USE_SAVI': 'True', 'USE_VIEWPOINT': '1', 'REWARD_FREE': 'True'}
    subprocess.run(['bash', str(root / 'run.sh')], env=env, check=True)
    calls = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(calls) == 2
    assert calls[0][calls[0].index('--agent.enc.typ')+1] == 'savi'
    assert calls[0][calls[0].index('--agent.loss_scales.repval')+1] == '0.0'
    assert calls[0][calls[0].index('--agent.use_viewpoint_robustness')+1] == 'True'
    assert calls[1][calls[1].index('--run.from_checkpoint')+1].endswith('pretrain/ckpt/test')
    assert calls[1][calls[1].index('--run.from_checkpoint_verify')+1] == 'True'
    log.unlink()
    env['FAIL_STAGE'] = 'pretrain'
    env['RUN_ROOT'] = str(tmp_path / 'failed run')
    result = subprocess.run(['bash', str(root / 'run.sh')], env=env)
    assert result.returncode == 7
    assert len(log.read_text().splitlines()) == 1


def test_verified_transfer_rejects_missing_and_mismatched_arrays():
    from types import SimpleNamespace
    import pytest
    from embodied.jax.agent import Agent
    agent = SimpleNamespace(params={'enc/weight': np.zeros((2, 3)),
                                   'val/weight': np.zeros((3, 1))})
    # Rejection happens before touching counters, locks, or any model weights.
    for params in ({'enc/weight': np.ones((2, 3))},
                   {'enc/weight': np.ones((3, 2)), 'val/weight': np.ones((3, 1))}):
        with pytest.raises(ValueError):
            Agent.load(agent, {'params': params}, regex='^(enc|val)/', verify=True)
    np.testing.assert_array_equal(agent.params['enc/weight'], np.zeros((2, 3)))


def test_counter_is_runtime_state_under_jit_and_restore():
    counter = nj.Variable(jnp.array, 0, jnp.int32, name='training_step')
    def update():
        value = counter.read()
        counter.write(value + 1)
        return value
    pure = nj.pure(update)
    state, value = pure({}, seed=0, create=True)
    fn = jax.jit(lambda state: pure(state, seed=0))
    checkpoint = jax.tree.map(np.asarray, state)
    for _ in range(3):
        state, value = fn(state)
    assert int(value) == 3
    _, restored_value = fn(checkpoint)
    assert int(restored_value) == 1
