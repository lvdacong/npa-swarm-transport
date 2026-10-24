"""Causal reward-to-advantage-to-actor check, not a transport success test."""
import copy
import pytest
import torch
from npa_swarm.config import EnvConfig, TrainConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.train import Trainer, capture_rng, restore_rng, load_policy_checkpoint


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rear_reward_changes_learning_with_identical_physics(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires physical CUDA")
    cfg = TrainConfig(hidden_dim=8, rollout_steps=4, bptt_steps=2, ppo_epochs=1)
    trainers = []
    for weight in (.8, 0.):
        env = SwarmEnv(EnvConfig(n_agents=8, rear_approach_weight=weight), device)
        env.positions[0] = torch.tensor([[-1.54,-.09],[-1.40,-.09],[-1.26,-.09],[-1.12,-.09],
                                        [-1.54,.09],[-1.40,.09],[-1.26,.09],[-1.12,.09]], device=device)
        env.object_pos.zero_(); env.goal_pos[:] = torch.tensor([1., 0.], device=device)
        env._refresh()
        trainers.append(Trainer(env, cfg))
    a, b = trainers
    b.policy.load_state_dict(a.policy.state_dict())
    rng = capture_rng(); ra = a.collect(); restore_rng(rng, device); rb = b.collect()
    torch.testing.assert_close(ra.latent_actions, rb.latent_actions, rtol=0, atol=0)
    torch.testing.assert_close(a.env.positions, b.env.positions, rtol=0, atol=0)
    torch.testing.assert_close(a.env.object_pos, b.env.object_pos, rtol=0, atol=0)
    assert ra.metrics['rear_active_agent_steps'] > 0
    assert ra.metrics['rear_reward_active_abs_mean'] > 0
    assert not torch.equal(ra.advantages, rb.advantages)
    before = copy.deepcopy(a.policy.state_dict())
    rng = capture_rng(); a.optimize(ra); restore_rng(rng, device); b.optimize(rb)
    assert not torch.equal(before['actor.weight'], a.policy.actor.weight)
    assert not torch.equal(a.policy.actor.weight, b.policy.actor.weight)


def test_legacy_missing_relay_field_keeps_original_time_decay(tmp_path):
    t = Trainer(SwarmEnv(EnvConfig(n_agents=8, rear_approach_weight=0)),
                TrainConfig(hidden_dim=8, relay_memory_decay=.97))
    path = tmp_path / 'legacy.pt'; t.save(path)
    data = torch.load(path, weights_only=False)
    del data['train_config']['relay_memory_decay']
    torch.save(data, path)
    assert Trainer.load(path).policy.relay_memory_decay == .97
    assert load_policy_checkpoint(path)[0].relay_memory_decay == .97


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rear_potential_never_rewards_front_turn_across_sensed_annulus(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires physical CUDA")
    from npa_swarm.env import rear_arc_gap
    cosine = torch.linspace(-1, 1, 201, device=device)
    direction = torch.stack((cosine, (1-cosine.square()).clamp_min(0).sqrt()), -1)
    distances = torch.linspace(.71, 1.85, 20, device=device)
    positions = distances[:, None, None] * direction[None]
    cargo = torch.zeros(20, 2, device=device)
    goal = torch.zeros_like(cargo); goal[:, 0] = 3
    gap = rear_arc_gap(positions, cargo, goal, .71, approach_limit=1.91)
    assert (gap[:, 1:] - gap[:, :-1] >= -1e-6).all()
    front = (cosine[:-1] > .2) & (cosine[1:] < .99)
    assert (gap[:, 1:] - gap[:, :-1])[:, front].amin() > 0
    old = rear_arc_gap(positions, cargo, goal, .71, version=1)
    assert (old[:, 1:] - old[:, :-1] < -1e-5).any()  # original counterexample


def test_old_rear_potential_restores_version_one(tmp_path):
    t = Trainer(SwarmEnv(EnvConfig(n_agents=8, rear_potential_version=1)), TrainConfig(hidden_dim=8))
    p = tmp_path / 'rear-v1.pt'; t.save(p)
    d = torch.load(p, weights_only=False)
    del d['env_config']['rear_potential_version']; del d['env_state']['config']['rear_potential_version']
    torch.save(d, p)
    assert Trainer.load(p).env.cfg.rear_potential_version == 1
    assert load_policy_checkpoint(p)[1].rear_potential_version == 1


def test_neighbor_eligibility_is_explicit_and_preserves_connectivity_task(tmp_path):
    envs = [SwarmEnv(EnvConfig(n_agents=8,rear_eligibility=mode)) for mode in ('main','neighbors')]
    for e in envs:
        e.positions[0] = torch.tensor([[-1.3,0.],[-1.1,0.],[-1.2,.7],
                                      [4.,4.],[4.2,4.],[4.4,4.],[4.,4.2],[4.2,4.2]])
        e.object_pos.zero_(); e.goal_pos[:] = torch.tensor([2.,0.]); e._refresh()
    a, b = envs
    assert not a.rear_membership()[0,:3].any()
    assert b.rear_membership()[0,:2].all() and not b.rear_membership()[0,2]
    action = torch.zeros_like(a.positions); action[0,:2,0] = 1
    x = a.step(action,torch.ones(1,8)); y = b.step(action,torch.ones(1,8))
    torch.testing.assert_close(a.positions,b.positions,atol=0,rtol=0)
    torch.testing.assert_close(a.observe(),b.observe(),atol=0,rtol=0)
    assert x[4]['rear_reward_sum'] == 0 and y[4]['rear_reward_sum'] > 0
    for key in ('connected','bridge','reward_connectivity'):
        torch.testing.assert_close(x[4][key],y[4][key],atol=0,rtol=0)
    t=Trainer(b,TrainConfig(hidden_dim=8)); path=tmp_path/'neighbors.pt';t.save(path)
    assert Trainer.load(path).env.cfg.rear_eligibility == 'neighbors'

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_local_batch_empty_fallback_and_active_update(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires physical CUDA")
    from dataclasses import replace
    torch.set_num_threads(2)
    base = Trainer(SwarmEnv(EnvConfig(n_agents=8),device), TrainConfig(hidden_dim=8, rollout_steps=4, bptt_steps=2, ppo_epochs=1))
    path = tmp_path/'initial.pt'; base.save(path)
    r = base.collect()
    r.rear_active = torch.zeros_like(r.advantages, dtype=torch.bool)
    uniform = Trainer.load(path,device); local = Trainer.load(path,device)
    local.cfg = replace(local.cfg, local_batch_fraction=.5)
    rng = capture_rng(); uniform.optimize(r); restore_rng(rng, device); local.optimize(r)
    for k,v in uniform.policy.state_dict().items():
        torch.testing.assert_close(v,local.policy.state_dict()[k],rtol=1e-4,atol=2e-6)
    r.rear_active.flatten()[0] = True
    assert local.optimize(r)["local_batch_fraction_effective"] == 0
    r.rear_active[...,0] = True
    local = Trainer.load(path,device); local.cfg = replace(local.cfg,local_batch_fraction=.5)
    restore_rng(rng, device); local.optimize(r)
    assert not torch.equal(uniform.policy.actor.weight,local.policy.actor.weight)
    local.save(tmp_path/'local.pt')
    assert Trainer.load(tmp_path/'local.pt').cfg.local_batch_fraction == .5


