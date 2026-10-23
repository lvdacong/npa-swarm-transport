import copy
import json

import numpy as np
import pytest
import torch

from npa_swarm.config import EnvConfig, TrainConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.evaluate import evaluate
from npa_swarm.graph import Graph, radius_graph
from npa_swarm.policy import LocalPolicy, squashed_log_prob
from npa_swarm.train import Trainer, compute_gae, load_policy_checkpoint, train


@pytest.fixture(scope="module", autouse=True)
def small_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def tiny_trainer(episode_steps=5, vector_actor=False):
    env = SwarmEnv(EnvConfig(n_agents=16, stage="contact", episode_steps=episode_steps))
    return Trainer(env, TrainConfig(hidden_dim=8, vector_actor=vector_actor, rollout_steps=4, bptt_steps=2, ppo_epochs=2))


def test_squashed_density_matches_torch_transform_and_extreme_values():
    mean = torch.tensor([[[0.2, -0.4]]])
    log_std = torch.tensor([-0.5, 0.1])
    latent = torch.tensor([[[0.7, -1.3]]])
    normal = torch.distributions.Normal(mean, log_std.exp())
    transformed = torch.distributions.TransformedDistribution(
        normal, [torch.distributions.transforms.TanhTransform(cache_size=1)])
    expected = transformed.log_prob(latent.tanh()).sum(-1)
    torch.testing.assert_close(squashed_log_prob(mean, log_std, latent), expected)
    assert torch.isfinite(squashed_log_prob(mean, log_std, latent * 100)).all()


def test_timeout_bootstraps_terminal_state_but_not_next_episode():
    rewards = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([10.0, 20.0, 30.0]).view(3, 1, 1)
    terminated = torch.tensor([[False], [False], [True]])
    truncated = torch.tensor([[True], [False], [False]])
    advantages, returns = compute_gae(rewards, values, next_values, terminated, truncated, 0.5, 1.0)
    # Step 0 is a timeout: 1 + .5*10, without any step-1 advantage.
    # Step 2 is a true terminal: 3, without .5*30.
    torch.testing.assert_close(advantages.flatten(), torch.tensor([6.0, 13.5, 3.0]))
    torch.testing.assert_close(returns, advantages)


def test_local_messages_take_one_hop_per_step_and_backpropagate():
    torch.manual_seed(7)
    policy = LocalPolicy(3, 8)
    graph = Graph(torch.tensor([0, 1, 1, 2]), torch.tensor([1, 0, 2, 1]), 3)
    baseline = torch.zeros(1, 3, 3)
    changed = baseline.clone()
    changed[0, 0, 0] = 1
    changed.requires_grad_()
    ha, hb = policy.initial_state(1, 3), policy.initial_state(1, 3)
    for t in range(3):
        _, _, _, ha = policy(baseline, ha, graph)
        _, _, _, hb = policy(changed, hb, graph)
        if t == 0:
            torch.testing.assert_close(ha[:, 1:], hb[:, 1:], rtol=0, atol=0)
        if t == 1:
            torch.testing.assert_close(ha[:, 2:], hb[:, 2:], rtol=0, atol=0)
    assert not torch.allclose(ha[:, 2], hb[:, 2], atol=1e-7)
    hb[:, 2].square().sum().backward()
    assert changed.grad[0, 0].abs().sum() > 0


def test_policy_is_particle_permutation_equivariant():
    torch.manual_seed(11)
    positions = torch.rand(2, 12, 2)
    graph = radius_graph(positions, 0.6)
    policy = LocalPolicy(5, 8)
    obs, hidden = torch.randn(2, 12, 5), torch.randn(2, 12, 8)
    order = torch.randperm(12)
    mean, _, value, memory = policy(obs, hidden, graph)
    pm, _, pv, ph = policy(obs[:, order], hidden[:, order], radius_graph(positions[:, order], .6))
    torch.testing.assert_close(pm, mean[:, order], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(pv, value[:, order], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ph, memory[:, order], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("vector_actor", [False, True])
def test_recurrent_ppo_updates_actor_memory_and_message_weights(vector_actor):
    trainer = tiny_trainer(vector_actor=vector_actor)
    before = copy.deepcopy(trainer.policy.state_dict())
    metrics = trainer.update()
    assert metrics["update"] == 1 and metrics["env_steps"] == 4
    assert np.isfinite(list(v for v in metrics.values() if isinstance(v, float))).all()
    after = trainer.policy.state_dict()
    for key in ("actor.weight", "memory.weight_hh", "encode.0.weight"):
        assert not torch.equal(before[key], after[key]), key
    assert not torch.equal(before["encode.0.weight"][:, -8:], after["encode.0.weight"][:, -8:])
    if vector_actor:
        assert not torch.equal(before["basis_actor.weight"], after["basis_actor.weight"])


def test_cpu_resume_reproduces_next_complete_update_across_pool_reset(tmp_path):
    trainer = tiny_trainer(episode_steps=3)
    trainer.update()
    path = tmp_path / "resume.pt"
    trainer.save(path)
    expected_metrics = trainer.update()
    expected_model = copy.deepcopy(trainer.policy.state_dict())
    expected_env = trainer.env.state_dict()
    expected_hidden = trainer.hidden.clone()
    restored = Trainer.load(path)
    actual_metrics = restored.update()
    assert restored.updates == 2 and restored.env_steps == 8
    assert restored.episodes == trainer.episodes > 0
    for key, value in expected_model.items():
        torch.testing.assert_close(restored.policy.state_dict()[key], value, rtol=0, atol=0)
    for key, value in expected_env.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(restored.env.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.hidden, expected_hidden, rtol=0, atol=0)
    assert actual_metrics["mean_reward"] == expected_metrics["mean_reward"]
    assert actual_metrics["policy_loss"] == expected_metrics["policy_loss"]


def test_train_resume_adds_budget_and_warmstart_resets_it(tmp_path):
    env_cfg = EnvConfig(n_agents=16, stage="contact", episode_steps=3)
    train_cfg = TrainConfig(hidden_dim=8, rollout_steps=2, bptt_steps=2, ppo_epochs=1)
    output = tmp_path / "first"
    first = train(env_cfg, train_cfg, output, 1)
    resumed = train(None, None, output, 1, resume=output / "last.pt")
    assert first["updates"] == 1 and resumed["updates"] == 2
    assert resumed["env_steps"] == 4
    with pytest.raises(FileExistsError):
        train(env_cfg, train_cfg, output, 1)
    new_cfg = EnvConfig(n_agents=16, stage="search", episode_steps=3)
    warm = train(new_cfg, train_cfg, tmp_path / "warm", 0, warmstart=output / "last.pt")
    assert warm["updates"] == 0 and warm["lineage"]["parent_updates"] == 2
    assert warm["lineage"]["kind"] == "warmstart"
    policy, cfg, metadata = load_policy_checkpoint(tmp_path / "warm" / "last.pt")
    assert cfg.stage == "search" and metadata["updates"] == 0
    assert metadata["has_trained_weights"]
    assert not metadata["learned_success_verified"]


def test_evaluation_preserves_timeouts_and_raw_trajectories(tmp_path):
    trainer = tiny_trainer(episode_steps=2)
    checkpoint = tmp_path / "policy.pt"
    trainer.save(checkpoint)
    output = tmp_path / "evaluation"
    summary = evaluate(checkpoint, output, seeds=(17, 18), episodes_per_seed=1)
    assert summary["requested"] == summary["completed"] == summary["timeout"] == 2
    assert summary["diagnostic"] and not summary["research_task_passed"]
    manifest = json.loads((output / "manifest.json").read_text())
    for request in manifest["requests"]:
        trajectory = np.load(output / request["trajectory"])
        assert trajectory["positions"].shape == (3, 16, 2)
        assert trajectory["actions"].shape == (2, 16, 2)
        assert len(json.loads(str(trajectory["metrics_json"]))) == 3


def test_evaluation_preserves_error_and_not_started_denominator(tmp_path, monkeypatch):
    trainer = tiny_trainer()
    checkpoint = tmp_path / "policy.pt"
    trainer.save(checkpoint)

    def fail(*args):
        raise RuntimeError("deliberate simulation failure")

    monkeypatch.setattr(SwarmEnv, "step", fail)
    output = tmp_path / "failed"
    summary = evaluate(checkpoint, output, seeds=(42, 43), episodes_per_seed=1)
    assert summary["requested"] == 2 and summary["error"] == 1 and summary["not_started"] == 1
    assert not summary["engineering_evaluation_completed"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert "deliberate simulation failure" in manifest["requests"][0]["error"]
    assert (output / manifest["requests"][0]["trajectory"]).exists()


def test_short_evaluation_is_censored_not_formal_success(tmp_path):
    trainer = tiny_trainer(episode_steps=10)
    checkpoint = tmp_path / "policy.pt"
    trainer.save(checkpoint)
    summary = evaluate(checkpoint, tmp_path / "short", seeds=(5,), episodes_per_seed=1, max_steps=1)
    assert summary["censored"] == 1 and summary["completed"] == 0
    assert summary["engineering_evaluation_completed"] and not summary["full_horizon_completed"]


def test_sampled_frozen_evaluation_repeats_recorded_action_seeds(tmp_path):
    trainer = tiny_trainer(episode_steps=2)
    checkpoint = tmp_path / "policy.pt"
    trainer.save(checkpoint)
    for name in ("a", "b"):
        summary = evaluate(checkpoint, tmp_path / name, seeds=(19,), episodes_per_seed=1,
                           stage="contact", deterministic=False)
        assert summary["policy_mode"] == "sampled_seeded" and summary["requested"] == 1
    a = np.load(tmp_path / "a" / "episode-0000.npz")
    b = np.load(tmp_path / "b" / "episode-0000.npz")
    np.testing.assert_array_equal(a["actions"], b["actions"])
    np.testing.assert_array_equal(a["positions"], b["positions"])
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["requests"][0]["action_seed"] == 19


def test_modified_task_scales_are_recorded_as_diagnostic(tmp_path):
    trainer = tiny_trainer(episode_steps=2)
    trainer.env.cfg.sense_radius = 1.8
    checkpoint = tmp_path / "policy.pt"
    trainer.save(checkpoint)
    summary = evaluate(checkpoint, tmp_path / "evaluation", seeds=(19,), episodes_per_seed=1,
                       stage="search")
    assert summary["diagnostic"]
    assert summary["task_config_differences"]["sense_radius"] == {"actual": 1.8, "reference": .8}


def test_checkpoint_keeps_loaded_source_identity_if_files_change(tmp_path, monkeypatch):
    trainer = tiny_trainer()
    loaded_identity = trainer.runtime_source_sha256
    monkeypatch.setattr("npa_swarm.train.source_sha256", lambda: "changed-on-disk")
    checkpoint = tmp_path / "policy.pt"
    trainer.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["source_sha256"] == loaded_identity


def test_vector_only_actor_has_no_unconditioned_world_direction():
    policy = LocalPolicy(20, 8, vector_actor=True, vector_only=True)
    with torch.no_grad():
        policy.actor.bias.fill_(3)
        policy.basis_actor.bias.fill_(.4)
    obs = torch.zeros(1, 3, 20)
    obs[..., 6] = .5  # scalar density cannot provide a direction
    obs[..., 16:] = .7
    graph = Graph(torch.tensor([0, 1]), torch.tensor([1, 0]), 3)
    mean, _, _, _ = policy(obs, torch.randn(1, 3, 8), graph)
    torch.testing.assert_close(mean, torch.zeros_like(mean), rtol=0, atol=0)
