import math

import pytest
import torch

from npa_swarm.config import EnvConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.graph import Graph, aggregate, components, radius_graph


def brute_edges(positions, radius):
    b, n, _ = positions.shape
    result = set()
    for e in range(b):
        for i in range(n):
            for j in range(n):
                if i != j and torch.sum((positions[e, i] - positions[e, j]) ** 2) <= radius ** 2:
                    result.add((e * n + i, e * n + j))
    return result


def brute_labels(graph, size):
    adjacency = [[] for _ in range(size)]
    for a, b in zip(graph.src.tolist(), graph.dst.tolist()):
        adjacency[a].append(b)
        adjacency[b].append(a)
    labels = [-1] * size
    for seed in range(size):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = seed
        while stack:
            for neighbor in adjacency[stack.pop()]:
                if labels[neighbor] < 0:
                    labels[neighbor] = seed
                    stack.append(neighbor)
    return labels


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_complete_spatial_graph_matches_all_pairs(seed):
    p = torch.randn(3, 23, 2, generator=torch.Generator().manual_seed(seed))
    p[0, 1] = p[0, 0]  # coincident, distinct IDs must still be neighbors
    original = p.clone()
    graph = radius_graph(p, .9)
    actual = set(zip(graph.src.tolist(), graph.dst.tolist()))
    assert actual == brute_edges(p, .9)
    assert len(actual) == graph.src.numel()
    assert torch.equal(p, original)
    assert components(graph, 3, 23).flatten().tolist() == brute_labels(graph, 69)


def test_dense_cell_is_not_silently_capped():
    p = torch.zeros(1, 200, 2)
    graph = radius_graph(p, .1)
    assert graph.src.numel() == 200 * 199


def test_exact_components_long_shuffled_chain():
    n = 513
    order = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    src = torch.cat((order[:-1], order[1:]))
    dst = torch.cat((order[1:], order[:-1]))
    assert torch.equal(components(Graph(src, dst), 1, n), torch.zeros(1, n, dtype=torch.long))


def test_message_mean_and_backward():
    state = torch.tensor([[[1.0], [3.0], [7.0], [9.0]]], requires_grad=True)
    graph = Graph(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 1]))
    actual = aggregate(state, graph)
    assert actual.flatten().tolist() == [0, 4, 3, 0]
    actual.sum().backward()
    assert state.grad.flatten().tolist() == [.5, 1, .5, 0]


@pytest.mark.parametrize("n", [32, 4096])
def test_search_initial_connected_and_targets_hidden(n):
    env = SwarmEnv(EnvConfig(n_agents=n, substeps=8))
    assert env.metrics()["connected"].all()
    assert not env.found_object.any() and not env.found_goal.any()
    obs = env.observe()
    assert obs.shape == (1, n, 20)
    assert torch.isfinite(obs).all()
    assert not obs[..., 9].any() and not obs[..., 12].any()
    pair_delta = env.positions.flatten(0, 1)[env.graph.src] - env.positions.flatten(0, 1)[env.graph.dst]
    assert pair_delta.norm(dim=-1).min() >= 2 * env.cfg.agent_radius - 1e-6


def test_observation_has_no_absolute_position_interior_translation():
    env = SwarmEnv(EnvConfig(n_agents=32, substeps=8))
    # Put both local target cues in range, then translate every physical object.
    env.object_pos[:] = torch.tensor([.7, .1])
    env.goal_pos[:] = torch.tensor([-.8, .2])
    env._refresh()
    before = env.observe()
    shift = torch.tensor([.35, -.42])
    env.positions += shift
    env.object_pos += shift
    env.goal_pos += shift
    env._refresh()
    assert torch.allclose(env.observe(), before, atol=2e-6)


@pytest.mark.parametrize("n", [32, 4096])
def test_contact_warmup_has_local_witnesses_for_both_endpoints(n):
    env = SwarmEnv(EnvConfig(n_agents=n, substeps=8, stage="contact"))
    assert env.metrics()["connected"].all()
    assert env.found_object.all() and env.found_goal.all()
    assert env.metrics()["bridge"].all()
    assert not env.metrics()["success"].any()


def test_no_remote_force_and_no_stationary_contact_bonus():
    env = SwarmEnv(EnvConfig(n_agents=16, substeps=8))
    before = env.object_pos.clone()
    env.step(torch.ones_like(env.positions))
    assert torch.equal(env.object_pos, before)
    _, _, _, _, info = env.step(torch.zeros_like(env.positions))
    assert info["reward_progress"].item() == 0
    assert info["reward_local_progress"].item() == 0


def arrange_push(env, active):
    n = env.cfg.n_agents
    env.object_pos.zero_()
    env.goal_pos[:] = torch.tensor([5.0, 0.0])
    env.object_vel.zero_()
    env.positions[0] = torch.stack((torch.linspace(-4, -3, n), torch.ones(n) * 3), -1)
    angles = torch.linspace(math.pi - .45, math.pi + .45, active) if active > 1 else torch.tensor([math.pi])
    env.positions[0, :active] = torch.stack((angles.cos(), angles.sin()), -1) * .687
    env.velocities.zero_()
    env._refresh()


def test_one_agent_cannot_push_heavy_object_but_group_can():
    one = SwarmEnv(EnvConfig(n_agents=16, substeps=8))
    many = SwarmEnv(EnvConfig(n_agents=16, substeps=8))
    arrange_push(one, 1)
    arrange_push(many, 8)
    a = torch.zeros_like(one.positions)
    a[0, 0, 0] = 1
    b = torch.zeros_like(many.positions)
    b[0, :8, 0] = 1
    for _ in range(30):
        one.step(a)
        many.step(b)
    assert one.object_pos[0, 0].abs() < 1e-5
    assert many.object_pos[0, 0] > .02


def test_disconnection_penalty_without_termination_or_protection():
    env = SwarmEnv(EnvConfig(n_agents=8, substeps=8))
    env.positions[0, :4] = torch.tensor([[-3., 0], [-2.9, 0], [-2.8, 0], [-2.7, 0]])
    env.positions[0, 4:] = torch.tensor([[3., 0], [3.1, 0], [3.2, 0], [3.3, 0]])
    env._refresh()
    before = env.positions.clone()
    _, _, terminated, truncated, info = env.step(torch.zeros_like(env.positions))
    assert not terminated.any() and not truncated.any()
    assert torch.equal(before, env.positions)
    assert not info["connected"].any()
    assert info["largest_fraction"].item() == .5
    assert info["reward_connectivity"].item() < 0


def test_whole_object_success_and_time_limit_separate():
    env = SwarmEnv(EnvConfig(n_agents=8, substeps=8, episode_steps=1))
    _, _, term, trunc, _ = env.step(torch.zeros_like(env.positions))
    assert not term.any() and trunc.all()
    env.reset()
    env.object_pos.copy_(env.goal_pos)
    env._refresh()
    _, _, term, trunc, info = env.step(torch.zeros_like(env.positions))
    assert term.all() and not trunc.any() and info["success"].all()


def test_checkpoint_restores_physics_metrics_and_reset_rng():
    cfg = EnvConfig(n_agents=32, num_envs=2, substeps=8)
    env = SwarmEnv(cfg)
    action = torch.rand(2, 32, 2, generator=torch.Generator().manual_seed(7)) * 2 - 1
    env.step(action)
    state = env.state_dict()
    restored = SwarmEnv(cfg)
    restored.load_state_dict(state)
    a, b = env.step(action), restored.step(action)
    for i in range(4):
        assert torch.equal(a[i], b[i])
    for key in a[4]:
        assert torch.equal(a[4][key], b[4][key])
    untouched = env.positions[1].clone()
    mask = torch.tensor([True, False])
    env.reset(mask)
    restored.reset(mask)
    assert torch.equal(env.positions, restored.positions)
    assert torch.equal(env.object_pos, restored.object_pos)
    assert torch.equal(env.positions[1], untouched)
