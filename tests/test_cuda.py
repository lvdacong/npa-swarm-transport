"""Physical CUDA checks. A CPU-only runner must report these as skipped."""

import copy
import math

import pytest
import torch

from npa_swarm.config import EnvConfig, TrainConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.graph import Graph, aggregate, radius_graph
from npa_swarm.policy import LocalPolicy
from npa_swarm.train import Trainer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a physical CUDA device")


@pytest.fixture(autouse=True)
def fp32_reference():
    old_matmul, old_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_matmul, old_cudnn


def test_cuda_radius_graph_and_local_policy_backward_match_cpu():
    generator = torch.Generator().manual_seed(61)
    positions = torch.rand(2, 64, 2, generator=generator)
    positions[0, 1] = positions[0, 0]  # separate stable IDs at the same location
    cpu_graph, gpu_graph = radius_graph(positions, .29), radius_graph(positions.cuda(), .29)
    cpu_edges = (cpu_graph.src * 128 + cpu_graph.dst).sort().values
    gpu_edges = (gpu_graph.src * 128 + gpu_graph.dst).sort().values.cpu()
    assert torch.equal(cpu_edges, gpu_edges)
    torch.manual_seed(12)
    cpu_policy = LocalPolicy(20, 16)
    gpu_policy = copy.deepcopy(cpu_policy).cuda()
    obs = torch.randn(2, 64, 20, generator=generator)
    hidden = torch.randn(2, 64, 16, generator=generator)
    cpu = cpu_policy(obs, hidden, cpu_graph)
    gpu = gpu_policy(obs.cuda(), hidden.cuda(), gpu_graph)
    for i in (0, 2, 3):
        torch.testing.assert_close(gpu[i].cpu(), cpu[i], atol=2e-5, rtol=2e-4)
    (cpu[0].square().mean() + cpu[2].square().mean() + cpu[3].square().mean()).backward()
    (gpu[0].square().mean() + gpu[2].square().mean() + gpu[3].square().mean()).backward()
    for (name, a), (_, b) in zip(cpu_policy.named_parameters(), gpu_policy.named_parameters()):
        if a.grad is not None:
            assert torch.isfinite(b.grad).all(), name
            torch.testing.assert_close(b.grad.cpu(), a.grad, atol=2e-5, rtol=2e-4, msg=name)


def test_cuda_environment_state_restore_continues_with_tolerance():
    cfg = EnvConfig(n_agents=48, num_envs=2, stage="contact", episode_steps=20)
    env = SwarmEnv(cfg, "cuda")
    generator = torch.Generator().manual_seed(83)
    actions = [torch.rand(2, 48, 2, generator=generator).cuda() * 2 - 1 for _ in range(5)]
    for action in actions[:2]:
        env.step(action)
    restored = SwarmEnv(cfg, "cuda")
    restored.load_state_dict(env.state_dict())
    for action in actions[2:]:
        expected, actual = env.step(action), restored.step(action)
        for index in (0, 1):
            torch.testing.assert_close(actual[index], expected[index], atol=5e-5, rtol=5e-4)
        for index in (2, 3):
            assert torch.equal(actual[index], expected[index])
    torch.testing.assert_close(restored.positions, env.positions, atol=5e-5, rtol=5e-4)
    torch.testing.assert_close(restored.object_pos, env.object_pos, atol=5e-5, rtol=5e-4)
    assert torch.equal(restored.steps, env.steps)
    assert torch.equal(restored.generator.get_state(), env.generator.get_state())


def test_cuda_ppo_checkpoint_next_update_preserves_counts_and_finite_state(tmp_path):
    env_cfg = EnvConfig(n_agents=32, stage="contact", episode_steps=3)
    train_cfg = TrainConfig(hidden_dim=16, rollout_steps=4, bptt_steps=2, ppo_epochs=1)
    trainer = Trainer(SwarmEnv(env_cfg, "cuda"), train_cfg)
    trainer.update()
    checkpoint = tmp_path / "cuda.pt"
    trainer.save(checkpoint)
    expected_metrics = trainer.update()
    expected_state = {k: v.detach().cpu().clone() for k, v in trainer.policy.state_dict().items()}
    expected_hidden = trainer.hidden.detach().cpu().clone()
    restored = Trainer.load(checkpoint, "cuda")
    actual_metrics = restored.update()
    assert restored.updates == 2 and restored.env_steps == 8
    assert restored.episodes == trainer.episodes
    assert all(math.isfinite(v) for v in actual_metrics.values() if isinstance(v, float))
    # Atomic reductions need not reproduce bit for bit. These are short bounded
    # comparisons, not a claim about long chaotic physics/learning trajectories.
    for name, actual in restored.policy.state_dict().items():
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual.cpu(), expected_state[name], atol=2e-5, rtol=2e-4, msg=name)
    torch.testing.assert_close(restored.hidden.cpu(), expected_hidden, atol=1e-4, rtol=1e-3)
    assert abs(actual_metrics["mean_reward"] - expected_metrics["mean_reward"]) < 1e-4


def test_cuda_sparse_mean_shuffled_duplicate_edges_preserves_values_and_gradients():
    graph = Graph(torch.tensor([2, 0, 1, 0]), torch.tensor([1, 1, 0, 1]), 3)
    gpu_graph = graph.to("cuda")
    x = torch.tensor([[[1., 2.], [3., 4.], [7., 8.]]], requires_grad=True)
    y = x.detach().cuda().requires_grad_()
    expected, actual = aggregate(x, graph), aggregate(y, gpu_graph)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y.grad.cpu(), x.grad, atol=2e-6, rtol=2e-5)
    matrix = gpu_graph._mean_matrix
    torch.sparse_csr_tensor(matrix.crow_indices(), matrix.col_indices(), matrix.values(),
                           size=matrix.shape, check_invariants=True)
