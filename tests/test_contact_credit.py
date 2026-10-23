"""Physical credit must reward signed, actual contact, never distant actions."""
import torch

from npa_swarm.config import EnvConfig
from npa_swarm.env import SwarmEnv


def arranged(weight):
    env = SwarmEnv(EnvConfig(n_agents=8, stage="contact", force_credit_weight=weight))
    env.positions[0] = torch.tensor([[-.687, 0], [3, 3], [3.1, 3], [3.2, 3],
                                     [3.3, 3], [3.4, 3], [3.5, 3], [3.6, 3]])
    env.object_pos.zero_()
    env.goal_pos[:] = torch.tensor([2., 0])
    env._refresh()
    env.found_goal[:] = True
    return env


def test_credit_changes_reward_but_never_actions_physics_or_observations():
    base, shaped = arranged(0), arranged(.5)
    action = torch.zeros_like(base.positions)
    action[0, 0, 0] = 1
    a, b = base.step(action), shaped.step(action)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(base.positions, shaped.positions, rtol=0, atol=0)
    torch.testing.assert_close(base.object_pos, shaped.object_pos, rtol=0, atol=0)
    assert b[4]["reward_contact_credit"].item() > 0
    assert b[1][0, 0] > a[1][0, 0]
    torch.testing.assert_close(a[1][0, 1:], b[1][0, 1:], rtol=0, atol=0)
    assert torch.equal(base.object_pos, torch.zeros_like(base.object_pos))  # one cannot overcome friction


def test_opposing_contact_has_negative_credit_and_hidden_goal_has_none():
    env = arranged(.5)
    env.goal_pos *= -1
    env._refresh()
    action = torch.zeros_like(env.positions)
    action[0, 0, 0] = 1
    assert env.step(action)[4]["reward_contact_credit"].item() < 0
    env = arranged(.5)
    env.goal_pos[:] = torch.tensor([8., 0])
    env.found_goal[:] = False
    env._refresh()
    assert env.step(action)[4]["reward_contact_credit"].item() == 0


def test_local_transport_credit_keeps_dynamics_and_excludes_distant_agents():
    base, local = arranged(0), arranged(0)
    local.cfg.local_transport_reward = True
    base.cfg.object_static_friction = local.cfg.object_static_friction = 0
    action = torch.zeros_like(base.positions)
    action[0, 0, 0] = 1
    a, b = base.step(action), local.step(action)
    assert a[4]["reward_progress"].item() > 0
    torch.testing.assert_close(base.positions, local.positions, rtol=0, atol=0)
    torch.testing.assert_close(base.object_pos, local.object_pos, rtol=0, atol=0)
    torch.testing.assert_close(a[1][0, 0], b[1][0, 0], rtol=0, atol=0)
    torch.testing.assert_close(a[1][0, 1:] - b[1][0, 1:], a[4]["reward_progress"].expand(7))
