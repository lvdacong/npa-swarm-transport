"""Training-only centralized team baseline and cooperative advantages.

This module is NOT imported by inference / server / visualization. All
privileged features feed only a value estimator; actor still receives only
its sensor observations, one-hop message relay and recurrent local state.
"""
import torch
from torch import nn


@torch.no_grad()
def world_features(env):
    c = env.cfg
    toward = env.goal_pos - env.object_pos
    direction = toward / toward.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack((
        env._distance / (2.8285 * c.arena_half),
        env.found_object.float(), env.found_goal.float(),
        env._largest_fraction, env._bridge.float(),
        env._contact.float().sum(-1) / c.n_agents,
        env.object_vel.norm(dim=-1) / c.max_speed,
        (env.object_vel * direction).sum(-1) / c.max_speed,
        (env._object_distance_agent - c.object_radius).clamp_min(0).amin(-1) / c.arena_half,
        (env._goal_delta.norm(dim=-1) - c.goal_radius).clamp_min(0).amin(-1) / c.arena_half,
        env._object_visible.float().mean(-1),
        env._goal_visible.float().mean(-1),
    ), dim=-1)


class TeamCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 48), nn.Tanh(),
                                 nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, context):
        return self.net(context).squeeze(-1)


def world_reward(info, terminated):
    """Team objective from real task outcomes, not per-particle fake success."""
    return (info["reward_progress"] + info["reward_discovery"]
            + 5. * terminated.float() + info["reward_connectivity"]
            + info["reward_endpoint_progress"] + info["reward_exploration"])


def actor_advantages(team_adv, local_adv, focus):
    """Independent normalization of team progress and local relative credits."""
    team = (team_adv - team_adv.mean()) / (team_adv.std(unbiased=False) + 1e-6)
    centered = local_adv - local_adv.mean(-1, keepdim=True)
    bonus = torch.zeros_like(centered)
    if bool(focus.any()):
        selected = centered[focus]
        if selected.numel() >= 2:
            bonus[focus] = ((selected - selected.mean()) /
                            (selected.std(unbiased=False) + 1e-6))
    return (team[..., None] + .5 * bonus).detach()
