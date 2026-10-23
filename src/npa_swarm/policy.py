"""Shared local recurrent policy. No coordinates or privileged task inputs."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .graph import Graph, aggregate


def squashed_log_prob(mean: torch.Tensor, log_std: torch.Tensor,
                      latent: torch.Tensor) -> torch.Tensor:
    """Log density of tanh(latent), using a stable exact Jacobian."""
    normal_logp = -0.5 * (((latent - mean) / log_std.exp()).square()
                         + 2 * log_std + math.log(2 * math.pi))
    log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
    return (normal_logp - log_jacobian).sum(dim=-1)


class LocalPolicy(nn.Module):
    """One-hop learned messages, persistent GRU memory, and bounded 2-D force.

    The same weights act on every particle. Hidden channels are learned scalar
    messages; they do not grant remote sensing. A message travels at most one
    graph hop per control step. Relative vectors use common simulation axes.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 32, vector_actor: bool = False, vector_only: bool = False):
        super().__init__()
        self.obs_dim, self.hidden_dim = obs_dim, hidden_dim
        self.encode = nn.Sequential(nn.Linear(obs_dim + hidden_dim, hidden_dim), nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        self.vector_actor = vector_actor
        if vector_only and not vector_actor:
            raise ValueError("vector_only requires vector_actor")
        self.vector_only = vector_only
        if vector_actor:
            if obs_dim != 20:
                raise ValueError("The local vector actor requires the documented 20-dimensional observation")
            # Every coefficient is learned. The bases are existing LOCAL
            # observations, not a privileged goal direction or a controller.
            # This gives the optimizer a shared direction parameter across
            # layouts instead of learning only a world-axis action bias.
            self.basis_actor = nn.Linear(hidden_dim, 6)
            nn.init.zeros_(self.basis_actor.weight)
            nn.init.zeros_(self.basis_actor.bias)

    def initial_state(self, batch: int, n_agents: int, device=None) -> torch.Tensor:
        if device is None:
            device = self.log_std.device
        return torch.zeros(batch, n_agents, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, graph: Graph):
        neighbor_memory = aggregate(hidden, graph)
        encoded = self.encode(torch.cat((obs, neighbor_memory), dim=-1))
        next_hidden = self.memory(encoded.flatten(0, 1), hidden.flatten(0, 1)).view_as(hidden)
        mean = torch.zeros_like(obs[..., :2]) if self.vector_only else self.actor(next_hidden)
        if self.vector_actor:
            bases = torch.stack((obs[..., 0:2], obs[..., 2:4], obs[..., 4:6],
                                 obs[..., 7:9], obs[..., 10:12], obs[..., 13:15]), dim=-2)
            mean = mean + (self.basis_actor(next_hidden)[..., None] * bases).sum(-2)
        return (mean, self.log_std.clamp(-5, 1),
                self.critic(next_hidden).squeeze(-1), next_hidden)

    def act(self, obs: torch.Tensor, hidden: torch.Tensor, graph: Graph,
            deterministic: bool = False, generator=None):
        mean, log_std, value, next_hidden = self(obs, hidden, graph)
        if deterministic:
            latent = mean
        else:
            noise = (torch.randn_like(mean) if generator is None else
                     torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator))
            latent = mean + log_std.exp() * noise
        return (latent.tanh(), latent, squashed_log_prob(mean, log_std, latent),
                value, next_hidden)

    def evaluate_actions(self, obs: torch.Tensor, hidden: torch.Tensor, graph: Graph,
                         latent: torch.Tensor):
        mean, log_std, value, next_hidden = self(obs, hidden, graph)
        log_prob = squashed_log_prob(mean, log_std, latent)
        # A reparameterized Monte Carlo estimate of the *squashed* entropy.
        entropy_latent = mean + log_std.exp() * torch.randn_like(mean)
        entropy = -squashed_log_prob(mean, log_std, entropy_latent)
        return log_prob, entropy, value, next_hidden
