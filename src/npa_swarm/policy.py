"""One shared recurrent policy, with learned scalar and directional messages."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .graph import Graph, aggregate


def squashed_log_prob(mean, log_std, latent):
    normal_logp = -.5 * (((latent - mean) / log_std.exp()).square() + 2 * log_std + math.log(2 * math.pi))
    log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
    return (normal_logp - log_jacobian).sum(-1)


class LocalPolicy(nn.Module):
    """Local observation -> GRU + two learned vector slots -> bounded force.

    Neighbor messages are OLD state, hence one hop per control step. Vector
    slots have no prescribed target identity or routing rule. Learned scalar
    coefficients write/retain/read local directions; there is no unconditional
    world-axis action bias. Relative vectors share simulation axes; this is not
    a claim of strict rotation equivariance of the scalar encoder.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 32):
        super().__init__()
        if obs_dim != 20 or hidden_dim < 8:
            raise ValueError("Expected the 20-channel local observation and hidden_dim >= 8")
        self.obs_dim, self.hidden_dim = obs_dim, hidden_dim
        scalar_dim = hidden_dim - 4
        self.encode = nn.Sequential(nn.Linear(obs_dim + hidden_dim, hidden_dim), nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, scalar_dim)
        self.vector_write = nn.Linear(scalar_dim, 2 * 10)
        self.vector_gate = nn.Linear(scalar_dim, 2)
        self.actor = nn.Linear(scalar_dim, 8)
        self.critic = nn.Linear(scalar_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), -.5))
        nn.init.orthogonal_(self.vector_write.weight, gain=.05)
        nn.init.zeros_(self.vector_write.bias)
        nn.init.zeros_(self.vector_gate.weight)
        nn.init.constant_(self.vector_gate.bias, 2.)
        nn.init.orthogonal_(self.actor.weight, gain=.05)
        nn.init.zeros_(self.actor.bias)

    def initial_state(self, batch: int, n_agents: int, device=None):
        return torch.zeros(batch, n_agents, self.hidden_dim,
                           device=self.log_std.device if device is None else device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, graph: Graph):
        neighbors = aggregate(hidden, graph)
        encoded = self.encode(torch.cat((obs, neighbors), -1))
        scalar = self.memory(encoded.flatten(0, 1), hidden[..., :-4].flatten(0, 1))
        scalar = scalar.view(*hidden.shape[:-1], self.hidden_dim - 4)
        vectors = hidden[..., -4:].reshape(*hidden.shape[:-1], 2, 2)
        neighbor_vectors = neighbors[..., -4:].reshape_as(vectors)
        local = torch.stack((obs[..., :2], obs[..., 2:4], obs[..., 4:6],
                             obs[..., 7:9], obs[..., 10:12], obs[..., 13:15]), -2)
        bases = torch.cat((local, vectors, neighbor_vectors), -2)
        coefficients = self.vector_write(scalar).view(*scalar.shape[:-1], 2, 10).tanh()
        candidate = (coefficients[..., None] * bases[..., None, :, :]).sum(-2)
        # Radial, not per-axis clipping; finite learned memory without a
        # hard-coded goal gradient or reconnection controller.
        candidate = candidate / candidate.norm(dim=-1, keepdim=True).clamp_min(1)
        gate = self.vector_gate(scalar).sigmoid()[..., None]
        next_vectors = gate * vectors + (1 - gate) * candidate
        next_hidden = torch.cat((scalar, next_vectors.flatten(-2)), -1)
        action_bases = torch.cat((local, next_vectors), -2)
        mean = (self.actor(scalar)[..., None] * action_bases).sum(-2)
        return mean, self.log_std.clamp(-5, 1), self.critic(scalar).squeeze(-1), next_hidden

    def act(self, obs, hidden, graph, deterministic=False, generator=None):
        mean, log_std, value, memory = self(obs, hidden, graph)
        noise = 0 if deterministic else torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        latent = mean + log_std.exp() * noise
        return latent.tanh(), latent, squashed_log_prob(mean, log_std, latent), value, memory

    def evaluate_actions(self, obs, hidden, graph, latent):
        mean, log_std, value, memory = self(obs, hidden, graph)
        entropy_latent = mean + log_std.exp() * torch.randn_like(mean)
        return (squashed_log_prob(mean, log_std, latent),
                -squashed_log_prob(mean, log_std, entropy_latent), value, memory)
