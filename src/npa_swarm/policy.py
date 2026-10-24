"""Learned local recurrent controller with one-hop, geometry-corrected endpoint relays."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .graph import Graph, aggregate
from .relay import relay_beacons


def squashed_log_prob(mean, log_std, latent):
    normal_logp = -.5 * (((latent-mean)/log_std.exp()).square()+2*log_std+math.log(2*math.pi))
    jacobian = 2*(math.log(2)-latent-F.softplus(-2*latent))
    return (normal_logp-jacobian).sum(-1)


class LocalPolicy(nn.Module):
    """One shared learned policy; destination messages never skip a graph hop.

    Six local beacon channels carry object/goal bearings and confidence. They
    start ONLY at sensor witnesses, travel one hop per control step, and decay
    when disconnected. The network learns whether/how to act on directions:
    no scripted pushing or reconnection, global map, ID or absolute coordinate.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 32,
                 relay_memory_decay: float = .97, task_head: bool = False):
        super().__init__()
        if obs_dim != 20 or hidden_dim < 8:
            raise ValueError("Expected 20-channel observation and hidden_dim >= 8")
        self.obs_dim, self.hidden_dim = obs_dim, hidden_dim
        if not 0 < relay_memory_decay < 1:
            raise ValueError("relay_memory_decay must be within (0,1)")
        self.relay_memory_decay = relay_memory_decay
        self.task_head = bool(task_head)
        self.state_dim = hidden_dim + 6
        scalar_dim = hidden_dim - 4
        self.encode = nn.Sequential(nn.Linear(obs_dim + hidden_dim + 6, hidden_dim), nn.Tanh())
        self.memory = nn.GRUCell(hidden_dim, scalar_dim)
        self.vector_write = nn.Linear(scalar_dim, 20)
        self.vector_gate = nn.Linear(scalar_dim, 2)
        self.actor = nn.Linear(scalar_dim, 10)
        self.critic = nn.Linear(scalar_dim, 1)
        if self.task_head:
            # Learned coefficients of purely LOCAL relational vector bases.
            # Zero initialization leaves the inherited search policy intact.
            self.task_actor = nn.Linear(scalar_dim, 4)
            nn.init.zeros_(self.task_actor.weight)
            nn.init.zeros_(self.task_actor.bias)
        self.log_std = nn.Parameter(torch.full((2,), -.5))
        nn.init.orthogonal_(self.vector_write.weight, gain=.05)
        nn.init.zeros_(self.vector_write.bias)
        nn.init.zeros_(self.vector_gate.weight)
        nn.init.constant_(self.vector_gate.bias, 2.)
        nn.init.orthogonal_(self.actor.weight, gain=.05)
        nn.init.zeros_(self.actor.bias)
        # Trainable local exploration bias; not a fixed world-frame heading.
        with torch.no_grad():
            self.actor.bias[:3] = torch.tensor([1.5, .5, .5])

    def initial_state(self, batch, n_agents, device=None):
        return torch.zeros(batch, n_agents, self.state_dim,
                           device=self.log_std.device if device is None else device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor, graph: Graph):
        if hidden.shape[-1] != self.state_dim:
            raise ValueError(f"Policy state last dimension must be {self.state_dim}")
        scalar_dim = self.hidden_dim - 4
        neighbors = aggregate(hidden, graph)
        beacons = relay_beacons(hidden[..., -6:], obs, graph,
                                memory_decay=self.relay_memory_decay).reshape(*obs.shape[:-1], 2, 3)
        directions = F.normalize(beacons[..., 1:], dim=-1, eps=1e-8) * beacons[..., 0:1]
        beacon_input = torch.cat((beacons[..., 0:1], directions), dim=-1).flatten(-2)
        encoded = self.encode(torch.cat((obs, neighbors[..., :self.hidden_dim], beacon_input), -1))
        scalar = self.memory(encoded.flatten(0, 1), hidden[..., :scalar_dim].flatten(0, 1))
        scalar = scalar.reshape(*hidden.shape[:-1], scalar_dim)
        vectors = hidden[..., scalar_dim:scalar_dim+4].reshape(*hidden.shape[:-1], 2, 2)
        neighbor_vectors = neighbors[..., scalar_dim:scalar_dim+4].reshape_as(vectors)
        local = torch.stack((obs[..., 0:2], obs[..., 2:4], obs[..., 4:6],
                             obs[..., 7:9], obs[..., 10:12], obs[..., 13:15]), -2)
        bases = torch.cat((local, vectors, neighbor_vectors), -2)
        coefficients = self.vector_write(scalar).reshape(*scalar.shape[:-1], 2, 10).tanh()
        candidate = (coefficients[..., None] * bases[..., None, :, :]).sum(-2)
        candidate = candidate / candidate.norm(dim=-1, keepdim=True).clamp_min(1)
        gate = self.vector_gate(scalar).sigmoid()[..., None]
        next_vectors = gate * vectors + (1 - gate) * candidate
        next_hidden = torch.cat((scalar, next_vectors.flatten(-2), beacons.flatten(-2)), -1)
        action_bases = torch.cat((local, next_vectors, directions), -2)
        mean = (self.actor(scalar)[..., None] * action_bases).sum(-2)
        if self.task_head:
            # Object/goal positions are beacons already received by the local
            # node, NOT simulator-global oracle vectors. A nonlinear rear
            # bearing offers an action basis, never an imposed action.
            object_vec = beacons[..., 0, 1:]
            objective_vec = beacons[..., 1, 1:] - object_vec
            goal_direction = F.normalize(objective_vec, dim=-1, eps=1e-8)
            object_direction = F.normalize(object_vec, dim=-1, eps=1e-8)
            rear_bearing = F.normalize(
                object_vec - graph.contact_radius * goal_direction,
                dim=-1, eps=1e-8)
            weights = self.task_actor(scalar)
            object_conf = beacons[..., 0, 0:1].clamp(0, 1)
            goal_conf = beacons[..., 1, 0:1].clamp(0, 1)
            residual = (weights[..., 0:1] * object_direction +
                        goal_conf * (weights[..., 1:2] * goal_direction +
                                     weights[..., 2:3] * rear_bearing))
            # A fourth, independently learned coefficient can act on local
            # neighbor-centroid bearing at sparsely connected graph fringes.
            # It is zero initially, not a tether/reconnect script or shield.
            density = obs[..., 6:7]
            fringe = ((.55 - density) / .55).clamp(0, 1) * (density > 0)
            local_cohesion = weights[..., 3:4] * obs[..., 2:4] * fringe
            mean = mean + object_conf * residual + local_cohesion
        return mean, self.log_std.clamp(-5, 1), self.critic(scalar).squeeze(-1), next_hidden

    def act(self, obs, hidden, graph, deterministic=False, generator=None):
        mean, log_std, value, memory = self(obs, hidden, graph)
        noise = 0 if deterministic else torch.randn(mean.shape, dtype=mean.dtype,
                                                       device=mean.device, generator=generator)
        latent = mean + log_std.exp() * noise
        return latent.tanh(), latent, squashed_log_prob(mean, log_std, latent), value, memory

    def evaluate_actions(self, obs, hidden, graph, latent):
        mean, log_std, value, memory = self(obs, hidden, graph)
        entropy_latent = mean + log_std.exp() * torch.randn_like(mean)
        return (squashed_log_prob(mean, log_std, latent),
                -squashed_log_prob(mean, log_std, entropy_latent), value, memory)
