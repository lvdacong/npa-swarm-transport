"""Small, GPU-capable 2D cooperative transport environment.

Overdamped disks exert bounded actuator forces. The cargo receives force only
through actual penetration contacts and has a static-friction threshold.
Connectivity is measured and rewarded; it never changes actions or dynamics.
"""
from dataclasses import asdict
import math

import torch

from .config import EnvConfig
from .graph import aggregate, components, radius_graph

OBS_DIM = 20


class SwarmEnv:
    obs_dim = OBS_DIM
    _state_names = (
        "positions", "velocities", "object_pos", "object_vel", "goal_pos", "steps",
        "found_object", "found_goal", "visited", "episode_return", "ever_disconnected",
        "disconnected_seconds", "min_largest_fraction", "bridge_eligible_steps", "bridge_steps",
    )

    def __init__(self, config: EnvConfig, device="cpu"):
        self.cfg, self.device = config, torch.device(device)
        c = config
        self._validate()
        self.generator = torch.Generator(device="cpu").manual_seed(c.seed)
        b, n = c.num_envs, c.n_agents
        self.positions = torch.zeros(b, n, 2, device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.object_pos = torch.zeros(b, 2, device=self.device)
        self.object_vel = torch.zeros_like(self.object_pos)
        self.goal_pos = torch.zeros_like(self.object_pos)
        self.steps = torch.zeros(b, dtype=torch.long, device=self.device)
        self.found_object = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.found_goal = torch.zeros_like(self.found_object)
        self._coverage_side = math.ceil(2 * c.arena_half / c.sense_radius)
        self.visited = torch.zeros(b, self._coverage_side ** 2, dtype=torch.bool, device=self.device)
        self.episode_return = torch.zeros(b, device=self.device)
        self.ever_disconnected = torch.zeros_like(self.found_object)
        self.disconnected_seconds = torch.zeros(b, device=self.device)
        self.min_largest_fraction = torch.ones(b, device=self.device)
        self.bridge_eligible_steps = torch.zeros_like(self.steps)
        self.bridge_steps = torch.zeros_like(self.steps)
        self.reset()

    def _validate(self):
        c = self.cfg
        if c.n_agents < 1 or c.num_envs < 1 or c.substeps < 1 or c.episode_steps < 1:
            raise ValueError("Population, batch, substeps and episode_steps must be positive")
        if c.stage not in ("contact", "search"):
            raise ValueError("stage must be contact or search")
        if min(c.dt, c.agent_radius, c.comm_radius, c.sense_radius, c.max_force,
               c.agent_drag, c.max_speed, c.object_drag, c.contact_stiffness) <= 0:
            raise ValueError("Physical scales must be positive")
        if c.goal_radius <= c.object_radius:
            raise ValueError("Destination must fit the entire cargo")
        if c.comm_radius < 2.4 * c.agent_radius:
            raise ValueError("comm_radius is too short for the connected initial lattice")
        if c.max_speed * c.dt / c.substeps > c.agent_radius:
            raise ValueError("Physics timestep too large: reduce dt or increase substeps")

    @torch.no_grad()
    def reset(self, mask=None):
        c = self.cfg
        if mask is None:
            mask = torch.ones(c.num_envs, dtype=torch.bool, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        ids = mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        # CPU RNG is used only for resets; its state is portable across devices.
        spacing = c.agent_radius * 2.4
        side = math.ceil(math.sqrt(c.n_agents))
        axis = torch.arange(side, dtype=torch.float32) - (side - 1) / 2
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        lattice = torch.stack((xx.flatten(), yy.flatten()), -1)[:c.n_agents] * spacing
        lattice -= lattice.mean(0)
        for i in ids:
            angle = float(torch.rand((), generator=self.generator)) * 2 * math.pi
            rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]])
            pos = lattice @ rotation.T
            if c.stage == "search":
                # Both targets are outside all initial local sensing disks.
                start_radius = float(pos.norm(dim=-1).max())
                target_radius = max(c.arena_half * .74,
                                    start_radius + c.sense_radius + c.goal_radius + .25)
                if target_radius + c.goal_radius >= c.arena_half:
                    raise ValueError("Arena too small to hide both endpoints; increase arena_half")
                # Target bearing is independent of the initial lattice frame.
                # Destination bearing is uniform subject only to separation.
                target_angle = float(torch.rand((), generator=self.generator)) * 2 * math.pi
                direction = torch.tensor([math.cos(target_angle), math.sin(target_angle)])
                cargo = direction * target_radius
                minimum_separation = c.object_radius + c.goal_radius + 2 * c.sense_radius
                minimum_angle = 2 * math.asin(minimum_separation / (2 * target_radius))
                goal_angle = target_angle + minimum_angle + (2 * math.pi - 2 * minimum_angle) * float(torch.rand((), generator=self.generator))
                goal = torch.tensor([math.cos(goal_angle), math.sin(goal_angle)]) * target_radius
            else:
                # A contact ring and an attached patch. Both endpoints have
                # actual local witnesses; no endpoint cue is globally injected.
                # A small debug population uses a connected partial ring.
                ring_radius = c.object_radius + c.agent_radius - .002
                full_ring_count = math.ceil(2 * math.pi * ring_radius / spacing)
                ring_count = min(c.n_agents, full_ring_count)
                if ring_count == full_ring_count:
                    angles = torch.arange(ring_count) * (2 * math.pi / ring_count)
                    goal_sign = 1
                else:
                    angles = (torch.arange(ring_count) - (ring_count - 1) / 2) * spacing / ring_radius + math.pi
                    goal_sign = -1
                ring = torch.stack((angles.cos(), angles.sin()), -1) * ring_radius
                remaining = c.n_agents - ring_count
                if remaining:
                    patch_side = math.ceil(math.sqrt(remaining))
                    patch_axis = torch.arange(patch_side, dtype=torch.float32) - (patch_side - 1) / 2
                    py, px = torch.meshgrid(patch_axis, patch_axis, indexing="ij")
                    patch = torch.stack((px.flatten(), py.flatten()), -1)[:remaining] * spacing
                    patch[:, 0] -= patch[:, 0].max() + c.object_radius + 3 * c.agent_radius + .02
                    ring = torch.cat((ring, patch), 0)
                pos = ring @ rotation.T
                cargo = torch.zeros(2)
                goal = torch.tensor([math.cos(angle), math.sin(angle)]) * (c.object_radius + c.goal_radius + .4) * goal_sign
                if float(pos.abs().max()) + c.agent_radius >= c.arena_half:
                    raise ValueError("Contact warm-up lattice does not fit the arena")
            self.positions[i] = pos.to(self.device)
            self.velocities[i].zero_()
            self.object_pos[i] = cargo.to(self.device)
            self.object_vel[i].zero_()
            self.goal_pos[i] = goal.to(self.device)
            self.steps[i] = 0
            self.found_object[i] = False
            self.found_goal[i] = False
            self.visited[i].zero_()
            self.episode_return[i] = 0
            self.ever_disconnected[i] = False
            self.disconnected_seconds[i] = 0
            self.min_largest_fraction[i] = 1
            self.bridge_eligible_steps[i] = 0
            self.bridge_steps[i] = 0
        self._refresh()
        self.found_object |= self._object_visible.any(-1)
        self.found_goal |= self._goal_visible.any(-1)
        self._mark_coverage(mask)
        return self.observe()

    def _coverage_ids(self):
        c = self.cfg
        xy = ((self.positions + c.arena_half) / c.sense_radius).floor().long()
        xy = xy.clamp(0, self._coverage_side - 1)
        return xy[..., 0] * self._coverage_side + xy[..., 1]

    def _mark_coverage(self, mask=None):
        cells = self._coverage_ids()
        before = self.visited.gather(1, cells)
        new = ~before
        if mask is not None:
            new &= mask[:, None]
        # Duplicates receive a bounded per-particle observation reward; no
        # persistent occupancy reward and no global map is exposed to the NN.
        self.visited.scatter_(1, cells, True)
        return new.to(self.positions.dtype)

    @torch.no_grad()
    def _refresh(self):
        c = self.cfg
        self.graph = radius_graph(self.positions, c.comm_radius)
        self._labels = components(self.graph, c.num_envs, c.n_agents)
        total = c.num_envs * c.n_agents
        counts = torch.zeros(total, device=self.device, dtype=torch.long)
        counts.scatter_add_(0, self._labels.flatten(), torch.ones(total, device=self.device, dtype=torch.long))
        self._largest_fraction = counts.reshape(c.num_envs, c.n_agents).amax(-1).float() / c.n_agents
        self._connected = self._largest_fraction == 1
        self._object_delta = self.object_pos[:, None] - self.positions
        self._goal_delta = self.goal_pos[:, None] - self.positions
        self._object_distance_agent = self._object_delta.norm(dim=-1)
        self._object_visible = self._object_distance_agent <= c.object_radius + c.sense_radius
        self._goal_visible = self._goal_delta.norm(dim=-1) <= c.goal_radius + c.sense_radius
        self._contact = self._object_distance_agent < c.object_radius + c.agent_radius
        near_object = torch.zeros(total, device=self.device, dtype=torch.long)
        near_goal = torch.zeros_like(near_object)
        near_object.scatter_add_(0, self._labels.flatten(), self._object_visible.flatten().long())
        near_goal.scatter_add_(0, self._labels.flatten(), self._goal_visible.flatten().long())
        self._bridge = ((near_object > 0) & (near_goal > 0)).reshape(c.num_envs, c.n_agents).any(-1)
        self._distance = (self.object_pos - self.goal_pos).norm(dim=-1)

    @torch.no_grad()
    def observe(self):
        c = self.cfg
        mean_position = aggregate(self.positions, self.graph)
        mean_velocity = aggregate(self.velocities, self.graph)
        degree = torch.zeros(c.num_envs * c.n_agents, device=self.device)
        degree.index_add_(0, self.graph.dst, torch.ones_like(self.graph.dst, dtype=self.positions.dtype))
        degree = degree.reshape(c.num_envs, c.n_agents, 1)
        relative = (mean_position - self.positions) / c.comm_radius
        relative = torch.where(degree > 0, relative, torch.zeros_like(relative))
        object_obs = self._object_delta / (c.sense_radius + c.object_radius)
        object_obs = object_obs * self._object_visible[..., None]
        goal_obs = self._goal_delta / (c.sense_radius + c.goal_radius)
        goal_obs = goal_obs * self._goal_visible[..., None]
        contact_normal = -self._object_delta / self._object_distance_agent.clamp_min(1e-8)[..., None]
        contact_normal = contact_normal * self._contact[..., None]
        p = self.positions
        wall = torch.stack((c.arena_half - c.agent_radius - p[..., 0],
                            c.arena_half - c.agent_radius + p[..., 0],
                            c.arena_half - c.agent_radius - p[..., 1],
                            c.arena_half - c.agent_radius + p[..., 1]), -1)
        wall = (wall / c.sense_radius).clamp(0, 1)
        return torch.cat((self.velocities / c.max_speed, relative,
                          mean_velocity / c.max_speed, (degree / 32).clamp_max(1),
                          object_obs, self._object_visible[..., None].float(),
                          goal_obs, self._goal_visible[..., None].float(),
                          contact_normal, self._contact[..., None].float(), wall), -1)

    @torch.no_grad()
    def step(self, actions):
        c = self.cfg
        if tuple(actions.shape) != (c.num_envs, c.n_agents, 2):
            raise ValueError("Actions must be [num_envs,n_agents,2]")
        actions = actions.to(self.device).clamp(-1, 1)
        # A circular actuator limit: one agent cannot apply sqrt(2) max_force.
        actuator = actions / actions.norm(dim=-1, keepdim=True).clamp_min(1) * c.max_force
        old_distance = self._distance.clone()
        old_local_distance = self._object_distance_agent.clone()
        old_visible = self._object_visible.clone()
        old_found_obj, old_found_goal = self.found_object.clone(), self.found_goal.clone()
        broad_radius = 2 * c.agent_radius + 2 * c.max_speed * c.dt + 1e-6
        broad = radius_graph(self.positions, broad_radius)
        flat = self.positions.reshape(-1, 2)
        sub_dt = c.dt / c.substeps
        contact_credit = torch.zeros_like(self._object_distance_agent)
        goal_direction = (self.goal_pos - self.object_pos) / self._distance.clamp_min(1e-8)[:, None]
        for _ in range(c.substeps):
            delta = flat[broad.dst] - flat[broad.src]
            distance = delta.norm(dim=-1)
            normal = delta / distance.clamp_min(1e-8)[:, None]
            overlap = (2 * c.agent_radius - distance).clamp_min(0)
            repulsion = normal * (c.contact_stiffness * overlap)[:, None]
            pair_force = torch.zeros_like(flat).index_add_(0, broad.dst, repulsion).reshape_as(self.positions)
            object_delta = self.positions - self.object_pos[:, None]
            object_distance = object_delta.norm(dim=-1)
            object_normal = object_delta / object_distance.clamp_min(1e-8)[..., None]
            object_overlap = (c.object_radius + c.agent_radius - object_distance).clamp_min(0)
            contact_force = object_normal * (c.contact_stiffness * object_overlap)[..., None]
            if c.force_credit_weight:
                # Credit the actual force that THIS particle transmits to the
                # cargo, including its sign. This is reward-only supervision:
                # no target vector is inserted into observations or actions.
                contact_credit -= (contact_force * goal_direction[:, None]).sum(-1) * sub_dt
            force = actuator + pair_force + contact_force
            velocity = force / c.agent_drag
            speed = velocity.norm(dim=-1, keepdim=True)
            velocity = velocity * (c.max_speed / speed.clamp_min(c.max_speed))
            cargo_force = -contact_force.sum(1)
            magnitude = cargo_force.norm(dim=-1, keepdim=True)
            effective = (magnitude - c.object_static_friction).clamp_min(0)
            self.object_vel = cargo_force / magnitude.clamp_min(1e-8) * effective / c.object_drag
            self.positions.add_(velocity * sub_dt)
            self.object_pos.add_(self.object_vel * sub_dt)
            # Hard walls are ordinary physical constraints, not a connectivity shield.
            before_wall = self.positions.clone()
            self.positions.clamp_(-c.arena_half + c.agent_radius, c.arena_half - c.agent_radius)
            self.velocities = velocity.masked_fill(before_wall != self.positions, 0)
            before_cargo_wall = self.object_pos.clone()
            self.object_pos.clamp_(-c.arena_half + c.object_radius, c.arena_half - c.object_radius)
            self.object_vel.masked_fill_(before_cargo_wall != self.object_pos, 0)
        self.steps += 1
        self._refresh()
        self.found_object |= self._object_visible.any(-1)
        self.found_goal |= self._goal_visible.any(-1)
        eligible = self.found_object & self.found_goal
        self.ever_disconnected |= ~self._connected
        self.disconnected_seconds += (~self._connected).float() * c.dt
        self.min_largest_fraction.copy_(torch.minimum(self.min_largest_fraction, self._largest_fraction))
        self.bridge_eligible_steps += eligible.long()
        self.bridge_steps += (eligible & self._bridge).long()
        success = self._distance <= c.goal_radius - c.object_radius
        discovery = ((self.found_object & ~old_found_obj).float()
                     + (self.found_goal & ~old_found_goal).float())
        progress = (old_distance - self._distance) * 3
        # After both endpoints were discovered, also encourage an actual live
        # local-message path between their current sensor witnesses.
        bridge_gap = (eligible & ~self._bridge).float()
        connectivity_cost = c.connectivity_weight * (1 - self._largest_fraction + bridge_gap) * c.dt
        exploration = self._mark_coverage() * c.exploration_weight
        local_progress = ((old_local_distance - self._object_distance_agent)
                          * (old_visible & self._object_visible)) * .05
        effort = .0005 * actions.square().sum(-1) * c.dt
        team_reward = discovery + success.float() * 5 - connectivity_cost
        transport_reward = progress[:, None]
        if c.local_transport_reward:
            # The cargo's movement is chiefly caused by local witnesses. Do
            # not train thousands of uninformed distant actions on the same
            # high-variance movement return. Success/discovery remain shared.
            transport_reward = transport_reward * (old_visible | self._object_visible)
        contact_credit *= c.force_credit_weight * self.found_goal[:, None]
        reward = team_reward[:, None] + transport_reward + exploration + local_progress - effort + contact_credit
        self.episode_return += reward.mean(-1)
        info = self.metrics()
        info.update(reward_progress=progress, reward_discovery=discovery,
                    reward_connectivity=-connectivity_cost, reward_exploration=exploration.mean(-1),
                    reward_local_progress=local_progress.mean(-1), reward_effort=-effort.mean(-1),
                    reward_contact_credit=contact_credit.mean(-1))
        return self.observe(), reward, success, (self.steps >= c.episode_steps) & ~success, info

    @torch.no_grad()
    def metrics(self):
        eligible = self.found_object & self.found_goal
        # Instantaneous physical contact resultant at the sampled state. This
        # includes opposing contacts; contact count alone cannot prove pushing.
        normal = -self._object_delta / self._object_distance_agent.clamp_min(1e-8)[..., None]
        overlap = (self.cfg.object_radius + self.cfg.agent_radius - self._object_distance_agent).clamp_min(0)
        cargo_force = -(normal * (self.cfg.contact_stiffness * overlap)[..., None]).sum(1)
        goal_direction = (self.goal_pos - self.object_pos) / self._distance.clamp_min(1e-8)[:, None]
        return {
            "success": self._distance <= self.cfg.goal_radius - self.cfg.object_radius,
            "connected": self._connected.clone(), "largest_fraction": self._largest_fraction.clone(),
            "bridge": (eligible & self._bridge), "bridge_eligible": eligible.clone(),
            "found_object": self.found_object.clone(), "found_goal": self.found_goal.clone(),
            "object_distance": self._distance.clone(), "contact_count": self._contact.sum(-1),
            "object_speed": self.object_vel.norm(dim=-1), "steps": self.steps.clone(),
            "cargo_force_norm": cargo_force.norm(dim=-1),
            "cargo_force_toward_goal": (cargo_force * goal_direction).sum(-1),
            "episode_return": self.episode_return.clone(), "ever_disconnected": self.ever_disconnected.clone(),
            "disconnected_seconds": self.disconnected_seconds.clone(),
            "min_largest_fraction": self.min_largest_fraction.clone(),
            "bridge_eligible_steps": self.bridge_eligible_steps.clone(), "bridge_steps": self.bridge_steps.clone(),
        }

    def state_dict(self):
        state = {name: getattr(self, name).detach().cpu().clone() for name in self._state_names}
        state["rng"] = self.generator.get_state().clone()
        state["config"] = asdict(self.cfg)
        return state

    @torch.no_grad()
    def load_state_dict(self, state):
        if asdict(EnvConfig(**state["config"])) != asdict(self.cfg):
            raise ValueError("Environment configuration differs from checkpoint")
        for name in self._state_names:
            target, saved = getattr(self, name), state[name]
            if target.shape != saved.shape or target.dtype != saved.dtype:
                raise ValueError(f"Invalid environment checkpoint field: {name}")
        self.generator.set_state(state["rng"].cpu())
        for name in self._state_names:
            getattr(self, name).copy_(state[name].to(self.device))
        self._refresh()
