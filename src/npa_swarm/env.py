"""Local RL transport, task v2. Contacts move cargo; connectivity never edits actions."""
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
        "detached_age", "last_displacement",
    )

    def __init__(self, config: EnvConfig, device="cpu"):
        self.cfg, self.device = config, torch.device(device)
        c = self.cfg
        if c.comm_radius < 2.4 * c.agent_radius:
            raise ValueError("Communication radius cannot connect the initial lattice")
        if c.max_speed * c.dt / c.substeps > c.agent_radius:
            raise ValueError("Physics timestep too large: reduce dt or increase substeps")
        self.generator = torch.Generator(device="cpu").manual_seed(c.seed)
        b, n = c.num_envs, c.n_agents
        self.positions = torch.zeros(b, n, 2, device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.last_displacement = torch.zeros_like(self.positions)
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
        self.detached_age = torch.zeros(b, n, device=self.device)
        self.use_cuda_graph_physics = True
        self._physics_cache = {}
        self.reset()

    @torch.no_grad()
    def reset(self, mask=None):
        c = self.cfg
        mask = (torch.ones(c.num_envs, dtype=torch.bool, device=self.device) if mask is None
                else torch.as_tensor(mask, dtype=torch.bool, device=self.device))
        ids = mask.nonzero(as_tuple=False).flatten().cpu().tolist()
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
                start_radius = float(pos.norm(dim=-1).max())
                radius = max(c.arena_half * .74, start_radius + c.sense_radius + c.goal_radius + .25)
                if radius + c.goal_radius >= c.arena_half:
                    raise ValueError("Arena too small to hide both endpoints")
                bearing = float(torch.rand((), generator=self.generator)) * 2 * math.pi
                if c.layout == "diagonal":
                    # Opposite corners, randomized quadrant. Evaluation stress
                    # layout, not a light-to-heavy or distance curriculum.
                    signs = torch.where(torch.rand(2, generator=self.generator) > .5, 1., -1.)
                    cargo = signs * (c.arena_half * .74)
                    goal = -cargo
                else:
                    cargo = torch.tensor([math.cos(bearing), math.sin(bearing)]) * radius
                    # At least a quarter turn: no near-overlapping endpoints.
                    goal_angle = bearing + math.pi / 2 + math.pi * float(torch.rand((), generator=self.generator))
                    goal = torch.tensor([math.cos(goal_angle), math.sin(goal_angle)]) * radius
            else:
                # Diagnostic only. Formal training starts in unknown search.
                ring_radius = c.object_radius + c.agent_radius - .002
                full_count = math.ceil(2 * math.pi * ring_radius / spacing)
                ring_count = min(c.n_agents, full_count)
                angles = (torch.arange(ring_count) * (2 * math.pi / ring_count)
                          if ring_count == full_count else
                          (torch.arange(ring_count) - (ring_count - 1) / 2) * spacing / ring_radius + math.pi)
                ring = torch.stack((angles.cos(), angles.sin()), -1) * ring_radius
                remaining = c.n_agents - ring_count
                if remaining:
                    patch_side = math.ceil(math.sqrt(remaining))
                    a = torch.arange(patch_side, dtype=torch.float32) - (patch_side - 1) / 2
                    py, px = torch.meshgrid(a, a, indexing="ij")
                    patch = torch.stack((px.flatten(), py.flatten()), -1)[:remaining] * spacing
                    patch[:, 0] -= patch[:, 0].max() + c.object_radius + 3 * c.agent_radius + .02
                    ring = torch.cat((ring, patch), 0)
                pos = ring @ rotation.T
                cargo = torch.zeros(2)
                sign = 1 if ring_count == full_count else -1
                goal = torch.tensor([math.cos(angle), math.sin(angle)]) * (c.object_radius + c.goal_radius + .4) * sign
            if float(pos.abs().max()) + c.agent_radius >= c.arena_half:
                raise ValueError("Initial swarm does not fit the arena")
            self.positions[i] = pos.to(self.device)
            self.object_pos[i] = cargo.to(self.device)
            self.goal_pos[i] = goal.to(self.device)
            for name in ("velocities", "last_displacement", "object_vel", "steps", "found_object", "found_goal", "visited",
                         "episode_return", "ever_disconnected", "disconnected_seconds",
                         "bridge_eligible_steps", "bridge_steps", "detached_age"):
                getattr(self, name)[i].zero_()
            self.min_largest_fraction[i] = 1
        self._refresh()
        self.found_object |= self._object_visible.any(-1)
        self.found_goal |= self._goal_visible.any(-1)
        self._mark_coverage(mask)
        return self.observe()

    def _coverage_ids(self):
        xy = ((self.positions + self.cfg.arena_half) / self.cfg.sense_radius).floor().long()
        xy = xy.clamp(0, self._coverage_side - 1)
        return xy[..., 0] * self._coverage_side + xy[..., 1]

    def _mark_coverage(self, mask=None):
        cells = self._coverage_ids()
        eligible = self._in_main.clone()
        if mask is not None:
            eligible &= mask[:, None]
        new = ~self.visited.gather(1, cells) & eligible
        count = torch.zeros_like(self.visited, dtype=torch.long)
        count.scatter_add_(1, cells, eligible.long())
        # A newly visited cell pays ONCE, divided among simultaneous visitors.
        # Detached particles cannot farm the exploration objective. The map
        # belongs only to reward bookkeeping and is never sent to the policy.
        credit = new.float() / count.gather(1, cells).clamp_min(1)
        self.visited |= count > 0
        return credit

    def _repair_distances(self):
        """Distance to the main component's communication envelope, reward only.

        Chunked N-by-N distance queries, only for disconnected worlds. This is
        not an MST or a claimed shortest multi-component repair route.
        """
        c = self.cfg
        gap = torch.zeros(c.num_envs, c.n_agents, device=self.device)
        detached = (~self._in_main).nonzero(as_tuple=False)
        # Only detached rows need distances. A few outliers previously made
        # every particle query every other particle in eight chunks.
        for begin in range(0, detached.shape[0], 128):
            world, node = detached[begin:begin+128].unbind(-1)
            query = self.positions[world, node]
            distance = (query[:, None] - self.positions[world]).norm(dim=-1)
            distance.masked_fill_(~self._in_main[world], float("inf"))
            gap[world, node] = (distance.amin(-1) - c.comm_radius).clamp_min(0)
        return gap

    @torch.no_grad()
    def _refresh(self):
        c = self.cfg
        self.graph = radius_graph(self.positions, c.comm_radius)
        self.graph.displacement = self.last_displacement
        # Only constant LOCAL sensor calibrations, not target coordinates.
        self.graph.cue_scales = (c.sense_radius + c.object_radius,
                                 c.sense_radius + c.goal_radius)
        self._labels = components(self.graph, c.num_envs, c.n_agents)
        total = c.num_envs * c.n_agents
        counts = torch.zeros(total, device=self.device, dtype=torch.long)
        counts.scatter_add_(0, self._labels.flatten(), torch.ones(total, device=self.device, dtype=torch.long))
        count = counts.reshape(c.num_envs, c.n_agents)
        main_root = count.argmax(-1) + torch.arange(c.num_envs, device=self.device) * c.n_agents
        self._in_main = self._labels == main_root[:, None]
        self._largest_fraction = count.amax(-1).float() / c.n_agents
        self._connected = self._largest_fraction == 1
        self._object_delta = self.object_pos[:, None] - self.positions
        self._goal_delta = self.goal_pos[:, None] - self.positions
        self._object_distance_agent = self._object_delta.norm(dim=-1)
        goal_distance = self._goal_delta.norm(dim=-1)
        self._object_visible = self._object_distance_agent <= c.object_radius + c.sense_radius
        self._goal_visible = goal_distance <= c.goal_radius + c.sense_radius
        self._contact = self._object_distance_agent < c.object_radius + c.agent_radius
        # Endpoints attach using COMMUNICATION reach to their surfaces, not
        # the larger sensing disks. Endpoints do not merge separate components.
        near_object = torch.zeros(total, device=self.device, dtype=torch.long)
        near_goal = torch.zeros_like(near_object)
        obj_port = self._object_distance_agent <= c.object_radius + c.comm_radius
        goal_port = goal_distance <= c.goal_radius + c.comm_radius
        near_object.scatter_add_(0, self._labels.flatten(), obj_port.flatten().long())
        near_goal.scatter_add_(0, self._labels.flatten(), goal_port.flatten().long())
        bridge_roots = (near_object > 0) & (near_goal > 0)
        self._bridge_members = bridge_roots[self._labels]
        self._bridge = self._bridge_members.any(-1)
        self._distance = (self.object_pos - self.goal_pos).norm(dim=-1)
        self._repair_gap = self._repair_distances()
        # Geometric progress of the connected MAIN component toward both
        # known endpoints. This is a potential, not a shortest-path claim.
        obj_gap = (self._object_distance_agent.masked_fill(~self._in_main, float("inf")).amin(-1)
                   - c.object_radius - c.comm_radius).clamp_min(0)
        goal_gap = (goal_distance.masked_fill(~self._in_main, float("inf")).amin(-1)
                    - c.goal_radius - c.comm_radius).clamp_min(0)
        self._endpoint_gap = obj_gap + goal_gap
        self._observation = None

    @torch.no_grad()
    def observe(self):
        if self._observation is not None:
            return self._observation
        c = self.cfg
        means = aggregate(torch.cat((self.positions, self.velocities), -1), self.graph)
        degree = torch.zeros(c.num_envs * c.n_agents, device=self.device)
        degree.index_add_(0, self.graph.dst, torch.ones_like(self.graph.dst, dtype=self.positions.dtype))
        degree = degree.reshape(c.num_envs, c.n_agents, 1)
        self._degree = degree.squeeze(-1)
        relative = (means[..., :2] - self.positions) / c.comm_radius
        relative = torch.where(degree > 0, relative, torch.zeros_like(relative))
        object_obs = self._object_delta / (c.sense_radius + c.object_radius) * self._object_visible[..., None]
        goal_obs = self._goal_delta / (c.sense_radius + c.goal_radius) * self._goal_visible[..., None]
        contact_normal = -self._object_delta / self._object_distance_agent.clamp_min(1e-8)[..., None]
        contact_normal = contact_normal * self._contact[..., None]
        p = self.positions
        wall = torch.stack((c.arena_half - c.agent_radius - p[..., 0],
                            c.arena_half - c.agent_radius + p[..., 0],
                            c.arena_half - c.agent_radius - p[..., 1],
                            c.arena_half - c.agent_radius + p[..., 1]), -1)
        wall = (wall / c.sense_radius).clamp(0, 1)
        self._observation = torch.cat((self.velocities / c.max_speed, relative,
                                       means[..., 2:] / c.max_speed, (degree / 32).clamp_max(1),
                                       object_obs, self._object_visible[..., None].float(),
                                       goal_obs, self._goal_visible[..., None].float(),
                                       contact_normal, self._contact[..., None].float(), wall), -1)
        return self._observation

    @torch.no_grad()
    def step(self, actions):
        c = self.cfg
        if tuple(actions.shape) != (c.num_envs, c.n_agents, 2):
            raise ValueError("Actions must be [num_envs,n_agents,2]")
        actions = actions.to(self.device).clamp(-1, 1)
        actuator = actions / actions.norm(dim=-1, keepdim=True).clamp_min(1) * c.max_force
        old_distance = self._distance.clone()
        old_positions = self.positions.clone()
        old_local_distance, old_visible = self._object_distance_agent, self._object_visible
        old_found_obj, old_found_goal = self.found_object.clone(), self.found_goal.clone()
        old_repair = torch.log1p(self._repair_gap / c.comm_radius)
        old_endpoint_gap = self._endpoint_gap.clone()
        broad = radius_graph(self.positions, 2 * c.agent_radius + 2 * c.max_speed * c.dt + 1e-6)
        from .physics import CapturedPhysics, integrate
        if self.device.type == "cuda" and self.use_cuda_graph_physics:
            capacity = 1 << max(0, (broad.src.numel() - 1).bit_length())
            key = (capacity, tuple(asdict(c).values()))
            if key not in self._physics_cache:
                # At most a few edge-size buckets; never truncate dense graphs.
                if len(self._physics_cache) >= 4:
                    self._physics_cache.clear()
                self._physics_cache[key] = CapturedPhysics(
                    self.positions, self.object_pos, self.goal_pos, actuator, capacity, c)
            outputs = self._physics_cache[key](self.positions, self.object_pos, self.goal_pos,
                                               actuator, broad.src, broad.dst)
        else:
            outputs = integrate(self.positions, self.object_pos, self.goal_pos,
                                actuator, broad.src, broad.dst, c)
        for target, value in zip((self.positions, self.velocities, self.object_pos, self.object_vel), outputs[:4]):
            target.copy_(value)
        work_credit = outputs[4].clone()
        self.last_displacement = self.positions - old_positions
        self.steps += 1
        self._refresh()
        self.found_object |= self._object_visible.any(-1)
        self.found_goal |= self._goal_visible.any(-1)
        eligible = self.found_object & self.found_goal
        self.ever_disconnected |= ~self._connected
        self.disconnected_seconds += (~self._connected).float() * c.dt
        self.detached_age = torch.where(self._in_main, 0., self.detached_age + c.dt)
        self.min_largest_fraction.copy_(torch.minimum(self.min_largest_fraction, self._largest_fraction))
        self.bridge_eligible_steps += eligible.long()
        self.bridge_steps += (eligible & self._bridge).long()
        success = self._distance <= c.goal_radius - c.object_radius
        discovery = ((self.found_object & ~old_found_obj).float() + (self.found_goal & ~old_found_goal).float())
        progress = (old_distance - self._distance) * 3
        detachment_cost = c.connectivity_weight * (~self._in_main) * (1 + (self.detached_age / 5).clamp_max(1)) * c.dt
        bridge_cost = c.bridge_weight * (eligible & ~self._bridge).float() * c.dt
        repair_reward = c.repair_weight * (old_repair - torch.log1p(self._repair_gap / c.comm_radius))
        endpoint_reward = (c.bridge_progress_weight * (old_endpoint_gap - self._endpoint_gap)
                           * (old_found_obj & old_found_goal))
        exploration = self._mark_coverage() * c.exploration_weight
        local_progress = ((old_local_distance - self._object_distance_agent)
                          * (old_visible & self._object_visible)) * .05
        effort = .0005 * actions.square().sum(-1) * c.dt
        work_credit *= c.work_credit_weight * old_found_goal[:, None]
        team_reward = discovery + success.float() * 5 + progress + endpoint_reward - bridge_cost
        reward = team_reward[:, None] + exploration + local_progress + repair_reward + work_credit - detachment_cost - effort
        self.episode_return += reward.mean(-1)
        info = self.metrics()
        info.update(reward_progress=progress, reward_discovery=discovery,
                    reward_connectivity=-detachment_cost.mean(-1) - bridge_cost,
                    reward_repair=repair_reward.mean(-1), reward_endpoint_progress=endpoint_reward,
                    reward_exploration=exploration.mean(-1), reward_local_progress=local_progress.mean(-1),
                    reward_effort=-effort.mean(-1), reward_contact_credit=work_credit.mean(-1))
        return self.observe(), reward, success, (self.steps >= c.episode_steps) & ~success, info

    @torch.no_grad()
    def metrics(self):
        eligible = self.found_object & self.found_goal
        normal = -self._object_delta / self._object_distance_agent.clamp_min(1e-8)[..., None]
        overlap = (self.cfg.object_radius + self.cfg.agent_radius - self._object_distance_agent).clamp_min(0)
        cargo_force = -(normal * (self.cfg.contact_stiffness * overlap)[..., None]).sum(1)
        goal_direction = (self.goal_pos - self.object_pos) / self._distance.clamp_min(1e-8)[:, None]
        return {
            "success": self._distance <= self.cfg.goal_radius - self.cfg.object_radius,
            "connected": self._connected.clone(), "largest_fraction": self._largest_fraction.clone(),
            "bridge": eligible & self._bridge, "bridge_eligible": eligible.clone(),
            "found_object": self.found_object.clone(), "found_goal": self.found_goal.clone(),
            "object_distance": self._distance.clone(), "contact_count": self._contact.sum(-1),
            "object_speed": self.object_vel.norm(dim=-1), "steps": self.steps.clone(),
            "cargo_force_norm": cargo_force.norm(dim=-1),
            "cargo_force_toward_goal": (cargo_force * goal_direction).sum(-1),
            "repair_gap_max": self._repair_gap.amax(-1), "endpoint_gap": self._endpoint_gap.clone(),
            "episode_return": self.episode_return.clone(), "ever_disconnected": self.ever_disconnected.clone(),
            "disconnected_seconds": self.disconnected_seconds.clone(),
            "min_largest_fraction": self.min_largest_fraction.clone(),
            "bridge_eligible_steps": self.bridge_eligible_steps.clone(), "bridge_steps": self.bridge_steps.clone(),
        }

    def state_dict(self):
        state = {name: getattr(self, name).detach().cpu().clone() for name in self._state_names}
        state["rng"], state["config"] = self.generator.get_state().clone(), asdict(self.cfg)
        return state

    @torch.no_grad()
    def load_state_dict(self, state):
        if state["config"] != asdict(self.cfg):
            raise ValueError("Environment configuration differs from checkpoint")
        for name in self._state_names:
            target, saved = getattr(self, name), state[name]
            if target.shape != saved.shape or target.dtype != saved.dtype:
                raise ValueError(f"Invalid environment checkpoint field: {name}")
        self.generator.set_state(state["rng"].cpu())
        for name in self._state_names:
            getattr(self, name).copy_(state[name].to(self.device))
        self._refresh()
