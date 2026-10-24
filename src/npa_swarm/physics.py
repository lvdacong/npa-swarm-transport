"""Exact existing overdamped substeps, optionally replayed as one CUDA graph.

Graphs only remove host launch overhead. Broad-phase edges remain complete;
padding uses zero-force self edges, never a neighbor cap or a new integrator.
"""
import torch


def integrate(positions, object_pos, goal_pos, actuator, src, dst, c):
    positions, object_pos = positions.clone(), object_pos.clone()
    flat = positions.reshape(-1, 2)
    sub_dt = c.dt / c.substeps
    work_credit = torch.zeros_like(positions[..., 0])
    for _ in range(c.substeps):
        delta = flat[dst] - flat[src]
        distance = delta.norm(dim=-1)
        normal = delta / distance.clamp_min(1e-8)[:, None]
        overlap = (2 * c.agent_radius - distance).clamp_min(0)
        repulsion = normal * (c.contact_stiffness * overlap)[:, None]
        pair_force = torch.zeros_like(flat).index_add_(0, dst, repulsion).reshape_as(positions)
        object_delta = positions - object_pos[:, None]
        object_distance = object_delta.norm(dim=-1)
        object_normal = object_delta / object_distance.clamp_min(1e-8)[..., None]
        object_overlap = (c.object_radius + c.agent_radius - object_distance).clamp_min(0)
        contact_force = object_normal * (c.contact_stiffness * object_overlap)[..., None]
        velocity = (actuator + pair_force + contact_force) / c.agent_drag
        speed = velocity.norm(dim=-1, keepdim=True)
        velocity = velocity * (c.max_speed / speed.clamp_min(c.max_speed))
        cargo_force = -contact_force.sum(1)
        magnitude = cargo_force.norm(dim=-1, keepdim=True)
        effective = (magnitude - c.object_static_friction).clamp_min(0)
        object_vel = cargo_force / magnitude.clamp_min(1e-8) * effective / c.object_drag
        before_cargo = object_pos.clone()
        positions.add_(velocity * sub_dt)
        object_pos.add_(object_vel * sub_dt)
        before_wall = positions.clone()
        positions.clamp_(-c.arena_half + c.agent_radius, c.arena_half - c.agent_radius)
        velocities = velocity.masked_fill(before_wall != positions, 0)
        before_cargo_wall = object_pos.clone()
        object_pos.clamp_(-c.arena_half + c.object_radius, c.arena_half - c.object_radius)
        object_vel.masked_fill_(before_cargo_wall != object_pos, 0)
        if c.work_credit_weight:
            if c.contact_credit_mode == "marginal":
                # Reward-only, one-substep counterfactual: remove this contact
                # force, retaining every other force and the exact drag/walls.
                # Unlike normalized actual work this can distinguish opposing
                # contacts when their net force cancels. Never an actor input.
                without = cargo_force[:, None] + contact_force
                norm = without.norm(dim=-1, keepdim=True)
                velocity_without = (without / norm.clamp_min(1e-8)
                                    * (norm - c.object_static_friction).clamp_min(0) / c.object_drag)
                position_without = (before_cargo[:, None] + velocity_without * sub_dt).clamp(
                    -c.arena_half + c.object_radius, c.arena_half - c.object_radius)
                work_credit += ((goal_pos[:, None] - position_without).norm(dim=-1)
                                - (goal_pos - object_pos).norm(dim=-1)[:, None])
            else:
                displacement = object_pos - before_cargo
                work = (-contact_force * displacement[:, None]).sum(-1)
                progress = ((goal_pos - before_cargo).norm(dim=-1)
                            - (goal_pos - object_pos).norm(dim=-1))
                work_credit += work / work.abs().sum(-1, keepdim=True).clamp_min(1e-8) * progress[:, None]
    return positions, velocities, object_pos, object_vel, work_credit


class CapturedPhysics:
    def __init__(self, positions, object_pos, goal_pos, actuator, capacity, c):
        self.inputs = [t.clone() for t in (positions, object_pos, goal_pos, actuator)]
        self.src = torch.zeros(capacity, dtype=torch.long, device=positions.device)
        self.dst = torch.zeros_like(self.src)
        stream = torch.cuda.Stream(device=positions.device)
        stream.wait_stream(torch.cuda.current_stream(positions.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                integrate(*self.inputs, self.src, self.dst, c)
        torch.cuda.current_stream(positions.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = integrate(*self.inputs, self.src, self.dst, c)

    def __call__(self, positions, object_pos, goal_pos, actuator, src, dst):
        for target, value in zip(self.inputs, (positions, object_pos, goal_pos, actuator)):
            target.copy_(value)
        self.src.zero_(); self.dst.zero_()
        self.src[:src.numel()].copy_(src)
        self.dst[:dst.numel()].copy_(dst)
        self.graph.replay()
        return self.outputs
