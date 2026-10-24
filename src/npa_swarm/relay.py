"""One-hop, agent-relative endpoint memory; no remote target observations."""
import torch


@torch.no_grad()
def relay_beacons(old, obs, graph, decay=0.97, memory_decay=None):
    """Return [B,N,6]: (object_conf, object_dx, object_dy, goal_conf, goal_dx, goal_dy).

    Only local sightings source a field; otherwise updates read last-step
    neighbors and physical edge-relative offsets. Disconnected memories decay.
    This is a message relay, not an action planner or connectivity protector.
    """
    b, n, dim = old.shape
    if dim != 6 or obs.shape[:2] != (b, n):
        raise ValueError("Expected [B,N,6] beacons and matching local observations")
    total = b * n
    prior = old.detach().reshape(total, 2, 3)
    if graph.displacement is not None:
        # Previous memory is relative to the previous body position. Apply
        # only each sender's own integrated motion before using current edges.
        # Moving remote cargo remains stale until a new sighting propagates.
        motion = graph.displacement.reshape(total, 1, 2)
        prior = torch.cat((prior[..., :1], prior[..., 1:] - motion), -1)
    src, dst = graph.src, graph.dst
    edge_count = src.numel()
    neighbor_conf = torch.zeros(total, 2, device=old.device, dtype=old.dtype)
    incoming_vec = torch.zeros(total, 2, 2, device=old.device, dtype=old.dtype)
    if edge_count:
        confidence = prior[src, :, 0] * decay
        indices = dst[:, None].expand(-1, 2)
        neighbor_conf.scatter_reduce_(0, indices, confidence, reduce="amax", include_self=True)
        eligible = (confidence > 0) & (confidence >= neighbor_conf[dst] - 1e-7)
        candidates = torch.arange(edge_count, device=src.device, dtype=torch.long)[:, None].expand(-1, 2)
        winner = torch.full((total, 2), edge_count, device=src.device, dtype=torch.long)
        winner.scatter_reduce_(0, indices, torch.where(eligible, candidates, edge_count),
                               reduce="amin", include_self=True)
        valid = winner < edge_count
        best_edge = winner.clamp_max(edge_count - 1)
        selected_src = src[best_edge]
        target = torch.arange(2, device=src.device)[None, :]
        relative = (graph.relative[best_edge] if graph.relative is not None
                    else torch.zeros(total, 2, 2, device=old.device, dtype=old.dtype))
        incoming_vec = torch.where(valid[..., None], prior[selected_src, target, 1:] + relative,
                                   incoming_vec)
    own_conf = prior[..., 0] * (decay if memory_decay is None else memory_decay)
    own_vec = prior[..., 1:]
    prefer_neighbor = (neighbor_conf > 0) & (neighbor_conf >= own_conf - 1e-7)
    conf = torch.where(prefer_neighbor, neighbor_conf, own_conf)
    vector = torch.where(prefer_neighbor[..., None], incoming_vec, own_vec)
    scales = getattr(graph, "cue_scales", (1.85, 2.2))
    local_visible = torch.stack((obs[..., 9], obs[..., 12]), -1).reshape(total, 2) > .5
    local_vec = torch.stack((obs[..., 7:9] * scales[0], obs[..., 10:12] * scales[1]), -2).reshape(total, 2, 2)
    conf = torch.where(local_visible, 1.0, conf)
    vector = torch.where(local_visible[..., None], local_vec, vector)
    alive = conf >= .02
    return torch.cat((conf[..., None], torch.where(alive[..., None], vector, 0.0)), -1).reshape(b, n, 6)


@torch.no_grad()
def relay_diagnostics(env, memory, geometry=False):
    """Decision-time diagnostics only. No result is fed to the controller."""
    obj, goal = memory[..., -6] > .05, memory[..., -3] > .05
    near = env._object_distance_agent <= env.cfg.object_radius + env.cfg.sense_radius
    contact = env._contact
    counts = torch.stack((obj, goal, near, near & goal, contact, contact & goal), -1).float().sum(1)
    result = dict(relay_decision_step=env.steps.clone(),
                  object_relay_reach_fraction=counts[:, 0] / env.cfg.n_agents,
                  goal_relay_reach_fraction=counts[:, 1] / env.cfg.n_agents,
                  cargo_near_agents=counts[:, 2], goal_relay_near_cargo_agents=counts[:, 3],
                  cargo_contact_agents=counts[:, 4], goal_relay_contact_agents=counts[:, 5],
                  goal_relay_near_cargo_fraction=counts[:, 3] / counts[:, 2].clamp_min(1),
                  goal_relay_contact_fraction=counts[:, 5] / counts[:, 4].clamp_min(1))
    if geometry:
        known_near = near & goal
        actual = env.goal_pos[:, None] - env.positions
        estimated = memory[..., -2:]
        error = (actual - estimated).norm(dim=-1)
        cosine = (actual * estimated).sum(-1) / (actual.norm(dim=-1) * estimated.norm(dim=-1)).clamp_min(1e-8)
        result.update(goal_bearing_error_near_cargo=(error * known_near).sum(-1) / counts[:, 3].clamp_min(1),
                      goal_bearing_alignment_near_cargo=(cosine * known_near).sum(-1) / counts[:, 3].clamp_min(1))
    return result
