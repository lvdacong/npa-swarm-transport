"""One-hop, agent-relative endpoint memory; no remote target observations."""
import torch


@torch.no_grad()
def relay_beacons(old, obs, graph, decay=0.97):
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
    own_conf = prior[..., 0] * decay
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
