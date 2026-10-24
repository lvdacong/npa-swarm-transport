"""Complete radius graphs using sorted spatial cells, without an N x N matrix.

Only cell keys are sorted. Particle arrays and their identities never move.
Candidate counts are variable: dense cells cost more memory, never lose neighbors.
"""
from dataclasses import dataclass, field
import math

import torch


@dataclass
class Graph:
    src: torch.Tensor
    dst: torch.Tensor
    num_nodes: int = 0
    _mean_matrix: torch.Tensor | None = field(default=None, repr=False)
    relative: torch.Tensor | None = field(default=None, repr=False)
    cue_scales: tuple[float, float] = (1.85, 2.2)
    displacement: torch.Tensor | None = field(default=None, repr=False)

    def to(self, device):
        return Graph(self.src.to(device), self.dst.to(device), self.num_nodes,
                     relative=None if self.relative is None else self.relative.to(device),
                     cue_scales=self.cue_scales,
                     displacement=None if self.displacement is None else self.displacement.to(device))


@torch.no_grad()
def radius_graph(positions: torch.Tensor, radius: float) -> Graph:
    if positions.ndim != 3 or positions.shape[-1] != 2 or radius <= 0:
        raise ValueError("Expected [B,N,2] positions and positive radius")
    batch, count, _ = positions.shape
    total = batch * count
    flat = positions.reshape(total, 2)
    cell = torch.floor(flat / radius).to(torch.long)
    low, high = cell.amin(0), cell.amax(0)
    span = high - low + 1
    cell = cell - low
    env_id = torch.arange(batch, device=positions.device).repeat_interleave(count)
    key = (env_id * span[0] + cell[:, 0]) * span[1] + cell[:, 1]
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    offset = torch.tensor([(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)],
                          device=positions.device, dtype=torch.long)
    neighbor_cell = cell[:, None, :] + offset[None, :, :]
    valid = ((neighbor_cell >= 0) & (neighbor_cell < span)).all(-1)
    query_key = ((env_id[:, None] * span[0] + neighbor_cell[..., 0]) * span[1]
                 + neighbor_cell[..., 1]).reshape(-1)
    begin = torch.searchsorted(sorted_key, query_key)
    end = torch.searchsorted(sorted_key, query_key, right=True)
    sizes = (end - begin) * valid.reshape(-1)
    query = torch.repeat_interleave(torch.arange(total * 9, device=positions.device), sizes)
    prefix = sizes.cumsum(0) - sizes
    within = torch.arange(query.numel(), device=positions.device) - prefix[query]
    src = order[begin[query] + within]
    dst = torch.div(query, 9, rounding_mode="floor")
    delta = flat[src] - flat[dst]
    keep = (src != dst) & (delta.square().sum(-1) <= radius * radius)
    return Graph(src[keep], dst[keep], total, relative=delta[keep])


def aggregate(hidden: torch.Tensor, graph: Graph) -> torch.Tensor:
    """Differentiable neighbor mean. An isolated node receives zeros."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    if flat.is_cuda:
        # radius_graph emits nondecreasing destination IDs. CSR SpMM avoids
        # materializing [edges, hidden_dim] gathers during every recurrent
        # forward/backward, which exhausted 8 GiB after dense-world rollouts.
        # Cache only fixed graph weights, never learned hidden states.
        matrix = graph._mean_matrix
        if matrix is None or matrix.dtype != flat.dtype or matrix.shape[0] != flat.shape[0]:
            with torch.no_grad():
                dst, src = graph.dst, graph.src
                degree = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.long)
                degree.scatter_add_(0, dst, torch.ones_like(dst))
                values = degree[dst].to(flat.dtype).reciprocal()
                # Canonicalize columns and preserve duplicate-edge weights for
                # hand-built graphs too. COO coalescing guarantees valid CSR.
                matrix = torch.sparse_coo_tensor(torch.stack((dst, src)), values,
                                                 size=(flat.shape[0], flat.shape[0]), device=flat.device,
                                                 check_invariants=False).coalesce().to_sparse_csr()
                graph._mean_matrix = matrix
        return torch.sparse.mm(matrix, flat).reshape_as(hidden)
    result = torch.zeros_like(flat).index_add(0, graph.dst, flat[graph.src])
    degree = torch.zeros(flat.shape[0], device=flat.device, dtype=flat.dtype)
    degree.index_add_(0, graph.dst, torch.ones_like(graph.dst, dtype=flat.dtype))
    return (result / degree.clamp_min(1)[:, None]).reshape_as(hidden)


@torch.no_grad()
def components(graph: Graph, batch: int, count: int) -> torch.Tensor:
    """Exact component labels by root hooking and pointer jumping.

    Labels are flattened particle IDs. Convergence is checked, rather than
    assuming a fixed message-hop count is enough for a long connected chain.
    CUDA uses a scalar convergence synchronization, not a CPU graph copy.
    """
    total = batch * count
    parent = torch.arange(total, device=graph.src.device)
    if graph.src.numel() == 0:
        return parent.reshape(batch, count)
    for _ in range(total):
        previous = parent
        a, b = parent[graph.src], parent[graph.dst]
        high, low = torch.maximum(a, b), torch.minimum(a, b)
        parent = parent.clone()
        parent.scatter_reduce_(0, high, low, reduce="amin", include_self=True)
        # Parent pointers strictly decrease, so log2(total) jumps reach roots.
        for _ in range(math.ceil(math.log2(max(total, 2)))):
            parent = parent[parent]
        if torch.equal(parent, previous):
            return parent.reshape(batch, count)
    raise RuntimeError("Connected-component calculation did not converge")
