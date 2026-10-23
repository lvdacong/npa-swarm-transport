"""Replay the densest saved world at B=1/2/4; never change task physics."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path

import torch

from npa_swarm.config import EnvConfig, TrainConfig
from npa_swarm.diagnostics import system_info, write_json
from npa_swarm.env import SwarmEnv
from npa_swarm.graph import radius_graph
from npa_swarm.train import Trainer, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--scatter-reference", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.scatter_reference:
        import npa_swarm.env as env_module
        import npa_swarm.policy as policy_module
        def scatter_reference(hidden, graph):
            flat = hidden.reshape(-1, hidden.shape[-1])
            out = torch.zeros_like(flat).index_add(0, graph.dst, flat[graph.src])
            degree = torch.zeros(flat.shape[0], device=flat.device, dtype=flat.dtype)
            degree.index_add_(0, graph.dst, torch.ones_like(graph.dst, dtype=flat.dtype))
            return (out / degree.clamp_min(1)[:, None]).reshape_as(hidden)
        env_module.aggregate = policy_module.aggregate = scatter_reference
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original = EnvConfig(**saved["env_config"])
    graph = radius_graph(saved["env_state"]["positions"], original.comm_radius)
    edges = torch.bincount(graph.src // original.n_agents, minlength=original.num_envs)
    selected = int(edges.argmax())
    result = {"system": system_info("cuda:0"), "checkpoint_sha256": file_sha256(args.checkpoint),
              "implementation": "gather_scatter_reference" if args.scatter_reference else "production_cuda_csr",
              "checkpoint_update": saved["updates"], "edges_per_original_world": edges.tolist(),
              "repeated_world": selected, "measurements": []}
    for batch in args.batches:
        trainer = None
        try:
            cfg = replace(original, num_envs=batch)
            env = SwarmEnv(cfg, "cuda:0")
            state = env.state_dict()
            for name in env._state_names:
                source = saved["env_state"][name]
                state[name] = source[selected:selected + 1].repeat(batch, *([1] * (source.ndim - 1)))
            env.load_state_dict(state)
            trainer = Trainer(env, TrainConfig(**saved["train_config"]))
            trainer.policy.load_state_dict(saved["model"])
            trainer.hidden = saved["hidden"][selected:selected + 1].to("cuda:0").repeat(batch, 1, 1)
            updates = [trainer.update() for _ in range(2)]
            row = {"batch": batch, "status": "PASS", "updates": updates,
                   "reserved_bytes": torch.cuda.memory_reserved(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        except Exception as exc:
            row = {"batch": batch, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        result["measurements"].append(row)
        write_json(args.output / "comparison.json", result)
        print(json.dumps(row), flush=True)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        if row["status"] == "FAIL":
            break  # driver-level sticky OOM requires a new process


if __name__ == "__main__":
    main()
