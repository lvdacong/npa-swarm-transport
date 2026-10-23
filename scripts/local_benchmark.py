"""Compare real 4096-agent CUDA throughput, sequentially on one idle device."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from npa_swarm.config import EnvConfig, TrainConfig
from npa_swarm.diagnostics import probe, system_info, write_json
from npa_swarm.env import SwarmEnv
from npa_swarm.train import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {"system": system_info("cuda:0"), "measurements": []}
    for batch in (1, 2, 4):
        cfg = EnvConfig(num_envs=batch)
        perf = probe(cfg, args.output / f"b{batch}", "cuda:0", steps=16, warmup=4)
        trainer = Trainer(SwarmEnv(cfg, "cuda:0"), TrainConfig())
        updates = [trainer.update() for _ in range(3)]
        row = {"env_config": asdict(cfg), "inference": perf, "default_ppo_updates": updates}
        results["measurements"].append(row)
        write_json(args.output / "comparison.json", results)
        print(json.dumps({"batch": batch, "inference_world_s": perf["environment_steps_per_second"],
                          "training_world_s": [m["world_steps_per_second"] for m in updates],
                          "peak_bytes": max(m["cuda_peak_allocated_bytes"] for m in updates)}), flush=True)
        del trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
