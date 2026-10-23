"""A bounded real CUDA trace of a saved 4096-agent rollout and backward pass."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

import npa_swarm.env as env_module
from npa_swarm.diagnostics import system_info, write_json
from npa_swarm.train import Trainer, file_sha256


def region(name, function):
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return function(*args, **kwargs)
    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trainer = Trainer.load(args.checkpoint, "cuda:0")
    parent_updates = trainer.updates
    trainer.cfg = replace(trainer.cfg, rollout_steps=8, bptt_steps=4, ppo_epochs=1)
    env_module.radius_graph = region("radius_graph", env_module.radius_graph)
    env_module.components = region("connectivity", env_module.components)
    trainer.env.observe = region("local_observation", trainer.env.observe)
    trainer.env.step = region("eight_physics_substeps_and_refresh", trainer.env.step)
    trainer.policy.forward = region("local_GRU_and_message", trainer.policy.forward)
    trainer.collect = region("rollout_collection", trainer.collect)
    trainer.optimize = region("PPO_replay_and_backward", trainer.optimize)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=True) as profile:
        measurement = trainer.update()
    profile.export_chrome_trace(str(args.output / "trace.json"))
    (args.output / "operators.txt").write_text(profile.key_averages().table(sort_by="self_device_time_total", row_limit=40), encoding="utf-8")
    write_json(args.output / "profile.json", {"kind": "profiled_checkpoint_diagnostic_not_learning_result",
               "parent_sha256": file_sha256(args.checkpoint), "parent_updates": parent_updates,
               "system": system_info("cuda:0"), "env_config": asdict(trainer.env.cfg),
               "probe_train_config": asdict(trainer.cfg), "measurement_with_profiler_overhead": measurement,
               "checkpoint_modified": False})


if __name__ == "__main__":
    main()
