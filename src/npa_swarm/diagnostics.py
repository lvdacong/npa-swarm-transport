"""Bounded measurements, never evidence that a policy has learned the task."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from .config import EnvConfig, TrainConfig


def resolve_device(value):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
    return device


def system_info(device="auto"):
    from .train import source_sha256
    selected = resolve_device(device)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "device": str(selected), "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "runtime_source_sha256": source_sha256()}
    if torch.cuda.is_available():
        result["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
            "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            result["nvidia_smi"] = proc.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["nvidia_smi_error"] = str(exc)
    try:
        root = Path(__file__).resolve().parents[2]
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        result["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def probe(cfg: EnvConfig, output, device="auto", steps=16, warmup=2, train_updates=0, trace=False):
    from .env import SwarmEnv
    from .policy import LocalPolicy
    from .train import Trainer
    import contextlib
    if steps < 1 or warmup < 0 or train_updates < 0:
        raise ValueError("Invalid probe counts")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    selected = resolve_device(device)
    result = {"kind": "bounded_untrained_probe", "research_task_passed": False,
              "system": system_info(str(selected)), "config": asdict(cfg), "requested_steps": steps,
              "warmup_steps": warmup, "train_updates_requested": train_updates}
    try:
        torch.manual_seed(cfg.seed)
        env = SwarmEnv(cfg, selected)
        policy = LocalPolicy(env.obs_dim).to(selected).eval()
        hidden = policy.initial_state(cfg.num_envs, cfg.n_agents, selected)
        if selected.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected)
        def sync():
            if selected.type == "cuda":
                torch.cuda.synchronize(selected)
        times, policy_times, physics_times, edges = [], [], [], []
        activities = [torch.profiler.ProfilerActivity.CPU]
        if selected.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        context = torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) if trace else contextlib.nullcontext()
        with context as profiler, torch.inference_mode():
            for i in range(warmup + steps):
                sync(); begin = time.perf_counter()
                with torch.profiler.record_function("local_policy"):
                    action, _, _, _, hidden = policy.act(env.observe(), hidden, env.graph)
                sync(); middle = time.perf_counter()
                with torch.profiler.record_function("physics_graph_observation"):
                    _, _, terminated, truncated, _ = env.step(
                        action, goal_relay=hidden[..., -3])
                sync(); end = time.perf_counter()
                if i >= warmup:
                    times.append(end-begin); policy_times.append(middle-begin); physics_times.append(end-middle)
                    edges.append(env.graph.src.numel())
                done = terminated | truncated
                if bool(done.any()):
                    env.reset(done); hidden[done] = 0
        if trace:
            profiler.export_chrome_trace(str(destination / "trace.json"))
        result.update(status="PASS", steps_measured=len(times),
            control_step_ms_mean=float(np.mean(times)*1000), control_step_ms_p95=float(np.percentile(times,95)*1000),
            policy_ms_mean=float(np.mean(policy_times)*1000), physics_graph_ms_mean=float(np.mean(physics_times)*1000),
            environment_steps_per_second=float(cfg.num_envs/np.mean(times)),
            simulated_seconds_per_wall_second=float(cfg.dt/np.mean(times)),
            agent_steps_per_second=float(cfg.num_envs*cfg.n_agents/np.mean(times)), directed_edges_mean=float(np.mean(edges)),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected) if selected.type=="cuda" else None)
        if train_updates:
            trainer = Trainer(SwarmEnv(cfg,selected),TrainConfig(rollout_steps=8,bptt_steps=4,ppo_epochs=1,seed=cfg.seed))
            result["training_probe_config"] = asdict(trainer.cfg)
            result["training_updates"] = [trainer.update() for _ in range(train_updates)]
            trainer.save(destination / "probe.pt")
    except Exception as exc:
        result.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        write_json(destination / "probe.json",result)
        raise
    write_json(destination / "probe.json",result)
    return result
