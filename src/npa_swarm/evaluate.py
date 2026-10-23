"""Frozen-policy evaluation with complete denominators and raw trajectories."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .env import SwarmEnv
from .train import file_sha256, load_policy_checkpoint, seed_all, source_sha256


def _json_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metrics(info):
    return {key: value.detach().cpu().tolist() if torch.is_tensor(value) else value
            for key, value in info.items()}


@torch.no_grad()
def evaluate(checkpoint, output, seeds=(42, 43, 44), episodes_per_seed=3,
             device="cpu", stage="search", max_steps=None, deterministic=True):
    """Frozen policy, fresh layouts; never silently omit failed requests.

    Every control-step position/action and task metric is saved. CPU diagnostic
    populations and shortened horizons are explicitly excluded from formal
    search-task validation. This does not assert continuous-time connectivity
    between the environment's sampled physical/control states.
    """
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds) or episodes_per_seed < 1:
        raise ValueError("Use distinct seeds and episodes_per_seed >= 1")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    # Evaluate a private frozen copy so concurrent training cannot replace it.
    frozen = output / "checkpoint.pt"
    shutil.copy2(checkpoint, frozen)
    policy, saved_cfg, checkpoint_metadata = load_policy_checkpoint(frozen, device)
    cfg = dataclasses.replace(saved_cfg, num_envs=1, stage=stage)
    horizon = cfg.episode_steps if max_steps is None else min(max_steps, cfg.episode_steps)
    reference = type(cfg)()
    physical_fields = ("arena_half", "agent_radius", "object_radius", "goal_radius", "comm_radius",
                       "sense_radius", "dt", "substeps", "max_force", "agent_drag", "max_speed",
                       "object_drag", "object_static_friction", "contact_stiffness", "episode_steps")
    task_differences = {name: {"actual": getattr(cfg, name), "reference": getattr(reference, name)}
                        for name in physical_fields if getattr(cfg, name) != getattr(reference, name)}
    diagnostic = cfg.n_agents != 4096 or horizon < cfg.episode_steps or stage != "search" or bool(task_differences)
    requests = [{"seed": seed, "episode": episode,
                 "layout_seed": seed + 1000003 * episode,
                 "action_seed": None if deterministic else seed + 1000003 * episode,
                 "status": "not_started"}
                for seed in seeds for episode in range(episodes_per_seed)]
    manifest = {"schema_version": 1, "checkpoint": checkpoint_metadata,
                "runtime_source_sha256": source_sha256(),
                "source_changed": source_sha256() != checkpoint_metadata["source_sha256"],
                "env_config": dataclasses.asdict(cfg), "requested": len(requests),
                "device": str(device), "policy_mode": "deterministic" if deterministic else "sampled_seeded",
                "max_steps": horizon, "diagnostic": diagnostic,
                "task_config_differences": task_differences,
                "connectivity_sampling": "control-step endpoints (not continuous-time proof)",
                "requests": requests}
    _json_dump(output / "manifest.json", manifest)
    stop = False
    start = time.perf_counter()
    for index, request in enumerate(requests):
        if stop:
            continue
        request["status"] = "running"
        _json_dump(output / "manifest.json", manifest)
        positions, objects, actions, metrics = [], [], [], []
        env = None
        try:
            seed_all(request["layout_seed"])
            episode_cfg = dataclasses.replace(cfg, seed=request["layout_seed"])
            env = SwarmEnv(episode_cfg, device=device)
            hidden = policy.initial_state(1, cfg.n_agents, device)
            positions.append(env.positions[0].cpu().numpy().copy())
            objects.append(env.object_pos[0].cpu().numpy().copy())
            metrics.append(_metrics(env.metrics()))
            for step in range(horizon):
                obs = env.observe()
                action, _, _, _, hidden = policy.act(obs, hidden, env.graph, deterministic=deterministic)
                _, _, terminated, truncated, info = env.step(action)
                actions.append(action[0].cpu().numpy().copy())
                positions.append(env.positions[0].cpu().numpy().copy())
                objects.append(env.object_pos[0].cpu().numpy().copy())
                metrics.append(_metrics(info))
                if bool(terminated[0]):
                    request["status"] = "success"
                    break
                if bool(truncated[0]):
                    request["status"] = "timeout"
                    break
            else:
                request["status"] = "censored" if horizon < cfg.episode_steps else "timeout"
            final = _metrics(env.metrics())
            request.update(steps=len(actions), final_metrics=final,
                           full_connectivity=not bool(final["ever_disconnected"][0]))
        except (Exception, KeyboardInterrupt) as error:
            request.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                           error_type=type(error).__name__, error=str(error),
                           traceback=traceback.format_exc(), steps=len(actions))
            # Preserve the remaining requests as not_started instead of repeatedly
            # running a broken configuration or silently shrinking the denominator.
            stop = True
        finally:
            if positions:
                trajectory = output / f"episode-{index:04d}.npz"
                np.savez_compressed(trajectory, positions=np.stack(positions),
                                    object_positions=np.stack(objects),
                                    actions=np.stack(actions) if actions else np.empty((0, cfg.n_agents, 2), np.float32),
                                    goal_position=env.goal_pos[0].cpu().numpy(),
                                    metrics_json=np.array(json.dumps(metrics, allow_nan=False)))
                request["trajectory"] = trajectory.name
                request["trajectory_sha256"] = file_sha256(trajectory)
            _json_dump(output / "manifest.json", manifest)
    counts = {status: sum(r["status"] == status for r in requests) for status in
              ("success", "timeout", "censored", "error", "interrupted", "not_started")}
    completed = counts["success"] + counts["timeout"]
    final_metrics = [r["final_metrics"] for r in requests if "final_metrics" in r]
    eligible_steps = sum(m["bridge_eligible_steps"][0] for m in final_metrics)
    bridge_steps = sum(m["bridge_steps"][0] for m in final_metrics)
    summary = {"requested": len(requests), "completed": completed, **counts,
               "transport_success_rate_all_requests": counts["success"] / len(requests),
               "full_connectivity_successes": sum(r.get("full_connectivity", False)
                                                   and r["status"] == "success" for r in requests),
               "episodes_fully_connected": sum(r.get("full_connectivity", False) for r in requests),
               "found_object_requests": sum(bool(m["found_object"][0]) for m in final_metrics),
               "found_goal_requests": sum(bool(m["found_goal"][0]) for m in final_metrics),
               "bridge_eligible_steps": eligible_steps, "bridge_connected_steps": bridge_steps,
               "bridge_keep_fraction": bridge_steps / eligible_steps if eligible_steps else None,
               "diagnostic": diagnostic, "stage": stage, "n_agents": cfg.n_agents,
               "task_config_differences": task_differences,
               "policy_mode": manifest["policy_mode"],
               "seconds": time.perf_counter() - start,
               "engineering_evaluation_completed": not (counts["error"] + counts["interrupted"] + counts["not_started"]),
               "full_horizon_completed": completed == len(requests),
               "research_task_passed": False,
               "note": "Task quality requires reviewed cross-seed outcomes; an executable evaluator is not a learned result."}
    _json_dump(output / "summary.json", summary)
    return summary
