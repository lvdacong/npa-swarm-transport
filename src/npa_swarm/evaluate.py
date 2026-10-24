"""Frozen evaluation with full request denominators and control-step trajectories."""
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
from .relay import relay_diagnostics
from .train import file_sha256, load_policy_checkpoint, seed_all, source_sha256


def _json_dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metrics(info):
    tensors = [(key, value) for key, value in info.items() if torch.is_tensor(value)]
    result = {key: value for key, value in info.items() if not torch.is_tensor(value)}
    if tensors:
        host = torch.cat([value.reshape(-1).float() for _, value in tensors]).detach().cpu().numpy()
        offset = 0
        for key, value in tensors:
            a = host[offset:offset + value.numel()].reshape(value.shape)
            offset += value.numel()
            if value.dtype == torch.bool:
                a = a.astype(bool)
            elif not value.is_floating_point():
                a = a.astype(np.int64)
            result[key] = a.tolist()
    return result


@torch.no_grad()
def evaluate(checkpoint, output, seeds=(100, 101, 102), episodes_per_seed=1,
             device="cpu", stage="search", max_steps=None, deterministic=True, layout=None):
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds) or episodes_per_seed < 1:
        raise ValueError("Use distinct seeds and episodes_per_seed >= 1")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "checkpoint.pt"
    shutil.copy2(checkpoint, frozen)
    policy, saved_cfg, metadata = load_policy_checkpoint(frozen, device)
    cfg = dataclasses.replace(saved_cfg, num_envs=1, stage=stage, layout=layout or saved_cfg.layout)
    horizon = cfg.episode_steps if max_steps is None else min(max_steps, cfg.episode_steps)
    reference = type(cfg)()
    physical_fields = ("arena_half", "agent_radius", "object_radius", "goal_radius", "comm_radius",
                       "sense_radius", "dt", "substeps", "max_force", "agent_drag", "max_speed",
                       "object_drag", "object_static_friction", "contact_stiffness", "episode_steps")
    differences = {name: {"actual": getattr(cfg, name), "reference": getattr(reference, name)}
                   for name in physical_fields if getattr(cfg, name) != getattr(reference, name)}
    diagnostic = cfg.n_agents != reference.n_agents or horizon < cfg.episode_steps or stage != "search" or bool(differences)
    requests = [{"seed": seed, "episode": episode, "layout_seed": seed + 1000003 * episode,
                 "action_seed": None if deterministic else seed + 1000003 * episode, "status": "not_started"}
                for seed in seeds for episode in range(episodes_per_seed)]
    manifest = {"schema_version": 2, "task_version": cfg.task_version, "checkpoint": metadata,
                "runtime_source_sha256": source_sha256(), "source_changed": source_sha256() != metadata["source_sha256"],
                "env_config": dataclasses.asdict(cfg), "requested": len(requests), "device": str(device),
                "policy_mode": "deterministic" if deterministic else "sampled_seeded",
                "max_steps": horizon, "diagnostic": diagnostic, "task_config_differences": differences,
                "connectivity_sampling": "control-step endpoints, not continuous-time proof",
                "bridge_definition": "same particle component within comm_radius of both endpoint surfaces; sensing is insufficient",
                "requests": requests}
    _json_dump(output / "manifest.json", manifest)
    stop, start = False, time.perf_counter()
    for index, request in enumerate(requests):
        if stop:
            continue
        request["status"] = "running"
        _json_dump(output / "manifest.json", manifest)
        positions, objects, actions, metrics = [], [], [], []
        env = None
        try:
            seed_all(request["layout_seed"])
            env = SwarmEnv(dataclasses.replace(cfg, seed=request["layout_seed"]), device=device)
            hidden = policy.initial_state(1, cfg.n_agents, device)
            positions.append(env.positions[0].cpu().numpy().copy())
            objects.append(env.object_pos[0].cpu().numpy().copy())
            metrics.append(_metrics(env.metrics()))
            for _ in range(horizon):
                action, _, _, _, hidden = policy.act(env.observe(), hidden, env.graph, deterministic=deterministic)
                message_info = relay_diagnostics(env, hidden, geometry=True)
                _, _, terminated, truncated, info = env.step(action)
                info.update(message_info)
                packet = torch.cat((env.positions[0].flatten(), env.object_pos[0], action[0].flatten())).cpu().numpy()
                positions.append(packet[:cfg.n_agents*2].reshape(cfg.n_agents, 2).copy())
                objects.append(packet[cfg.n_agents*2:cfg.n_agents*2+2].copy())
                actions.append(packet[cfg.n_agents*2+2:].reshape(cfg.n_agents, 2).copy())
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
            stop = True
        finally:
            if positions:
                trajectory = output / f"episode-{index:04d}.npz"
                np.savez_compressed(trajectory, positions=np.stack(positions), object_positions=np.stack(objects),
                    actions=np.stack(actions) if actions else np.empty((0, cfg.n_agents, 2), np.float32),
                    goal_position=env.goal_pos[0].cpu().numpy(), metrics_json=np.array(json.dumps(metrics, allow_nan=False)))
                request["trajectory"], request["trajectory_sha256"] = trajectory.name, file_sha256(trajectory)
            _json_dump(output / "manifest.json", manifest)
    counts = {status: sum(r["status"] == status for r in requests)
              for status in ("success", "timeout", "censored", "error", "interrupted", "not_started")}
    completed = counts["success"] + counts["timeout"]
    finals = [r["final_metrics"] for r in requests if "final_metrics" in r]
    eligible = sum(m["bridge_eligible_steps"][0] for m in finals)
    bridges = sum(m["bridge_steps"][0] for m in finals)
    summary = {"requested": len(requests), "completed": completed, **counts,
        "transport_success_rate_all_requests": counts["success"] / len(requests),
        "full_connectivity_successes": sum(r.get("full_connectivity", False) and r["status"] == "success" for r in requests),
        "episodes_fully_connected": sum(r.get("full_connectivity", False) for r in requests),
        "found_object_requests": sum(bool(m["found_object"][0]) for m in finals),
        "found_goal_requests": sum(bool(m["found_goal"][0]) for m in finals),
        "bridge_eligible_steps": eligible, "bridge_connected_steps": bridges,
        "bridge_keep_fraction": bridges / eligible if eligible else None,
        "diagnostic": diagnostic, "stage": stage, "layout": cfg.layout, "task_version": cfg.task_version,
        "n_agents": cfg.n_agents, "task_config_differences": differences, "policy_mode": manifest["policy_mode"],
        "seconds": time.perf_counter() - start,
        "engineering_evaluation_completed": not (counts["error"] + counts["interrupted"] + counts["not_started"]),
        "full_horizon_completed": completed == len(requests), "research_task_passed": False,
        "note": "Task v2 has changed physics/population/bridge semantics. Do not pool with v1; learned quality needs reviewed outcomes."}
    _json_dump(output / "summary.json", summary)
    return summary
