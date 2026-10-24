"""Recurrent PPO: ordered world/time replay, one-hop messages, full checkpoints."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .config import EnvConfig, TrainConfig
from .env import SwarmEnv
from .policy import LocalPolicy
from .relay import relay_diagnostics

SCHEMA_VERSION = 3


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.device(device).type == "cuda" and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs; use a new run for a new topology")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compute_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        boot = (~terminated[t]).unsqueeze(-1)
        continuation = (~(terminated[t] | truncated[t])).unsqueeze(-1)
        delta = rewards[t] + gamma * next_values[t] * boot - values[t]
        carry = delta + gamma * lam * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


@dataclasses.dataclass
class Rollout:
    observations: list
    graphs: list
    latent_actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    done: torch.Tensor
    initial_hidden: torch.Tensor
    metrics: dict


class Trainer:
    def __init__(self, env: SwarmEnv, config: TrainConfig, policy=None):
        self.env, self.cfg, self.device = env, config, torch.device(env.device)
        seed_all(config.seed)
        self.policy = (policy or LocalPolicy(env.obs_dim, config.hidden_dim, config.relay_memory_decay)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.hidden = self.policy.initial_state(env.cfg.num_envs, env.cfg.n_agents, self.device)
        self.updates = self.env_steps = self.episodes = self.successes = 0
        self.lineage = {"kind": "new"}
        self.resume_metadata = None
        self.runtime_source_sha256 = source_sha256()
        self.best_progress = None

    @torch.no_grad()
    def collect(self):
        c = self.cfg
        observations, graphs, latent, logp, values, rewards, terms, truncs = ([] for _ in range(8))
        truncated_boot = {}
        physical = {key: [] for key in (
            "connected", "largest_fraction", "bridge", "object_distance", "object_speed", "contact_count",
            "cargo_force_norm", "cargo_force_toward_goal", "reward_progress", "reward_contact_credit",
            "found_object", "found_goal", "repair_gap_max", "endpoint_gap",
            "reward_repair", "reward_connectivity", "reward_rear_approach", "reward_transport_local")}
        initial_hidden = self.hidden.detach().clone()
        contact_peaks, contact_active = [], []
        relay_statistics = {}
        for t in range(c.rollout_steps):
            obs, graph = self.env.observe(), self.env.graph
            action, z, lp, value, memory = self.policy.act(obs, self.hidden, graph)
            for key, statistic in relay_diagnostics(self.env, memory).items():
                relay_statistics.setdefault(key, []).append(statistic.float().mean())
            next_obs, reward, terminated, truncated, info = self.env.step(
                action, goal_relay=memory[..., -3])
            observations.append(obs.detach())
            graphs.append(graph)
            latent.append(z)
            logp.append(lp)
            values.append(value)
            rewards.append(reward)
            terms.append(terminated.clone())
            truncs.append(truncated.clone())
            for key in physical:
                if key in info:
                    physical[key].append(info[key].float().mean())
            contact_peaks.append(info["contact_count"].max())
            contact_active.append((info["contact_count"] > 0).float().mean())
            done = terminated | truncated
            self.hidden = memory * (~done).view(-1, 1, 1)
            if bool(done.any()):
                if bool(truncated.any()):
                    # Only a timeout needs the PRE-reset value. On ordinary
                    # steps, the next iteration already computes this value.
                    _, _, boot, _ = self.policy(next_obs, memory, self.env.graph)
                    truncated_boot[t] = boot
                self.env.reset(done)
            self.env_steps += self.env.cfg.num_envs
        values_t, rewards_t = torch.stack(values), torch.stack(rewards)
        term_t, trunc_t = torch.stack(terms), torch.stack(truncs)
        _, _, last_value, _ = self.policy(self.env.observe(), self.hidden, self.env.graph)
        next_values = torch.cat((values_t[1:], last_value[None]))
        for t, boot in truncated_boot.items():
            next_values[t] = torch.where(trunc_t[t, :, None], boot, next_values[t])
        episode_count, success_count = int((term_t | trunc_t).sum()), int(term_t.sum())
        self.episodes += episode_count
        self.successes += success_count
        advantages, returns = compute_gae(rewards_t, values_t, next_values, term_t, trunc_t, c.gamma, c.gae_lambda)
        metrics = {"mean_reward": float(rewards_t.mean()), "rollout_episodes": episode_count,
                   "rollout_successes": success_count,
                   "contact_count_max": int(torch.stack(contact_peaks).max()),
                   "contact_active_fraction": float(torch.stack(contact_active).mean())}
        metrics.update({key: float(torch.stack(values).mean()) for key, values in relay_statistics.items()})
        aliases = {"connected": "connected_fraction", "largest_fraction": "largest_fraction", "bridge": "bridge_fraction"}
        for key, collected in physical.items():
            if collected:
                metrics[aliases.get(key, key + "_mean")] = float(torch.stack(collected).mean())
        if physical["reward_progress"]:
            metrics["rollout_distance_delta"] = float(torch.stack(physical["reward_progress"]).sum() / 3)
        return Rollout(observations, graphs, torch.stack(latent), torch.stack(logp), values_t,
                       advantages, returns, term_t | trunc_t, initial_hidden, metrics)

    def optimize(self, rollout: Rollout):
        c, total_steps = self.cfg, len(rollout.observations)
        advantages = rollout.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        stats = {key: torch.zeros((), device=self.device) for key in
                 ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
        for _ in range(c.ppo_epochs):
            hidden = rollout.initial_hidden.detach()
            self.optimizer.zero_grad(set_to_none=True)
            chunk_loss = None
            for t in range(total_steps):
                lp, entropy, value, hidden = self.policy.evaluate_actions(
                    rollout.observations[t], hidden, rollout.graphs[t], rollout.latent_actions[t])
                log_ratio = lp - rollout.old_log_probs[t]
                ratio = log_ratio.exp()
                raw = ratio * advantages[t]
                clipped = ratio.clamp(1 - c.clip_ratio, 1 + c.clip_ratio) * advantages[t]
                policy_loss = -torch.minimum(raw, clipped).mean()
                value_loss = .5 * (value - rollout.returns[t]).square().mean()
                entropy_mean = entropy.mean()
                loss = (policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy_mean) / total_steps
                chunk_loss = loss if chunk_loss is None else chunk_loss + loss
                hidden = hidden * (~rollout.done[t]).view(-1, 1, 1)
                with torch.no_grad():
                    scale = 1 / (total_steps * c.ppo_epochs)
                    stats["policy_loss"] += policy_loss.detach() * scale
                    stats["value_loss"] += value_loss.detach() * scale
                    stats["entropy"] += entropy_mean.detach() * scale
                    stats["approx_kl"] += ((ratio - 1) - log_ratio).mean() * scale
                    stats["clip_fraction"] += ((ratio - 1).abs() > c.clip_ratio).float().mean() * scale
                if (t + 1) % c.bptt_steps == 0 or t + 1 == total_steps:
                    chunk_loss.backward()
                    chunk_loss, hidden = None, hidden.detach()
            norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            stats["gradient_norm"] = norm.detach()
        # Refresh live memory with updated parameters, retaining the detached
        # pre-window carry; this is finite-window TBPTT, not full-episode BPTT.
        with torch.no_grad():
            hidden = rollout.initial_hidden.detach()
            for obs, graph, done in zip(rollout.observations, rollout.graphs, rollout.done):
                _, _, _, hidden = self.policy(obs, hidden, graph)
                hidden = hidden * (~done).view(-1, 1, 1)
            self.hidden = hidden
        return {key: float(value) for key, value in stats.items()}

    def update(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        rollout = self.collect()
        stats = self.optimize(rollout)
        self.updates += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        duration = time.perf_counter() - start
        worlds = self.env.cfg.num_envs * self.cfg.rollout_steps
        result = {"update": self.updates, "env_steps": self.env_steps,
                  "agent_steps": self.env_steps * self.env.cfg.n_agents,
                  "episodes": self.episodes, "successes": self.successes,
                  "seconds": duration, "world_steps_per_second": worlds / duration,
                  "agent_decisions_per_second": worlds * self.env.cfg.n_agents / duration,
                  "device": str(self.device), "stage": self.env.cfg.stage, "n_agents": self.env.cfg.n_agents,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None,
                  "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else None,
                  **rollout.metrics, **stats}
        if self.resume_metadata is not None:
            result["resume"] = self.resume_metadata
        return result

    def save(self, path):
        atomic_torch_save({"schema_version": SCHEMA_VERSION,
            "env_config": dataclasses.asdict(self.env.cfg), "train_config": dataclasses.asdict(self.cfg),
            "obs_dim": self.env.obs_dim, "model": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
            "hidden": self.hidden, "env_state": self.env.state_dict(), "rng": capture_rng(),
            "updates": self.updates, "env_steps": self.env_steps, "episodes": self.episodes, "successes": self.successes,
            "lineage": self.lineage, "source_sha256": self.runtime_source_sha256, "best_progress": self.best_progress,
            "torch_version": torch.__version__, "device": str(self.device)}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        data = _read_checkpoint(path, device)
        before = capture_rng()
        try:
            env_cfg = dict(data["env_config"])
            env_cfg.setdefault("rear_approach_weight", 0.0)
            env = SwarmEnv(EnvConfig(**env_cfg), device=device)
            trainer = cls(env, TrainConfig(**data["train_config"]))
            trainer.policy.load_state_dict(data["model"])
            trainer.optimizer.load_state_dict(data["optimizer"])
            env.load_state_dict(data["env_state"])
            trainer.hidden = data["hidden"].to(device)
            expected = (env.cfg.num_envs, env.cfg.n_agents, trainer.policy.state_dim)
            if tuple(trainer.hidden.shape) != expected or not bool(torch.isfinite(trainer.hidden).all()):
                raise ValueError("Invalid recurrent memory")
            for field in ("updates", "env_steps", "episodes", "successes"):
                value = data[field]
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid checkpoint counter: {field}")
                setattr(trainer, field, value)
            restore_rng(data["rng"], device)
        except Exception:
            restore_rng(before, "cuda" if before["cuda"] else "cpu")
            raise
        trainer.lineage = data.get("lineage", {"kind": "unknown"})
        trainer.best_progress = data.get("best_progress")
        trainer.resume_metadata = {"checkpoint_sha256": file_sha256(path),
            "source_changed": source_sha256() != data["source_sha256"],
            "device_changed": str(device) != data["device"], "trajectory_bitwise_claimed": False}
        return trainer


def _read_checkpoint(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Task/policy v3 requires a new run; v1/v2 weights are incompatible, preserve them separately.")
    return data


def has_trained_weights(data):
    lineage = data.get("lineage", {})
    return bool(data["updates"] > 0 or lineage.get("inherited_trained_weights", False) or lineage.get("parent_updates", 0) > 0)


def load_policy_checkpoint(path, device="cpu"):
    data = _read_checkpoint(path, device)
    env_cfg = dict(data["env_config"])
    env_cfg.setdefault("rear_approach_weight", 0.0)
    cfg = EnvConfig(**env_cfg)
    tc = TrainConfig(**data["train_config"])
    policy = LocalPolicy(data["obs_dim"], tc.hidden_dim, tc.relay_memory_decay).to(device)
    policy.load_state_dict(data["model"])
    policy.eval()
    return policy, cfg, {"checkpoint_sha256": file_sha256(path), "updates": data["updates"],
        "env_steps": data["env_steps"], "stage": cfg.stage, "source_sha256": data["source_sha256"],
        "model_schema_version": data["schema_version"], "relay_memory_decay": tc.relay_memory_decay,
        "learned_success_verified": False, "has_trained_weights": has_trained_weights(data), "lineage": data.get("lineage", {})}


def train(env_cfg, train_cfg, output, updates, device="cpu", resume=None, warmstart=None):
    if updates < 0 or (resume is not None and warmstart is not None):
        raise ValueError("Require updates >= 0 and at most one of resume/warmstart")
    output = Path(output)
    if not resume and ((output / "last.pt").exists() or (output / "metrics.jsonl").exists()):
        raise FileExistsError("Run output exists; use resume or choose a new directory")
    if resume and (output / "last.pt").exists() and file_sha256(resume) != file_sha256(output / "last.pt"):
        raise FileExistsError("Output contains another checkpoint; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        trainer = Trainer.load(resume, device)
        for supplied, actual in ((env_cfg, trainer.env.cfg), (train_cfg, trainer.cfg)):
            if supplied is not None and dataclasses.asdict(supplied) != dataclasses.asdict(actual):
                raise ValueError("Resume uses saved configuration; use warmstart for a new task")
    else:
        if env_cfg is None or train_cfg is None:
            raise ValueError("New training requires environment and training configurations")
        trainer = Trainer(SwarmEnv(env_cfg, device=device), train_cfg)
        if warmstart:
            parent = _read_checkpoint(warmstart, device)
            trainer.policy.load_state_dict(parent["model"])
            trainer.lineage = {"kind": "warmstart", "parent_sha256": file_sha256(warmstart),
                "inherited_trained_weights": has_trained_weights(parent), "parent_updates": parent["updates"],
                "parent_stage": parent["env_config"]["stage"]}
    trainer.save(output / "last.pt")
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        for _ in range(updates):
            result = trainer.update()
            if trainer.device.type == "cuda":
                reserved, allocated = torch.cuda.memory_reserved(trainer.device), torch.cuda.memory_allocated(trainer.device)
                if reserved > 4 * 1024 ** 3 and reserved - allocated > 2 * 1024 ** 3:
                    torch.cuda.empty_cache()
            # Rank a TRAINING candidate by actual signed movement, not by
            # absolute remaining distance across unrelated random layouts.
            score = result.get("rollout_distance_delta", 0.)
            if trainer.best_progress is None or score > trainer.best_progress:
                trainer.best_progress = score
                trainer.save(output / "candidate.pt")
            trainer.save(output / "last.pt")
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(result, allow_nan=False), flush=True)
    summary = {"updates": trainer.updates, "additional_updates": updates, "env_steps": trainer.env_steps,
        "episodes": trainer.episodes, "successes": trainer.successes, "checkpoint": str(output / "last.pt"),
        "checkpoint_sha256": file_sha256(output / "last.pt"), "stage": trainer.env.cfg.stage,
        "n_agents": trainer.env.cfg.n_agents, "device": str(trainer.device), "research_task_passed": False,
        "lineage": trainer.lineage}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
