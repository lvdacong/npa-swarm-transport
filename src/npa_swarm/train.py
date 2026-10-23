"""A small recurrent PPO trainer and complete, atomic training checkpoints.

Training batches retain time and particle identity. Each PPO epoch replays the
whole rollout in order; only BPTT boundaries detach memory. No time/particle
shuffle and no frozen-neighbor-message shortcut is used.
"""

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


SCHEMA_VERSION = 1


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
            raise ValueError("CUDA RNG device count differs; use warmstart for a new device topology")
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
    """Timeout bootstraps its terminal state but never the next episode.

    Inputs have time first. Rewards/values: [T,B,N]; done flags: [T,B].
    """
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
        self.env, self.cfg = env, config
        self.device = torch.device(env.device)
        seed_all(config.seed)
        self.policy = (policy or LocalPolicy(env.obs_dim, config.hidden_dim, config.vector_actor, config.vector_only)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.hidden = self.policy.initial_state(env.cfg.num_envs, env.cfg.n_agents, self.device)
        self.updates = 0
        self.env_steps = 0
        self.episodes = 0
        self.successes = 0
        self.lineage = {"kind": "new"}
        self.resume_metadata = None
        self.runtime_source_sha256 = source_sha256()
        self.best_rollout_distance = None

    @torch.no_grad()
    def collect(self):
        c = self.cfg
        observations, graphs, latent, logp, values = [], [], [], [], []
        rewards, next_values, terms, truncs = [], [], [], []
        connected, largest, bridge = [], [], []
        physical = {key: [] for key in ("object_distance", "object_speed", "contact_count",
                                       "cargo_force_norm", "cargo_force_toward_goal",
                                       "reward_progress", "reward_contact_credit", "found_object", "found_goal")}
        initial_hidden = self.hidden.detach().clone()
        start_episodes, start_successes = self.episodes, self.successes
        for _ in range(c.rollout_steps):
            obs = self.env.observe()
            graph = self.env.graph
            action, z, lp, value, memory = self.policy.act(obs, self.hidden, graph)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            # This is the actual post-step observation, before any reset.
            _, _, boot_value, _ = self.policy(next_obs, memory, self.env.graph)
            observations.append(obs.detach().clone())
            graphs.append(graph)
            latent.append(z)
            logp.append(lp)
            values.append(value)
            rewards.append(reward)
            next_values.append(boot_value)
            terms.append(terminated.clone())
            truncs.append(truncated.clone())
            for key, destination in (("connected", connected), ("largest_fraction", largest), ("bridge", bridge)):
                if key in info:
                    destination.append(torch.as_tensor(info[key], device=self.device).float().mean())
            for key, destination in physical.items():
                if key in info:
                    destination.append(info[key].float().mean())
            done = terminated | truncated
            self.hidden = memory * (~done).view(-1, 1, 1)
            if bool(done.any()):
                self.env.reset(done)
            self.env_steps += self.env.cfg.num_envs
        values_t, rewards_t = torch.stack(values), torch.stack(rewards)
        term_t, trunc_t = torch.stack(terms), torch.stack(truncs)
        self.episodes += int((term_t | trunc_t).sum())
        self.successes += int(term_t.sum())
        advantages, returns = compute_gae(rewards_t, values_t, torch.stack(next_values),
                                         term_t, trunc_t, c.gamma, c.gae_lambda)
        metrics = {"mean_reward": float(rewards_t.mean()),
                   "rollout_episodes": self.episodes - start_episodes,
                   "rollout_successes": self.successes - start_successes}
        for key, collected in (("connected_fraction", connected), ("largest_fraction", largest), ("bridge_fraction", bridge)):
            if collected:
                metrics[key] = float(torch.stack(collected).mean())
        for key, collected in physical.items():
            if collected:
                metrics[key + "_mean"] = float(torch.stack(collected).mean())
        if physical["reward_progress"]:
            # Sum actual per-step changes, so resets cannot look like progress.
            metrics["rollout_distance_delta"] = float(torch.stack(physical["reward_progress"]).sum() / 3)
        return Rollout(observations, graphs, torch.stack(latent), torch.stack(logp),
                       values_t, advantages, returns, term_t | trunc_t,
                       initial_hidden, metrics)

    def optimize(self, rollout: Rollout):
        c = self.cfg
        total_steps = len(rollout.observations)
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
                value_loss = 0.5 * (value - rollout.returns[t]).square().mean()
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
                    chunk_loss = None
                    hidden = hidden.detach()
            norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm,
                                                 error_if_nonfinite=True)
            self.optimizer.step()
            stats["gradient_norm"] = norm.detach()
        # Refresh live memory using the updated policy over this rollout.
        # The pre-rollout carry remains detached: this is finite-window TBPTT,
        # not a claim to reconstruct the entire episode under updated weights.
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
                  "device": str(self.device), "stage": self.env.cfg.stage,
                  "n_agents": self.env.cfg.n_agents,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda" else None,
                  "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device)
                    if self.device.type == "cuda" else None,
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
                    if self.device.type == "cuda" else None,
                  **rollout.metrics, **stats}
        if self.resume_metadata is not None:
            result["resume"] = self.resume_metadata
        return result

    def save(self, path):
        payload = {"schema_version": SCHEMA_VERSION,
                   "env_config": dataclasses.asdict(self.env.cfg),
                   "train_config": dataclasses.asdict(self.cfg),
                   "obs_dim": self.env.obs_dim, "model": self.policy.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "hidden": self.hidden,
                   "env_state": self.env.state_dict(), "rng": capture_rng(),
                   "updates": self.updates, "env_steps": self.env_steps,
                   "episodes": self.episodes, "successes": self.successes,
                   "lineage": self.lineage, "source_sha256": self.runtime_source_sha256,
                   "best_rollout_distance": self.best_rollout_distance,
                   "torch_version": torch.__version__, "device": str(self.device)}
        atomic_torch_save(payload, path)

    @classmethod
    def load(cls, path, device="cpu"):
        data = _read_checkpoint(path, device)
        env = SwarmEnv(EnvConfig(**data["env_config"]), device=device)
        trainer = cls(env, TrainConfig(**data["train_config"]))
        trainer.policy.load_state_dict(data["model"])
        trainer.optimizer.load_state_dict(data["optimizer"])
        env.load_state_dict(data["env_state"])
        trainer.hidden = data["hidden"].to(device)
        expected = (env.cfg.num_envs, env.cfg.n_agents, trainer.cfg.hidden_dim)
        if tuple(trainer.hidden.shape) != expected or not bool(torch.isfinite(trainer.hidden).all()):
            raise ValueError("Checkpoint recurrent memory has invalid shape or values")
        for field in ("updates", "env_steps", "episodes", "successes"):
            value = data[field]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid checkpoint counter: {field}")
            setattr(trainer, field, value)
        restore_rng(data["rng"], device)
        trainer.lineage = data.get("lineage", {"kind": "unknown"})
        trainer.best_rollout_distance = data.get("best_rollout_distance")
        trainer.resume_metadata = {"checkpoint_sha256": file_sha256(path),
                                   "source_changed": source_sha256() != data["source_sha256"],
                                   "device_changed": str(device) != data["device"],
                                   "trajectory_bitwise_claimed": False}
        return trainer


def _read_checkpoint(path, device):
    # These are project-owned trusted checkpoints, not an arbitrary upload API.
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    return data


def has_trained_weights(data):
    lineage = data.get("lineage", {})
    return bool(data["updates"] > 0 or lineage.get("inherited_trained_weights", False)
                or lineage.get("parent_updates", 0) > 0)


def load_policy_checkpoint(path, device="cpu"):
    """Inference-only loading: never resumes a training environment or RNG."""
    data = _read_checkpoint(path, device)
    cfg = EnvConfig(**data["env_config"])
    policy = LocalPolicy(data["obs_dim"], data["train_config"]["hidden_dim"],
                         data["train_config"].get("vector_actor", False),
                         data["train_config"].get("vector_only", False)).to(device)
    policy.load_state_dict(data["model"])
    policy.eval()
    metadata = {"checkpoint_sha256": file_sha256(path), "updates": data["updates"],
                "env_steps": data["env_steps"], "stage": cfg.stage,
                "source_sha256": data["source_sha256"], "learned_success_verified": False,
                "has_trained_weights": has_trained_weights(data), "lineage": data.get("lineage", {})}
    return policy, cfg, metadata


def train(env_cfg, train_cfg, output, updates, device="cpu", resume=None, warmstart=None):
    """Run an explicit number of *additional* updates, with a checkpoint per update."""
    if updates < 0 or (resume is not None and warmstart is not None):
        raise ValueError("Require updates >= 0 and at most one of resume/warmstart")
    output = Path(output)
    if not resume and ((output / "last.pt").exists() or (output / "metrics.jsonl").exists()):
        raise FileExistsError("Run output exists; use resume or choose a new directory")
    if resume and (output / "last.pt").exists() and file_sha256(resume) != file_sha256(output / "last.pt"):
        raise FileExistsError("Output contains another checkpoint; branch into a new directory instead of overwriting it")
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        trainer = Trainer.load(resume, device)
        for supplied, actual in ((env_cfg, trainer.env.cfg), (train_cfg, trainer.cfg)):
            if supplied is not None and dataclasses.asdict(supplied) != dataclasses.asdict(actual):
                raise ValueError("Resume uses saved configuration; use warmstart to change the task")
    else:
        if env_cfg is None or train_cfg is None:
            raise ValueError("New training requires environment and training configurations")
        trainer = Trainer(SwarmEnv(env_cfg, device=device), train_cfg)
        if warmstart:
            parent = _read_checkpoint(warmstart, device)
            trainer.policy.load_state_dict(parent["model"])
            trainer.lineage = {"kind": "warmstart", "parent_sha256": file_sha256(warmstart),
                               "inherited_trained_weights": has_trained_weights(parent),
                               "parent_updates": parent["updates"], "parent_stage": parent["env_config"]["stage"]}
    trainer.save(output / "last.pt")
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        for _ in range(updates):
            result = trainer.update()
            # Dynamic edge counts leave differently sized cached blocks after
            # a rollout. Trim only large idle pools, between complete updates,
            # rather than allowing an 8-GiB WDDM device to page or fail.
            trim_start = time.perf_counter()
            trimmed = False
            if trainer.device.type == "cuda":
                reserved = torch.cuda.memory_reserved(trainer.device)
                allocated = torch.cuda.memory_allocated(trainer.device)
                if reserved > 4 * 1024 ** 3 and reserved - allocated > 2 * 1024 ** 3:
                    torch.cuda.empty_cache()
                    trimmed = True
                result["cuda_reserved_after_update_bytes"] = torch.cuda.memory_reserved(trainer.device)
            result["cache_trimmed"] = trimmed
            result["cache_trim_seconds"] = time.perf_counter() - trim_start if trimmed else 0.0
            score = result.get("object_distance_mean")
            if score is not None and (trainer.best_rollout_distance is None or score < trainer.best_rollout_distance):
                trainer.best_rollout_distance = score
                # A training-metric candidate only; independent evaluation is
                # still required. Preserve it when later updates regress.
                trainer.save(output / "best-distance.pt")
            trainer.save(output / "last.pt")
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(result, allow_nan=False), flush=True)
    summary = {"updates": trainer.updates, "additional_updates": updates,
               "env_steps": trainer.env_steps, "episodes": trainer.episodes,
               "successes": trainer.successes, "checkpoint": str(output / "last.pt"),
               "checkpoint_sha256": file_sha256(output / "last.pt"),
               "stage": trainer.env.cfg.stage, "n_agents": trainer.env.cfg.n_agents,
               "device": str(trainer.device), "research_task_passed": False,
               "lineage": trainer.lineage}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
