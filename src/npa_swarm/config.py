"""Version 2 task: 900 particles, continuous drag, one search training task."""
from dataclasses import asdict, dataclass


@dataclass
class EnvConfig:
    n_agents: int = 900
    num_envs: int = 1
    arena_half: float = 10.0
    agent_radius: float = 0.06
    object_radius: float = 0.65
    goal_radius: float = 1.0
    comm_radius: float = 0.48
    sense_radius: float = 1.2
    dt: float = 0.1
    substeps: int = 8
    max_force: float = 1.0
    agent_drag: float = 1.0
    max_speed: float = 1.0
    object_drag: float = 12.0
    object_static_friction: float = 0.0
    contact_stiffness: float = 60.0
    episode_steps: int = 1200
    stage: str = "search"  # contact retained only as an isolated physical diagnostic
    layout: str = "random"
    task_version: int = 2
    connectivity_weight: float = 0.15
    repair_weight: float = 0.2
    bridge_weight: float = 0.05
    bridge_progress_weight: float = 0.25
    work_credit_weight: float = 1.0
    # New run: reward-only rear-arc geometry and localized cargo progress.
    # 0 preserves historical v3 rewards for existing checkpoints.
    rear_approach_weight: float = 0.8
    contact_credit_mode: str = "work"
    exploration_weight: float = 0.2
    seed: int = 42

    def __post_init__(self):
        if self.n_agents < 2 or self.num_envs < 1:
            raise ValueError("n_agents >= 2 and num_envs >= 1 required")
        if self.stage not in ("contact", "search") or self.layout not in ("random", "diagonal"):
            raise ValueError("Invalid stage or layout")
        if self.task_version != 2:
            raise ValueError("This runtime implements task v2; use the original revision for v1")
        if self.contact_credit_mode not in ("work", "marginal"):
            raise ValueError("contact_credit_mode must be work or marginal")
        for name in ("arena_half", "agent_radius", "object_radius", "goal_radius", "comm_radius",
                     "sense_radius", "dt", "max_force", "agent_drag", "max_speed", "object_drag", "contact_stiffness"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.substeps < 1 or self.episode_steps < 1 or self.goal_radius <= self.object_radius:
            raise ValueError("Positive step counts and a goal containing the complete cargo are required")
        for name in ("object_static_friction", "connectivity_weight", "repair_weight", "bridge_weight",
                     "bridge_progress_weight", "work_credit_weight", "rear_approach_weight", "exploration_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    hidden_dim: int = 32  # 28 trainable channels + 6 local endpoint relay channels
    rollout_steps: int = 128
    bptt_steps: int = 32
    ppo_epochs: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.97
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    seed: int = 42
    relay_memory_decay: float = 0.997

    def __post_init__(self):
        if self.hidden_dim < 8 or min(self.rollout_steps, self.bptt_steps, self.ppo_epochs) < 1:
            raise ValueError("hidden_dim >= 8 and positive rollout/optimization sizes required")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid optimizer/return configuration")
        if not 0 < self.relay_memory_decay < 1:
            raise ValueError("relay_memory_decay must decay gradually within (0,1)")

    def to_dict(self):
        return asdict(self)
