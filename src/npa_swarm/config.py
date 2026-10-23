"""Small, serializable experiment configuration. Formal population stays at 4096."""
from dataclasses import asdict, dataclass


@dataclass
class EnvConfig:
    n_agents: int = 4096
    num_envs: int = 1
    arena_half: float = 10.0
    agent_radius: float = 0.04
    object_radius: float = 0.65
    goal_radius: float = 1.0
    comm_radius: float = 0.32
    sense_radius: float = 0.8
    dt: float = 0.1
    substeps: int = 8
    max_force: float = 1.0
    agent_drag: float = 2.0
    max_speed: float = 0.5
    object_drag: float = 15.0
    object_static_friction: float = 4.0
    contact_stiffness: float = 100.0
    episode_steps: int = 1200
    stage: str = "search"
    connectivity_weight: float = 0.1
    force_credit_weight: float = 0.0
    local_transport_reward: bool = False
    exploration_weight: float = 0.01
    seed: int = 42

    def __post_init__(self):
        if self.n_agents < 2 or self.num_envs < 1:
            raise ValueError("n_agents >= 2 and num_envs >= 1 required")
        if self.stage not in ("contact", "search"):
            raise ValueError("stage must be contact or search")
        for name in ("arena_half", "agent_radius", "object_radius", "goal_radius", "comm_radius", "sense_radius", "dt", "max_force", "agent_drag", "max_speed", "object_drag", "contact_stiffness"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.substeps < 1 or self.episode_steps < 1:
            raise ValueError("substeps and episode_steps must be positive")
        if self.goal_radius <= self.object_radius:
            raise ValueError("destination must contain the complete object")
        if min(self.object_static_friction, self.connectivity_weight, self.force_credit_weight, self.exploration_weight) < 0:
            raise ValueError("friction and connectivity weight cannot be negative")

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    hidden_dim: int = 32
    vector_actor: bool = False
    vector_only: bool = False
    rollout_steps: int = 32
    bptt_steps: int = 8
    ppo_epochs: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    seed: int = 42

    def __post_init__(self):
        if self.vector_only and not self.vector_actor:
            raise ValueError("vector_only requires vector_actor")
        if min(self.hidden_dim, self.rollout_steps, self.bptt_steps, self.ppo_epochs) < 1:
            raise ValueError("network and rollout dimensions must be positive")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid optimizer/return configuration")

    def to_dict(self):
        return asdict(self)
