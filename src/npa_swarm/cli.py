import argparse
from dataclasses import asdict
import json
import sys

import torch

from .config import EnvConfig, TrainConfig
from .diagnostics import probe, resolve_device, system_info, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="NPA-inspired local-message RL swarm transport")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="record actual system/device information")
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--output")
    doctor.add_argument("--threads", type=int, default=2)

    def env_args(p):
        p.add_argument("--agents", type=int, default=4096)
        p.add_argument("--envs", type=int, default=1)
        p.add_argument("--stage", choices=("contact", "search"), default="search")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--episode-steps", type=int, default=1200)
        p.add_argument("--device", default="auto")
        p.add_argument("--threads", type=int, default=2)

    perf = sub.add_parser("probe", help="bounded untrained device/performance probe")
    env_args(perf)
    perf.add_argument("--output", required=True)
    perf.add_argument("--steps", type=int, default=16)
    perf.add_argument("--warmup", type=int, default=2)
    perf.add_argument("--train-updates", type=int, default=0)
    perf.add_argument("--trace", action="store_true")

    training = sub.add_parser("train", help="train; --updates is the number of ADDITIONAL PPO updates")
    env_args(training)
    training.add_argument("--output", required=True)
    training.add_argument("--updates", type=int, required=True)
    training.add_argument("--resume", help="strict continuation: saved configs, optimizer, RNG, world and memory")
    training.add_argument("--warmstart", help="policy-only initialization for a new run/stage; resets counts and optimizer")
    training.add_argument("--rollout-steps", type=int, default=32)
    training.add_argument("--bptt-steps", type=int, default=8)
    training.add_argument("--ppo-epochs", type=int, default=3)
    training.add_argument("--hidden-dim", type=int, default=32)
    training.add_argument("--learning-rate", type=float, default=3e-4)

    evaluation = sub.add_parser("evaluate", help="frozen checkpoint evaluation, preserving failures and trajectories")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    evaluation.add_argument("--episodes-per-seed", type=int, default=1)
    evaluation.add_argument("--stage", choices=("contact", "search"), default="search")
    evaluation.add_argument("--max-steps", type=int, help="diagnostic horizon override; not formal evaluation")
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--threads", type=int, default=2)

    viewer = sub.add_parser("serve", help="explicit checkpoint inference or labelled untrained diagnostics")
    env_args(viewer)
    viewer.add_argument("--checkpoint")
    viewer.add_argument("--untrained", action="store_true")
    viewer.add_argument("--port", type=int, default=8765)
    viewer.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    # Comparable FP32 baseline; performance changes need measured evidence.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(args.device)
    if args.command == "doctor":
        result = system_info(str(device))
        if args.output:
            write_json(args.output, result)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args.checkpoint, args.output, args.seeds, args.episodes_per_seed, device=str(device), stage=args.stage, max_steps=args.max_steps)
    else:
        cfg = EnvConfig(n_agents=args.agents, num_envs=args.envs, stage=args.stage,
                        seed=args.seed, episode_steps=args.episode_steps)
        if args.command == "probe":
            result = probe(cfg, args.output, str(device), args.steps, args.warmup, args.train_updates, args.trace)
        elif args.command == "train":
            from .train import train
            if args.resume and args.warmstart:
                parser.error("choose strict --resume OR policy-only --warmstart")
            if args.resume:
                forbidden = {"--agents", "--envs", "--stage", "--seed", "--episode-steps", "--rollout-steps", "--bptt-steps", "--ppo-epochs", "--hidden-dim", "--learning-rate"}
                provided = sys.argv[1:] if argv is None else argv
                if any(token.split("=", 1)[0] in forbidden for token in provided):
                    parser.error("strict resume uses saved configs; omit config overrides, or use --warmstart for a new experiment")
            tc = TrainConfig(hidden_dim=args.hidden_dim, rollout_steps=args.rollout_steps,
                             bptt_steps=args.bptt_steps, ppo_epochs=args.ppo_epochs,
                             learning_rate=args.learning_rate, seed=args.seed)
            result = train(None if args.resume else cfg, None if args.resume else tc,
                           args.output, args.updates, str(device), resume=args.resume, warmstart=args.warmstart)
        else:
            from .server import serve
            serve(config=cfg, checkpoint=args.checkpoint, untrained=args.untrained,
                  device=str(device), port=args.port, host=args.host)
            return
    if result is not None:
        if not isinstance(result, (dict, list, str, int, float, bool)):
            result = {"result": str(result)}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == "evaluate" and isinstance(result, dict) and not result.get("engineering_evaluation_completed", True):
            raise SystemExit(1)
