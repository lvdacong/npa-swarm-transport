"""Small warm-state CUDA stage timing; instrumented timings are not production rates."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
import npa_swarm.env as module
from npa_swarm.config import EnvConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.train import load_policy_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--snapshot', help='optional actual viewer binary positions; hidden state starts at zero')
    args = p.parse_args()
    torch.set_num_threads(2)
    policy, cfg, _ = load_policy_checkpoint(args.checkpoint, 'cuda:0')
    cfg = replace(cfg, num_envs=1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['env_state']
    state = {k: v[:1] if torch.is_tensor(v) and k != 'rng' else v for k, v in saved.items()}
    state['config'] = cfg.to_dict()
    result = {}
    with torch.inference_mode():
        for name in ('initial', 'saved_world') + (('viewer_world',) if args.snapshot else ()):
            env = SwarmEnv(cfg, 'cuda:0')
            if name == 'saved_world':
                env.load_state_dict(state)
            if name == 'viewer_world':
                packet = Path(args.snapshot).read_bytes()
                f = np.frombuffer(packet, '<f4', 24, 32)
                agents = np.frombuffer(packet, '<f4', offset=128).reshape(cfg.n_agents, 8)
                env.positions[0].copy_(torch.tensor(agents[:, :2].copy(), device='cuda:0'))
                env.velocities[0].copy_(torch.tensor(agents[:, 2:4].copy(), device='cuda:0'))
                env.object_pos[0].copy_(torch.tensor(f[:2].copy(), device='cuda:0'))
                env.goal_pos[0].copy_(torch.tensor(f[2:4].copy(), device='cuda:0'))
                env._refresh()
            hidden = policy.initial_state(1, cfg.n_agents)
            for _ in range(5):
                a, _, _, _, hidden = policy.act(env.observe(), hidden, env.graph)
                env.step(a)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(32):
                a, _, _, _, hidden = policy.act(env.observe(), hidden, env.graph)
                env.step(a)
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            totals = defaultdict(float)
            def instrument(owner, attr, label):
                original = getattr(owner, attr)
                def measured(*a, **kw):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    value = original(*a, **kw)
                    torch.cuda.synchronize()
                    totals[label] += time.perf_counter() - begin
                    return value
                setattr(owner, attr, measured)
                return lambda: setattr(owner, attr, original)
            undo = [instrument(module, 'radius_graph', 'radius_graph'),
                    instrument(module, 'components', 'components'),
                    instrument(env, '_repair_distances', 'repair_gaps'),
                    instrument(env, '_refresh', 'refresh_including_nested'),
                    instrument(policy, 'act', 'policy'),
                    instrument(env, 'step', 'step_including_nested')]
            try:
                for _ in range(16):
                    a, _, _, _, hidden = policy.act(env.observe(), hidden, env.graph)
                    env.step(a)
            finally:
                for restore in reversed(undo):
                    restore()
            result[name] = dict(production_step_ms=duration / 32 * 1000,
                production_actual_speed=32 * cfg.dt / duration,
                stage_ms_instrumented={k: v / 16 * 1000 for k, v in totals.items()},
                largest_fraction=float(env._largest_fraction[0]),
                edges=env.graph.src.numel(), checkpoint=str(Path(args.checkpoint).resolve()))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
