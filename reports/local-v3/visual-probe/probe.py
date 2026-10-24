"""900-agent hardware display probe: real relay/contact states, synthetic geometry.
No actions, reward, optimization or claimed learned transport.
"""
import json
from pathlib import Path
import sys
import threading
import torch
from npa_swarm.config import EnvConfig
from npa_swarm.server import create_server, pack_snapshot

torch.set_num_threads(2)
root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
server = create_server(EnvConfig(), device='cpu', port=8773, untrained=True)
sim = server.simulation
sim.stop_event.set(); sim.worker.join(3)
env = sim.env
angle = torch.arange(30) * (2 * torch.pi / 30)
ring = torch.stack((angle.cos(), angle.sin()), -1) * .77
ring[[0, 15]] *= .69 / .77
ring[:, 0] -= 7.4
x, y = torch.linspace(-6.3, 6., 29), (torch.arange(30) - 14.5) * .144
yy, xx = torch.meshgrid(y, x, indexing='ij')
env.positions[0] = torch.cat((ring, torch.stack((xx.flatten(), yy.flatten()), -1)), 0)
env.object_pos[:] = torch.tensor([-7.4, 0.])
env.goal_pos[:] = torch.tensor([7.4, 0.])
env.last_displacement.zero_(); env._refresh()
env.found_object[:] = env._object_visible.any(-1)
env.found_goal[:] = env._goal_visible.any(-1)
with torch.no_grad():
    for _ in range(64):
        _, _, _, sim.hidden = sim.policy(env.observe(), sim.hidden, env.graph)
    before = env.positions.clone()
    env.positions[0, -1] = torch.tensor([0., 7.])
    env.last_displacement = env.positions - before
    env._refresh()
    _, _, _, sim.hidden = sim.policy(env.observe(), sim.hidden, env.graph)
sim.info = env.metrics()
sim.sequence = 1
sim.snapshot = pack_snapshot(env, sim.hidden, sim.info, sim.sequence, True, 0)
(root/'snapshot.bin').write_bytes(sim.snapshot)
import numpy as np
a = np.frombuffer(sim.snapshot, '<f4', offset=128).reshape(900, 8)
result = dict(scope='synthetic geometry, actual 65 one-hop relay transitions and instantaneous spring contact; no policy actions',
              particles=900, packet_bytes=len(sim.snapshot),
              both_messages=int(((a[:, 4]>.05)&(a[:, 5]>.05)).sum()),
              selected={str(i): a[i].tolist() for i in (0, 15, 899)})
(root/'states.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result), flush=True)
threading.Thread(target=server.serve_forever, daemon=True).start()
print('http://127.0.0.1:8773; quit to close', flush=True)
try:
    for line in sys.stdin:
        if line.strip() == 'quit': break
finally:
    server.shutdown(); server.server_close()
