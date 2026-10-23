"""Single inference worker, latest-only snapshots, WebGPU viewer (no training)."""
from __future__ import annotations
import json
import struct
import threading
import time
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import torch
from .config import EnvConfig
from .env import SwarmEnv
from .policy import LocalPolicy

MAGIC = 0x3153504E
HEADER_BYTES = 128
PARTICLE_FLOATS = 8
WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


def pack_snapshot(env, hidden, info, sequence, paused, revision, step_ms=0., realtime_factor=0.):
    """Pack dynamic scalars and display particles into ONE device->host transfer.

    CUDA -> host -> HTTP -> WebGPU is still not end-to-end zero-copy. No GPU
    readback is performed by the browser and only new snapshots are uploaded.
    """
    env.observe()  # cached; also supplies degree
    signals = torch.stack((hidden[0].square().mean(-1).sqrt(), env._degree[0],
                           env._bridge_members[0].float(), env._in_main[0].float()), -1)
    particles = torch.cat((env.positions[0], env.velocities[0], signals), -1)
    keys = ("largest_fraction", "connected", "bridge", "object_distance", "contact_count", "object_speed",
            "steps", "bridge_eligible", "found_object", "found_goal", "endpoint_gap")
    dynamic = torch.cat((env.object_pos[0], env.goal_pos[0],
                         torch.stack([info[k].reshape(-1)[0].float() for k in keys])))
    host = torch.cat((dynamic, particles.flatten())).detach().cpu().numpy().astype("<f4", copy=False)
    d, payload = host[:15], host[15:]
    c = env.cfg
    success = d[7] <= c.goal_radius - c.object_radius
    flags = int(paused) | (int(success) << 1) | (int(d[10] >= c.episode_steps and not success) << 2)
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8I", header, 0, MAGIC, 1, c.n_agents, HEADER_BYTES, sequence, flags, int(d[10]), revision)
    struct.pack_into("<24f", header, 32, *d[:4], c.object_radius, c.goal_radius,
                     c.agent_radius, c.arena_half, c.comm_radius, c.sense_radius,
                     *d[4:10], step_ms, time.monotonic() % 1_000_000, c.dt,
                     d[11], realtime_factor, d[12], d[13], d[14])
    return bytes(header) + payload.tobytes()


class Simulation:
    def __init__(self, config=None, checkpoint=None, device="cpu", untrained=False, stochastic=False):
        if checkpoint is None and not untrained:
            raise ValueError("Pass --checkpoint or explicitly opt into --untrained diagnostics")
        if checkpoint is not None:
            from .train import load_policy_checkpoint
            self.policy, config, metadata = load_policy_checkpoint(checkpoint, device)
            self.checkpoint = {**metadata, "name": Path(checkpoint).name}
        else:
            config = config or EnvConfig()
            with torch.random.fork_rng(devices=[] if torch.device(device).type == "cpu" else [torch.device(device)]):
                torch.manual_seed(config.seed)
                self.policy = LocalPolicy(20).to(device)
            self.checkpoint = {"name": "未训练", "updates": 0, "checkpoint_sha256": None,
                               "has_trained_weights": False, "stage": config.stage}
        self.env = SwarmEnv(replace(config, num_envs=1), device=device)
        self.policy.eval()
        self.stochastic = stochastic
        self.action_generator = torch.Generator(device=self.env.device).manual_seed(config.seed)
        self.hidden = self.policy.initial_state(1, config.n_agents, self.env.device)
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.paused, self.finished, self.speed = True, False, 4.0
        self.reset_pending = False
        self.revision = self.sequence = 0
        self.error = None
        self.info = self.env.metrics()
        self.step_ms = self.realtime_factor = 0.
        self.snapshot = pack_snapshot(self.env, self.hidden, self.info, 0, True, 0)
        self.worker = threading.Thread(target=self._run, name="swarm-inference", daemon=True)
        self.worker.start()

    def command(self, body):
        command = body.get("command")
        with self.lock:
            if command == "play":
                if self.error or self.finished:
                    raise ValueError("Reset before playing after an error or terminal state")
                self.paused = False
            elif command == "pause":
                self.paused = True
            elif command == "reset":
                self.reset_pending, self.paused = True, True
            elif command == "speed":
                speed = float(body.get("value", 4))
                if not np.isfinite(speed) or (speed != 0 and not .1 <= speed <= 16):
                    raise ValueError("Speed must be 0 (unpaced) or between 0.1 and 16")
                self.speed = speed
            else:
                raise ValueError("Unknown command")
            self.revision += 1
            return {"accepted_revision": self.revision, "paused": self.paused, "speed": self.speed}

    def status(self):
        with self.lock:
            return {"schema": 1, "config": asdict(self.env.cfg), "checkpoint": self.checkpoint,
                    "device": str(self.env.device), "paused": self.paused, "speed": self.speed,
                    "sequence": self.sequence, "revision": self.revision, "step_ms": self.step_ms,
                    "realtime_factor": self.realtime_factor, "publication_hz_limit": 30,
                    "error": self.error, "finished": self.finished, "trained_transport_verified": False,
                    "policy_mode": "sampled_seeded" if self.stochastic else "deterministic",
                    "action_seed": self.env.cfg.seed if self.stochastic else None,
                    "transport": "One batched CUDA-to-host copy per published snapshot; HTTP-to-WebGPU upload. Not zero-copy."}

    def _run(self):
        last_revision, last_publish = 0, time.monotonic()
        window_start, window_steps = time.monotonic(), 0
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.lock:
                reset = self.reset_pending
                self.reset_pending = False
                paused, speed, revision = self.paused, self.speed, self.revision
            changed = revision != last_revision
            try:
                with torch.inference_mode():
                    if reset:
                        self.env.reset()
                        self.hidden.zero_()
                        self.info = self.env.metrics()
                        self.error, self.finished = None, False
                        self.realtime_factor = 0.
                        changed = True
                    if paused or changed:
                        window_start, window_steps = started, 0
                        self.realtime_factor = 0.
                    if not paused:
                        action, _, _, _, self.hidden = self.policy.act(
                            self.env.observe(), self.hidden, self.env.graph, deterministic=not self.stochastic,
                            generator=self.action_generator if self.stochastic else None)
                        _, _, terminated, truncated, self.info = self.env.step(action)
                        # The termination query synchronizes the executed step;
                        # no redundant synchronize() + many scalar CPU copies.
                        ended = bool((terminated | truncated).any())
                        self.step_ms = (time.monotonic() - started) * 1000
                        window_steps += 1
                        elapsed = time.monotonic() - window_start
                        if elapsed >= .5:
                            self.realtime_factor = window_steps * self.env.cfg.dt / elapsed
                            window_start, window_steps = time.monotonic(), 0
                        if ended:
                            with self.lock:
                                self.paused, self.finished = True, True
                            paused, changed = True, True
                    # Simulation pacing is independent of display publication.
                    # Commands/terminal states publish immediately, otherwise
                    # snapshots are latest-only and limited to 30 Hz.
                    now = time.monotonic()
                    if changed or (not paused and now - last_publish >= 1 / 30):
                        with self.lock:
                            self.sequence += 1
                            seq = self.sequence
                        frame = pack_snapshot(self.env, self.hidden, self.info, seq, paused, revision,
                                              self.step_ms, self.realtime_factor)
                        with self.lock:
                            self.snapshot = frame
                        last_publish = now
                last_revision = revision
            except Exception as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.paused = True
                    header = bytearray(self.snapshot[:HEADER_BYTES])
                    self.sequence += 1
                    struct.pack_into("<I", header, 16, self.sequence)
                    struct.pack_into("<I", header, 20, struct.unpack_from("<I", header, 20)[0] | 1 | 8)
                    self.snapshot = bytes(header) + self.snapshot[HEADER_BYTES:]
                paused = True
            delay = .02 if paused else (0 if speed == 0 else max(0, self.env.cfg.dt / speed - (time.monotonic() - started)))
            self.stop_event.wait(delay)

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=10)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, simulation):
        self.simulation = simulation
        super().__init__(address, ViewerHandler)

    def server_close(self):
        super().server_close()
        self.simulation.close()


class ViewerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, code, payload=b"", content_type="application/json", headers=None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            if payload:
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/api/status", "/health"):
            self._reply(200, json.dumps(self.server.simulation.status(), ensure_ascii=False).encode())
        elif path == "/api/snapshot":
            with self.server.simulation.lock:
                payload = self.server.simulation.snapshot
            tag = f'"{struct.unpack_from("<I", payload, 16)[0]}"'
            if self.headers.get("If-None-Match") == tag:
                self._reply(304, headers={"ETag": tag})
            else:
                self._reply(200, payload, "application/octet-stream", {"ETag": tag})
        elif path == "/favicon.ico":
            self._reply(204)
        elif path in ("/", "/index.html", "/app.js", "/style.css", "/fonts/swarm-cjk.woff"):
            filename = "index.html" if path == "/" else path[1:]
            file = WEB_ROOT / filename
            if not file.exists():
                self._reply(404, b'{"error":"Viewer assets missing"}')
                return
            types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                     ".css": "text/css; charset=utf-8", ".woff": "font/woff"}
            self._reply(200, file.read_bytes(), types[file.suffix])
        else:
            self._reply(404, b'{"error":"Not found"}')

    def do_POST(self):
        if self.path != "/api/control":
            self._reply(404, b'{"error":"Not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("Invalid control payload size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Control payload must be an object")
            self._reply(200, json.dumps(self.server.simulation.command(body)).encode())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._reply(400, json.dumps({"error": str(exc)}).encode())


def create_server(config=None, checkpoint=None, device="cpu", port=8765, host="127.0.0.1", untrained=False, stochastic=False):
    simulation = Simulation(config, checkpoint, device, untrained, stochastic)
    try:
        return ViewerServer((host, port), simulation)
    except Exception:
        simulation.close()
        raise


def serve(config=None, checkpoint=None, device="cpu", port=8765, host="127.0.0.1", untrained=False, stochastic=False):
    with create_server(config, checkpoint, device, port, host, untrained, stochastic) as server:
        print(f"http://{host}:{server.server_address[1]}", flush=True)
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass
