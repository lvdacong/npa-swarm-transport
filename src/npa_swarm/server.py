"""Local inference viewer: one worker, latest-only binary snapshots, no training."""
from __future__ import annotations

import json
import struct
import threading
import time
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import EnvConfig
from .env import SwarmEnv
from .policy import LocalPolicy

MAGIC = 0x3153504E  # little-endian bytes NPS1
HEADER_BYTES = 128
PARTICLE_FLOATS = 8
WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(value)


def pack_snapshot(env: SwarmEnv, hidden: torch.Tensor, info: dict,
                  sequence: int, paused: bool, revision: int,
                  step_ms: float = 0.0) -> bytes:
    """One explicit device→host copy of the display payload, not zero-copy CUDA."""
    n = env.cfg.n_agents
    degree = torch.zeros(n, device=env.positions.device)
    if env.graph.dst.numel():
        first = env.graph.dst < n
        degree.scatter_add_(0, env.graph.dst[first], torch.ones_like(env.graph.dst[first], dtype=degree.dtype))
    signals = torch.stack((hidden[0].square().mean(-1).sqrt(), degree,
                          torch.zeros_like(degree), torch.zeros_like(degree)), dim=-1)
    payload = torch.cat((env.positions[0], env.velocities[0], signals), dim=-1).detach().cpu().numpy().astype("<f4", copy=False)
    obj = env.object_pos[0].detach().cpu().tolist()
    goal = env.goal_pos[0].detach().cpu().tolist()
    success = bool(_scalar(info.get("success")))
    timeout = int(_scalar(env.steps)) >= env.cfg.episode_steps and not success
    flags = int(paused) | (int(success) << 1) | (int(timeout) << 2)
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<8I", header, 0, MAGIC, 1, n, HEADER_BYTES, sequence, flags,
                     int(_scalar(env.steps)), revision)
    values = [*obj, *goal, env.cfg.object_radius, env.cfg.goal_radius,
              env.cfg.agent_radius, env.cfg.arena_half, env.cfg.comm_radius,
              env.cfg.sense_radius, _scalar(info.get("largest_fraction"), -1),
              _scalar(info.get("connected"), -1), _scalar(info.get("bridge"), -1),
              _scalar(info.get("object_distance"), float(np.linalg.norm(np.array(obj)-np.array(goal)))),
              _scalar(info.get("contact_count")), _scalar(info.get("object_speed")),
              step_ms, time.monotonic() % 1_000_000, float(env.cfg.dt),
              _scalar(info.get("bridge_eligible"), -1), 0, 0, 0, 0]
    struct.pack_into("<24f", header, 32, *values)
    return bytes(header) + payload.tobytes()


class Simulation:
    def __init__(self, config: EnvConfig | None = None, checkpoint: str | Path | None = None,
                 device: str = "cpu", untrained: bool = False):
        if checkpoint is None and not untrained:
            raise ValueError("Pass --checkpoint or explicitly opt into --untrained diagnostics")
        self.checkpoint = None
        if checkpoint is not None:
            from .train import load_policy_checkpoint
            self.policy, loaded_cfg, metadata = load_policy_checkpoint(checkpoint, device)
            config = replace(loaded_cfg, num_envs=1)
            self.checkpoint = {**metadata, "name": Path(checkpoint).name}
        else:
            config = replace(config or EnvConfig(), num_envs=1)
        self.env = SwarmEnv(config, device=device)
        if checkpoint is None:
            # Avoid changing process-global training RNG; server is inference only.
            with torch.random.fork_rng(devices=[] if device == "cpu" else [torch.device(device)]):
                torch.manual_seed(config.seed)
                self.policy = LocalPolicy(self.env.obs_dim).to(device)
            self.checkpoint = {"name": "随机初始化 / 未训练", "updates": 0,
                               "checkpoint_sha256": None, "stage": config.stage}
        self.policy.eval()
        self.hidden = self.policy.initial_state(1, config.n_agents, self.env.device)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.paused = True
        self.finished = False
        self.speed = 1.0
        self.reset_pending = False
        self.revision = 0
        self.sequence = 0
        self.error: str | None = None
        self.info: dict = self.env.metrics()
        self.step_ms = 0.0
        self.snapshot = pack_snapshot(self.env, self.hidden, self.info, 0, True, 0)
        self.worker = threading.Thread(target=self._run, name="swarm-inference", daemon=True)
        self.worker.start()

    def command(self, body: dict) -> dict:
        command = body.get("command")
        with self.lock:
            if command == "play":
                if self.error:
                    raise ValueError("Reset first after a simulation error")
                if self.finished:
                    raise ValueError("Episode ended; reset before playing again")
                self.paused = False
            elif command == "pause":
                self.paused = True
            elif command == "reset":
                self.reset_pending = True
                self.paused = True
            elif command == "speed":
                speed = float(body.get("value", 1))
                if not np.isfinite(speed) or not 0.1 <= speed <= 4:
                    raise ValueError("Speed must be between 0.1 and 4")
                self.speed = speed
            else:
                raise ValueError("Unknown command")
            self.revision += 1
            return {"accepted_revision": self.revision, "paused": self.paused, "speed": self.speed}

    def status(self) -> dict:
        with self.lock:
            return {"schema": 1, "config": asdict(self.env.cfg), "checkpoint": self.checkpoint,
                    "device": str(self.env.device), "paused": self.paused, "speed": self.speed,
                    "sequence": self.sequence, "revision": self.revision, "step_ms": self.step_ms,
                    "error": self.error, "finished": self.finished, "trained_transport_verified": False,
                    "transport": "CPU/CUDA to host binary snapshot to browser GPU; not end-to-end zero-copy"}

    def _run(self):
        last_revision = 0
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
                        self.error = None
                        self.finished = False
                        changed = True
                    if not paused:
                        obs = self.env.observe()
                        action, _, _, _, self.hidden = self.policy.act(obs, self.hidden, self.env.graph, deterministic=True)
                        _, _, terminated, truncated, self.info = self.env.step(action)
                        if self.env.device.type == "cuda":
                            torch.cuda.synchronize(self.env.device)
                        self.step_ms = (time.monotonic() - started) * 1000
                        changed = True
                        if bool((terminated | truncated).any()):
                            with self.lock:
                                self.paused = True
                                self.finished = True
                            paused = True
                    if changed:
                        with self.lock:
                            self.sequence += 1
                            seq = self.sequence
                        frame = pack_snapshot(self.env, self.hidden, self.info, seq, paused, revision, self.step_ms)
                        with self.lock:
                            self.snapshot = frame
                last_revision = revision
            except Exception as exc:
                with self.lock:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.paused = True
                    # Publish the paused/error state without advancing the failed physics.
                    header = bytearray(self.snapshot[:HEADER_BYTES])
                    self.sequence += 1
                    struct.pack_into("<I", header, 16, self.sequence)
                    struct.pack_into("<I", header, 20, struct.unpack_from("<I", header, 20)[0] | 1)
                    self.snapshot = bytes(header) + self.snapshot[HEADER_BYTES:]
                paused = True
            delay = 0.04 if paused else max(0.001, self.env.cfg.dt/speed - (time.monotonic()-started))
            self.stop_event.wait(delay)

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=10)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, simulation: Simulation):
        self.simulation = simulation
        super().__init__(address, ViewerHandler)

    def server_close(self):
        super().server_close()
        self.simulation.close()


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer
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
            seq = struct.unpack_from("<I", payload, 16)[0]
            tag = f'"{seq}"'
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
                self._reply(404, b'{"error":"Viewer assets missing; run from repository checkout"}')
                return
            types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".woff": "font/woff"}
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
            result = self.server.simulation.command(body)
            self._reply(200, json.dumps(result).encode())
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._reply(400, json.dumps({"error": str(exc)}).encode())


def create_server(config: EnvConfig | None = None, checkpoint=None, device="cpu", port=8765,
                  host="127.0.0.1", untrained=False) -> ViewerServer:
    simulation = Simulation(config, checkpoint, device, untrained)
    try:
        return ViewerServer((host, port), simulation)
    except Exception:
        simulation.close()
        raise


def serve(config: EnvConfig | None = None, checkpoint=None, device="cpu", port=8765,
          host="127.0.0.1", untrained=False):
    with create_server(config, checkpoint, device, port, host, untrained) as server:
        print(f"NPA Swarm viewer: http://{host}:{server.server_address[1]} (inference only)", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
