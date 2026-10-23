"""Bounded API driver while a real visible browser renders the trained viewer.

This is an engineering soak, not frozen-policy research evaluation. It never
claims hardware graphics from HTTP alone; browser evidence is collected apart.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def request(path, body=None, binary=False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.url + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read() if binary else json.load(response)
    def status():
        value = request("/api/status")
        if value["checkpoint"].get("checkpoint_sha256") != args.checkpoint_sha256:
            raise RuntimeError("Viewer checkpoint identity changed; refusing to control it")
        if value["error"]:
            raise RuntimeError(value["error"])
        return value
    def command(name, value=None):
        status()
        return request("/api/control", {"command": name, "value": value})
    result = {"kind": "trained_viewer_engineering_soak", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "requested_seconds": args.seconds, "checkpoint_sha256": args.checkpoint_sha256,
              "research_task_passed": False, "samples": [], "terminal_resets": 0}
    start = time.monotonic()
    last_sample = -10
    try:
        status()
        command("speed", .5)
        command("play")
        while time.monotonic() - start < args.seconds:
            current = status()
            elapsed = time.monotonic() - start
            if elapsed - last_sample >= 10:
                data = request("/api/snapshot", binary=True)
                header = struct.unpack_from("<8I", data)
                result["samples"].append({"elapsed_seconds": elapsed, "sequence": header[4],
                    "step": header[6], "flags": header[5], "step_ms": current["step_ms"],
                    "finished": current["finished"]})
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(result["samples"][-1]), flush=True)
                last_sample = elapsed
            if current["finished"]:
                command("reset")
                for _ in range(100):
                    if not status()["finished"]:
                        break
                    time.sleep(.05)
                command("play")
                result["terminal_resets"] += 1
            time.sleep(.5)
        command("pause")
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
