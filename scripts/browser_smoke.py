"""Bounded browser integration probe against the project's own local server.

No trained-task-success claim. Run with `python scripts/browser_smoke.py`.
"""
import argparse
import json
import io
import os
import threading
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from npa_swarm.config import EnvConfig
from npa_swarm.server import create_server


def run(output, n_agents=4096, renderer="canvas"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    server = create_server(EnvConfig(n_agents=n_agents, episode_steps=30), untrained=True, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors, console_errors = [], []
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=os.environ.get("SWARM_BROWSER_EXECUTABLE"), args=["--enable-unsafe-webgpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"])
            page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.goto(base + f"/?renderer={renderer}")
            page.wait_for_function("window.__swarm?.diagnostics.particles === " + str(n_agents) + " && window.__swarm.diagnostics.renderedFrames > 0", timeout=45000)
            page.evaluate("document.fonts.load('16px \"Swarm CJK\"')")
            initial = page.evaluate("window.__swarm.diagnostics")
            assert initial["particles"] == n_agents and initial["step"] == 0
            assert "未训练策略" in page.locator("#training-notice").inner_text()
            if renderer == "webgpu":
                assert initial["backend"] == "WebGPU", initial
            # Pixel evidence: at least many varying pixels; do not accept a blank canvas.
            scene_png = page.locator("#scene").screenshot()
            pixels = np.asarray(Image.open(io.BytesIO(scene_png)).convert("RGB")).astype(np.int32)
            teal = (pixels[..., 1] > 100) & (pixels[..., 1] > pixels[..., 0] * 1.25) & (pixels[..., 2] > pixels[..., 0] * 1.1)
            orange = (pixels[..., 0] > 120) & (pixels[..., 0] > pixels[..., 1] * 1.15) & (pixels[..., 1] > pixels[..., 2] * 1.2)
            pixel_counts = {"teal": int(teal.sum()), "orange": int(orange.sum())}
            assert pixel_counts["teal"] > min(n_agents, 1000) and pixel_counts["orange"] > 80, pixel_counts
            page.screenshot(path=str(output / f"{renderer}-initial.png"), full_page=True)
            page.locator("#play").click()
            page.wait_for_function("window.__swarm.diagnostics.step >= 2", timeout=45000)
            page.locator("#play").click()
            page.wait_for_function("window.__swarm.diagnostics.paused", timeout=15000)
            playing = page.evaluate("window.__swarm.diagnostics")
            position = page.evaluate("""async () => {
              const b=await (await fetch('/api/snapshot')).arrayBuffer();
              const u=new Uint32Array(b,0,8),f=new Float32Array(b,32,24),d=new Float32Array(b,128);
              const id=Math.floor(u[2]/2),r=document.getElementById('scene').getBoundingClientRect();
              const a=r.width/r.height,half=f[7]*1.08/Math.min(a,1);
              return {x:r.width/2+d[id*8]*r.width/(2*half*a),y:r.height/2-d[id*8+1]*r.height/(2*half),id};
            }""")
            page.locator("#scene").click(position={"x": position["x"], "y": position["y"]})
            page.wait_for_function("window.__swarm.diagnostics.selected === " + str(position["id"]))
            page.locator("#scene").hover()
            page.mouse.wheel(0, -300)
            page.locator("#fit").click()
            page.locator("#reset").click()
            page.wait_for_function("window.__swarm.diagnostics.step === 0 && window.__swarm.diagnostics.sequence > " + str(playing["sequence"]), timeout=30000)
            # Exercise renderer replacement after an initialized context.
            other = "canvas" if renderer == "webgpu" else "auto"
            page.select_option("#renderer", other)
            page.wait_for_function("window.__swarm.diagnostics.backend.length > 0")
            page.set_viewport_size({"width": 430, "height": 900})
            page.screenshot(path=str(output / f"{renderer}-mobile.png"), full_page=True)
            final = page.evaluate("window.__swarm.diagnostics")
            result = {"n_agents": n_agents, "requested_renderer": renderer,
                      "initial": initial, "after_play": playing, "final": final,
                      "page_errors": errors, "console_errors": console_errors, "scene_pixels": pixel_counts,
                      "physical_gpu_validated": False,
                      "trained_transport_validated": False,
                      "scope": "real headless Chromium, software graphics, local inference/control/snapshot/render integration"}
            (output / f"{renderer}-observed.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            assert not errors, errors
            assert not final["errors"], final["errors"]
            assert not console_errors, console_errors
            (output / f"{renderer}-browser.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            browser.close()
            return result
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/browser-smoke")
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--renderer", choices=["canvas", "webgpu", "auto"], default="canvas")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.output, args.particles, args.renderer), ensure_ascii=False))
    except Exception as exc:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / f"{args.renderer}-failure.json").write_text(json.dumps({"error": str(exc), "physical_gpu_validated": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
