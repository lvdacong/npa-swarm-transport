# Current entrypoints

Use `python -m npa_swarm` for probe/train/evaluate/serve, and `browser_smoke.py` for the v2 WebGPU-only interface. `analyze_trajectories.py` remains compatible with evaluation arrays. The earlier dense_benchmark/local_benchmark/profile_checkpoint/viewer_soak scripts are v1 research helpers; use them at `8e07995` for historical reproduction, not as mandatory v2 work. Do not run their old 4096/renderer-menu/curriculum presets on v2. No script starts scheduled tasks.
