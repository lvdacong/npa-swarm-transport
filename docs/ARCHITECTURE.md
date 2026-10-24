# v2 implementation

`SwarmEnv -> local observations + radius graph -> LocalPolicy -> force -> physics`.

## Local learned mechanism

20 observation channels retain velocity, neighbor-relative centroid/mean velocity/degree, locally visible endpoint vectors and visibility, contact normal/contact flag, and locally clipped wall ranges. No global information or connectivity labels are input. Observations cache until the world changes; position/velocity aggregation shares one operation.

Default memory width 32 contains 28 GRU scalars plus two learned 2D vector slots. Shared learned coefficients write and gate a mixture of six currently available local vectors, old own vector memory, and neighboring old vector memory. The actor learns coefficients of current local vectors and updated vector slots. All initial vectors are zero; no unconditional world-axis action bias, predefined destination channel or routing controller exists. Vector memory permits a direction after a local cue disappears, but its useful meaning must be learned. Each update reads OLD neighbor state, so information travels at most one graph hop per control step.

The sorted-cell radius graph and canonical CUDA CSR aggregation from the local OOM repair are retained. No N×N hidden-message tensor or silent neighbor truncation. Component calculation is exact at sampled states. Reward-only repair gaps use chunked distances, which may need local profiling if disconnection makes them expensive.

The local CUDA runtime captures only the unchanged eight physics substeps in edge-size buckets. Zero-force self edges pad unused capacity; larger complete graphs allocate a larger bucket. CPU and an explicit eager CUDA diagnostic use the same integrator. Repair gaps query detached rows only. These execution changes do not change the timestep, physics, observation or neighbor semantics.

## Recurrent PPO

Default 128-step rollout, 32-step truncated backpropagation, three epochs, gamma .999. Whole worlds/ordered time are replayed; particle/time samples are not randomly detached from communication histories. Ordinary next-state values reuse the next forward pass. Timeouts explicitly bootstrap the PRE-reset state, true terminals do not, and GAE never crosses an episode boundary. A regression compares this with eager per-step bootstrapping through resets.

Full checkpoint v2 contains policy, optimizer, recurrent carry, world, RNG and cumulative counters. Failed load restores RNG. v1 checkpoints are rejected with a message; history remains replayable at the old commit. v2 warmstart is explicitly a new run with lineage. Candidate selection uses realized rollout distance progress, not a smaller absolute initial layout distance. Independent frozen evaluation still determines quality.

## WebGPU-only viewer

One native inference worker, newest-only immutable binary snapshots, ETag for unchanged data. Publication is capped at 30Hz independently of simulation speed; commands and terminal states publish immediately. Dynamic header and particle display payload use a single device-to-host transfer per publication. The browser creates TypedArray views, reuses two GPU storage buffers and cached bind groups, interpolates in the vertex shader, coalesces frames, and draws on requestAnimationFrame only when changed/interpolating. Sparse GPU timestamps are diagnostic-only.

This architecture still has **CUDA -> host -> HTTP -> browser GPU** copies. It is not CUDA/WebGPU shared-memory zero-copy. Eliminating that boundary would require a different execution/integration backend; do not build a second simulator merely to claim it. At 900 agents each packet is 28,928 bytes. No per-particle DOM nodes. No Canvas or renderer selector. Unsupported or lost WebGPU is explicit; device loss pauses the local simulation.

Controls cycle requested 1×/4×/8×/unpaced (∞); the diagnostics show actual simulated-seconds/wall-second separately. Neither rAF nor draw submission rate is monitor presentation FPS. A low-power adapter preference is a hint, not an AMD device guarantee. Actual browser/torch devices must be recorded on the laptop.

Windows Python 3.12 may implement `monotonic()` using coarse GetTickCount64. Playback therefore uses high-resolution `perf_counter()` and accumulated deadlines, carrying wait overshoot into the following step. Status also records simulated seconds, active wall seconds and their cumulative ratio, excluding pauses and resetting with the scene.

## State-driven WebGPU cues

The viewer retains the same 32-byte particle record and one device-to-host snapshot transfer. Its four signal floats now mean [cargo relay confidence, destination relay confidence, actual signed projected physical contact force normalized by actuator, exact integer bit flags]. Flags: bit0 largest component, bit1 direct cargo sighting, bit2 direct destination sighting, bit3 physical contact, bit4 membership in live two-endpoint bridge. Direct witnesses are displayed at full confidence; relays fade according to their **actual** stored message state. The scene uses amber for cargo messages, periwinkle for destination messages, mint for both, golden yellow for helpful contact force, coral for force against the destination and desaturated slate for detached particles. A subtle radius pulse identifies **direct sightings only**. A concise legend appears in the existing optional diagnostics, not as a new mode or a persistent text overlay. Picking a particle shows its real confidence, contact direction, direct sensor flags and connectivity. These colours are visualization only; privileged force/goal comparison is never fed to NN policy, reward, or physics. WebGPU instancing / interpolated snapshots / binary size stay unchanged; hardware shader compilation and actual performance need real-browser verification.

Important: the colour of a contacting particle visualizes its **individual instantaneous signed contact force**, not the net cargo force, cargo displacement, or a completed RL task. Opposing particles can cancel. Success remains the whole cargo inside the goal disk.
