# Cloud revision: task v2 (2026-10-23)

Base: `8e07995cb8ad348950ed6b0d3b56538955bc9a7b`. This is a new task/model baseline, not continuation or reproduction of the v1 research result. Existing reports/weights remain historical evidence.

## What was actually executed

- `python -m pytest -q --junitxml=reports/cloud-revision/pytest.xml`: **45 passed, 4 skipped in 3.17s**. Skips require physical CUDA. Raw logs are retained; publication-pytest.xml is the final suite XML. Counts are one suite, not summed overlapping runs.
- One real default PPO update with **900 agents, 128 rollout steps, 32-step TBPTT, three epochs**: **5.4386s**, CPU; zero complete episodes, zero successes. Peak process RSS 572,764 KiB includes diagnostic setup. Not a GPU benchmark or convergence result. `default900.json` records all fields and runtime SHA-256 `024689173d28542db2cb2cc68f25fa97abaedb029ad34256377c55e9b13fdaf0`.
- Exact CPU full-checkpoint next-update comparison across resets; eager vs reused bootstrap equality; one-hop message causality/backpropagation and particle permutation; persistent learned-vector representation; no absolute coordinates under interior translation; graph identity/dense completeness; bridge counterexamples; detached survival/no action modification; coverage splitting; actual-work reward with blocked-wall negative example. See tests.
- Fixed-action physical diagnostic at 0.8 simulated seconds: one pusher moved cargo **0.06051**, eight pushers **0.20625**. At 3 seconds the same eight fixed-direction pushers had slipped off, giving **0.33468** vs one **0.22974**. This deliberately retained long-window loss of contact shows that force availability does NOT prove learned recruitment or contact maintenance. `physics.json` includes 8- and 16-substep traces.
- JavaScript syntax check passed. Static HTML/CSS in actual Chromium checked at 1440×900 and 390×844 with no horizontal overflow. HUD values in this static check were manually populated; it is not particle rendering.
- Full WebGPU browser check was **BLOCKED before navigation**, `net::ERR_BLOCKED_BY_ADMINISTRATOR` on local HTTP, under the cloud container's managed Chromium policy. No policy was changed. Raw `browser/browser.json` preserves the failure. No browser GPU/shader/pixel pass is claimed for this revision. The real executable smoke script is supplied for local Chrome/Edge.
- **No GitHub Actions run was launched.** Automatic matrices were replaced with an optional manually triggered Windows job. No repeated Linux CI.

## Relevant findings and changes

The previous distance reward already existed. Static friction 4 prevented the low-force search episodes from moving the cargo, and pure contact-impulse credit could pay despite no displacement. v2 eliminates the default deadzone and pays movement-conditioned work. Simplified drag/stiffness/speed/geometry have changed, explicitly documented in docs/TASK.md.

The old sensor-visible bridge could be true without agents near endpoint communication range. New fixtures reproduce that distinction and require the same particle component attaching at both surfaces. Endpoint bodies cannot join disconnected components.

The vector-only v1 action head combined CURRENT local vectors (including velocity, which can implicitly carry direction), without dedicated persistent vector slots. Generic learned vector memory now permits a direction when a cue is absent, with one-hop neighbor updates. This is a new learnable mechanism, not a coded target-direction flood fill and not proof of useful learned communication.

Actual simulation and display publication are decoupled, metadata+particles use one host transfer per snapshot, frames/bind groups reuse GPU resources, and WebGPU-only UI hides diagnostics by default. End-to-end CUDA/WebGPU zero-copy is NOT implemented or claimed.

## Preserved failed checks and scope

The first CPU regression run had 39 passes and one incorrect *diagnostic expectation*: it demanded >2× displacement after three seconds with eight fixed-x pushers, but those pushers slid away. The corrected physical feasibility check uses an explicit 0.8-second maintained-contact window; both short and long traces remain in physics.json, and the first failure log is retained as core-tests.log. We did not modify physics to conceal the failure or interpret the fixed-action test as an RL result.

Full new task learning, physical CUDA, steady laptop throughput, dense/disconnected-world performance, and the new hardware WebGPU interface remain unvalidated. Previous v1 hardware success does not close those new deltas. See docs/LOCAL_HANDOFF.md for a bounded, single-line continuation without curriculum or repetitive CI.

## Publication check

The final local suite remains **45 passed / 4 CUDA skipped (3.27s)**, recorded in publication-pytest.log and publication-pytest.xml. Final Python runtime digest is `28eb0fddbe032597722f4e3f9aeb9e7d259133b1d72088cdd968f7bb57168842`. Compared with the earlier default900.json snapshot, the Python difference is only the equivalent constructor alias `c = config` to `c = self.cfg`; old probe evidence retains its original digest and timing. The frontend control queue now computes play/speed toggles when executing, rather than capturing stale values during rapid clicks. Node syntax passed; no actual WebGPU browser pass is added. No further training or CI was run for these publication checks.
