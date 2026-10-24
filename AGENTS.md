# Working agreement — task v2

User-authorized autonomous implementation, profiling, training and evaluation in this repository. No approval gates or artificial frozen-file rules. Protect unrelated running tasks and existing modifications; leave the independent NPA lizard unchanged. Never import the old Swarm modules.

## Current task (supersedes earlier 4096/contact requirements)

- Default **900 agents**, <=1000 preferred. Jointly tune density, communication/sensing range and measured compute; do not silently use a tiny population to claim success.
- One unknown-object/unknown-destination search-and-push task; **no mandatory contact curriculum or light-to-heavy curriculum**.
- No absolute coordinates, IDs, global map or global target vector in policy input. Local relative observations, memory and one-hop messages allowed.
- No action shield, scripted recruiting, fixed roles, automatic reconnection or deletion/termination of detached agents. Use soft costs and signed geometric progress.
- Whole-cargo arrival, full swarm connectivity and live endpoint bridge are different measures. A bridge is a particle component attaching within communication radius of both surfaces, not just seeing both.
- v2 changes physics/population/model/checkpoint schema. Preserve v1 outcomes; replay v1 at `8e07995`, never pass old weights off as v2 or silently relax a resume.

## Efficient workflow

Use the current Issue and docs/LOCAL_HANDOFF.md. Reuse already working Windows/CUDA installations. Cloud-verified CPU work is not to be duplicated by Linux CI. CI is manual Windows only; no automatic matrix or repeat browser suites. Run targeted tests for actual changes, then bounded GPU feedback. Keep success/failure/timeout/error denominators and necessary trajectories; do not generate a large secondary evidence framework.

Improve actual transport and usable viewing speed, not GPU utilization percentages or test counts. Profile only the observed bottleneck. Do not use cold/profiler timings as steady throughput. Do not mechanically add training seeds or fill budgets without mechanism progress. A first observation window is not a convergence deadline; changes need a reason, not random reward-weight rotation.

Codex may modify algorithms, rewards and settings based on evidence, then revalidate the changed behavior. Return code, actual hardware, configuration, useful trajectories, throughput and limitations in the same Issue. A successor Issue is warranted only for a material unresolved problem, not routine fixes. No scheduled tasks.


v3 research: local beacons are a one-hop geometry-corrected communication primitive, never privileged world observations or a scripted controller. All neural action coefficients remain learned. New model schema 3 requires a new search run; do not overwrite the preserved v2 checkpoints. Prefer targeted CUDA and relay tests, no large CI matrix.


Next train-only rear-sector reward is simulator-privileged **reward computation**, not an NN observation/action, scripted agent, controller or curriculum. `rear_approach_weight=0` retains old v3 reward semantics; old checkpoints missing this config field must load as 0. Keep physical task unchanged and run only focused hardware validation, never duplicate broad CI.
