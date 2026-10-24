# v5 cloud handoff — structural RL research revision

Source parent: `8633359eef3e6dcbd3830b5864336f5023b1e6cc`.

The previous v4 local reward/conditional PPO optimization failed to
maintain contact or move cargo (reported in `reports/local-v4/README.md`).
Changing the magnitude of sparse contact rewards again would not address
the loss of previously learned search behavior or the 900-agent dilution.

This revision leaves the **full fixed 900-agent unknown-endpoint search,
overdamped cargo physics, contact requirements, communication radius, soft
connectivity constraints and WebGPU viewer unchanged**. It introduces a
separate *optional* `--ctde-task` learner:

- A zero-initialized four-coefficient, shared, locally conditioned action
  residual learns cargo approach, cargo-to-goal orientation, rear contact
  geometry, and graph-fringe cohesion from actor-local cues. All four
  coefficients are learned; no scripted role, tether, force or reconnection.
- The old search actor/recurrent trunk is frozen to prevent replaying the
  observed search-to-no-contact collapse. An existing **direct search** A20
  checkpoint initializes it; this is not contact/weight curriculum.
- A 12-feature centralized value model sees privileged global task state
  **only during training**. Team GAE and particle-local advantage residuals
  are normalized separately. Physical cargo witnesses and graph-fringe
  samples have separate importance normalization (70%/30% when both present).
- Training mode, parent checkpoint SHA, separate team critic/optimizer and
  all RNG/world state are saved. Older schema-3 training files lacking
  `learning_mode` remain explicitly `local`, rather than silently adopting
  CTDE. The new head has zero initial residual for a source-preserving
  warmstart; conditional mode is not used by old checkpoints.
- No new frontend, physics branch, CI workflow or dependency.

## Tests actually performed in this cloud container

Container: Torch 2.10.0+cpu, **no CUDA**. Five focused numerical
tests on the independent `ctde.py` helper equivalent passed in 1.26s:
12-feature team critic forward/backward, translation invariance, sparse
advantage normalization, empty/singleton witness masks, and separate
witness/fringe weight normalization. A synthetic PPO update also produced
finite gradients and changed its local task head. Agent-relative
cargo/goal/rear vector basis was checked for directional consistency.

The entire checked-out project and GPU/device tests **have not** run here
because repository checkout/network access is unavailable. The new
repository tests in `tests/test_learning.py`/ `test_cli.py` are authored,
but **not represented as passed**. They must be run once locally before
CUDA training. A successful PPO step, frozen transport, all-agent connectivity
and WebGPU performance under v5 are NOT yet established.
