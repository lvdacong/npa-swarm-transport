# Project working agreement

The user authorizes Codex to autonomously implement, modify, profile, train, evaluate and fix this project, including changing algorithms/configuration when real evidence warrants it. No artificial approval gates, frozen-file rules, periodic tasks or mandatory repetition budgets are imposed by this repository.

## Goal

Deliver a minimal NPA-inspired local-message RL swarm that finds an unknown heavy object and destination, maintains useful local connections and physically pushes the object into the destination. Formal target is approximately 4096 particles, chosen jointly with sensing radius/density and measured resources. Keep the existing independent lizard project unchanged; do not import old Swarm modules.

## Design decisions

- No absolute coordinates/global target vector/map/IDs in NN input. Local relative vectors, proprioception, contacts and learned messages/memory are allowed.
- No connectivity shield, automated reconnect, scripted agent controller or fixed roles. Physical collisions, walls and actuator bounds remain ordinary environmental dynamics.
- Start with soft connectivity penalty without terminating disconnected episodes. Report transport, sampled full connectivity and endpoint bridge separately.
- Contact warm-up, small CPU diagnostics and search evaluation must be named accurately. Keep failures, censored episodes and requested denominators.
- Prefer a single working implementation and targeted measurements. If hardware-only checks are unavailable, report that directly instead of pretending CPU/software rendering proves hardware results.

## Workflow

Use the current handoff Issue and `docs/LOCAL_HANDOFF.md`. Work through real feedback, fix issues directly, keep raw evidence and report relevant results in that same Issue. A material new unresolved problem may warrant closing the old handoff and publishing a specific replacement; routine fixes do not require new Issues.

All training counts are cumulative. `resume` restores state; `warmstart` is a new run with parent identity. Choose resource-efficient settings from measured throughput and memory. The initial 50-update command is a diagnostic segment, not a directive to fill a budget blindly. Existing unrelated training should be observed before scheduling competing hardware work.

Tests are `python -m pytest -q`; browser smoke is documented in `docs/LOCAL_HANDOFF.md`. Record exact runtime versions and revision. Changes to task semantics or evaluation criteria must be explicit in reports.
