# Task v2: unknown endpoints, collective push

## Fixed first baseline

900 overdamped disk agents in a 20×20 arena. Agent radius 0.06; communication radius 0.48; local sensing extends 1.2 beyond endpoint surfaces. The connected initial lattice has spacing 0.144 and random rotation. Random-layout cargo/goal centers are on radius 7.4 with angular separation at least 90°, so their center separation is 10.47–14.8. Both are outside initial sensing. `--layout diagonal` places endpoints at opposite (±7.4, ±7.4) corners, separation 20.93, for a separately reported stress evaluation. No curriculum modifies object difficulty.

No globally observed coordinates, ID, full map, discovered flags, path labels, rewards or privileged target direction enter the actor. Position exists in the simulator and visualization only. Local relative vectors share the simulation axes; absence of absolute position is NOT a claim of a compass-free, strictly rotation-equivariant embodied robot model.

## Physics and motion

`dt=0.1`, eight substeps, force limit 1, agent drag 1, free speed cap 1, contact stiffness 60. Cargo radius 0.65, drag 12; destination radius 1. No static-friction threshold in the default baseline. Cargo velocity is resultant contact force divided by drag; positions advance through this physical law, not a neural coordinate update. These are simplified overdamped dynamics, not an inertial rigid-body simulator. Reactions and walls remain physical constraints, never connectivity control.

A single steady aligned pusher has approximately 1/(12+1) speed in the simple one-contact case. Several aligned pushers can be faster, but must actually maintain contact and avoid cancellation. Merely removing a deadzone does not teach coordination. Default stiffness was reduced with drag to keep the substep integration reasonable; local hardware should preserve numerics when optimizing, not take arbitrary larger physical timesteps to claim speed.

## Rewards (training supervision only)

- Main task: `3 * (previous cargo-goal distance - current distance)`, plus 5 on whole-cargo arrival. This distance reward already existed in v1; the previous static threshold blocked its displacement signal in failed search episodes.
- Each endpoint's first discovery pays 1 to the team; it is not a persistent sighting bonus.
- Work credit uses each particle's signed actual contact force dotted with the actual cargo displacement, normalized by total absolute work and multiplied by the signed goal-distance progress. No displacement means zero extra credit, including a cargo blocked by a wall. It is enabled after the goal has been discovered; it does not inject a goal vector into the actor.
- Detached-agent cost is `0.15 * dt * (1 + min(detached_seconds/5, 1))` per agent outside the largest component. It does not kill, teleport or constrain that agent. Largest-component ties use deterministic label ordering; the actor does not receive labels.
- Signed reconnection progress is `0.2 * (log1p(old_gap/comm_radius) - log1p(new_gap/comm_radius))`. Gap means nearest distance to the main component's communication envelope, **not MST length or the globally shortest repair route**. Distances are chunked and only calculated in disconnected worlds.
- After both endpoints were discovered, a signed reduction of the main component's two endpoint gaps is weighted by 0.25, and absent live bridge costs 0.05 per simulated second. This helps connect useful endpoints without requiring the whole swarm to stretch to a predetermined shape.
- A new spatial coverage cell pays 0.2 ONCE, divided among simultaneous visitors from the main component; detached agents cannot repeatedly farm coverage. Local approach progress is weighted by 0.05 and effort by 0.0005 per second.

These auxiliary differences are practical shaping, not a proof of policy-invariant reward transformation. Watch for stationary clustering, reward farming and bridge/transport tradeoffs. Do not fix them solely by repeatedly increasing penalties.

## Connectivity and success

Agent edges are complete radius-0.48 edges, with no degree cap. A live bridge exists only when the SAME particle component has at least one agent within communication radius of the cargo surface and at least one within that radius of the goal surface. Endpoint nodes do not join otherwise disconnected agent components. Seeing either endpoint at sensing radius 1.2 is not enough.

Arrival requires the complete cargo disk inside the destination (`center distance <= 1-0.65`). Report transport success, whether every sampled control state was connected, and bridge fraction after both discoveries separately. Current connectivity samples are control-step endpoints: no continuous-time guarantee between them. Detached particles remain present and can rejoin. The episode ends on delivery or 1200 control steps, not on detachment.

900 particles and the revised radius are a starting engineering choice, not a theorem that the task is learnable. Full search success must be shown by training and frozen evaluation. The contact placement remains only for physical/software diagnostics; it is not the prescribed training path.

## Local information relay (v3 research candidate)

A particle seeing cargo or destination can start a six-channel, agent-relative message (object/goal confidence plus local bearing). On each control step, neighbors can forward only the **previous** step's message across an actual communication edge; their relative edge displacement updates the bearing. Message confidence decays at each hop and in disconnected memory. This solves a communication bottleneck without granting an oracle global target direction: no local witness and no connected route means no remote target information. The relay never acts, pulls, reconnects, assigns roles or chooses a route for the agents. The shared neural actor learns coefficients for these local bearings along with the original learned recurrent messages. Training reward, environment geometry, 900 population, no-course baseline and WebGPU are otherwise unchanged. This is a **hypothesis for training**, not a claim of learned recruitment or arrival.

The model adds six non-learned relay-state channels outside the existing 32 learned recurrent channels. Checkpoint schema **3** is not compatible with schema 2/v2 weights; retain the prior checkpoint and use a new run directory. No old model comparisons may be claimed under unchanged task identifiers.
