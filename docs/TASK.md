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

## Informed rear-arc learning reward (next study; no curriculum)

Issue #4 established that a correctly aligned goal message may reach almost every
contactor while the trained policy still pushes away from the goal. The next
minimal, train-only intervention is a **signed per-agent change in rear-sector
distance** for agents that directly *see* cargo, belong to the largest live
communication component, and have received an actual local goal message.
`rear_arc_gap` is defined by cargo-relative orientation, contact-rim distance
and a continuous rear-sector weighting; approaching cargo radially from the
**front** does not earn the bonus. It is a change in potential, not a per-step
standing bonus. Simulator truth for cargo and destination is used ONLY to
compute that reward. The actor still sees exactly the local 20 observations,
shared recurrent state and one-hop local goal messages. No teacher action,
role, privileged target coordinate in NN, scripted pushing, connectivity shield,
teleport, agent deletion or easier reset distribution is introduced.

Under the same new-reward switch, true cargo-distance improvement is credited
only to local witnesses, with amplification capped at 16, rather than
broadcasting the same motion event to all 900 uninformed actions. Actual
physics, terminal success, discovery, exploration and graph objectives are
unchanged. The default new experiment uses `rear_approach_weight=0.8`
and trained relay memory decay `.997` (chosen because it greatly improved
same-weight contactor message reach, **not** because prior B20 RL learned).
A value of **0** keeps the original v3 progress broadcast and no rear reward.
Earlier schema-3 checkpoints that lack this new field restore with 0.0;
their saved model, reward and optimiser meanings must not silently change.
A new v3 training run may warmstart the preserved A20 weights with the new
reward and retention but must record its parent hash, new optimizer,
separate output directory and new evaluation denominators. Historical frozen
B/C failures must not be reported as successes.


## Issue #5 本机修正

新rear势版本3修正远侧错误方向与前侧角向平台；版本1/2仅供旧检查点准确恢复。默认最大分量资格仍main；显式neighbors实验仅允许具有真实邻居的非孤立粒子获rear信用，不改变连通代价/严格桥/完成标准。可选local_batch_fraction混合全体与局部活跃样本PPO及critic目标，默认0，空集/单例回退；它只作用于训练，不改变局部actor观察。实验没有学会搬运，详见reports/local-v4。


## CTDE task-head experiment (research revision)

Repeated reward shaping and conditional replay weighting failed to keep contact.
A short-lived local reward cannot compensate for a collapsed base explorer,
and the earlier rollouts only covered ~2 complete training episodes in v4.
The new **optional** `--ctde-task` training mode changes learning structure,
**not** the environment or particle action limits:

- **Training-only centralized team critic.** It sees 12 normalized invariant
  world metrics (distance, discovery, connectivity, contact and cargo motion).
  The critic predicts long-term cooperative return, while agent-local GAE
  residuals give relative action credit. This global state and critic are
  *never* sent to the actor or browser inference.
- **Preserve inherited navigation.** The old recurrent local backbone,
  action head, entropy and message relay are frozen. A tiny zero-initialized
  4-coefficient *trainable* local task head adds goal-conditioned action
  residuals using only previously received cargo/goal-relative bearings and
  a local physical cargo contact radius, plus an independent learned
  neighbor-centroid coefficient at thin communication fringes. That fourth
  coefficient is also initialized to zero, and its sign is trained rather
  than hard-coded; it permits learning graph cohesion without sacrificing
  center-of-swarm exploration. It is not a scripted pushing
  controller: there are no prescribed forces, roles or guaranteed connections.
  With zero residual, the old explorer is mathematically unchanged.
- **Focus optimization on real witnesses.** PPO policy terms are normalized
  over samples that actually have a local cargo sighting, rather than 900
  mostly irrelevant agents. The joint team objective uses actual delivery,
  signed cargo progress, discovery, bridge/connectivity, and exploration.
  Rare actor-local GAE residuals remain available, with a *separate*
  centralized baseline. A completely empty witness batch updates only the
  team critic. No synthetic successes or new partial resets.
- It is a **full unknown-endpoint, fixed 900-agent search task throughout**;
  no contact/wall/force change, contact→search or light-object curriculum,
  oracle NN inputs, deletion, shield or scripted recruitment.

New mode requires a clear A20 direct-search warmstart from the preserved v3
model. This only initializes the *old search policy*, NOT a pre-trained
transport policy. The new task head/critic/optimizer start fresh; history
and weights are preserved. Legacy schema-3 checkpoints without the new
train-config `learning_mode` retain the original `local` optimizer and
action semantics for strict resume and playback. Train in a new directory.
The CTDE option is a research hypothesis; success remains unknown until
frozen full-length rollout and connectivity checks.
