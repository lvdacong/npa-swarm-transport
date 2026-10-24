"""One frozen full-size search rollout; quantify reward's actual actor gradient."""
import json
import argparse
a=argparse.ArgumentParser();a.add_argument("checkpoint");a.add_argument("output");args_cli=a.parse_args()
from pathlib import Path
import torch
from npa_swarm.env import rear_arc_gap
from npa_swarm.train import Trainer
from npa_swarm.policy import squashed_log_prob

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
t = Trainer.load(args_cli.checkpoint, 'cuda:0')
rear_rewards = []
main_rewards = []
original = t.env.step
def step(action, goal_relay=None):
    e, c = t.env, t.env.cfg
    args = (c.object_radius+c.agent_radius, c.rear_potential_version, c.sense_radius+c.object_radius+c.agent_radius)
    before = rear_arc_gap(e.positions, e.object_pos, e.goal_pos, *args)
    seen = e._object_visible.clone()
    result = original(action, goal_relay=goal_relay)
    after = rear_arc_gap(e.positions, e.object_pos, e.goal_pos, *args)
    reward = c.rear_approach_weight*(before-after)*(goal_relay/.2).clamp(0,1)*seen*e._object_visible*e.rear_membership()
    torch.testing.assert_close(reward.sum(-1), result[4]['rear_reward_sum'])
    rear_rewards.append(reward)
    main_rewards.append(reward*e._in_main)
    return result
t.env.step = step
r = t.collect()
contribution = torch.zeros_like(r.advantages)
carry = torch.zeros_like(r.advantages[0])
for i in reversed(range(len(rear_rewards))):
    carry = rear_rewards[i] + t.cfg.gamma*t.cfg.gae_lambda*(~r.done[i])[...,None]*carry
    contribution[i] = carry
def norm(a): return (a-a.mean())/(a.std(unbiased=False)+1e-8)
full, ablated = norm(r.advantages), norm(r.advantages-contribution)
params = (t.policy.actor.weight, t.policy.actor.bias)
totals = {name: torch.zeros(sum(p.numel() for p in params),device=t.device) for name in ('full','without_rear','active')}
h = r.initial_hidden
for i, (obs,g,z) in enumerate(zip(r.observations,r.graphs,r.latent_actions)):
    mean,ls,_,h = t.policy(obs,h.detach(),g)
    lp = squashed_log_prob(mean,ls,z)
    for name,adv in (('full',full[i]),('without_rear',ablated[i]),('active',full[i]*r.rear_active[i])):
        grad = torch.autograd.grad(-(lp*adv).mean()/len(rear_rewards), params, retain_graph=True)
        totals[name] += torch.cat([p.flatten() for p in grad]).detach()
    h = h * (~r.done[i])[...,None,None]
result = dict(scope='900x2 frozen A20 warmstart, 384 direct-search steps, no optimization; reward ablated on identical states/actions',
    checkpoint=args_cli.checkpoint, eligibility=t.env.cfg.rear_eligibility, main_reward_abs_sum=float(torch.stack(main_rewards).abs().sum()), neighbors_reward_abs_sum=float(torch.stack(rear_rewards).abs().sum()), metrics=r.metrics, actor_gradient_norms={k:float(v.norm()) for k,v in totals.items()},
    rear_gradient_change_norm=float((totals['full']-totals['without_rear']).norm()),
    rear_advantage_contribution_norm=float(contribution.norm()),
    all_advantage_norm=float(r.advantages.norm()))
Path(args_cli.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
