"""Reward-only actual-work credit; no threshold deadzone, no static-force bonus."""
import torch
from npa_swarm.config import EnvConfig
from npa_swarm.env import SwarmEnv


def arranged(weight):
    env=SwarmEnv(EnvConfig(n_agents=8,work_credit_weight=weight))
    env.positions[0]=torch.tensor([[-.708,0],[3,3],[3.2,3],[3.4,3],[3.6,3],[3.8,3],[4,3],[4.2,3]])
    env.object_pos.zero_();env.goal_pos[:]=torch.tensor([2.,0]);env._refresh();env.found_goal[:]=True
    return env


def test_work_credit_changes_reward_not_physics_or_observation():
    base,shaped=arranged(0),arranged(1)
    a=torch.zeros_like(base.positions);a[0,0,0]=1
    x,y=base.step(a),shaped.step(a)
    for left,right in ((x[0],y[0]),(base.positions,shaped.positions),(base.object_pos,shaped.object_pos)):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert y[4]["reward_contact_credit"]>0 and y[1][0,0]>x[1][0,0]
    torch.testing.assert_close(x[1][0,1:],y[1][0,1:],rtol=0,atol=0)
    assert shaped.object_pos[0,0]>0


def test_backward_progress_is_negative_and_unknown_goal_has_no_extra_credit():
    env=arranged(1);env.goal_pos*=-1;env._refresh()
    action=torch.zeros_like(env.positions);action[0,0,0]=1
    assert env.step(action)[4]["reward_contact_credit"]<0
    env=arranged(1);env.goal_pos[:]=torch.tensor([8.,0]);env.found_goal[:]=False;env._refresh()
    assert env.step(action)[4]["reward_contact_credit"]==0


def test_wall_blocked_push_cannot_farm_work_credit():
    env=arranged(1);shift=env.cfg.arena_half-env.cfg.object_radius
    env.positions[0,0,0]+=shift;env.object_pos[0,0]=shift;env.goal_pos[:]=torch.tensor([9.8,0]);env._refresh()
    a=torch.zeros_like(env.positions);a[0,0,0]=1
    before=env.object_pos.clone()
    for _ in range(3):
        info=env.step(a)[4]
        assert torch.equal(before,env.object_pos) and info["reward_contact_credit"]==0 and info["reward_progress"]==0


def test_physical_threshold_is_only_an_explicit_diagnostic():
    env=arranged(1);env.cfg.object_static_friction=4
    a=torch.zeros_like(env.positions);a[0,0,0]=1
    for _ in range(10):
        info=env.step(a)[4]
        assert torch.equal(env.object_pos,torch.zeros_like(env.object_pos))
        assert info["reward_contact_credit"]==0
