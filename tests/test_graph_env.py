"""Exact graph, local observations, and v2 physical invariants."""
import math
import pytest
import torch
from npa_swarm.config import EnvConfig
from npa_swarm.env import SwarmEnv
from npa_swarm.graph import Graph, aggregate, components, radius_graph


def brute_edges(p, radius):
    return {(e*p.shape[1]+i,e*p.shape[1]+j) for e in range(p.shape[0])
            for i in range(p.shape[1]) for j in range(p.shape[1])
            if i != j and (p[e,i]-p[e,j]).square().sum() <= radius**2}


def brute_labels(graph, size):
    adjacency=[[] for _ in range(size)]
    for a,b in zip(graph.src.tolist(),graph.dst.tolist()):
        adjacency[a].append(b);adjacency[b].append(a)
    labels=[-1]*size
    for seed in range(size):
        if labels[seed]>=0:continue
        stack=[seed];labels[seed]=seed
        while stack:
            for j in adjacency[stack.pop()]:
                if labels[j]<0:labels[j]=seed;stack.append(j)
    return labels


@pytest.mark.parametrize("seed",[1,7,23])
def test_complete_graph_matches_all_pairs(seed):
    p=torch.randn(3,23,2,generator=torch.Generator().manual_seed(seed));p[0,1]=p[0,0]
    original=p.clone();g=radius_graph(p,.9)
    edges=set(zip(g.src.tolist(),g.dst.tolist()))
    assert edges==brute_edges(p,.9) and len(edges)==g.src.numel()
    assert torch.equal(p,original)
    assert components(g,3,23).flatten().tolist()==brute_labels(g,69)


def test_dense_cell_and_shuffled_long_chain():
    assert radius_graph(torch.zeros(1,200,2),.1).src.numel()==200*199
    n=513;o=torch.randperm(n,generator=torch.Generator().manual_seed(42))
    g=Graph(torch.cat((o[:-1],o[1:])),torch.cat((o[1:],o[:-1])))
    assert torch.equal(components(g,1,n),torch.zeros(1,n,dtype=torch.long))


def test_mean_and_backward():
    x=torch.tensor([[[1.],[3.],[7.],[9.]]],requires_grad=True)
    g=Graph(torch.tensor([0,1,2]),torch.tensor([1,2,1]))
    y=aggregate(x,g);assert y.flatten().tolist()==[0,4,3,0]
    y.sum().backward();assert x.grad.flatten().tolist()==[.5,1,.5,0]


@pytest.mark.parametrize("layout",["random","diagonal"])
def test_900_search_initially_connected_endpoints_hidden(layout):
    env=SwarmEnv(EnvConfig(layout=layout))
    assert env.cfg.n_agents==900 and env.metrics()["connected"].all()
    assert not env.found_object.any() and not env.found_goal.any()
    obs=env.observe();assert obs.shape==(1,900,20) and torch.isfinite(obs).all()
    assert not obs[...,9].any() and not obs[...,12].any()
    assert env._distance.item()>10
    if layout=="diagonal":assert env._distance.item()==pytest.approx(2*7.4*math.sqrt(2))
    delta=env.positions.flatten(0,1)[env.graph.src]-env.positions.flatten(0,1)[env.graph.dst]
    assert delta.norm(dim=-1).min()>=2*env.cfg.agent_radius-1e-6


def test_observation_translation_invariance_and_cache_refresh():
    env=SwarmEnv(EnvConfig(n_agents=32))
    env.object_pos[:]=torch.tensor([.7,.1]);env.goal_pos[:]=torch.tensor([-.8,.2]);env._refresh()
    before=env.observe().clone();assert env.observe() is env.observe()
    shift=torch.tensor([.35,-.42]);env.positions+=shift;env.object_pos+=shift;env.goal_pos+=shift;env._refresh()
    torch.testing.assert_close(env.observe(),before,atol=2e-6,rtol=1e-5)


def arrange_push(env, active):
    n=env.cfg.n_agents
    env.object_pos.zero_();env.object_vel.zero_();env.goal_pos[:]=torch.tensor([5.,0.])
    env.positions[0]=torch.stack((torch.linspace(-5,-2,n),torch.ones(n)*3),-1)
    angles=torch.linspace(math.pi-.9,math.pi+.9,active) if active>1 else torch.tensor([math.pi])
    env.positions[0,:active]=torch.stack((angles.cos(),angles.sin()),-1)*(env.cfg.object_radius+env.cfg.agent_radius-.002)
    env.velocities.zero_();env._refresh();env.found_goal[:]=True


def test_single_push_is_nonzero_and_cooperation_faster():
    one=SwarmEnv(EnvConfig(n_agents=16));many=SwarmEnv(EnvConfig(n_agents=16))
    arrange_push(one,1);arrange_push(many,8)
    a=torch.zeros_like(one.positions);a[0,0,0]=1
    b=torch.zeros_like(many.positions);b[0,:8,0]=1
    for _ in range(8):one.step(a);many.step(b)
    assert .04 < one.object_pos[0,0] < .08
    assert many.object_pos[0,0]>one.object_pos[0,0]*2


def test_no_remote_force_or_static_contact_bonus():
    env=SwarmEnv(EnvConfig(n_agents=16));before=env.object_pos.clone()
    env.step(torch.ones_like(env.positions));assert torch.equal(env.object_pos,before)
    info=env.step(torch.zeros_like(env.positions))[4]
    assert info["reward_progress"].item()==info["reward_contact_credit"].item()==0


def test_disconnect_penalty_no_deletion_termination_or_shield():
    env=SwarmEnv(EnvConfig(n_agents=8))
    env.positions[0,:4]=torch.tensor([[-3.,0],[-2.85,0],[-2.7,0],[-2.55,0]])
    env.positions[0,4:]=torch.tensor([[3.,0],[3.15,0],[3.3,0],[3.45,0]])
    env._refresh();before=env.positions.clone()
    _,reward,term,trunc,info=env.step(torch.zeros_like(env.positions))
    assert not term.any() and not trunc.any() and torch.equal(before,env.positions)
    assert not info["connected"].any() and info["largest_fraction"].item()==.5
    assert info["reward_connectivity"].item()<0 and (env.detached_age[0,4:]>0).all()
    # Move the detached component toward the main component: signed repair progress.
    act=torch.zeros_like(env.positions);act[0,4:,0]=-1
    info=env.step(act)[4];assert info["reward_repair"].item()>0
    assert env.positions.shape==(1,8,2)


def test_endpoint_bridge_requires_comm_reach_and_same_component():
    env=SwarmEnv(EnvConfig(n_agents=2))
    env.positions[0]=torch.tensor([[-.1,0],[.1,0]])
    env.object_pos[:]=torch.tensor([-1.5,0]);env.goal_pos[:]=torch.tensor([1.8,0]);env._refresh()
    assert env._object_visible.any() and env._goal_visible.any()
    assert not env._bridge.any()  # old sensor-based definition would pass
    env.object_pos[:]=torch.tensor([-1.1,0]);env.goal_pos[:]=torch.tensor([1.5,0]);env._refresh()
    assert env._bridge.all()
    env.positions[0]=torch.tensor([[-2.,0],[2.,0]])
    env.object_pos[:]=torch.tensor([-2.8,0]);env.goal_pos[:]=torch.tensor([3.1,0]);env._refresh()
    assert not env._bridge.any()  # both attached, but distinct particle components


def test_endpoint_does_not_merge_two_particle_components():
    env=SwarmEnv(EnvConfig(n_agents=2))
    env.positions[0]=torch.tensor([[-.9,0],[.9,0]])
    env.object_pos.zero_();env.goal_pos[:]=torch.tensor([2.2,0]);env._refresh()
    assert not env._connected.any()  # cargo is not a free communication relay


def test_coverage_pays_once_and_detached_agent_cannot_farm():
    env=SwarmEnv(EnvConfig(n_agents=4))
    env.positions[0]=torch.tensor([[0.,0],[.15,0],[.30,0],[4.,4.]])
    env._refresh();env.visited.zero_()
    credit=env._mark_coverage()
    assert credit.sum().item()==pytest.approx(1) and credit[0,3]==0
    assert env._mark_coverage().sum()==0


def test_success_requires_whole_cargo_and_timeout_separate():
    env=SwarmEnv(EnvConfig(n_agents=8,episode_steps=1))
    _,_,term,trunc,_=env.step(torch.zeros_like(env.positions));assert not term.any() and trunc.all()
    env.reset();env.object_pos.copy_(env.goal_pos);env._refresh()
    _,_,term,trunc,info=env.step(torch.zeros_like(env.positions))
    assert term.all() and not trunc.any() and info["success"].all()


def test_state_restore_and_masked_reset():
    cfg=EnvConfig(n_agents=32,num_envs=2)
    env=SwarmEnv(cfg);action=torch.rand(2,32,2,generator=torch.Generator().manual_seed(7))*2-1
    env.step(action);restored=SwarmEnv(cfg);restored.load_state_dict(env.state_dict())
    a,b=env.step(action),restored.step(action)
    for i in range(4):assert torch.equal(a[i],b[i])
    for key in a[4]:assert torch.equal(a[4][key],b[4][key]),key
    untouched=env.positions[1].clone();mask=torch.tensor([True,False])
    env.reset(mask);restored.reset(mask)
    assert torch.equal(env.positions,restored.positions) and torch.equal(env.positions[1],untouched)
    assert torch.equal(env.generator.get_state(),restored.generator.get_state())
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_detached_gap_matches_full_distance_reference(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    env = SwarmEnv(EnvConfig(n_agents=48, num_envs=2), device)
    env.positions[0, -2:] += torch.tensor([5., 0.], device=device)
    env.positions[1, -15:] += torch.tensor([-5., 0.], device=device)
    env._refresh()
    distance = torch.cdist(env.positions, env.positions)
    distance.masked_fill_(~env._in_main[:, None], float("inf"))
    reference = (distance.amin(-1) - env.cfg.comm_radius).clamp_min(0).masked_fill(env._in_main, 0)
    torch.testing.assert_close(env._repair_gap, reference, atol=2e-5, rtol=2e-5)
