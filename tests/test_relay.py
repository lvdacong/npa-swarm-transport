"""Local route messages are strictly causal and geometrically relative."""
import torch
import pytest
from npa_swarm.graph import Graph
from npa_swarm.relay import relay_beacons
from npa_swarm.policy import LocalPolicy


def chain():
    pos = torch.tensor([[0.,0.],[.3,0.],[.6,0.],[.9,0.]])
    edges = [(i,i+1) for i in range(3)] + [(i+1,i) for i in range(3)]
    src = torch.tensor([a for a,b in edges])
    dst = torch.tensor([b for a,b in edges])
    return Graph(src,dst,4,relative=pos[src]-pos[dst]),pos


def observations():
    obs = torch.zeros(1,4,20)
    obs[0,3,12] = 1
    obs[0,3,10] = .2 / 2.2
    obs[0,0,9] = 1
    obs[0,0,7] = -.2 / 1.85
    return obs


def test_goal_bearing_spreads_one_hop_and_never_teleports():
    graph,_ = chain()
    obs = observations()
    beacons = torch.zeros(1,4,6)
    for t in range(4):
        beacons = relay_beacons(beacons,obs,graph)
        assert (beacons[0,:max(0,3-t),3]==0).all()
    assert beacons[0,0,3] > .9 ** 3
    torch.testing.assert_close(beacons[0,0,4:6],torch.tensor([1.1,0.]),atol=1e-5,rtol=0)


def test_disconnected_agents_never_receive_remote_goal_message():
    graph,_ = chain()
    graph.src = torch.tensor([0,1,2,3])
    graph.dst = torch.tensor([1,0,3,2])
    graph.relative = torch.tensor([[-.3,0],[.3,0],[-.3,0],[.3,0]])
    beacons = torch.zeros(1,4,6)
    for _ in range(12):
        beacons = relay_beacons(beacons,observations(),graph)
    assert (beacons[0,:2,3] == 0).all()
    assert (beacons[0,2:,3] > 0).all()


def test_policy_uses_causal_bearing_only_as_learned_action_basis():
    graph,_ = chain()
    policy = LocalPolicy(20,8)
    with torch.no_grad():
        policy.actor.weight.zero_()
        policy.actor.bias.zero_()
        policy.actor.bias[-1] = 1.
    memory = policy.initial_state(1,4)
    obs = observations()
    for t in range(4):
        mean,_,_,memory = policy(obs,memory,graph)
        if t < 3: assert mean[0,0].abs().max() == 0
    assert mean[0,0,0] > 0 and mean[0,0,1] == 0
    assert torch.isfinite(memory).all()


def test_translation_preserves_message_and_action():
    graph, pos = chain()
    p = LocalPolicy(20,8)
    obs = observations()
    state = p.initial_state(1,4)
    shifted = Graph(graph.src,graph.dst,4,relative=(pos+42)[graph.src]-(pos+42)[graph.dst])
    a=p(obs,state,graph);b=p(obs,state,shifted)
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=0,atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_moving_agents_keep_bearings_to_static_occluded_endpoints(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    graph, pos = chain()
    graph = graph.to(device)
    obs = observations().to(device)
    memory = torch.zeros(1, 4, 6, device=device)
    for _ in range(4):
        memory = relay_beacons(memory, obs, graph)
    motion = torch.tensor([[.05,.01],[.02,-.03],[-.03,.02],[-.05,-.01]], device=device)
    moved = pos.to(device) + motion
    graph.relative = moved[graph.src] - moved[graph.dst]
    graph.displacement = motion[None]
    result = relay_beacons(memory, torch.zeros_like(obs), graph)
    # No local sightings now. Both endpoints remain static while every agent
    # moves differently, so old own and old neighbor bearings need odometry.
    for channel, endpoint in ((1, [-.2, 0.]), (4, [1.1, 0.])):
        expected = torch.tensor(endpoint, device=device) - moved
        torch.testing.assert_close(result[0, :, channel:channel+2], expected, atol=1e-6, rtol=0)


def test_retention_changes_only_known_memory_not_hop_latency_or_unseen_nodes():
    graph = Graph(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), 2)
    known = torch.zeros(1, 2, 6); known[0, 0, 3:] = torch.tensor([1., 2., 1.])
    old, retained = known.clone(), known.clone()
    obs = torch.zeros(1, 2, 20)
    for _ in range(100):
        old = relay_beacons(old, obs, graph)
        retained = relay_beacons(retained, obs, graph, memory_decay=.997)
    assert old[0, 0, 3] < .05 < retained[0, 0, 3] < 1
    assert torch.count_nonzero(retained[0, 1]) == 0
    chain_graph, _ = chain(); state = torch.zeros(1, 4, 6)
    state = relay_beacons(state, observations(), chain_graph, memory_decay=.997)
    assert (state[0, :3, 3] == 0).all()
    state = relay_beacons(state, observations(), chain_graph, memory_decay=.997)
    torch.testing.assert_close(state[0, 2, 3], torch.tensor(.97))
    assert (state[0, :2, 3] == 0).all()
