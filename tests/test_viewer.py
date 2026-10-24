import json
import struct
import threading
import time
import urllib.error
import urllib.request
import pytest
import torch
from npa_swarm.config import EnvConfig
from npa_swarm.server import HEADER_BYTES,MAGIC,Simulation,create_server


def request(base,path,payload=None,headers=None):
    data=None if payload is None else json.dumps(payload).encode()
    headers={**(headers or {}),**({"Content-Type":"application/json"} if data else {})}
    return urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers),timeout=10)


def wait_for(predicate,timeout=8):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError("Viewer worker did not reach expected state")


@pytest.fixture
def viewer():
    server=create_server(EnvConfig(n_agents=16,episode_steps=3),untrained=True,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f"http://127.0.0.1:{server.server_address[1]}",server
    finally:server.shutdown();server.server_close();thread.join(5)


def test_snapshot_identity_etag_and_minimal_webgpu_only_ui(viewer):
    base,server=viewer
    with request(base,"/api/status") as r:s=json.load(r)
    assert s["checkpoint"]["updates"]==0 and not s["trained_transport_verified"]
    with request(base,"/api/snapshot") as r:body,tag=r.read(),r.headers["ETag"]
    m,v,n,h,seq,flags,step,rev=struct.unpack_from("<8I",body)
    assert (m,v,n,h,step)==(MAGIC,1,16,HEADER_BYTES,0)
    assert len(body)==HEADER_BYTES+n*32 and flags&1
    f=struct.unpack_from("<24f",body,32);assert f[10]==f[11]==1
    with pytest.raises(urllib.error.HTTPError) as e:request(base,"/api/snapshot",headers={"If-None-Match":tag})
    assert e.value.code==304
    with request(base,"/") as r:html=r.read().decode()
    with request(base,"/app.js") as r:js=r.read().decode()
    assert '<select' not in html and '<h1' not in html and "CanvasRenderer" not in js
    assert "createRenderPipelineAsync" in js and "writeBuffer" in js


def test_control_terminal_hold_reset_and_unpaced(viewer):
    base,server=viewer
    for command in ({"command":"speed","value":0},{"command":"play"}):
        with request(base,"/api/control",command):pass
    wait_for(lambda:server.simulation.finished)
    assert int(server.simulation.env.steps[0])==3
    with pytest.raises(urllib.error.HTTPError) as e:request(base,"/api/control",{"command":"play"})
    assert e.value.code==400
    with request(base,"/api/control",{"command":"reset"}) as r:rev=json.load(r)["accepted_revision"]
    wait_for(lambda:struct.unpack_from("<I",server.simulation.snapshot,28)[0]==rev)
    assert int(server.simulation.env.steps[0])==0 and torch.count_nonzero(server.simulation.hidden)==0
    assert not server.simulation.finished and server.simulation.paused


def test_untrained_requires_opt_in():
    with pytest.raises(ValueError,match="untrained"):create_server(EnvConfig(n_agents=16),port=0)


def test_sampled_inference_keeps_global_rng_and_reports_mode():
    before=torch.get_rng_state().clone()
    sim=Simulation(EnvConfig(n_agents=16,episode_steps=3),untrained=True,stochastic=True)
    try:
        assert sim.status()["policy_mode"]=="sampled_seeded" and sim.status()["action_seed"]==42
        sim.command({"command":"speed","value":0});sim.command({"command":"play"});wait_for(lambda:sim.finished)
        assert torch.equal(before,torch.get_rng_state())
    finally:sim.close()


def test_publication_rate_decoupled_from_simulation():
    sim=Simulation(EnvConfig(n_agents=16,episode_steps=50),untrained=True,stochastic=True)
    try:
        sim.command({"command":"speed","value":0});sim.command({"command":"play"});wait_for(lambda:sim.finished)
        wait_for(lambda:struct.unpack_from("<I",sim.snapshot,24)[0]==50)
        assert sim.sequence < 50  # no mandatory GPU->CPU frame copy at every simulated step
        assert struct.unpack_from("<I",sim.snapshot,20)[0]&4
    finally:sim.close()

def test_snapshot_visual_channels_match_real_sensor_contact_and_relay():
    import numpy as np
    from npa_swarm.policy import LocalPolicy
    from npa_swarm.server import pack_snapshot
    cfg = EnvConfig(n_agents=12, stage="contact")
    env = SwarmEnv(cfg)
    hidden = LocalPolicy(env.obs_dim).initial_state(1, cfg.n_agents)
    hidden[0, :, -6] = .35
    hidden[0, :, -3] = .6
    def unpack():
        data = np.frombuffer(pack_snapshot(env, hidden, env.metrics(), 0, True, 0),
                             dtype="<f4", offset=HEADER_BYTES).reshape(cfg.n_agents, 8)
        return data
    data = unpack()
    assert data.nbytes == cfg.n_agents * 8 * 4
    np.testing.assert_allclose(data[:,4], torch.maximum(hidden[0,:,-6],env._object_visible[0].float()).numpy())
    np.testing.assert_allclose(data[:,5], torch.maximum(hidden[0,:,-3],env._goal_visible[0].float()).numpy())
    flags = data[:,7].astype(np.int32)
    np.testing.assert_array_equal((flags&1)!=0,env._in_main[0].numpy())
    np.testing.assert_array_equal((flags&2)!=0,env._object_visible[0].numpy())
    np.testing.assert_array_equal((flags&4)!=0,env._goal_visible[0].numpy())
    np.testing.assert_array_equal((flags&8)!=0,env._contact[0].numpy())
    np.testing.assert_array_equal((flags&16)!=0,env._bridge_members[0].numpy())
    # Reverse the physical contact side: signed force must reverse, not the policy.
    env.object_pos.zero_()
    env.goal_pos[:] = torch.tensor([2.,0.])
    env.positions[0,0] = torch.tensor([-(cfg.object_radius+cfg.agent_radius-.025),0.])
    env._refresh()
    assert unpack()[0,6] > .04
    env.positions[0,0,0] *= -1
    env._refresh()
    assert unpack()[0,6] < -.04
