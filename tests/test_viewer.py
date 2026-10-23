import json
import struct
import threading
import time
import urllib.error
import urllib.request

import pytest
import torch

from npa_swarm.config import EnvConfig
from npa_swarm.server import HEADER_BYTES, MAGIC, Simulation, create_server


def request(base, path, payload=None, headers=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {**(headers or {}), **({"Content-Type": "application/json"} if data else {})}
    return urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=10)


def wait_for(predicate, timeout=8):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(.02)
    raise AssertionError("Viewer worker did not reach expected state")


@pytest.fixture
def viewer():
    torch.set_num_threads(1)
    server = create_server(EnvConfig(n_agents=16, episode_steps=3), untrained=True, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_snapshot_identity_and_unchanged_etag(viewer):
    base, server = viewer
    with request(base, "/api/status") as response:
        status = json.load(response)
    assert status["checkpoint"]["updates"] == 0
    assert status["trained_transport_verified"] is False
    with request(base, "/fonts/swarm-cjk.woff") as response:
        assert response.headers["Content-Type"] == "font/woff"
        assert response.read(4) == b"wOFF"
    with request(base, "/api/snapshot") as response:
        body, etag = response.read(), response.headers["ETag"]
    magic, version, n, header, sequence, flags, step, revision = struct.unpack_from("<8I", body)
    assert (magic, version, n, header, step) == (MAGIC, 1, 16, HEADER_BYTES, 0)
    assert len(body) == HEADER_BYTES + n * 32
    assert flags & 1
    # The first snapshot already contains valid connectivity, not a made-up success.
    values = struct.unpack_from("<24f", body, 32)
    assert values[10] == 1 and values[11] == 1
    with pytest.raises(urllib.error.HTTPError) as error:
        request(base, "/api/snapshot", headers={"If-None-Match": etag})
    assert error.value.code == 304


def test_control_terminal_hold_and_reset(viewer):
    base, server = viewer
    with request(base, "/api/control", {"command": "speed", "value": 4}):
        pass
    with request(base, "/api/control", {"command": "play"}):
        pass
    wait_for(lambda: server.simulation.finished)
    assert int(server.simulation.env.steps[0]) == 3
    with pytest.raises(urllib.error.HTTPError) as error:
        request(base, "/api/control", {"command": "play"})
    assert error.value.code == 400
    with request(base, "/api/control", {"command": "reset"}) as response:
        revision = json.load(response)["accepted_revision"]
    wait_for(lambda: struct.unpack_from("<I", server.simulation.snapshot, 28)[0] == revision)
    assert int(server.simulation.env.steps[0]) == 0
    assert torch.count_nonzero(server.simulation.hidden) == 0
    assert not server.simulation.finished and server.simulation.paused


def test_random_policy_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="untrained"):
        create_server(EnvConfig(n_agents=16), port=0)


def test_sampled_inference_reports_mode_and_does_not_consume_global_rng():
    before = torch.get_rng_state().clone()
    simulation = Simulation(EnvConfig(n_agents=16, episode_steps=3), untrained=True, stochastic=True)
    try:
        assert simulation.status()["policy_mode"] == "sampled_seeded"
        assert simulation.status()["action_seed"] == 42
        simulation.command({"command": "speed", "value": 4})
        simulation.command({"command": "play"})
        wait_for(lambda: simulation.finished)
        assert int(simulation.env.steps[0]) == 3
        assert torch.equal(before, torch.get_rng_state())
    finally:
        simulation.close()
