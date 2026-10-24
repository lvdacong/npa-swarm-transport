import pytest
from npa_swarm.cli import main
from npa_swarm.config import EnvConfig,TrainConfig


def test_minimal_v2_defaults():
    assert EnvConfig().n_agents==900 and EnvConfig().stage=="search"
    assert EnvConfig().object_static_friction==0
    assert TrainConfig().rollout_steps==128 and TrainConfig().bptt_steps==32


@pytest.mark.parametrize("override", [["--agents", "16"], ["--relay-memory-decay", ".997"],
                                     ["--contact-credit-mode", "marginal"]])
def test_resume_cannot_silently_override_task(tmp_path, override):
    with pytest.raises(SystemExit):
        main(["train","--resume","missing.pt",*override,"--updates","1","--output",str(tmp_path)])


def test_bad_config():
    with pytest.raises(ValueError):EnvConfig(n_agents=1)
    with pytest.raises(ValueError):TrainConfig(hidden_dim=4)



def test_ctde_requires_known_search_warmstart(tmp_path):
    from npa_swarm.cli import main
    import pytest
    with pytest.raises(SystemExit):
        main(["train", "--ctde-task", "--updates", "0", "--output", str(tmp_path/"invalid"), "--device", "cpu"])
