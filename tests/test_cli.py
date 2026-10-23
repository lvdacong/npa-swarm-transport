import pytest

from npa_swarm.cli import main


def test_evaluation_engineering_error_has_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr("npa_swarm.evaluate.evaluate", lambda *a, **k: {"engineering_evaluation_completed": False, "error": 1})
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--checkpoint", "unused.pt", "--device", "cpu", "--output", str(tmp_path)])
    assert error.value.code == 1


def test_task_timeout_is_a_result_not_cli_failure(monkeypatch, tmp_path):
    monkeypatch.setattr("npa_swarm.evaluate.evaluate", lambda *a, **k: {"engineering_evaluation_completed": True, "timeout": 3, "success": 0})
    main(["evaluate", "--checkpoint", "unused.pt", "--device", "cpu", "--output", str(tmp_path)])


def test_resume_rejects_silently_ignored_cli_configuration(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["train", "--resume", "unused.pt", "--agents", "32", "--device", "cpu", "--updates", "1", "--output", str(tmp_path)])
    assert error.value.code == 2
