import json
import time

from litellm.budget_manager import BudgetManager


def _wait_for_disk_current_cost(path, expected, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
            if data["test_user"]["current_cost"] == expected:
                return
        except Exception:
            pass
        time.sleep(0.05)


def test_reset_cost_persists_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    bm = BudgetManager(project_name="test_project", client_type="local")
    bm.user_dict["test_user"] = {
        "total_budget": 100.0,
        "current_cost": 5.0,
        "model_cost": {"gpt-3.5-turbo": 5.0},
    }
    bm.save_data()

    with open(tmp_path / "user_cost.json", "r") as json_file:
        persisted = json.load(json_file)
    assert persisted["test_user"]["current_cost"] == 5.0

    bm.reset_cost("test_user")
    _wait_for_disk_current_cost(tmp_path / "user_cost.json", 0)

    reloaded = BudgetManager(project_name="test_project", client_type="local")
    assert reloaded.get_current_cost("test_user") == 0
    assert reloaded.get_model_cost("test_user") == {}
