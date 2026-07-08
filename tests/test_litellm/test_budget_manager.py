import time

from litellm.budget_manager import BudgetManager


def test_create_budget_records_distinct_creation_times():
    manager = BudgetManager(project_name="test")

    manager.create_budget(total_budget=10, user="u1", duration="daily")
    time.sleep(0.01)
    manager.create_budget(total_budget=10, user="u2", duration="daily")

    # created_at must be captured per call, not frozen at import time
    assert manager.user_dict["u1"]["created_at"] != manager.user_dict["u2"]["created_at"]
