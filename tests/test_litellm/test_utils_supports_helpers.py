import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.utils import get_llm_provider, get_model_info, supports_function_calling


@pytest.fixture
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


class TestSupportsHelpersBareModelName:
    """supports_* helpers must stay consistent with get_model_info for bare model
    keys that live in model_cost with the capability set to True but that
    get_llm_provider cannot map to a provider (e.g. "deepseek-chat").

    Regression for _supports_factory returning a wrong False: get_llm_provider
    raised for these bare keys and short-circuited the whole capability lookup
    before the static cost-map entry was consulted, while get_model_info found
    the same key. See litellm/utils.py::_supports_factory.
    """

    CANDIDATE_KEYS = [
        "deepseek-chat",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "deepseek-v3-2-251201",
        "glm-4-7-251222",
        "kimi-k2-thinking-251104",
        "gemini-flash-latest",
        "gemini-pro-latest",
        "gemini-flash-lite-latest",
        "gemini-exp-1206",
        "ft:o4-mini-2025-04-16",
        "computer-use-preview",
    ]

    def _affected_keys(self):
        affected = []
        for key in self.CANDIDATE_KEYS:
            entry = litellm.model_cost.get(key)
            if not isinstance(entry, dict):
                continue
            if entry.get("supports_function_calling") is not True:
                continue
            try:
                get_llm_provider(model=key)
            except Exception:
                affected.append(key)
        return affected

    def test_bug_precondition_deepseek_chat(self, local_model_cost_map):
        assert litellm.model_cost["deepseek-chat"]["supports_function_calling"] is True
        with pytest.raises(Exception):
            get_llm_provider(model="deepseek-chat")

    def test_supports_function_calling_matches_get_model_info(self, local_model_cost_map):
        affected = self._affected_keys()
        assert affected, "expected at least one bare mapped model unresolvable by get_llm_provider"

        for key in affected:
            assert (
                get_model_info(key)["supports_function_calling"] is True
            ), f"get_model_info regressed for {key}"
            assert (
                supports_function_calling(key) is True
            ), f"supports_function_calling returned False for bare mapped key {key}"
