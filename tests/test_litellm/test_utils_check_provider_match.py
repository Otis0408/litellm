import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.utils import _check_provider_match, get_model_info, supports_tool_choice


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


def test_check_provider_match_tolerates_same_family_runtime_variants():
    """get_llm_provider auto-detects a provider *variant* that differs from the cost-map
    entry's stored litellm_provider (e.g. 'jamba-large-1.7' -> 'ai21_chat' while the entry
    says 'ai21'; 'azure_ai/gpt-*' -> 'azure' while the entry says 'azure_ai'). _check_provider_match
    must treat these same-family variants as a match, otherwise the exact-key cost-map hit is
    dropped and get_model_info raises 'This model isn't mapped yet'."""
    assert (
        _check_provider_match(
            model_info={"litellm_provider": "ai21"}, custom_llm_provider="ai21_chat"
        )
        is True
    )
    assert (
        _check_provider_match(
            model_info={"litellm_provider": "azure_ai"}, custom_llm_provider="azure"
        )
        is True
    )
    assert (
        _check_provider_match(
            model_info={"litellm_provider": "azure_text"}, custom_llm_provider="azure"
        )
        is True
    )
    assert (
        _check_provider_match(
            model_info={"litellm_provider": "azure"}, custom_llm_provider="azure_text"
        )
        is True
    )

    assert (
        _check_provider_match(
            model_info={"litellm_provider": "anthropic"}, custom_llm_provider="ai21_chat"
        )
        is False
    )
    assert (
        _check_provider_match(
            model_info={"litellm_provider": "anthropic"}, custom_llm_provider="azure"
        )
        is False
    )


def test_get_model_info_resolves_exact_key_for_provider_variants(local_model_cost_map):
    """Every model that is an exact key in litellm.model_cost must resolve via get_model_info
    without an explicit custom_llm_provider. Regression: ai21 / azure_ai / azure_text keys were
    auto-detected as a provider variant (ai21_chat / azure) that _check_provider_match rejected,
    so get_model_info raised and every supports_* helper returned a wrong False."""
    candidate_keys = [
        "jamba-large-1.7",
        "jamba-mini-1.7",
        "jamba-1.5-large",
        "jamba-1.5-mini",
        "j2-ultra",
        "azure_ai/gpt-5.4",
        "azure_ai/gpt-5.4-mini",
        "azure/gpt-35-turbo-instruct",
    ]
    present = [k for k in candidate_keys if k in litellm.model_cost]
    assert "jamba-large-1.7" in present

    for key in present:
        info = get_model_info(key)
        assert info["key"] == key

    assert litellm.model_cost["jamba-large-1.7"].get("supports_tool_choice") is True
    assert supports_tool_choice("jamba-large-1.7") is True
