import os

import pytest

import litellm
from litellm.llms.vertex_ai.cost_calculator import cost_per_character
from litellm.types.utils import Usage


@pytest.mark.parametrize(
    "prompt_characters, completion_characters, expected_prompt_cost, expected_completion_cost",
    [
        (32001, 33000, 32001 * 5e-6, 33000 * 1.5e-5),
        (100, 200, 100 * 5e-6, 200 * 1.5e-5),
    ],
)
def test_medlm_character_pricing_below_128k_tokens(
    prompt_characters, completion_characters, expected_prompt_cost, expected_completion_cost
):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    usage = Usage(prompt_tokens=8000, completion_tokens=50, total_tokens=8050)
    prompt_cost, completion_cost = cost_per_character(
        model="medlm-large",
        custom_llm_provider="vertex_ai",
        usage=usage,
        prompt_characters=prompt_characters,
        completion_characters=completion_characters,
    )

    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(expected_completion_cost)
