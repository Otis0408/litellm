import pytest

from litellm.litellm_core_utils.llm_response_utils.get_api_base import get_api_base


@pytest.mark.parametrize(
    "model, optional_params, expected_api_base",
    [
        (
            "gemini/gemini-1.5-pro",
            {"stream": True},
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent",
        ),
        (
            "gemini/gemini-1.5-pro",
            {"stream": False},
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        ),
        (
            "gemini/gemini-1.5-pro",
            {},
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        ),
    ],
)
def test_get_api_base_gemini_stream_flag_from_dict(model, optional_params, expected_api_base):
    assert get_api_base(model=model, optional_params=optional_params) == expected_api_base


def test_get_api_base_vertex_stream_flag_from_dict():
    optional_params = {
        "vertex_project": "my-project",
        "vertex_location": "us-central1",
        "stream": True,
    }
    api_base = get_api_base(model="gemini-1.5-pro", optional_params=optional_params)
    assert api_base is not None
    assert api_base.endswith(":streamGenerateContent")
