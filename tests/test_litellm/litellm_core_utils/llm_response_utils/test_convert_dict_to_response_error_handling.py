import pytest

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import ModelResponse


def _raise_from_error(error_obj):
    with pytest.raises(Exception) as exc_info:
        convert_to_model_response_object(
            response_object={"choices": [], "error": error_obj},
            model_response_object=ModelResponse(),
            response_type="completion",
        )
    return exc_info.value


def test_string_symbolic_error_code_keeps_integer_status_code():
    exc = _raise_from_error({"message": "bad", "code": "invalid_request_error"})
    status_code = getattr(exc, "status_code", None)
    assert isinstance(status_code, int)
    assert status_code == 422


def test_integer_error_code_is_used_as_status_code():
    exc = _raise_from_error({"message": "rate limited", "code": 429})
    assert getattr(exc, "status_code", None) == 429


def test_numeric_string_error_code_is_coerced_to_int():
    exc = _raise_from_error({"message": "server error", "code": "500"})
    status_code = getattr(exc, "status_code", None)
    assert isinstance(status_code, int)
    assert status_code == 500
