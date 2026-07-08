from litellm.secret_managers.main import get_secret, get_secret_bool


def test_get_secret_returns_default_value_when_missing():
    assert get_secret("DOES_NOT_EXIST_XYZ_123", default_value="fallback") == "fallback"


def test_get_secret_bool_returns_default_value_when_missing():
    assert get_secret_bool("DOES_NOT_EXIST_XYZ_123", default_value=True) is True
