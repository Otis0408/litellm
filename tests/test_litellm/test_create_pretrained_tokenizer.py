from unittest.mock import MagicMock, patch

from litellm.utils import create_pretrained_tokenizer


def test_create_pretrained_tokenizer_passes_token_kwarg():
    with patch("litellm.utils.Tokenizer.from_pretrained", return_value=MagicMock()) as mock_from_pretrained:
        create_pretrained_tokenizer("some-model", auth_token="secret")

    mock_from_pretrained.assert_called_once_with("some-model", revision="main", token="secret")
