import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../.."))

import litellm.utils as litellm_utils
from litellm.utils import shorten_message_to_fit_limit


def _fake_token_count(messages, model):
    return len(messages[0]["content"])


def test_tiny_target_does_not_grow_content():
    original_content = "x" * 100
    message = {"role": "user", "content": original_content}

    with patch.object(litellm_utils, "get_token_count", _fake_token_count):
        pre_trim_tokens = _fake_token_count([message], "claude")
        out = shorten_message_to_fit_limit(dict(message), tokens_needed=1, model="claude")
        post_trim_tokens = _fake_token_count([out], "claude")

    assert len(out["content"]) < len(original_content)
    assert post_trim_tokens <= pre_trim_tokens
