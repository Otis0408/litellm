import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.prompt_templates.factory import (
    cohere_messages_pt_v2,
    convert_to_cohere_tool_invoke,
)


def test_convert_to_cohere_tool_invoke_empty_string_arguments():
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": ""}}
    ]
    result = convert_to_cohere_tool_invoke(tool_calls)
    assert result == [{"name": "get_time", "parameters": {}}]


def test_convert_to_cohere_tool_invoke_missing_arguments():
    tool_calls = [
        {"id": "call_2", "type": "function", "function": {"name": "get_time"}}
    ]
    result = convert_to_cohere_tool_invoke(tool_calls)
    assert result == [{"name": "get_time", "parameters": {}}]


def test_cohere_messages_pt_v2_no_argument_tool_call_does_not_crash():
    messages = [
        {"role": "user", "content": "what time is it?"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_time", "arguments": ""}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
        {"role": "user", "content": "thanks"},
    ]
    _returned_message, chat_history = cohere_messages_pt_v2(messages, "command-r", "cohere_chat")
    chatbot_turns = [m for m in chat_history if m["role"] == "CHATBOT"]
    assert len(chatbot_turns) == 1
    assert chatbot_turns[0]["tool_calls"] == [{"name": "get_time", "parameters": {}}]
