"""Catch native tool result IDs rejected by DeepSeek's OpenAI API."""
from pickparts_agent.cloud import Endpoint


def test_native_tool_result_serialization():
    from qwen_agent.llm.base import BaseChatModel
    config = Endpoint("test-key", "https://api.deepseek.com", "deepseek-chat").qwen_config()
    assert config["generate_cfg"].get("use_raw_api") is True
    messages = BaseChatModel._conv_qwen_agent_messages_to_oai([
        {"role": "user", "content": "把零件 A 放进盒子"},
        {"role": "assistant", "content": "", "function_call": {
            "name": "pick_and_place", "arguments": '{"target":"A"}'},
         "extra": {"function_id": "call-42"}},
        {"role": "function", "name": "pick_and_place", "content": '{"success":true}',
         "extra": {"function_id": "call-42"}},
    ])
    assert messages[1]["tool_calls"][0]["id"] == "call-42"
    assert messages[2]["tool_call_id"] == "call-42"
    assert "id" not in messages[2]
