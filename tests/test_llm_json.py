import json

import httpx
import pytest

from pickparts_agent.agent.llm import JSONChat, LLMError


def client_with(payloads, requests=None, base_url="https://cloud.invalid/v1",
                reasoning=None):
    it = iter(payloads)
    def handle(request):
        if requests is not None:
            requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": next(it),
                                    "reasoning_content": reasoning}}]})
    from openai import OpenAI
    return OpenAI(api_key="k", base_url=base_url,
                  http_client=httpx.Client(transport=httpx.MockTransport(handle)),
                  max_retries=0)


def test_json_returns_parsed_object():
    chat = JSONChat(client_with(['{"a": 1}']), "m")
    assert chat.json("sys", "user") == {"a": 1}


def test_bad_json_triggers_one_repair_then_succeeds():
    chat = JSONChat(client_with(['not-json', '{"a": 2}']), "m")
    assert chat.json("sys", "user") == {"a": 2}


def test_bad_json_after_repair_raises_llm_error():
    chat = JSONChat(client_with(['bad', 'still-bad']), "m")
    with pytest.raises(LLMError):
        chat.json("sys", "user")


def test_network_error_becomes_sanitized_llm_error():
    from openai import APITimeoutError
    def raise_once(request):
        raise APITimeoutError(request=request)
    from openai import OpenAI
    c = OpenAI(api_key="k", base_url="https://cloud.invalid/v1",
               http_client=httpx.Client(transport=httpx.MockTransport(raise_once)),
               max_retries=0)
    with pytest.raises(LLMError, match="APITimeoutError"):
        JSONChat(c, "m").json("sys", "user")


@pytest.mark.parametrize("raw", [
    '```json\n{"a": 1}\n```',
    ' \n```JSON\r\n{"a": 1}\r\n```\n ',
    '```\n{"a": 1}\n```',
    '```json {"a": 1}```',
])
def test_fenced_json_parses_without_a_repair(raw):
    requests = []
    assert JSONChat(client_with([raw], requests), "m").json("s", "u") == {"a": 1}
    assert len(requests) == 1


@pytest.mark.parametrize("raw", [
    '[]', 'true', 'null', '{"a": NaN}', '{"a": Infinity}',
    'explanation\n```json\n{"a": 1}\n```', None,
])
def test_non_object_or_non_json_output_is_rejected(raw):
    with pytest.raises(LLMError):
        JSONChat(client_with([raw]), "m", max_repair=0).json("s", "u")


def test_reasoning_content_never_used_as_answer_or_sent_for_repair():
    requests = []
    chat = JSONChat(client_with(
        [None, '{"rationale":"Locate both objects."}'], requests,
        reasoning='{"rationale":"PRIVATE_REASONING"}'), "m")
    assert chat.json("s", "u") == {"rationale": "Locate both objects."}
    assert "PRIVATE_REASONING" not in json.dumps(requests)


@pytest.mark.parametrize("thinking", [False, True])
def test_supported_provider_thinking_toggle_in_request_body(thinking):
    requests = []
    chat = JSONChat(client_with(
        ['{"ok":true}'], requests, base_url="https://api.deepseek.com/v1"),
        "deepseek-flash", thinking=thinking)
    assert chat.json("s", "u") == {"ok": True}
    assert requests[0]["thinking"] == {
        "type": "enabled" if thinking else "disabled"}
    if thinking:
        assert "temperature" not in requests[0]


def test_generic_default_has_no_provider_extension():
    requests = []
    JSONChat(client_with(['{}'], requests), "m").json("s", "u")
    assert "thinking" not in requests[0]
    assert requests[0]["temperature"] == 0


@pytest.mark.parametrize("model,url,expected", [
    ("deepseek-flash", "https://api.deepseek.com", True),
    ("deepseek-v4-pro", "https://api.deepseek.com/v1", True),
    ("deepseek-v4-flash", "https://api.deepseek.com", True),
    ("deepseek-chat", "https://api.deepseek.com", False),
    ("deepseek-flash", "https://cloud.invalid/v1", False),
    ("deepseek-flash", "https://api.deepseek.com.evil.invalid", False),
    ("deepseek-flash", "http://api.deepseek.com", False),
    ("other", "https://api.deepseek.com", False),
])
def test_capability_is_explicit_about_provider_and_model(model, url, expected):
    from pickparts_agent.agent.llm import supports_thinking
    assert supports_thinking(model, url) is expected


def test_unsupported_thinking_opt_in_fails_before_any_request():
    with pytest.raises(ValueError, match="thinking"):
        JSONChat(client_with([]), "deepseek-chat", thinking=True)


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_provider_error_is_sanitized(status):
    from openai import OpenAI
    c = OpenAI(api_key="k", base_url="https://cloud.invalid/v1",
               http_client=httpx.Client(transport=httpx.MockTransport(
                   lambda request: httpx.Response(status, json={
                       "error": {"message": "SECRET_PROVIDER_DETAIL"}}))),
               max_retries=0)
    with pytest.raises(LLMError) as caught:
        JSONChat(c, "m").json("s", "u")
    assert "SECRET_PROVIDER_DETAIL" not in str(caught.value)


@pytest.mark.parametrize("payload", [
    {}, {"choices": []}, {"choices": [{"message": None}]},
])
def test_malformed_completion_envelope_is_an_llm_error(payload):
    from openai import OpenAI
    c = OpenAI(api_key="k", base_url="https://cloud.invalid/v1",
               http_client=httpx.Client(transport=httpx.MockTransport(
                   lambda request: httpx.Response(200, json=payload))),
               max_retries=0)
    with pytest.raises(LLMError):
        JSONChat(c, "m").json("s", "u")


def test_repair_preserves_thinking_toggle_without_replaying_raw_content():
    requests = []
    chat = JSONChat(client_with(
        ["PRIVATE_INVALID_CONTENT", '{"rationale":"Retry localization."}'],
        requests, base_url="https://api.deepseek.com",
        reasoning="PRIVATE_REASONING"), "deepseek-flash", thinking=True)
    assert chat.json("s", "u") == {"rationale": "Retry localization."}
    assert len(requests) == 2
    assert all(r["thinking"] == {"type": "enabled"} for r in requests)
    assert "PRIVATE_" not in json.dumps(requests)
