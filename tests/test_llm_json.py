import httpx
import pytest

from pickparts_agent.llm import JSONChat, LLMError


def client_with(payloads):
    it = iter(payloads)
    def handle(request):
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": next(it)}}]})
    from openai import OpenAI
    return OpenAI(api_key="k", base_url="https://cloud.invalid/v1",
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


def test_network_error_passes_through_openai_exception():
    from openai import APITimeoutError
    def raise_once(request):
        raise APITimeoutError(request=request)
    from openai import OpenAI
    c = OpenAI(api_key="k", base_url="https://cloud.invalid/v1",
               http_client=httpx.Client(transport=httpx.MockTransport(raise_once)),
               max_retries=0)
    with pytest.raises(APITimeoutError):
        JSONChat(c, "m").json("sys", "user")
