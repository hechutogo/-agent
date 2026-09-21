"""OpenAI-compatible chat with strict JSON output and a single repair retry."""
import json
import re
from urllib.parse import urlsplit


_THINKING_MODELS = {
    "deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
}


def supports_thinking(model: str, base_url: str) -> bool:
    """Whether the endpoint documents the DeepSeek thinking body extension.

    Official models: https://api-docs.deepseek.com/quick_start/pricing
    Legacy deepseek-chat and third-party gateways are not assumed compatible.
    """
    try:
        url = urlsplit(str(base_url))
        return (
            model in _THINKING_MODELS
            and url.scheme == "https" and url.hostname == "api.deepseek.com"
            and url.port in (None, 443) and url.path.rstrip("/") in ("", "/v1")
            and not url.username and not url.password and not url.query
            and not url.fragment
        )
    except (TypeError, ValueError):
        return False


class LLMError(RuntimeError):
    pass


class JSONChat:
    def __init__(self, client, model, max_repair=1, *, thinking=False):
        self.client = client
        self.model = model
        self.max_repair = max_repair
        self.thinking = thinking
        self._supports_thinking = supports_thinking(
            model, getattr(client, "base_url", ""))
        if type(thinking) is not bool:
            raise ValueError("thinking must be a boolean")
        if thinking and not self._supports_thinking:
            raise ValueError("Endpoint/model does not support the thinking toggle")

    def _complete(self, system, user):
        options = {"model": self.model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]}
        if not self.thinking:
            options["temperature"] = 0
        if self._supports_thinking:
            # Supported models default to thinking; disable it unless opted in.
            options["extra_body"] = {
                "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        try:
            response = self.client.chat.completions.create(**options)
            # Never read, return, or replay the provider's reasoning_content.
            return response.choices[0].message.content
        except Exception as exc:
            raise LLMError(f"LLM completion failed ({type(exc).__name__})") from None

    @staticmethod
    def _parse(raw):
        if not isinstance(raw, str):
            raise ValueError("Expected JSON text")
        text = raw.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text,
                             flags=re.IGNORECASE | re.DOTALL)
        if fence:
            text = fence.group(1)

        def reject_constant(value):
            raise ValueError("Non-finite JSON number")

        data = json.loads(text, parse_constant=reject_constant)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def json(self, system, user):
        for attempt in range(self.max_repair + 1):
            prompt = user if attempt == 0 else (
                user + "\n只输出合法 JSON，不要解释或 Markdown 代码块。")
            raw = self._complete(system, prompt)
            try:
                return self._parse(raw)
            except (TypeError, ValueError):
                continue
        raise LLMError("LLM did not return valid JSON")
