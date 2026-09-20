"""OpenAI-compatible chat with strict JSON output and a single repair retry."""
import json


class LLMError(RuntimeError):
    pass


class JSONChat:
    def __init__(self, client, model, max_repair=1):
        self.client = client
        self.model = model
        self.max_repair = max_repair

    def _complete(self, system, user):
        response = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return response.choices[0].message.content

    def json(self, system, user):
        raw = self._complete(system, user)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (TypeError, json.JSONDecodeError):
            pass
        for _ in range(self.max_repair):
            repaired = self._complete(
                system, user + "\n只输出合法 JSON，不要解释或 Markdown 代码块。")
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    return data
            except (TypeError, json.JSONDecodeError):
                continue
        raise LLMError("LLM did not return valid JSON")
