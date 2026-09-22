from pickparts_agent.agent.llm import JSONChat


class _Message:
    content = '{"ok": true}'


class _Choice:
    message = _Message()


class _Completions:
    def create(self, **kwargs):
        return type("R", (), {"choices": [_Choice()]})()


class _Chat:
    completions = _Completions()


class _Client:
    chat = _Chat()


def test_json_opens_llm_span_and_saves_transcript(tmp_path):
    from pickparts_agent.observability import Recorder
    rec = Recorder(tmp_path)
    chat = JSONChat(_Client(), "m", recorder=rec)
    with rec.run("command", "g"):
        data = chat.json("sys", "usr", label="plan")
    assert data == {"ok": True}
    artifacts = list((tmp_path / "artifacts").rglob("*.resp.txt"))
    assert artifacts and artifacts[0].read_text() == '{"ok": true}'


def test_json_defaults_to_null_recorder():
    chat = JSONChat(_Client(), "m")
    data = chat.json("sys", "usr")
    assert data == {"ok": True}
