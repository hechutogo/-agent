import json

from pickparts_agent.observability import events


def test_now_iso_has_timezone():
    assert "+" in events.now_iso() or events.now_iso().count("-") >= 3


def test_redact_masks_secret_shapes():
    out = events.redact("key is sk-abcdefgh12345678 ok")
    assert "abcdefgh12345678" not in out and "sk-" in out


def test_redact_masks_named_credentials_without_provider_prefix():
    secret = "0123456789abcdef0123456789abcdef"
    out = events.redact(
        f'{{"api_key":"{secret}","access_token"="{secret}"}}')
    assert secret not in out
    assert out.count("***REDACTED***") == 2


def test_make_event_carries_common_fields():
    evt = events.make_event("log", "r1", 3, level="info", message="你好")
    assert evt["kind"] == "log" and evt["run_id"] == "r1" and evt["seq"] == 3
    assert evt["level"] == "info" and evt["message"] == "你好"


def test_dumps_is_compact_jsonline():
    line = events.dumps(events.make_event("run_start", "r", 1, task="reset"))
    assert "\n" not in line
    assert json.loads(line)["task"] == "reset"
    assert ", " not in line and ": " not in line
