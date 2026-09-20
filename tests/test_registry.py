import pytest

from pickparts_agent.atoms.base import Atom
from pickparts_agent.atoms.registry import AtomRegistry


class GreetAtom(Atom):
    name = "greet"
    description = "Say hello to a named object."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string", "enum": ["A", "B"]}},
        "required": ["target"],
    }

    def check_pre(self, ctx, args): ...
    def run(self, ctx, args): ...
    def verify(self, ctx, args, result): ...


def reg():
    return AtomRegistry([GreetAtom()])


def test_validate_accepts_well_formed_call():
    reg().validate_call({"atom": "greet", "args": {"target": "A"}})


@pytest.mark.parametrize("call", [
    {"atom": "missing", "args": {}},
    {"atom": "greet"},
    {"atom": "greet", "args": {"target": "Z"}},
    {"atom": "greet", "args": {}},
    {"atom": "greet", "args": {"target": "A", "point": [1, 2, 3]}},
    "greet",
])
def test_validate_rejects_unknown_atom_bad_enum_or_extra_args(call):
    with pytest.raises(ValueError):
        reg().validate_call(call)


def test_get_and_has():
    r = reg()
    assert r.has("greet") and not r.has("nope")
    assert isinstance(r.get("greet"), GreetAtom)
    with pytest.raises(KeyError):
        r.get("nope")


def test_catalog_mentions_name_and_param():
    text = reg().catalog_for_prompt()
    assert "greet" in text and "target" in text
