from pickparts_agent.agent.atoms.base import Atom, AtomContext, AtomResult, Check


class DemoAtom(Atom):
    name = "demo"
    description = "demo atom"
    parameters = {"type": "object"}

    def __init__(self, pre=True, post=True):
        self.pre, self.post = pre, post

    def check_pre(self, ctx, args):
        return Check(self.pre, "pre-failed")

    def run(self, ctx, args):
        return AtomResult(True, message="ran", observed={"v": 1})

    def verify(self, ctx, args, result):
        return Check(self.post, "post-failed")


CTX = AtomContext(None, None, None, None, lambda s: None, lambda t: None)


def test_precondition_failure_short_circuits():
    r = DemoAtom(pre=False).call(CTX, {})
    assert r.success is False and r.error_kind == "precondition"
    assert r.message == "pre-failed"


def test_postcondition_failure_maps_to_verify():
    r = DemoAtom(post=False).call(CTX, {})
    assert r.success is False and r.error_kind == "verify"
    assert r.observed == {"v": 1}


def test_happy_path():
    r = DemoAtom().call(CTX, {})
    assert r.success is True and r.observed == {"v": 1}
