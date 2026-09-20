"""Atom contract: precondition -> grounded run -> postcondition verify."""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Check:
    ok: bool
    reason: str = ""
    facts: dict = field(default_factory=dict)


@dataclass
class AtomContext:
    sim: object
    locate: Callable
    kin: object
    state: object
    on_stage: Callable[[str], None]
    on_trace: Callable[[dict], None]


@dataclass
class AtomResult:
    success: bool
    error_kind: str | None = None
    message: str = ""
    observed: dict = field(default_factory=dict)


class Atom:
    name = ""
    description = ""
    parameters = {}

    def check_pre(self, ctx, args):
        raise NotImplementedError

    def run(self, ctx, args):
        raise NotImplementedError

    def verify(self, ctx, args, result):
        raise NotImplementedError

    def call(self, ctx, args):
        pre = self.check_pre(ctx, args)
        if not pre.ok:
            return AtomResult(False, "precondition", pre.reason, pre.facts)
        out = self.run(ctx, args)
        if out.success:
            post = self.verify(ctx, args, out)
            if not post.ok:
                return AtomResult(False, "verify", post.reason, post.facts)
        return out
