"""Bounded sensor sampling; no robot motion or semantic planning decisions."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisualEvidence:
    confirmed: bool
    value: Any
    samples: int
    error: Exception | None = None


def sample_until_stable(sample, accept, wait, *, max_samples=3,
                        recoverable=(ValueError,)):
    """Require two consecutive positive samples, separated by ``wait``.

    A failed observation clears the previous value: stale evidence must never
    masquerade as the current frame. Runtime/programming errors propagate.
    The caller supplies a same-thread settling operation, not a motion plan.
    """
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("Stable verification requires at least two samples")
    consecutive, value, error = 0, None, None
    for count in range(1, max_samples + 1):
        if count > 1:
            wait()
        try:
            value = sample()
            accepted = bool(accept(value))
            error = None
        except recoverable as exc:
            value, error, accepted = None, exc, False
        consecutive = consecutive + 1 if accepted else 0
        if consecutive >= 2:
            return VisualEvidence(True, value, count)
    return VisualEvidence(False, value, max_samples, error)
