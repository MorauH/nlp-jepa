"""Choosing the horizon k (``docs/adr/0003-random-hidden-horizon.md``).

There is one k per sequence of work: a training batch, a generation, a decoded
plan.  It is drawn here, outside the model, and the predictor never sees it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def draw_horizon(horizon_range: Sequence[int], *key: int) -> int:
    """k uniform over the inclusive ``horizon_range``, a function of ``key`` only.

    Training keys on ``(seed, global_step)``, so a resumed run replays the same
    k without extra state.  The generator is private: no global RNG moves.
    """
    lo, hi = horizon_range
    # A string seed is hashed, so neighbouring keys give unrelated draws.
    return random.Random(":".join(str(int(part)) for part in key)).randint(lo, hi)


def cycle_horizon(horizon_range: Sequence[int], index: int) -> int:
    """The ``index``-th k of a cycle through ``horizon_range`` (validation)."""
    lo, hi = horizon_range
    return lo + index % (hi - lo + 1)
