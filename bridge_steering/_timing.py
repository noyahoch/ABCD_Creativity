"""Lightweight stage/section timing hooks.

Stages already get their wall-time recorded in ``_manifest.json`` by
``pipeline.run_stage``. ``timed(label)`` adds a *within-stage* section
timer that prints to stderr so you can see where the seconds go without
having to attach a profiler.

Usage::

    from .._timing import timed

    with timed("load model"):
        loader.load()
    with timed("generate", n=len(prompts)):
        for batch in batches: ...

Output looks like::

    [t] load model            13.4s
    [t] generate              247.1s  (956 items, 3.9 it/s)

Numbers are rough wall-time, not perf counters — pick the biggest bar and
optimise that. Reading the output is the whole point; if you ever want to
suppress it, set ``BRIDGE_STEERING_TIMING=0`` in the env.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Iterator, Optional


def _enabled() -> bool:
    return os.environ.get("BRIDGE_STEERING_TIMING", "1") != "0"


@contextlib.contextmanager
def timed(label: str, n: Optional[int] = None) -> Iterator[None]:
    """Print ``[t] <label>  <elapsed>s [(<n> items, <rate> it/s)]`` on exit."""
    if not _enabled():
        yield
        return
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        suffix = ""
        if n is not None and n > 0 and dt > 0:
            suffix = f"  ({n} items, {n / dt:.1f} it/s)"
        print(f"[t] {label:<32}{dt:8.1f}s{suffix}", file=sys.stderr, flush=True)
