"""Seeded span and trace ids, so a run's trace refs are reproducible from its seed.

Inside `seeded_ids(seed)` every new id comes from one `random.Random` seeded from that
string, so the same agent run gets the same ids whatever thread or worker count ran it.
Outside it, ids are random as usual.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from opentelemetry import trace
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator

_RNG: ContextVar[random.Random | None] = ContextVar("auditlab_id_rng", default=None)


@contextmanager
def seeded_ids(seed: str) -> Iterator[None]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    token = _RNG.set(random.Random(int.from_bytes(digest[:16], "big")))
    try:
        yield
    finally:
        _RNG.reset(token)


class SeededIdGenerator(IdGenerator):
    def __init__(self) -> None:
        self._fallback = RandomIdGenerator()

    def generate_span_id(self) -> int:
        rng = _RNG.get()
        if rng is None:
            return self._fallback.generate_span_id()
        value = rng.getrandbits(64)
        while value == trace.INVALID_SPAN_ID:
            value = rng.getrandbits(64)
        return value

    def generate_trace_id(self) -> int:
        rng = _RNG.get()
        if rng is None:
            return self._fallback.generate_trace_id()
        value = rng.getrandbits(128)
        while value == trace.INVALID_TRACE_ID:
            value = rng.getrandbits(128)
        return value
