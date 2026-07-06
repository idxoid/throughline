"""Artifact store: the data plane behind lightweight payloads.

Payloads stay small (control plane); bulk data — corpora, embeddings,
generated reports — lives in a store and travels through flows and across
the agent boundary as an ``ArtifactRef`` handle.

A handle is a *lease*, not a reference: artifacts expire (TTL) and get
evicted (session caps), and ArtifactExpired is a normal condition, not a
bug. If the producing flow is *replayable* — same inputs, same config, same
artifact sources — a re-run re-creates the artifact; otherwise (an LLM step
without caching, a mutable source) the caller must handle expiration
explicitly: pin a longer TTL, persist the output, or treat expiry as data
loss. Replayability is a property of the flow, not a guarantee of the store.
Session namespaces are dropped wholesale when a session ends; that is the
garbage collector for data whose consumers live outside this process.

Store contract (duck-typed, like Cache stores):

    put(value, *, session="default", key=None, ttl=None, meta=None) -> ArtifactRef
    get(ref)                 -> value            (raises ArtifactExpired on miss)
    slice(ref, start, stop)  -> portion of a list/str/bytes artifact
    drop_session(session)    -> int              (artifacts removed)

``MemoryArtifactStore`` is the zero-dependency default. External backends
(Redis, Arrow/Parquet with zero-copy slices) plug in via the same contract —
distributed as ``store.artifact:`` components (``store:`` is the umbrella
kind covering both this protocol and cache stores; see registry kinds).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .errors import ArtifactExpired, StoreError


@dataclass(frozen=True)
class ArtifactRef:
    """Serializable handle to a stored artifact — safe to embed in payloads.

    Carries a summary (``meta``) so consumers (and agents) can decide whether
    to materialize the artifact without fetching it.
    """

    id: str                      # "<session>/<key>"
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def session(self) -> str:
        return self.id.partition("/")[0]

    @property
    def key(self) -> str:
        return self.id.partition("/")[2]

    def to_dict(self) -> dict:
        return {"$artifact": self.id, "meta": self.meta}

    @classmethod
    def from_dict(cls, data: dict) -> "ArtifactRef":
        if not isinstance(data, dict) or "$artifact" not in data:
            raise StoreError(f"not an artifact ref: {data!r}")
        return cls(id=data["$artifact"], meta=data.get("meta", {}))

    def __repr__(self) -> str:
        return f"ArtifactRef({self.id!r}, meta={self.meta!r})"


def _approx_size(value: Any) -> int:
    if isinstance(value, (str, bytes)):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))


class MemoryArtifactStore:
    """In-memory artifact store with TTL leases and per-session caps.

    Args:
        default_ttl:      seconds an artifact lives unless ``put`` overrides
                          (None = no expiry).
        max_per_session:  artifact count cap per session (oldest evicted).
        max_bytes_per_session: approximate byte cap per session.
    """

    def __init__(self, default_ttl: float | None = None,
                 max_per_session: int = 256,
                 max_bytes_per_session: int = 64 * 1024 * 1024):
        self.default_ttl = default_ttl
        self.max_per_session = max_per_session
        self.max_bytes_per_session = max_bytes_per_session
        # (session, key) -> (value, expires_at | None, size, created_at)
        self._data: dict[tuple[str, str], tuple[Any, float | None, int, float]] = {}
        self._lock = threading.Lock()

    # -- writing --------------------------------------------------------------
    def put(self, value: Any, *, session: str = "default", key: str | None = None,
            ttl: float | None = None, meta: dict | None = None) -> ArtifactRef:
        key = key or uuid.uuid4().hex[:12]
        ttl = self.default_ttl if ttl is None else ttl
        expires_at = time.monotonic() + ttl if ttl is not None else None
        size = _approx_size(value)
        summary = {"size": size, **self._shape(value), **(meta or {})}
        with self._lock:
            self._data[(session, key)] = (value, expires_at, size, time.monotonic())
            self._evict(session)
        return ArtifactRef(id=f"{session}/{key}", meta=summary)

    @staticmethod
    def _shape(value: Any) -> dict:
        if isinstance(value, (list, tuple)):
            return {"kind": "list", "items": len(value)}
        if isinstance(value, str):
            return {"kind": "text", "lines": value.count("\n") + 1}
        if isinstance(value, dict):
            return {"kind": "dict", "keys": len(value)}
        return {"kind": type(value).__name__}

    def _evict(self, session: str) -> None:
        entries = [(k, v) for k, v in self._data.items() if k[0] == session]
        entries.sort(key=lambda item: item[1][3])  # oldest first
        while len(entries) > self.max_per_session:
            k, _ = entries.pop(0)
            del self._data[k]
        total = sum(v[2] for _, v in entries)
        while total > self.max_bytes_per_session and entries:
            k, v = entries.pop(0)
            total -= v[2]
            del self._data[k]

    # -- reading --------------------------------------------------------------
    def _lookup(self, ref: ArtifactRef | str) -> Any:
        artifact_id = ref.id if isinstance(ref, ArtifactRef) else str(ref)
        session, _, key = artifact_id.partition("/")
        with self._lock:
            entry = self._data.get((session, key))
            if entry is not None and entry[1] is not None and time.monotonic() > entry[1]:
                del self._data[(session, key)]
                entry = None
        if entry is None:
            raise ArtifactExpired(
                f"artifact {artifact_id!r} is missing or expired; "
                f"re-run the producing flow if it is replayable, "
                f"otherwise the data is gone", artifact_id=artifact_id)
        return entry[0]

    def get(self, ref: ArtifactRef | str) -> Any:
        return self._lookup(ref)

    def slice(self, ref: ArtifactRef | str, start: int = 0, stop: int | None = None) -> Any:
        value = self._lookup(ref)
        if isinstance(value, (list, tuple, str, bytes)):
            return value[start:stop]
        raise StoreError(f"artifact {ref!r} of type {type(value).__name__} is not sliceable")

    # -- lifecycle --------------------------------------------------------------
    def drop_session(self, session: str) -> int:
        with self._lock:
            doomed = [k for k in self._data if k[0] == session]
            for k in doomed:
                del self._data[k]
        return len(doomed)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
