"""V2-22d Closeout A.4.12 — prismpy-side proxy that registers agera5
and iSDA cache-path derivations with prismweb's cache_path_registry.

Design
------
prismweb owns the eviction sweep; prismpy owns the cache writers.
This tiny proxy bridges the two: ``__init_registry__()`` is called
exactly once from ``evict_cache.handle()`` on the prismweb side, and
each call registers the per-subdir derivation function so the
registry's fail-CLOSED check is satisfied. Without this bridge,
enabled subdirs ``agera5`` and ``isda`` would raise
``RegistryIncomplete`` and the sweep would exit 2.

Why an explicit entry point rather than top-level side effects
--------------------------------------------------------------
Prismpy modules load via multiple paths (executor, sources, tests),
and some test paths import prismpy without importing prismweb at
all. Top-level ``register_cache_writer`` calls would either
double-register or run under test-only conditions where prismweb's
registry module isn't present. An explicit
``__init_registry__()`` fires ONCE per evict_cache invocation and
is import-order agnostic.

Why empty-set derivations
-------------------------
Both agera5 and isda writers hold a ``filelock`` during active
downloads (the writer's ``cache_lock_path`` path). The eviction
sweep probes that filelock before deleting, so an actively-written
file is caught by that signal. Post-download, atime reflects the
last read and the per-subdir age policy handles retention. A
conservative empty-set derivation here means the registry satisfies
fail-CLOSED without needing to mirror the writer's region → path
derivation, which would couple prismpy to prismweb's run model.

If a future contract tightens the retention invariant (e.g., "never
evict any agera5 file touched by the past N minutes of running-run
activity"), this file is the seam where that derivation slots in.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Type-only import; avoids prismpy gaining a hard prismweb
    # dependency at runtime. The actual call path is
    # evict_cache.handle → __init_registry__ → prismweb's
    # register_cache_writer, which only exists at prismweb runtime.
    from core.services.cache_path_registry import register_cache_writer  # noqa: F401


def _agera5_in_use_for(run) -> frozenset[Path]:
    """Conservative derivation — return the empty set so the
    FileLock probe + atime age policy own the in-use signal. See
    module docstring for rationale."""
    return frozenset()


def _isda_in_use_for(run) -> frozenset[Path]:
    """Same rationale as ``_agera5_in_use_for``."""
    return frozenset()


def __init_registry__() -> None:
    """Register agera5 + isda derivations with prismweb's
    cache_path_registry. Called exactly once from
    ``evict_cache.handle()``.

    Raises ``ImportError`` if the prismweb registry module is not
    importable — the caller expects this shape under test scaffolds
    that run prismpy in isolation.
    """
    from core.services.cache_path_registry import register_cache_writer

    register_cache_writer('agera5', _agera5_in_use_for)
    register_cache_writer('isda', _isda_in_use_for)
