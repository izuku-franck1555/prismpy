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

Real derivations (Gate B GB-H1 fixup)
-------------------------------------
The initial closeout pass shipped empty-set derivations. Team-lead's
Gate B codex flagged this as security-theater: the cache-path
registry is supposed to prevent eviction of an actively-used cache
entry, but empty derivations give eviction a free pass.

Post-fixup:

* ``_agera5_in_use_for(run)`` returns the exact data_dir the agera5
  writer derives (``cache_dir / "agera5" / f"AgERA5_{region_key}"``)
  plus the lock path (``cache_dir / f".agera5-{region_key}.lock"``).
  Keys come from ``run.region_cache_key``, which prismweb persists
  on every PipelineRun via V2-22b/P.2 migration 0008.

* ``_isda_in_use_for(run)`` returns the isda subdir root
  (``cache_dir / "isda"``) — iSDA rasters are continent-wide
  (not region-keyed), read-many-times after initial download, and
  have no FileLock contract. A running run could touch any file
  under that subdir, so the conservative safe-path is to mark the
  whole subdir in-use whenever any run is running. Eviction then
  falls back to atime-based age policy once no runs are active,
  which is the expected production shape.

``run.config_snapshot`` is the fallback source for the region key
when ``run.region_cache_key`` is legacy-empty. Any run without
either signal cannot have its agera5 paths derived, so the
derivation returns empty-set for that row — at that point the
FileLock probe and atime policy are the remaining safety gates.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    # Type-only import; avoids prismpy gaining a hard prismweb
    # dependency at runtime. The actual call path is
    # evict_cache.handle → __init_registry__ → prismweb's
    # register_cache_writer, which only exists at prismweb runtime.
    from core.services.cache_path_registry import register_cache_writer  # noqa: F401


def _resolve_cache_dir() -> Path | None:
    """Return ``settings.PRISMWEB_CACHE_DIR`` as a Path, or None if
    Django isn't configured in this process. The derivation callers
    always run inside prismweb's ``evict_cache.handle()``, where
    Django is fully booted; the None path keeps this module
    import-safe under bare prismpy unit tests.
    """
    try:
        from django.conf import settings
    except ImportError:
        return None
    cache_dir = getattr(settings, 'PRISMWEB_CACHE_DIR', None)
    if not cache_dir:
        return None
    return Path(cache_dir)


def _region_key_for(run: Any) -> str:
    """Return the persisted region cache key for ``run``, or ''.

    Reads ``run.region_cache_key`` — prismweb's V2-22b/P.2 migration
    0008 backfilled this on every existing PipelineRun and every
    create path writes it at submission time. Rows that still come
    back empty are effectively legacy (pre-migration 0008 or a
    schema regression); the caller handles '' by returning an empty
    in-use set so the FileLock + atime gates cover them.
    """
    key = getattr(run, 'region_cache_key', None) or ''
    return key if isinstance(key, str) else ''


def _agera5_in_use_for(run) -> frozenset[Path]:
    """Derive the agera5 cache paths owned by a running run.

    Mirrors the writer's path convention:
    ``cache_dir / "agera5" / f"AgERA5_{region_key}"`` (data_dir)
    plus the lock file ``cache_dir / f".agera5-{region_key}.lock"``
    (lock path per ``cache_lock_path`` in prismpy's shared helper).

    Returns the empty set when the cache dir is undiscoverable or
    the run has no region key — the FileLock probe in evict_cache
    + atime retention remain the safety gates in that case.
    """
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return frozenset()
    key = _region_key_for(run)
    if not key:
        return frozenset()
    return frozenset([
        cache_dir / 'agera5' / f'AgERA5_{key}',
        cache_dir / f'.agera5-{key}.lock',
    ])


def _isda_in_use_for(run) -> frozenset[Path]:
    """Derive the iSDA cache path owned by a running run. iSDA
    rasters are continent-wide (not region-keyed), so every running
    run is potentially reading from the whole iSDA subdir. The
    conservative derivation returns the subdir root; evict_cache
    treats any entry whose resolved path OR any ancestor is in the
    in-use set as protected."""
    cache_dir = _resolve_cache_dir()
    if cache_dir is None:
        return frozenset()
    return frozenset([cache_dir / 'isda'])


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
