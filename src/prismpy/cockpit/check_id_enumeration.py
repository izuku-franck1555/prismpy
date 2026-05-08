"""Canonical enumeration of every check_id the producer side emits.

Sprint E.2 AC-E2-25 + Codex Gate A MEDIUM B2 + Builder Sub-CA #4.

The cockpit consumer side ships per-check assets (humanized
labels, plain-language descriptions, "why this check exists"
copy) keyed by ``check_id``. Without a canonical enumeration of
every emitted ``check_id`` the producer side carries, the
consumer-side registries (``CHECK_ID_DESCRIPTIONS`` per AC-E2-26
+ ``check_id_reasons.json`` per AC-E2-25) drift silently — a new
producer-emitted check_id ships without a consumer-side
description, the structural pin doesn't catch it (because the
"completeness" assertion can't enumerate what doesn't have a
canonical roster), and the cockpit renders the raw snake_case
string in a body-copy slot that violates the F-E2-3 ban.

Per durable §24 canonical-source-or-pin:

* :func:`enumerate_emitted_check_ids` returns the union of
  every ``check_id`` the producer side emits — the canonical
  truth set the consumer-side completeness pin asserts against.
* The 3 source pools below cover the producer landscape: the
  validators/scientific.py module + the validators/post_translate.py
  module + the executor.py prefix-derived families. A structural
  pin AST-walks each pool's source file + asserts the helper's
  union equals the AST-derived emission set; drift in either
  direction (unenumerated emit / phantom enumeration) fires loud
  at CI time.

Per durable §27 two-vocabulary substrate-drift: the producer's
``check_id`` namespace is the canonical vocabulary; consumer-side
registries (descriptions / reasons / labels) MUST key off this
namespace + structural pins MUST trace back to the helper's
output.
"""
from __future__ import annotations

from typing import Final

from prismpy.validators.scientific import (
    CLIMATE_RANGES,
    CLIMATE_TIER1_CHECKS,
    PLATFORM_SOIL_REQUIREMENTS,
    SOIL_CHECKS_ALL,
    SOIL_RANGES,
)


# ── Pool 1: validators/scientific.py ────────────────────────────────


# Sentinel + axis-level + per-variable + per-layer checks the
# Sprint E.0 ``run_scientific_validation`` orchestrator emits.
# Aggregates the existing canonical iteration constants
# (:data:`CLIMATE_TIER1_CHECKS` + :data:`SOIL_CHECKS_ALL`) +
# the additional checks (``format_compliance`` /
# ``spatial_temporal_coverage``) that aren't in the climate /
# soil tiers but ride alongside.
#
# Note: ``CLIMATE_TIER1_CHECKS`` already expands
# ``value_range_<var>`` over :data:`CLIMATE_RANGES` and
# ``SOIL_CHECKS_ALL`` already expands ``value_range_soil_<var>``
# over :data:`SOIL_RANGES`; reading those same source-of-truth
# dicts here would duplicate canonical iteration. Instead we
# import the already-expanded tuples so adding a new variable
# to ``CLIMATE_RANGES`` / ``SOIL_RANGES`` automatically
# propagates here without an extra rename pass.
_VALIDATOR_AXIS_LEVEL: Final[frozenset[str]] = frozenset({
    "format_compliance",
    "spatial_temporal_coverage",
})


def _validator_check_ids() -> frozenset[str]:
    """Compose the validator pool's full check_id set.

    Combines :data:`CLIMATE_TIER1_CHECKS` + :data:`SOIL_CHECKS_ALL`
    + the axis-level extras + the per-platform soil completeness
    fan-out (``soil_completeness_<platform>`` for each platform
    in :data:`PLATFORM_SOIL_REQUIREMENTS`).
    """
    out: set[str] = set()
    out.update(CLIMATE_TIER1_CHECKS)
    out.update(SOIL_CHECKS_ALL)
    out.update(_VALIDATOR_AXIS_LEVEL)
    # Per-platform soil completeness checks — the validators emit
    # ``soil_completeness_<platform>`` where ``<platform>`` is a
    # key in :data:`PLATFORM_SOIL_REQUIREMENTS`.
    out.update(
        f"soil_completeness_{platform}"
        for platform in PLATFORM_SOIL_REQUIREMENTS
    )
    return frozenset(out)


VALIDATOR_CHECK_IDS: Final[frozenset[str]] = _validator_check_ids()


# ── Pool 2: validators/post_translate.py ────────────────────────────


# The post-translate validators emit one check per platform per
# variable. The platform set covers all 4 PRISMWEB-supported
# platforms; the variable set mirrors the climate range vocabulary.
#
# Static check ids (per-platform climate aggregator) + parametric
# check ids (per-platform per-variable + per-platform date
# continuity + per-platform consistency) are aggregated via the
# helper below.
_POST_TRANSLATE_PLATFORMS: Final[tuple[str, ...]] = (
    "acea", "craft", "pythia", "sarra_py",
)


def _post_translate_check_ids() -> frozenset[str]:
    """Compose the post-translate pool's full check_id set.

    Per-platform aggregator + per-platform per-variable range
    + per-platform date continuity + per-platform consistency
    + SARRA-Py completeness sentinel.
    """
    out: set[str] = set()
    for platform in _POST_TRANSLATE_PLATFORMS:
        # Per-platform climate aggregator (one record per platform).
        out.add(f"post_translate_climate_{platform}")
        # Per-platform per-variable range (one record per
        # platform × climate variable).
        out.update(
            f"post_translate_range_{platform}_{var}"
            for var in CLIMATE_RANGES
        )
        # Per-platform date continuity + consistency.
        out.add(f"post_translate_date_continuity_{platform}")
        out.add(f"post_translate_consistency_{platform}")
    # SARRA-Py file-coverage sentinel — only emitted on the SARRA
    # path because that's the only platform with a daily-file
    # shape rather than a per-record series.
    out.add("post_translate_completeness_sarra_py")
    return frozenset(out)


POST_TRANSLATE_CHECK_IDS: Final[frozenset[str]] = _post_translate_check_ids()


# ── Pool 3: executor.py prefix-derived families ─────────────────────


# The pipeline executor's per-cell pivot at
# ``executor.py:_CATEGORY_FROM_PREFIX`` (see executor.py:3434)
# uses prefix matching to project per-cell ``failed_checks``
# into dimension-toggle category strings. The prefixes here
# match the AST-walkable tuple in executor.py; the structural
# pin re-reads the executor's source to verify parity.
#
# ``VALUE_RANGE_PREFIX_FAMILIES`` — the prefix strings whose
# match-set covers per-cell-scoped check ids that the producer
# emits but might not exist in the static enumerations above
# (e.g., a future ``value_range_<new_var>``). Including the
# prefixes keeps the helper's union closed under future
# variable additions to :data:`CLIMATE_RANGES` /
# :data:`SOIL_RANGES`.
VALUE_RANGE_PREFIX_FAMILIES: Final[tuple[str, ...]] = (
    "value_range_",
    "value_range_soil_",
    "soil_completeness_",
    "post_translate_range_",
    "post_translate_climate_",
    "post_translate_date_continuity_",
    "post_translate_consistency_",
)


# ── Public helper ───────────────────────────────────────────────────


def enumerate_emitted_check_ids() -> frozenset[str]:
    """Return the union of every check_id the producer side emits.

    Aggregates all 3 source pools — :data:`VALIDATOR_CHECK_IDS`
    (validators/scientific.py expansions) + :data:`POST_TRANSLATE_CHECK_IDS`
    (validators/post_translate.py per-platform fan-out) + the
    executor.py prefix-derived families.
    """
    return frozenset(VALIDATOR_CHECK_IDS | POST_TRANSLATE_CHECK_IDS)


def matches_known_prefix(check_id: str) -> bool:
    """True iff ``check_id`` starts with one of the canonical
    prefix families.

    Used by the consumer-side completeness pin's relaxation rule
    — a check_id that doesn't appear in the static enumeration
    but matches a known prefix (e.g., a future
    ``value_range_<new_var>``) is allowed to exist in the
    consumer registry too. The static enumeration enforces the
    strict floor; the prefix relaxation closes the union under
    legitimate future additions.
    """
    return any(
        check_id.startswith(prefix)
        for prefix in VALUE_RANGE_PREFIX_FAMILIES
    )


__all__ = [
    "POST_TRANSLATE_CHECK_IDS",
    "VALIDATOR_CHECK_IDS",
    "VALUE_RANGE_PREFIX_FAMILIES",
    "enumerate_emitted_check_ids",
    "matches_known_prefix",
]
