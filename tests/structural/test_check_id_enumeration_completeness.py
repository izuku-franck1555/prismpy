"""Sprint E.2 AC-E2-25 + Codex Gate A MEDIUM B2 + Builder Sub-CA
#4 — :func:`prismpy.cockpit.check_id_enumeration.enumerate_emitted_check_ids`
covers every check_id the producer side emits.

Producer-side ``check_id`` strings are emitted from 3 source
pools per the canonical aggregation:

1. :mod:`prismpy.validators.scientific` — climate + soil tier
   validators (axis-level + per-variable + per-layer + texture-
   sum + coverage families).
2. :mod:`prismpy.validators.post_translate` — per-platform
   post-translate fan-out (per-platform climate aggregator,
   per-platform per-variable range, per-platform date
   continuity, per-platform consistency, SARRA file-coverage
   sentinel).
3. :mod:`prismpy.pipeline.executor` — the ``_CATEGORY_FROM_PREFIX``
   tuple at line 3434 narrows per-cell pivot prefix-derived
   families.

The consumer-side description registry (``CHECK_ID_DESCRIPTIONS``
at prismweb / Phase 2) MUST cover every producer-emitted
``check_id``. Per durable §24 + §27 the helper composes the
canonical truth set + this pin asserts the helper's union is
LARGER than (or equal to) the AST-walk-derived emission set
across the 3 producer source files.

Pin pattern: AST-walk each pool's source file, collect every
literal ``"check": "..."`` string + every f-string expansion
template (``f"value_range_{var}"``), and assert the helper's
union is a superset (or matches via the prefix-tolerance
relaxation in :func:`matches_known_prefix`).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from prismpy.cockpit.check_id_enumeration import (
    enumerate_emitted_check_ids,
    matches_known_prefix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_FILE = (
    REPO_ROOT / "src" / "prismpy" / "validators" / "scientific.py"
)
POST_TRANSLATE_FILE = (
    REPO_ROOT / "src" / "prismpy" / "validators" / "post_translate.py"
)
EXECUTOR_FILE = (
    REPO_ROOT / "src" / "prismpy" / "pipeline" / "executor.py"
)


def _extract_check_id_literals(source: str) -> set[str]:
    """Walk the AST + collect every literal string assigned to a
    ``"check"`` key in a dict literal. Captures concrete check
    ids (e.g., ``"temporal_completeness"``) but not f-string
    parametric forms (those go through prefix-tolerance below).
    """
    tree = ast.parse(source)
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "check"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                out.add(value.value)
    return out


def _extract_fstring_check_id_patterns(source: str) -> set[str]:
    """Walk the AST + collect f-string check-id patterns. The
    f-strings are intentionally parametric (``f"value_range_{var}"``);
    the pin's prefix-tolerance handler covers them via the
    ``VALUE_RANGE_PREFIX_FAMILIES`` constant — this helper just
    extracts the literal prefix portion for sanity-checking that
    the producer's f-string template prefixes match the
    canonical prefix family.
    """
    tree = ast.parse(source)
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "check"
                and isinstance(value, ast.JoinedStr)
            ):
                # f-string — extract the leading literal portion.
                first_part = value.values[0]
                if (
                    isinstance(first_part, ast.Constant)
                    and isinstance(first_part.value, str)
                ):
                    out.add(first_part.value)
    return out


@pytest.fixture(scope="module")
def emitted_concrete_check_ids() -> set[str]:
    """Collect every concrete (non-parametric) check_id the
    producer side emits across all 3 pools."""
    out: set[str] = set()
    for src_file in (SCIENTIFIC_FILE, POST_TRANSLATE_FILE, EXECUTOR_FILE):
        try:
            text = src_file.read_text(encoding="utf-8")
        except OSError:
            continue
        out |= _extract_check_id_literals(text)
    return out


@pytest.fixture(scope="module")
def emitted_fstring_prefixes() -> set[str]:
    """Collect every parametric (f-string) check_id prefix the
    producer side emits."""
    out: set[str] = set()
    for src_file in (SCIENTIFIC_FILE, POST_TRANSLATE_FILE, EXECUTOR_FILE):
        try:
            text = src_file.read_text(encoding="utf-8")
        except OSError:
            continue
        out |= _extract_fstring_check_id_patterns(text)
    return out


def test_enumerate_emitted_check_ids_returns_non_empty() -> None:
    """Sanity floor — the helper returns a non-empty frozenset.
    Without callsites the pin would silently pass even with
    all-broken pool composition."""
    enumerated = enumerate_emitted_check_ids()
    assert len(enumerated) >= 50, (
        f"enumerate_emitted_check_ids() returned {len(enumerated)} "
        f"check ids; expected >= 50 across 3 pools (validators "
        f"+ post-translate + prefix families). Did the pool "
        f"composition break?"
    )


def test_concrete_emitted_check_ids_covered_by_helper(
    emitted_concrete_check_ids: set[str],
) -> None:
    """Every concrete check_id literal the producer emits MUST
    be either:

    1. In :func:`enumerate_emitted_check_ids` direct output, OR
    2. Match a known prefix via :func:`matches_known_prefix`
       (the prefix-tolerance relaxation closes the union under
       legitimate parametric expansions).
    """
    enumerated = enumerate_emitted_check_ids()
    offenders: list[str] = []
    # Filter sentinels — ``"<check_name>"`` is the docstring
    # placeholder convention at scientific.py:14; not a real
    # check_id. Skip explicitly.
    docstring_sentinels = {
        "<check_name>",
    }
    for check_id in sorted(emitted_concrete_check_ids):
        if check_id in docstring_sentinels:
            continue
        if check_id in enumerated:
            continue
        if matches_known_prefix(check_id):
            continue
        offenders.append(check_id)
    assert not offenders, (
        f"Producer-emitted check ids missing from helper "
        f"enumeration AND not matching a known prefix family "
        f"(per durable §24 + §27 the consumer-side description "
        f"registry can't catch these): {offenders}"
    )


def test_fstring_prefixes_match_canonical_families(
    emitted_fstring_prefixes: set[str],
) -> None:
    """Every f-string check-id prefix the producer uses MUST be
    a known family per :data:`VALUE_RANGE_PREFIX_FAMILIES`.
    Catches a future producer that ships a new f-string template
    without updating the canonical prefix list."""
    from prismpy.cockpit.check_id_enumeration import (
        VALUE_RANGE_PREFIX_FAMILIES,
    )
    offenders: list[str] = []
    for prefix in sorted(emitted_fstring_prefixes):
        # An f-string prefix matches a known family iff the
        # prefix string starts-with one of the canonical
        # prefixes (allows ``"value_range_"`` vs full
        # ``"post_translate_range_<platform>_"`` mid-string).
        if any(prefix.startswith(family) for family in VALUE_RANGE_PREFIX_FAMILIES):
            continue
        offenders.append(prefix)
    assert not offenders, (
        f"Producer f-string check-id prefixes outside the "
        f"canonical VALUE_RANGE_PREFIX_FAMILIES set: {offenders}"
    )


# ── AC-α-7 axis-unavailable synthetic prefix family ──────────────────


def test_axis_unavailable_synthetic_prefix_accepts_climate_and_soil() -> None:
    """The consumer-side cockpit records "Acknowledge as known gap"
    decisions on data-unavailable cells via a synthetic check_id of
    shape ``__axis_unavailable__:<axis>``. There's no real producer
    check to acknowledge against — the cell has no value to flag —
    so the validator's prefix-family allowlist accepts the synthetic
    token without forcing the consumer to fabricate a fake check.
    """
    assert matches_known_prefix("__axis_unavailable__:climate")
    assert matches_known_prefix("__axis_unavailable__:soil")
    assert matches_known_prefix("__axis_unavailable__:climate_and_soil")


def test_axis_unavailable_synthetic_prefix_round_trips_decision_record() -> None:
    """End-to-end: a ``CellDecisionRecord`` carrying the synthetic
    token constructs cleanly through the canonical-membership
    validator. Prior to this prefix-family extension the pydantic
    model raised ``ValueError`` because the token matched no
    enumerated producer check_id.
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from prismpy.models.decision_log import CellDecisionRecord

    record = CellDecisionRecord(
        decision_id=uuid4(),
        cell_id="42",
        action="acknowledge",
        check_id="__axis_unavailable__:climate",
        timestamp=datetime.now(tz=timezone.utc),
        user_identity="builder",
        method_or_rationale="ack",
        sequence_number=1,
    )
    assert record.check_id == "__axis_unavailable__:climate"


def test_unknown_synthetic_prefix_still_rejected() -> None:
    """Belt-and-suspenders — only the documented
    ``__axis_unavailable__:<axis>`` family relaxes the validator;
    an arbitrary leading-underscore token still fails, and the
    delimiter is part of the family (a no-colon variant like
    ``__axis_unavailable__climate`` must NOT silently widen).
    """
    assert not matches_known_prefix("__foo__:bar")
    assert not matches_known_prefix("__not_a_known_prefix__:anything")
    # Delimiter-stripped variant is malformed and must reject.
    assert not matches_known_prefix("__axis_unavailable__climate")
    # Empty-suffix variant carries no axis info and must reject.
    assert not matches_known_prefix("__axis_unavailable__")


def test_real_check_ids_still_validated_unchanged() -> None:
    """The prefix-family extension is additive — every concrete
    producer-side check_id that was accepted before still is.
    """
    for sample in (
        "value_range_soil_ph",
        "value_range_climate_tmax",
        "post_translate_climate_tmin_continuity",
        "soil_completeness_layer_ph",
    ):
        assert matches_known_prefix(sample), (
            f"prefix-family regression on {sample!r}"
        )
