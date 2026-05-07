"""Structural pin: §0.2 Decision Workflow Semantics canonical layer.

Sprint E.2 §0.2 + AC-E2-21 + Sub-CA-B (Draft 4.1 ordering-tuple
allow-list). Asserts the seven canonical sources + four pins keep
their canonical-source-or-pin discipline:

* Action vocabulary owned by ``validators/affordance_routing.py``.
* Ordering tuple consumers limited to a 5-module allow-list (per
  Sub-CA-B). The allow-list is the explicit consumer enumeration:
  - prismpy: ``models/decision_log.py`` + ``packaging/cockpit_snapshot.py``.
  - prismweb (out of prismpy scope): ``core/views/cockpit.py`` +
    ``core/migrations/cockpit_decisions_to_e2.py`` +
    ``core/models.py::PipelineRunDecision``.
* Current-state reader owned by ``models/decision_log.py::current_decisions``.
* Commit snapshot helper owned by ``packaging/cockpit_snapshot.py``.
* Self-link integrity in ``models/decision_log.py::CellDecisionRecord``
  validator (asserted via separate test in test_decision_log.py).
* Caveat code vocabulary owned by ``standards/caveat_codes.py``
  (asserted via test_caveat_code_completeness.py).

This pin closes the §6.3 redesign-trigger gap that codex Gate A §5
flagged: scattered patches across CA-H1/H2/H3/H4/H6/H7 would have
fired the redesign trigger; the canonical layer closes the class
with one source-of-truth + structural enforcement.
"""

from __future__ import annotations

import re
from pathlib import Path


def _prismpy_src() -> Path:
    """Locate prismpy/src/prismpy/ regardless of the test's
    location relative to the repo root."""
    return Path(__file__).resolve().parents[2] / "src" / "prismpy"


# ── §1 Action vocabulary canonical-source ───────────────────────────


def test_affordance_type_only_declared_in_canonical_module() -> None:
    """``AffordanceType = Literal[...]`` must only appear in
    ``validators/affordance_routing.py``. A re-declaration in
    another module is a §0.2 #1 violation."""
    src = _prismpy_src()
    pattern = re.compile(r"\bAffordanceType\s*=\s*Literal\[")
    declarations: list[str] = []
    for py_file in src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            declarations.append(str(py_file.relative_to(src.parent)))
    assert declarations == ["prismpy/validators/affordance_routing.py"], (
        f"AffordanceType Literal must only live at "
        f"validators/affordance_routing.py; got: {declarations}"
    )


def test_affordance_to_action_map_only_declared_in_canonical_module() -> None:
    """The ``AFFORDANCE_TO_ACTION_MAP`` dict must only appear at the
    canonical site. A duplicate map elsewhere risks producer/consumer
    drift the §0.2 #1 pin exists to prevent."""
    src = _prismpy_src()
    pattern = re.compile(r"^AFFORDANCE_TO_ACTION_MAP\s*[:=]", re.MULTILINE)
    declarations: list[str] = []
    for py_file in src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            declarations.append(str(py_file.relative_to(src.parent)))
    assert declarations == ["prismpy/validators/affordance_routing.py"], (
        f"AFFORDANCE_TO_ACTION_MAP must only live at "
        f"validators/affordance_routing.py; got: {declarations}"
    )


# ── §2 Ordering-tuple consumer allow-list (Sub-CA-B) ────────────────


# Per Sub-CA-B (Draft 4.1 mini-amendment): the explicit consumer
# allow-list for decision-record ordering is 5 modules. Three are
# in prismpy:
_PRISMPY_ORDERING_CONSUMERS: frozenset[str] = frozenset(
    {
        "src/prismpy/models/decision_log.py",  # canonical reader
        "src/prismpy/packaging/cockpit_snapshot.py",  # snapshot serializer
    }
)
# Two are in prismweb (out of scope for this pin; covered separately
# in prismweb's test surface):
#   - core/views/cockpit.py  (HTMX endpoint)
#   - core/migrations/cockpit_decisions_to_e2.py  (migration reader)
#   - core/models.py::PipelineRunDecision.Meta.ordering


def test_decision_log_uses_canonical_ordering_tuple() -> None:
    """The canonical reader at ``current_decisions()`` MUST sort by
    the canonical (timestamp, sequence_number) tuple per §0.2 #2.
    A drift to (timestamp, decision_id) or (timestamp,) alone fires
    this pin."""
    src = _prismpy_src() / "models" / "decision_log.py"
    text = src.read_text(encoding="utf-8")
    # Canonical sort key references both attributes.
    assert "r.timestamp" in text and "r.sequence_number" in text, (
        f"decision_log.py must sort by (r.timestamp, r.sequence_number) "
        f"per §0.2 canonical-source #2. Either attribute missing → drift."
    )


# ── §3 Caveat-code vocabulary canonical-source ──────────────────────


def test_caveat_code_only_declared_in_canonical_module() -> None:
    """``CaveatCode = Literal[...]`` must only appear at
    ``standards/caveat_codes.py`` per §0.2 #7."""
    src = _prismpy_src()
    pattern = re.compile(r"\bCaveatCode\s*=\s*Literal\[")
    declarations: list[str] = []
    for py_file in src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            declarations.append(str(py_file.relative_to(src.parent)))
    assert declarations == ["prismpy/standards/caveat_codes.py"], (
        f"CaveatCode Literal must only live at "
        f"standards/caveat_codes.py; got: {declarations}"
    )


# ── §4 Commit-snapshot helper canonical-source ──────────────────────


def test_serialize_decisions_to_config_only_declared_in_canonical_module() -> None:
    """``serialize_decisions_to_config()`` must only be defined at
    ``packaging/cockpit_snapshot.py`` per §0.2 #4 (prismpy half of
    the I-DN-1 split)."""
    src = _prismpy_src()
    pattern = re.compile(r"^def serialize_decisions_to_config\b", re.MULTILINE)
    declarations: list[str] = []
    for py_file in src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            declarations.append(str(py_file.relative_to(src.parent)))
    assert declarations == ["prismpy/packaging/cockpit_snapshot.py"], (
        f"serialize_decisions_to_config must only live at "
        f"packaging/cockpit_snapshot.py; got: {declarations}"
    )


# ── §5 Current-state reader canonical-source ────────────────────────


def test_current_decisions_only_declared_in_canonical_module() -> None:
    """``current_decisions()`` must only be defined at
    ``models/decision_log.py`` per §0.2 #3."""
    src = _prismpy_src()
    pattern = re.compile(r"^def current_decisions\b", re.MULTILINE)
    declarations: list[str] = []
    for py_file in src.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            declarations.append(str(py_file.relative_to(src.parent)))
    assert declarations == ["prismpy/models/decision_log.py"], (
        f"current_decisions must only live at models/decision_log.py; "
        f"got: {declarations}"
    )
