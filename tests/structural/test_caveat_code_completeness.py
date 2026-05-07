"""Structural pin: ``CaveatCode`` Literal ↔ ``METHODS_TEXT_CAVEAT_PHRASES`` parity.

Sprint E.2 §0.2 canonical-source #7 + AC-E2-6. The ``CaveatCode``
Literal at ``prismpy/standards/caveat_codes.py`` and the
``METHODS_TEXT_CAVEAT_PHRASES`` dict in the same module are a paired
canonical-source: every Literal value MUST have a methods-text
phrase, and every dict key MUST be a Literal value. A new caveat
added to one side without the other fires this pin loud per durable
§24 canonical-source-or-pin discipline.

Sprint S precedent: ``WarningCategory`` enum mirrors a registry-of-
truth via the same set-equality pattern. Sprint G's
``ISIMIP_TO_SARRA_VAR_MAPPING`` enforces a similar producer-vs-
consumer parity.
"""

from __future__ import annotations

import typing

from prismpy.standards.caveat_codes import (
    METHODS_TEXT_CAVEAT_PHRASES,
    CaveatCode,
)


# ── §1 set-equality between Literal args and phrase-dict keys ────────


def test_caveat_literal_args_match_phrase_dict_keys() -> None:
    literal_args = set(typing.get_args(CaveatCode))
    phrase_keys = set(METHODS_TEXT_CAVEAT_PHRASES.keys())

    missing_phrases = literal_args - phrase_keys
    orphan_phrases = phrase_keys - literal_args

    assert not missing_phrases, (
        f"CaveatCode Literal members without methods-text phrases: "
        f"{sorted(missing_phrases)}. Each Literal value MUST have "
        f"exactly one entry in METHODS_TEXT_CAVEAT_PHRASES so the "
        f"methods-text generator (AC-E2-7) has a phrase to emit."
    )
    assert not orphan_phrases, (
        f"METHODS_TEXT_CAVEAT_PHRASES keys without a Literal member: "
        f"{sorted(orphan_phrases)}. Either remove the orphan phrases "
        f"or extend the Literal."
    )


# ── §2 the three Sprint E.2 caveats are all present ─────────────────


def test_caveat_literal_carries_three_e2_codes() -> None:
    """Sprint E.2 ships three caveats (sahel-precip / sahel-wind /
    highland-orographic-excluded). This pin documents the contract
    scope so a future expansion lands as an intentional change."""
    expected = {
        "sahel-precip-convective",
        "sahel-wind-convective",
        "highland-orographic-excluded",
    }
    literal_args = set(typing.get_args(CaveatCode))
    assert literal_args == expected, (
        f"Sprint E.2 CaveatCode scope: {sorted(expected)}. Got: "
        f"{sorted(literal_args)}."
    )


# ── §3 phrases satisfy persona-tone discipline ──────────────────────


def test_phrases_are_non_empty_strings() -> None:
    """A blank phrase emits silently in methods text — that's a
    silent-skip class violation per ``feedback_no_data_cooking.md``.
    Pin the bar at non-empty so a future contributor can't ship a
    placeholder ``""``."""
    for code, phrase in METHODS_TEXT_CAVEAT_PHRASES.items():
        assert phrase, f"CaveatCode {code!r} maps to empty phrase"
        assert phrase.strip() == phrase, (
            f"CaveatCode {code!r} phrase has leading/trailing "
            f"whitespace: {phrase!r}"
        )


def test_phrases_cite_peer_reviewed_evidence() -> None:
    """Every caveat phrase references the underlying domain
    citation (Mathon 2002 / Daly 2006) so a paper reviewer reading
    the manifest can trace the claim. The exact phrase text is
    specified in the AC-E2-7 contract; this is the
    citation-presence-only check."""
    citations = {
        "sahel-precip-convective": "Mathon et al. 2002",
        "sahel-wind-convective": "Mathon et al. 2002",
        "highland-orographic-excluded": "Daly 2006",
    }
    for code, expected_citation in citations.items():
        phrase = METHODS_TEXT_CAVEAT_PHRASES[code]
        assert expected_citation in phrase, (
            f"CaveatCode {code!r} phrase MUST cite {expected_citation!r}; "
            f"got: {phrase!r}"
        )


def test_phrases_avoid_marketing_words() -> None:
    """Persona-tone discipline per VISION.md + Sprint E.2 vocabulary
    contract (AC-E2-14). Forbidden marketing words: seamless /
    powerful / revolutionary / AI-powered / one-click / simple."""
    forbidden = {"seamless", "powerful", "revolutionary", "AI-powered", "one-click"}
    for code, phrase in METHODS_TEXT_CAVEAT_PHRASES.items():
        lowered = phrase.lower()
        for word in forbidden:
            assert word.lower() not in lowered, (
                f"CaveatCode {code!r} phrase contains forbidden "
                f"marketing word {word!r}: {phrase!r}"
            )


# ── §4 dunder-all is the canonical export surface ───────────────────


def test_module_exports_only_canonical_pair_in_dunder_all() -> None:
    """Only ``CaveatCode`` + ``METHODS_TEXT_CAVEAT_PHRASES`` are the
    canonical exports. Internal constants stay private."""
    from prismpy.standards import caveat_codes
    assert sorted(caveat_codes.__all__) == ["CaveatCode", "METHODS_TEXT_CAVEAT_PHRASES"]
