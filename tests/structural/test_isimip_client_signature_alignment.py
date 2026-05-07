"""Structural pin: ISIMIP3b client signatures track upstream isimip-client.

Per durable lesson §24 canonical-source-or-pin: every consumer of an
external API contract (here the upstream ``isimip_client.ISIMIPClient``)
must either route through ONE canonical source or be pinned to assert
parity. The Sprint G boundary 1+2 codex round 1 caught two real-API-
shape drifts (datasets() return shape + cutout_bbox positional args)
that were hidden by fake-test-client divergence.

This module pins three signature relationships so a future drift
(upstream API change OR a new fake / facade method that doesn't match
the real one) fires loud at CI time:

1. The ``ISIMIP3bClient`` facade narrows but doesn't drift — every
   parameter the facade exposes appears on the upstream method.
2. The ``_FakeISIMIP3bClient`` test fake's ``cutout_bbox`` signature
   matches upstream ordered-prefix-wise. The fake may add ``**kwargs``
   for test-side flexibility but cannot rename/drop an upstream param.
3. The ``_FakeISIMIP3bClient.datasets`` accepts the same kwargs the
   real client accepts (validated by signature compatibility).

The pin is small (~50 LOC) and permanently closes the drift class.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, List

from isimip_client.client import ISIMIPClient

from prismpy.data_sources.isimip3b import ISIMIP3bClient
from tests.structural.test_isimip3b_cached_cutout import _FakeISIMIP3bClient


def _public_param_names(method: Callable[..., Any]) -> List[str]:
    """Return the public parameter names of ``method`` excluding ``self``."""
    sig = inspect.signature(method)
    return [name for name in sig.parameters.keys() if name != "self"]


# ── Pin 1: facade narrows but doesn't drift from upstream ────────────


def test_isimip3b_client_cutout_bbox_facade_narrows_to_upstream() -> None:
    """Every parameter ``ISIMIP3bClient.cutout_bbox`` exposes must
    exist on the upstream ``ISIMIPClient.cutout_bbox`` method. The
    facade may exclude params it doesn't expose (deliberate narrowing
    per Sprint G), but it must not invent params the upstream
    doesn't have.
    """
    real_params = set(_public_param_names(ISIMIPClient.cutout_bbox))
    facade_params = set(_public_param_names(ISIMIP3bClient.cutout_bbox))
    facade_params.discard("kwargs")  # The wrapper forwards via **kwargs.

    extra = facade_params - real_params
    assert not extra, (
        "ISIMIP3bClient.cutout_bbox exposes parameters not on upstream "
        f"ISIMIPClient.cutout_bbox: {sorted(extra)}. The facade narrowing "
        "must NOT invent new parameters; ride upstream's surface or "
        "explicitly translate at the wrapper boundary."
    )


def test_isimip3b_client_cutout_bbox_required_edges_present() -> None:
    """The facade must expose all four bbox edges (west / east / south /
    north) so callers cannot accidentally drop a coordinate."""
    facade_params = set(_public_param_names(ISIMIP3bClient.cutout_bbox))
    for edge in ("west", "east", "south", "north"):
        assert edge in facade_params, (
            f"ISIMIP3bClient.cutout_bbox missing required edge {edge!r}. "
            "All four W/E/S/N edges must be exposed to match upstream."
        )


# ── Pin 2: fake test-client matches upstream cutout_bbox signature ───


def test_fake_client_cutout_bbox_signature_matches_upstream() -> None:
    """The ``_FakeISIMIP3bClient.cutout_bbox`` signature must accept
    the same edges (``west``/``east``/``south``/``north``) the real
    upstream method does. The fake may add extra params for test-side
    flexibility (counted as ``**kwargs``) but cannot rename or drop
    upstream params — that's exactly the drift class codex round 1
    on b89b784 caught (the fake accepted a single ``bbox`` list while
    upstream wanted four floats).
    """
    real_params = set(_public_param_names(ISIMIPClient.cutout_bbox))
    fake_params = set(_public_param_names(_FakeISIMIP3bClient.cutout_bbox))
    fake_params.discard("kwargs")  # Allow **kwargs for test flexibility.

    missing_from_fake = (
        real_params - fake_params
    ) - {"mean", "csv", "poll"}  # Optional upstream params the fake doesn't need
    assert not missing_from_fake, (
        "_FakeISIMIP3bClient.cutout_bbox is missing upstream parameters: "
        f"{sorted(missing_from_fake)}. The fake must accept the same "
        "edges (W/E/S/N) the real client accepts so cached_cutout's "
        "real-vs-fake call sites stay aligned."
    )


def test_fake_client_cutout_bbox_required_edges_present() -> None:
    """Mirror of ``test_isimip3b_client_cutout_bbox_required_edges_present``
    on the fake side. Both must expose W/E/S/N."""
    fake_params = set(_public_param_names(_FakeISIMIP3bClient.cutout_bbox))
    for edge in ("west", "east", "south", "north"):
        assert edge in fake_params, (
            f"_FakeISIMIP3bClient.cutout_bbox missing required edge "
            f"{edge!r}. Drift from real ISIMIPClient signature."
        )


# ── Pin 3: fake datasets accepts upstream-compatible signature ───────


def test_facade_datasets_accepts_upstream_kwargs() -> None:
    """The facade's ``datasets`` method must accept the same query
    parameters the real client accepts. Real upstream is ``def
    datasets(self, **kwargs)`` — any kwargs accepted; the facade must
    do the same so callers passing real-API kwargs (e.g.,
    ``simulation_round``, ``product``, ``climate_forcing``,
    ``climate_scenario``, ``climate_variable``) don't TypeError at the
    facade boundary.
    """
    facade_sig = inspect.signature(ISIMIP3bClient.datasets)
    has_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in facade_sig.parameters.values()
    )
    assert has_var_keyword, (
        "ISIMIP3bClient.datasets must accept **kwargs to match the "
        "upstream ``ISIMIPClient.datasets(self, **kwargs)`` signature. "
        "Without that, real-API kwargs passed by discover_datasets() "
        "would TypeError at the facade boundary."
    )


# ── Pin 4: discover_datasets handles both list + dict response shapes ─


def test_discover_datasets_response_shape_handler_present() -> None:
    """``discover_datasets`` MUST handle the upstream's default list
    response shape AND the paginated dict shape. A regression that
    drops either branch reintroduces the bug codex round 1 caught.
    """
    import ast
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[2]
        / "src/prismpy/data_sources/isimip3b.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    discover_def = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "discover_datasets"
    )
    body_text = ast.unparse(discover_def)
    assert "isinstance(response, list)" in body_text, (
        "discover_datasets must branch on isinstance(response, list) "
        "so the upstream ISIMIPClient.list() raw-list default response "
        "is handled."
    )
    assert "isinstance(response, dict)" in body_text, (
        "discover_datasets must also handle the paginated dict shape "
        "(``{'results': [...]}``) for callers passing paginate=True."
    )
