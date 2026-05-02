"""Harmonize-stage transformation thresholds.

Threshold rationale (Sprint D.1, AC-1.2):

``TEXTURE_RENORMALIZE_THRESHOLD_PCT`` (3.0) is the conservative
inner shell within the existing validator's [95, 105] tolerance
band at ``validators/scientific.py:1477`` — i.e., the
renormalize-eligible band sits inside the validator's accept
band (+/- 5%) so a renormalized layer is always within bounds
the validator already considers acceptable. The outer band
``TEXTURE_WARN_THRESHOLD_PCT`` (5.0) matches the validator's
upper edge: layers whose texture sum drifts more than 5% from
100 cannot be renormalized without misrepresenting the source
fractions; the harmonize stage routes those cells to
``unavailable`` instead of silently transforming them.

Citations:

* Jones et al. 2003 — DSSAT v4.7 Soil File Format Specification
  (the ICASA convention for texture-fraction reporting).
* ``validators/scientific.py:1477`` — the [95, 105] tolerance
  band the validator already enforces. The harmonize stage's
  renormalize-eligible band (+/- 3%) is the strict inner shell
  of that validator window so renormalization can never push a
  layer outside the validator's accept band.

``RH_CLIP_THRESHOLD_PCT`` (102.0) is the physical-rounding
tolerance NASA POWER and AgERA5 reanalyses are known to produce
near 100% saturation; values within (100, 102] are clipped to
100 with a provenance entry; values above 102 fall outside any
defensible rounding window and route the cell's climate axis to
``unavailable`` with cause ``climate_rh_invalid``.
"""
TEXTURE_RENORMALIZE_THRESHOLD_PCT: float = 3.0
TEXTURE_WARN_THRESHOLD_PCT: float = 5.0
RH_PHYSICAL_MAX_PCT: float = 100.0
RH_CLIP_THRESHOLD_PCT: float = 102.0
