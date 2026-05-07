"""Shared helpers used across the platform-specific translators.

Modules in this subpackage hold logic that more than one translator (or a
translator plus an ancillary subsystem such as the eGHR substrate builder)
needs to call. Each helper is responsible for emitting a single canonical
artifact format, so consumers cannot drift apart on format details.
"""

from prismpy.translators._shared.dssat_sol_writer import write_dssat_sol

__all__ = ["write_dssat_sol"]
