"""The soil each engine reads, declared from the files it reads.

The code path that selects a package's soil stamps the files it writes (a record line in
each ``.SOL``, a netCDF attribute for ACEA) and, after the last write of the set the engine
reads, a binding file holding the sha256 of every file in that set. ``declared_soil``
verifies the binding, reads the records and derives the package's soil declaration; the
manifest and the README take the soil they name from it and from nothing else.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

RECORD_PREFIX = "! prismpy soil record:"
BINDING_FILE = "prismpy_soil_binding.txt"
BINDING_HEADER = "! prismpy soil binding v1"
DETAIL_FILE = "soil_record.json"
ACEA_SOIL_FILE = "soil/HWSD_soil_data_on_cropland_v2.3.nc"
ACEA_RECORD_ATTR = "prismpy_soil_record"


class SoilDeclarationError(Exception):
    """A package's soil files do not carry a bound, consistent soil declaration."""


#: ``SoilProfile.source`` -> the profile-source token written in SLSOUR and the record.
PROFILE_SOURCE_TOKENS: Dict[str, str] = {
    "iSDA S3 (30m)": "isda_s3", "isda": "isda", "hwsd": "hwsd", "eghr": "eghr",
    "placeholder": "placeholder", "default": "default",
}
#: Tokens of the generic soils: a CRAFT package declares their cells through the clause below.
GENERIC_TOKENS = ("default", "placeholder")
SOL_SOURCES = ("profiles", "default_profile", "none")
EGHR_DATABASE = "eghr_database"
ACEA_RECORD_KEYS: Dict[str, Tuple[str, ...]] = {
    "hwsd_upper_layer_texture": ("source", "cells", "hwsd", "default", "masked"),
    "default_values": ("source", "cells"),
    "profiles_by_list_position": ("source", "profile_sources", "cells", "profile", "field_default", "default"),
}
#: The ``.SOL`` record's keys, in the order the writer writes them.
SOL_RECORD_KEYS = (
    "source", "profile_sources", "no_profile_cells", "override_cells", "ptf_layers",
    "profiles", "layers", "organic_layers", "andic_layers", "chem_defaulted", "chem_default_values",
    "cells_with_flagged_layers", "cells_with_defaulted_chemistry", "cells", "default_cells",
    "default_cause", "default_fraction", "default_warning", "default_profile", "default_depth_cm",
    "default_paw_mm",
)
_COUNT_KEYS = {"profile_sources", "chem_defaulted", "default_cause"}
_TEXT_KEYS = {"source", "chem_default_values", "default_fraction", "default_profile", "default_depth_cm"}
HYDRAULICS = {"method": "saxton_rawls_2006_simplified",
              "reference": "Saxton KE, Rawls WJ (2006) Soil Sci. Soc. Am. J. 70:1569–1578"}


def profile_source_token(source: str) -> str:
    """The closed token for a profile's source; an unknown source is refused."""
    try:
        return PROFILE_SOURCE_TOKENS[source]
    except KeyError:
        raise SoilDeclarationError(f"unknown soil profile source {source!r}") from None


@dataclass(frozen=True)
class SoilStamp:
    """What the code path that selected the soil knows; the writer adds what it writes."""

    source: str
    no_profile_cells: int = 0
    override_cells: int = 0

    def __post_init__(self) -> None:
        if self.source not in SOL_SOURCES:
            raise SoilDeclarationError(f"a .SOL cannot be stamped with source {self.source!r}")


def profile_origin(profile: Any) -> Dict[str, Any]:
    """Where a written profile came from, for the detail file."""
    token = profile_source_token(profile.source)
    if token == "hwsd":
        return {"hwsd_smu_id": int(profile.metadata["hwsd_smu_id"]),
                "component_rule": "max_share_then_lowest_sequence"}
    if token == "default":
        return {"generic_default": "branch-4 profile"}
    if token == "placeholder":
        return {"executor_placeholder": True}
    return {"source_cell_id": profile.metadata.get("source_cell_id")}


# ── the record grammar ─────────────────────────────────────────────────────────


def format_counts(counts: Iterable[Tuple[str, int]]) -> str:
    """``key:n,key:n`` in the order given, or ``-`` when there is none."""
    return ",".join(f"{key}:{n}" for key, n in counts) or "-"


def parse_counts(text: str) -> Dict[str, int]:
    if text == "-":
        return {}
    counts: Dict[str, int] = {}
    for part in text.split(","):
        key, sep, value = part.partition(":")
        if not sep or not key or key in counts or not value.isdigit():
            raise SoilDeclarationError(f"malformed count list {text!r}")
        counts[key] = int(value)
    return counts


def format_record(fields: Sequence[Tuple[str, Any]]) -> str:
    """The one-line record: the prefix, then ``key=value`` fields; no value holds a space."""
    parts = []
    for key, value in fields:
        text = str(value)
        if not text or any(ch.isspace() for ch in text) or not key.isidentifier():
            raise ValueError(f"record field {key}={text!r} is not writable")
        parts.append(f"{key}={text}")
    return f"{RECORD_PREFIX} {' '.join(parts)}"


def parse_record(text: str) -> Dict[str, str]:
    if not text.startswith(RECORD_PREFIX):
        raise SoilDeclarationError("not a prismpy soil record")
    fields: Dict[str, str] = {}
    for part in text[len(RECORD_PREFIX):].split():
        key, sep, value = part.partition("=")
        if not sep or not key or key in fields:
            raise SoilDeclarationError(f"malformed soil record field {part!r}")
        fields[key] = value
    return fields


def _typed(fields: Mapping[str, str], keys: Sequence[str]) -> Dict[str, Any]:
    """The record with its exact keys in order, counts and integers parsed."""
    if tuple(fields) != tuple(keys):
        raise SoilDeclarationError(f"soil record keys {list(fields)} are not {list(keys)}")
    typed: Dict[str, Any] = {}
    for key, value in fields.items():
        if key in _COUNT_KEYS:
            typed[key] = parse_counts(value)
        elif key in _TEXT_KEYS:
            typed[key] = None if value == "-" and key in ("default_profile", "default_depth_cm") else value
        elif key == "default_paw_mm" and value == "-":
            typed[key] = None
        elif value.isdigit():
            typed[key] = int(value)
        else:
            raise SoilDeclarationError(f"soil record field {key}={value!r} is not a count")
    for token in typed.get("profile_sources", {}):
        if token not in PROFILE_SOURCE_TOKENS.values():
            raise SoilDeclarationError(f"unknown profile-source token {token!r}")
    return typed


# ── binding and detail files ───────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_binding(digests: Mapping[str, str], source: Optional[str] = None) -> str:
    lines = [BINDING_HEADER] + ([f"source={source}"] if source else [])
    lines += [f"{digests[path]}  {path}" for path in sorted(digests)]
    return "\n".join(lines) + "\n"


def parse_binding(text: str) -> Tuple[Optional[str], Dict[str, str]]:
    lines = text.splitlines()
    if not lines or lines[0] != BINDING_HEADER:
        raise SoilDeclarationError("not a prismpy soil binding")
    source = None
    if len(lines) > 1 and lines[1].startswith("source="):
        source, lines = lines[1][len("source="):], lines[:1] + lines[2:]
    digests: Dict[str, str] = {}
    for line in lines[1:]:
        digest, sep, path = line.partition("  ")
        if not sep or len(digest) != 64 or path in digests or path.startswith("/") or ".." in Path(path).parts:
            raise SoilDeclarationError(f"malformed soil binding line {line!r}")
        digests[path] = digest
    return source, digests


def write_binding(package_dir: Path, binding_dir: str, files: Iterable[Path],
                  source: Optional[str] = None) -> Path:
    """Bind ``files`` (inside ``package_dir``) by their sha256; written after their last write."""
    package_dir = Path(package_dir)
    digests = {Path(f).resolve().relative_to(package_dir.resolve()).as_posix(): sha256_file(Path(f))
               for f in files}
    path = package_dir / binding_dir / BINDING_FILE
    path.write_text(format_binding(digests, source))
    return path


def check_binding(package_dir: Path, binding_dir: str, expected: Iterable[str]) -> Optional[str]:
    """The binding lists exactly ``expected`` and every digest matches; returns its source line."""
    path = Path(package_dir) / binding_dir / BINDING_FILE
    if not path.is_file():
        raise SoilDeclarationError(f"{binding_dir}/{BINDING_FILE} is missing")
    source, digests = parse_binding(path.read_text())
    if set(digests) != set(expected):
        raise SoilDeclarationError(
            f"{binding_dir}/{BINDING_FILE} binds {sorted(digests)}, not the soil files {sorted(expected)}")
    for rel, digest in digests.items():
        target = Path(package_dir) / rel
        if not target.is_file() or sha256_file(target) != digest:
            raise SoilDeclarationError(f"{rel} changed after it was bound")
    return source


def format_detail(detail: Mapping[str, Any]) -> str:
    return json.dumps(detail, indent=1, sort_keys=True) + "\n"


# ── reading a .SOL ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SolFile:
    record: Optional[Dict[str, str]]
    profiles: Dict[str, Tuple[str, int]]  # profile name -> (SLSOUR token, header depth cm)


def read_sol(path: Path) -> SolFile:
    """The record line (before the first profile, at most one) and each profile's header."""
    record: Optional[Dict[str, str]] = None
    profiles: Dict[str, Tuple[str, int]] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("*SOILS"):
            continue
        if line.startswith(RECORD_PREFIX):
            if record is not None or profiles:
                raise SoilDeclarationError(f"{Path(path).name}: a soil record out of place")
            record = parse_record(line)
        elif line.startswith("*"):
            name, token, depth = line[1:11].strip(), line[13:24].strip(), line[31:36].strip()
            if name in profiles or not depth.isdigit():
                raise SoilDeclarationError(f"{Path(path).name}: unreadable profile header {line!r}")
            profiles[name] = (token, int(depth))
    return SolFile(record, profiles)


# ── the declaration ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SoilDeclaration:
    source_id: str
    label: str
    record: Dict[str, Any]

    def inputs_used(self) -> Dict[str, Any]:
        return {"source_id": self.source_id, "label": self.label, "record": self.record}


def declared_soil(package_dir: Path, platform: Any) -> SoilDeclaration:
    """The soil ``platform``'s engine reads from this package, verified and derived from the
    files themselves. It reads files, writes none and keeps no state between calls."""
    name = str(getattr(platform, "value", platform)).lower()
    reader = {"craft": _craft, "pythia": _pythia, "acea": _acea, "sarra_py": _sarra}.get(name)
    if reader is None:
        raise SoilDeclarationError(f"no soil declaration for platform {name!r}")
    return reader(Path(package_dir))


def _craft(package_dir: Path) -> SoilDeclaration:
    sols = sorted((package_dir / "soil").glob("*.SOL"))
    if len(sols) != 1:
        raise SoilDeclarationError(f"a CRAFT package holds one soil/*.SOL, found {len(sols)}")
    sol_rel = f"soil/{sols[0].name}"
    check_binding(package_dir, "soil", [sol_rel, "soil/soil_mask.txt", f"soil/{DETAIL_FILE}"])
    sol = read_sol(sols[0])
    if sol.record is None:
        raise SoilDeclarationError(f"{sol_rel} carries no prismpy soil record")
    record = _typed(sol.record, SOL_RECORD_KEYS)
    if record["source"] not in ("profiles", "default_profile"):
        raise SoilDeclarationError(f"a CRAFT soil record cannot say source={record['source']}")
    rows = []
    for line in (package_dir / "soil" / "soil_mask.txt").read_text().splitlines()[1:]:
        if line.strip():
            profile = line.split("\t")[1]
            if profile not in sol.profiles:
                raise SoilDeclarationError(f"soil_mask.txt names {profile}, which {sol_rel} lacks")
            rows.append(profile)
    derived = Counter(_token(sol, profile) for profile in rows)
    if dict(derived) != record["profile_sources"]:
        raise SoilDeclarationError(
            f"the mask gives {dict(derived)} cells per source, the record {record['profile_sources']}")
    _check_generic_cells(record, rows, derived)
    return SoilDeclaration(record["source"], _label("craft", record, sol.profiles), _with_hydraulics(record))


def _token(sol: SolFile, profile: str) -> str:
    token = sol.profiles[profile][0]
    if token not in PROFILE_SOURCE_TOKENS.values():
        raise SoilDeclarationError(f"profile {profile} has the unknown source token {token!r}")
    return token


def _check_generic_cells(record: Dict[str, Any], rows: List[str], derived: Counter) -> None:
    """The record's generic-soil fields agree with the mask; its numbers are consumed, not re-derived."""
    n_default, n_cells = record["default_cells"], record["cells"]
    if record["default_warning"] != int(20 * n_default > n_cells):
        raise SoilDeclarationError("default_warning does not follow 20 x default_cells > cells")
    generic = sum(derived[token] for token in GENERIC_TOKENS)
    if n_default:
        if len(record["default_cause"]) != 1 or sum(record["default_cause"].values()) != n_default:
            raise SoilDeclarationError(
                f"default_cause {record['default_cause']} does not give one cause for {n_default} cells")
        named = sum(1 for profile in rows if profile == record["default_profile"])
        if not named == n_default == generic:
            raise SoilDeclarationError(
                f"default_cells={n_default}, but {named} mask rows name {record['default_profile']} "
                f"and {generic} cells run on a generic soil")
    elif record["default_profile"] is not None or record["default_cause"] or generic:
        raise SoilDeclarationError("no default cells are declared, but a generic soil is used")


def _with_hydraulics(record: Dict[str, Any]) -> Dict[str, Any]:
    return {**record, "hydraulics": {**HYDRAULICS, "estimated_layers": record["ptf_layers"],
                                     "layers": record["layers"]}}


def _pythia(package_dir: Path) -> SoilDeclaration:
    ghr = package_dir / "eGHR"
    sols = sorted(ghr.glob("*.SOL"))
    present = [f"eGHR/{s.name}" for s in sols] + [
        rel for rel in ("eGHR/GHR.db", "raster/soil.tif", f"eGHR/{DETAIL_FILE}") if (package_dir / rel).is_file()]
    binding = ghr / BINDING_FILE
    if not binding.is_file():
        raise SoilDeclarationError(f"eGHR/{BINDING_FILE} is missing")
    source = parse_binding(binding.read_text())[0]
    check_binding(package_dir, "eGHR", present)
    parsed = [read_sol(s) for s in sols]
    if source == EGHR_DATABASE:
        if any(p.record is not None for p in parsed):
            raise SoilDeclarationError("a database-copied eGHR package carries a prismpy soil record")
        record = {"source": EGHR_DATABASE}
        return SoilDeclaration(EGHR_DATABASE, _label("pythia", record, {}), record)
    if source is not None or not parsed or any(p.record is None for p in parsed):
        raise SoilDeclarationError("eGHR soil files are neither prismpy-written nor a bound database copy")
    records = [_typed(p.record, SOL_RECORD_KEYS) for p in parsed]
    first = records[0]
    if any(r["source"] != first["source"] or set(r["profile_sources"]) != set(first["profile_sources"])
           for r in records):
        raise SoilDeclarationError("the eGHR .SOL records disagree on their soil sources")
    record = dict(first)
    for key in ("no_profile_cells", "override_cells", "ptf_layers", "profiles", "layers", "organic_layers",
                "andic_layers", "cells_with_flagged_layers", "cells_with_defaulted_chemistry", "cells"):
        record[key] = sum(r[key] for r in records)
    for key in ("profile_sources", "chem_defaulted"):
        record[key] = {token: sum(r[key].get(token, 0) for r in records) for token in first[key]}
    if record["source"] not in ("profiles", "none") or any(r["default_cells"] for r in records):
        raise SoilDeclarationError(f"an eGHR soil record cannot say source={record['source']} with default cells")
    profiles = {name: header for p in parsed for name, header in p.profiles.items()}
    return SoilDeclaration(record["source"], _label("pythia", record, profiles), _with_hydraulics(record))


def _acea(package_dir: Path) -> SoilDeclaration:
    if not (package_dir / ACEA_SOIL_FILE).is_file():
        record = {"source": "engine_installed"}
        return SoilDeclaration("engine_installed", _label("acea", record, {}), record)
    check_binding(package_dir, "soil", [ACEA_SOIL_FILE])
    import netCDF4

    with netCDF4.Dataset(package_dir / ACEA_SOIL_FILE) as ds:
        if ACEA_RECORD_ATTR not in ds.ncattrs():
            raise SoilDeclarationError(f"{ACEA_SOIL_FILE} carries no {ACEA_RECORD_ATTR}")
        text = str(ds.getncattr(ACEA_RECORD_ATTR))
    fields = parse_record(text)
    keys = ACEA_RECORD_KEYS.get(fields.get("source", ""))
    if keys is None:
        raise SoilDeclarationError(f"an ACEA soil record cannot say source={fields.get('source')}")
    record = _typed(fields, keys)
    return SoilDeclaration(record["source"], _label("acea", record, {}), record)


def _sarra(package_dir: Path) -> SoilDeclaration:
    record = {"source": "engine_bundled_africa"}
    return SoilDeclaration("engine_bundled_africa", _label("sarra_py", record, {}), record)


# ── labels: the declaration's only wording ────────────────────────────────────

SOURCE_DESCRIPTIONS = {
    "isda_s3": "iSDA Africa soil properties, 0–50 cm, read at each cell centre",
    "isda": "iSDA Africa topsoil properties, 0–20 cm, read at each cell centre",
    "hwsd": ("HWSD v2.0 dominant soil component (max share), layered 0–100 cm from its own HWSD "
             "layers, read at each cell centre"),
    "eghr": "eGHR soil profiles",
    "placeholder": "a generic placeholder soil profile (no soil data was retrieved)",
}
NO_PROFILE = "No soil profile is available for any cell in this package"
BASE = {
    ("acea", "hwsd_upper_layer_texture"): (
        "HWSD v2.0 upper-layer (0–20 or 20–40 cm) texture (sand, clay) of one soil component, not "
        "necessarily the dominant one, read from the ~1 km pixel at each 0.5° cell centre and applied "
        "to the whole profile"),
    ("acea", "profiles_by_list_position"): (
        "{S}, one per 0.5° cell, assigned by list order, not by location, so a cell's soil can come "
        "from elsewhere in the region"),
    ("acea", "default_values"): "A generic default soil for every cell (sand 40%, clay 25%); no HWSD value was available",
    ("acea", "engine_installed"): "No soil in this package; ACEA reads the soil file installed with the engine",
    ("craft", "profiles"): "{S}, one profile per grid cell",
    ("pythia", "profiles"): "{S}, one profile per grid cell, stored in the eGHR file format and looked up at each simulated site",
    ("pythia", "eghr_database"): (
        "eGHR global soil profiles from the database configured for PYTHIA, looked up at each simulated site"),
    ("pythia", "none"): NO_PROFILE,
    ("sarra_py", "engine_bundled_africa"): (
        "SARRA-Py's bundled Africa soil data, read by the engine and not carried in this package: "
        "root-zone depth from a GYGA rootable-depth map; soil water capacity, surface-layer thickness, "
        "initial deep-soil water and runoff from iSDA soil classes at TAMSAT resolution via an "
        "HWSD-derived class table (versions not recorded). The package's parameters/soil.yaml is "
        "loaded, but the SARRA-Py version the runner pins (ac769ea) uses none of its values"),
}
PER_CELL_CAUSES = {
    "no_hwsd_soil_at_cell_centre": "the HWSD soil unit there is not a soil (e.g. water or urban) or has no data",
    "no_retrieved_soil_at_cell": "the retrieved soil source returned no value there",
}
PACKAGE_CAUSES = {
    "no_soil_source": "no soil data source was available for this package",
    "retrieve_stage_placeholder": "the soil data for this package could not be retrieved when it was built",
}
#: The direction sentence of the WARNING form ({kind}: default or placeholder).
WARNING_DIRECTION = ("the {kind} soil holds more water than most local soils, and its organic matter may be "
                     "higher or lower than theirs, so these cells' yields may be over- or understated")
_CHEM_TEXT = {"bulk_density": "bulk density {v} g/cm³", "organic_carbon": "organic carbon {v} %", "ph": "pH {v}"}


def _sources_text(counts: Mapping[str, int], leave_out: Sequence[str]) -> str:
    shown = sorted(((n, t) for t, n in counts.items() if t not in leave_out and n > 0), key=lambda x: (-x[0], x[1]))
    if len(shown) == 1:
        return SOURCE_DESCRIPTIONS[shown[0][1]]
    if not shown:
        return ""
    return "Retrieved soil: " + "; ".join(f"{SOURCE_DESCRIPTIONS[t]} for {n} cells" for n, t in shown)


def _generic_clause(record: Mapping[str, Any]) -> str:
    (cause, k), = record["default_cause"].items()
    kind = "placeholder" if cause == "retrieve_stage_placeholder" else "default"
    n = record["cells"]
    water = f"({record['default_depth_cm']} cm, about {record['default_paw_mm']} mm plant-available water)"
    if cause in PER_CELL_CAUSES:
        return (f"{k} of {n} candidate grid cells run on a generic default soil profile {water}, because no "
                f"soil value exists at the cell centre: {PER_CELL_CAUSES[cause]}. Their results reflect "
                f"this default soil, not the local soil.")
    return (f"{k} of {n} candidate grid cells run on a generic {kind} soil profile {water}, because "
            f"{PACKAGE_CAUSES[cause]}. Their results reflect this {kind} soil, not the local soil.")


def _suffixes(platform: str, record: Mapping[str, Any], profiles: Mapping[str, Tuple[str, int]]) -> List[str]:
    items: List[str] = []
    n = record.get("cells", 0)
    if platform == "acea":
        if record.get("default"):
            items.append(f"{record['default']} of {n} cells use a generic default soil (sand 40%, clay 25%)")
        if record.get("masked"):
            items.append(f"{record['masked']} of {n} cells have no soil value and are skipped by the engine")
        if record.get("field_default"):
            items.append(f"{record['field_default']} of {n} cells lacked sand or clay and use the default for it")
        return items
    if "ptf_layers" not in record:
        return items
    if record["ptf_layers"]:
        items.append("water limits estimated with a pedotransfer function (Saxton & Rawls 2006, simplified) "
                     f"for {record['ptf_layers']} of {record['layers']} layers")
    shallow = sorted(depth for _, depth in profiles.values() if depth < 100)
    if shallow:
        items.append(f"{len(shallow)} of {len(profiles)} profiles are shallower than 100 cm "
                     f"(depth {shallow[0]}–{shallow[-1]} cm)")
    if record["no_profile_cells"]:
        items.append(f"{record['no_profile_cells']} of {n} grid cells have no soil profile in the package")
    if record["override_cells"]:
        items.append(f"user soil overrides applied to the top layer of {record['override_cells']} cells")
    values = dict(part.split(":", 1) for part in record["chem_default_values"].split(","))
    used = [_CHEM_TEXT[f].format(v=values[f]) + f" ×{c}" for f, c in record["chem_defaulted"].items() if c]
    if used:
        items.append(f"chemistry defaults used ({', '.join(used)})")
    if record["organic_layers"] or record["andic_layers"]:
        items.append(f"{record['organic_layers']} organic and {record['andic_layers']} andic layers (of "
                     f"{record['layers']} layers in {record['profiles']} profiles; "
                     f"{record['cells_with_flagged_layers']} cells), outside the pedotransfer function's "
                     f"calibration range")
    return items


def _chain(base: str, items: Sequence[str]) -> str:
    """BASE, then each suffix after "; "; without a BASE the first suffix opens capitalised."""
    tail = "; ".join(items)
    if not base:
        return tail[:1].upper() + tail[1:]
    return f"{base}; {tail}" if tail else base


def _label(platform: str, record: Mapping[str, Any], profiles: Mapping[str, Tuple[str, int]]) -> str:
    source = record["source"]
    if source == "none":
        return NO_PROFILE
    leave_out = GENERIC_TOKENS if platform == "craft" else ("default",)
    sources = _sources_text(record.get("profile_sources", {}), leave_out)
    template = BASE.get((platform, source))
    base = template.format(S=sources) if template and (sources or "{S}" not in template) else ""
    chain = _chain(base, _suffixes(platform, record, profiles))
    if not record.get("default_cells"):
        return chain
    clause = _generic_clause(record)
    if record["default_warning"]:
        kind = "placeholder" if "retrieve_stage_placeholder" in record["default_cause"] else "default"
        warning = f"WARNING: {clause[:-1]}; {WARNING_DIRECTION.format(kind=kind)}."
        return f"{warning} {chain}" if chain else warning
    return f"{chain}. {clause}" if chain else clause
