"""Each engine's roster read back from the file its translator wrote, so a restricted run can prove
that every package holds exactly the final roster: PYTHIA's sites by CellID, CRAFT's soil mask, and
ACEA's gridcells (its configuration is parsed, never executed)."""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from prismpy.cells.roster_digest import roster_id_digest

ROSTER_FILES: Dict[str, tuple] = {  # engine -> (file, id field, grid)
    "pythia": ("shapes/sites.shp", "CellID", "5arcmin"),
    "craft": ("soil/soil_mask.txt", "CellID", "5arcmin"),
    "acea": ("config/*_config.py", "gridcells", "30arcmin"),
}
_SOIL_MASK_HEADER = "CellID\tSoilProfile\tSharePCT"


@dataclass(frozen=True)
class RosterReadback:
    ids: List[int]
    file: str
    field: str
    grid: str

    @property
    def id_digest(self) -> str:
        return roster_id_digest(self.ids)


def _strict_ids(values, source: str) -> List[int]:
    ids = list(values)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in ids):
        raise ValueError(f"{source}: a cell id is not an integer")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source}: a cell id appears more than once")
    return ids


def _pythia(package: Path) -> RosterReadback:
    shp, csv_path = package / "shapes" / "sites.shp", package / "shapes" / "sites.csv"
    if shp.is_file():
        import pyogrio

        frame = pyogrio.read_dataframe(shp, read_geometry=False)
        if "CellID" not in frame.columns:
            raise ValueError("shapes/sites.shp has no CellID field")
        return RosterReadback(_strict_ids(frame["CellID"].tolist(), "shapes/sites.shp"),
                              "shapes/sites.shp", "CellID", "5arcmin")
    if csv_path.is_file():  # the translator's fallback when geopandas is absent
        rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
        if not rows or "CellID" not in rows[0]:
            raise ValueError("shapes/sites.csv has no CellID column")
        return RosterReadback(_strict_ids([_int(r["CellID"], "shapes/sites.csv") for r in rows],
                                          "shapes/sites.csv"), "shapes/sites.csv", "CellID", "5arcmin")
    raise ValueError("shapes/sites.shp is missing")


def _int(text: str, source: str) -> int:
    text = (text or "").strip()
    if not text.isdigit():
        raise ValueError(f"{source}: a cell id is not an integer: {text!r}")
    return int(text)


def _tsv_value(token: str):
    """A soil-mask cell as the runner coerces it: an int, else a float, else the stripped text."""
    text = token.strip()
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _craft(package: Path) -> RosterReadback:
    """The soil mask read with the runner's row semantics: the header's three columns on every row,
    a text SoilProfile and a numeric SharePCT; blank rows are skipped, as the runner skips them."""
    path, source = package / "soil" / "soil_mask.txt", "soil/soil_mask.txt"
    if not path.is_file():
        raise ValueError(f"{source} is missing")
    rows = list(csv.reader(path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if not rows or "\t".join(rows[0]) != _SOIL_MASK_HEADER:
        raise ValueError(f"{source} does not start with its CellID header")
    ids = []
    for number, row in enumerate(rows[1:], start=2):
        if all(not cell.strip() for cell in row):
            continue
        if len(row) != 3:
            raise ValueError(f"{source} line {number}: {len(row)} columns where the header has 3")
        _, profile, share = (_tsv_value(cell) for cell in row)
        if not isinstance(profile, str) or isinstance(share, str):
            raise ValueError(f"{source} line {number}: SoilProfile must be text, SharePCT a number")
        ids.append(_int(row[0], source))
    return RosterReadback(_strict_ids(ids, source), source, "CellID", "5arcmin")


def _acea(package: Path) -> RosterReadback:
    configs = sorted((package / "config").glob("*_config.py"))
    if len(configs) != 1:
        raise ValueError(f"config/ holds {len(configs)} *_config.py files, not exactly one")
    tree = ast.parse(configs[0].read_text(encoding="utf-8"), filename=configs[0].name)
    values = [node.value for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "project_conf"
              for node in cls.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "gridcells" for t in node.targets)]
    if len(values) != 1:
        raise ValueError("project_conf does not assign gridcells exactly once")
    gridcells = ast.literal_eval(values[0])
    if not isinstance(gridcells, list):
        raise ValueError("gridcells is not a list")
    rel = f"config/{configs[0].name}"
    return RosterReadback(_strict_ids(gridcells, rel), rel, "gridcells", "30arcmin")


_READERS: Dict[str, Callable[[Path], RosterReadback]] = {"pythia": _pythia, "craft": _craft, "acea": _acea}


def read_back_roster(platform: str, package_dir) -> RosterReadback:
    """The roster the platform's translator wrote in ``package_dir``; ValueError when the file is
    missing or malformed (a duplicate id included)."""
    if platform not in _READERS:
        raise ValueError(f"no roster read-back for {platform}")
    return _READERS[platform](Path(package_dir))
