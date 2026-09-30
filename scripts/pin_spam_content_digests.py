"""Pin the content digest and header of every registered SPAM source.

Writes ``src/prismpy/sources/crop_areas/spam_content_digests.json`` from the MapSPAM
harvested-area archives: one entry per registered (year, release, crop code, technology
R/I/A), read in place through GDAL's ``/vsizip/``, and the sha256 of each archive read.
Run it when a vintage is registered or re-provisioned::

    python scripts/pin_spam_content_digests.py \\
        spam2020V2r2_global_harvested_area.geotiff.zip spam2010v2r0_global_harv_area.geotiff.zip

A registered source missing from every archive stops the script: the registry pins each one.
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import rasterio

from prismpy.sources.crop_areas import spam_vintage as sv

OUT = Path(__file__).resolve().parents[1] / "src/prismpy/sources/crop_areas" / sv._CONTENT_DIGESTS_FILE
TECHS = ("A", "I", "R")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(archives):
    members, sources = {}, {}
    for archive in map(Path, archives):
        sources[archive.name] = _sha256(archive)
        for name in zipfile.ZipFile(archive).namelist():
            members.setdefault(Path(name).name, f"/vsizip/{archive.resolve()}/{name}")
    entries = []
    for (year, release), spec in sorted(sv.SPAM_VINTAGES.items()):
        for code in sorted(spec.crops):
            for tech in TECHS:
                name = spec.pattern.format(code=code, tech=tech)
                if name not in members:
                    sys.exit(f"registered source {name} is in none of the archives")
                with rasterio.open(members[name]) as ds:
                    entries.append({"year": year, "release": release, "crop_code": code, "tech": tech,
                                    "content_digest": sv.content_digest(ds),
                                    "header": sv._content_header(ds)})
    head = json.dumps({"schema": sv._CONTENT_DIGEST_SCHEMA, "sources": sources}, sort_keys=True)
    lines = [json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in entries]
    OUT.write_text(head[:-1] + ', "entries": [\n' + ",\n".join(lines) + "\n]}\n", encoding="utf-8")
    print(f"{len(entries)} sources pinned in {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
