"""A provisioned SPAM directory for ACEA package tests: small synthetic R, I and A layers under a
registered vintage's file names, with the registry made to pin their content digests, so a package
build resolves, verifies and carries them as it does the real layers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from prismpy.sources.crop_areas import spam_vintage as sv


def harvested_area_entries(year="2020", release="V2r2", crop="Maize", code="MAIZ", fao=56) -> dict:
    """The harvested-area entries of an ACEA package's data_sources, for tests that write its
    manifest or README without building the layer (the default vintage, the registry's own pins)."""
    selection = "default" if (year, release) == sv.DEFAULT_SPAM_VINTAGE else "selected"
    files = {tech: {"name": f"spam2020V2r0_global_H_{fao}_{tech}.tif",
                    "content_digest": sv.SPAM_CONTENT_DIGESTS[(year, release, code, tech)]} for tech in "RIA"}
    return {"harvested_areas": sv.derive_harvested_area_label(year, release, selection),
            "harvested_areas_layer": {"contract": "acea-spam-identity/1", "year": year, "release": release,
                                      "selection": selection, "crop": crop, "crop_code": code, "fao": fao,
                                      "files": files}}


def provision_spam(spam_dir, monkeypatch, *, vintage=("2020", "V2r2"), codes=("MAIZ",), nodata=np.nan,
                   shape=(36, 72)) -> Path:
    """Write R, I and A for each code of ``vintage`` into ``spam_dir`` and pin their digests."""
    spam_dir = Path(spam_dir)
    spam_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(codes))
    transform = from_origin(-180.0, 90.0, 360 / shape[1], 180 / shape[0])
    for code in codes:
        for tech in "RIA":
            band = rng.uniform(0.0, 1000.0, shape).astype("float32")
            band[0, :2] = np.float32(np.nan if nodata is None else nodata)
            path = spam_dir / sv.SPAM_VINTAGES[vintage].pattern.format(code=code, tech=tech)
            with rasterio.open(path, "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                               dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata) as ds:
                ds.write(band, 1)
            with rasterio.open(path) as ds:
                monkeypatch.setitem(sv.SPAM_CONTENT_DIGESTS, (*vintage, code, tech), sv.content_digest(ds))
    return spam_dir
