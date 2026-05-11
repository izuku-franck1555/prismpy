"""Pydantic schemas for FAO Esri Image Service /exportImage requests.

The Esri ImageServer answers two endpoints we care about: ``/exportImage``
(returns TIFF bytes) and ``/query`` (returns metadata JSON). Both accept
the same attribute-filter vocabulary embedded in the ``mosaicRule`` JSON
payload. These schemas validate every parameter that crosses the boundary
into the network call. (The canonical host URL lives in ``esri_client``.)

Per durable §6.4 schema-layer discipline:
  - Models are ``frozen=True`` so a constructed request is immutable.
  - ``EsriExportImageRequest`` uses ``extra="forbid"`` — every parameter we
    ship must be on the schema; an unrecognized kwarg is a programmer bug.
  - ``EsriErrorResponse`` is the ONE documented ``extra="ignore"`` exception
    per codex M2: Esri's JSON error envelope evolves independently of our
    consumer and we only care about the documented ``error.code`` + ``message``
    fields. Ignoring extras lets us parse stale-schema responses cleanly.

The canonical host URL is owned by ``esri_client.py`` per durable §24:
these schemas validate parameters in the abstract and never embed the
production URL.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


_BBOX_RE = re.compile(r"^-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?$")
_SIZE_RE = re.compile(r"^\d+,\d+$")
_SR_RE = re.compile(r"^\d+$")


class EsriExportImageRequest(BaseModel):
    """Validated parameter bundle for ``/exportImage`` GET requests.

    ``mosaic_rule_json`` carries the ``esriMosaicAttribute`` mosaicRule
    pre-serialized as JSON; the client url-encodes it on the wire. The
    schema does NOT validate the JSON shape (that's the ``EsriQuerySpec``
    contract in ``raster_mapping.py``) — this class is the wire-level
    invariant net: bbox/size/SR shapes are well-formed strings, format and
    pixel type are restricted to the values ACEA's downstream reader
    expects.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bbox: str = Field(
        description="Comma-separated 'minx,miny,maxx,maxy' in bboxSR coords.",
    )
    bbox_sr: str = Field(default="4326", description="Spatial reference WKID for bbox.")
    image_sr: str = Field(default="4326", description="Spatial reference WKID for output image.")
    size: str = Field(
        description="Comma-separated 'width,height' pixel size.",
    )
    image_format: str = Field(default="tiff", description="Output image format.")
    pixel_type: str = Field(default="S16", description="Output pixel type.")
    interpolation: str = Field(
        default="RSP_NearestNeighbor",
        description="Resampling method; nearest-neighbor preserves integer class indices.",
    )
    mosaic_rule_json: str = Field(
        description="JSON-encoded ``esriMosaicAttribute`` mosaicRule.",
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: str) -> str:
        if not _BBOX_RE.match(v):
            raise ValueError(
                f"bbox must match 'minx,miny,maxx,maxy' (got {v!r}); "
                "use floats with optional decimal point."
            )
        return v

    @field_validator("size")
    @classmethod
    def _check_size(cls, v: str) -> str:
        if not _SIZE_RE.match(v):
            raise ValueError(f"size must match 'width,height' (got {v!r}).")
        return v

    @field_validator("bbox_sr", "image_sr")
    @classmethod
    def _check_sr(cls, v: str) -> str:
        if not _SR_RE.match(v):
            raise ValueError(f"spatial reference must be a positive WKID (got {v!r}).")
        return v

    @field_validator("image_format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        # ACEA's rasterio reader expects TIFF; the Esri service supports
        # png/jpg/etc. but those drop pixel-type fidelity. Pin to tiff.
        if v.lower() not in ("tiff", "tif"):
            raise ValueError(
                f"image_format must be 'tiff' for ACEA pixel fidelity (got {v!r})."
            )
        return v.lower()

    @field_validator("pixel_type")
    @classmethod
    def _check_pixel_type(cls, v: str) -> str:
        # GAEZ rasters are int16 (LUT class indices + S16 yield); the
        # full Esri enumeration includes U8/F32/etc. but ACEA only reads S16.
        if v not in ("S16", "U16", "S32", "F32"):
            raise ValueError(
                f"pixel_type must be in Esri's documented enum (got {v!r})."
            )
        return v

    def as_query(self) -> dict:
        """Project the validated parameters onto the Esri kwarg names.

        The Esri service uses camelCase parameter names. This method centralizes
        the camelCase mapping so the client never hand-rolls the parameter dict.
        """
        return {
            "bbox": self.bbox,
            "bboxSR": self.bbox_sr,
            "imageSR": self.image_sr,
            "size": self.size,
            "format": self.image_format,
            "pixelType": self.pixel_type,
            "interpolation": self.interpolation,
            "mosaicRule": self.mosaic_rule_json,
            "f": "image",
        }


class EsriErrorResponse(BaseModel):
    """In-band JSON error envelope returned by the Esri ImageServer.

    The ImageServer sometimes replies ``200 OK`` with ``Content-Type:
    application/json`` and a body like::

        {"error": {"code": 400, "message": "Invalid mosaicRule", "details": [...]}}

    This shape is parsed by ``esri_client.fetch_image`` to surface the
    actual Esri-level error to the caller rather than silently writing
    JSON to a ``.tif`` file. Per codex M2 + module-docstring contract:
    this is the ONE schema with ``extra="ignore"`` because Esri evolves
    the envelope (e.g., adds ``messageCode``) and we only care about
    ``error.code`` and ``error.message`` for the audit trail.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    code: int = Field(description="Esri-defined error code (often mirrors HTTP).")
    message: str = Field(description="Human-readable error narrative.")
    details: Optional[list] = Field(default=None, description="Optional detail list.")
