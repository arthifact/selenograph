"""Small synthetic rasters; no dependency on the optional poster tools."""
import hashlib
import json
import numpy as np
import rasterio
from rasterio.crs import CRS

MOON = CRS.from_string("+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m")
TRANSFORM = rasterio.Affine(5, 0, 1000, 0, -5, 2000)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def raster(path, values, transform=TRANSFORM, nodata=np.nan, crs=MOON):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", count=1, height=values.shape[0],
                       width=values.shape[1], dtype=values.dtype, transform=transform,
                       crs=crs, nodata=nodata) as dst:
        dst.write(values, 1)
