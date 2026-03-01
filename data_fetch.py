"""
data_fetch.py — DEM and OSM data acquisition
=============================================
Phase 1/2:
  DEM fallback chain: COP30 → SRTMGL1 → SRTMGL3 → mock
  NoData sentinel: DEM_NODATA_SENTINEL (-9999.0) — not 0 (valid sea-level terrain).

Phase 3 — DEM stream fallback:
  D8 flow accumulation derives a stream network when OSM water is sparse.

Phase 5 — Performance:
  _flow_accumulation uses vectorised level-batch processing instead of a
  Python deque loop — O(max_tree_depth) numpy ops vs O(n_cells) Python calls.
  try_jit from jit_utils applies numba JIT when installed.

Phase 14.5 — Data pipeline audit:
  - OSM Overpass retry with exponential backoff + fail-hard.
  - WorldCover memory cleanup after tile merge.
  - Overture dedup chunked sjoin for bounded memory.
  - DEM void-fill hardened edge-case handling.
  - Quantised tile cache key (_tile_key) for Phase 15 tile routing.
  - RasterCache: content-addressed intermediate raster caching.
"""
import gc
import hashlib
import math
import os
import time as _time
import requests
import logging
import numpy as np
from scipy.ndimage import generic_filter
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

from jit_utils import try_jit   # Phase 5: numba fallback JIT

from config import (
    DATA_DIR, UTM_EPSG, OPENTOPO_URL, RESOLUTION,
    DEM_PREFERENCE, DEM_NODATA_SENTINEL,
    OSM_WATER_FALLBACK_TRIGGER, STREAM_ACCUM_THRESHOLD_KM2,
    OSM_LULC_WARN_THRESHOLD,
    OVERPASS_MAX_RETRIES, OVERPASS_BACKOFF_BASE_S,
    CACHE_TILE_SIZE_DEG, ENABLE_RASTER_CACHE,
)
from geometry_utils import wgs84_to_utm

log = logging.getLogger("highway_alignment")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _bbox_key(bbox_wgs84):
    """Legacy cache key — kept for backward compatibility with existing cached files."""
    w, s, e, n = [round(v, 4) for v in bbox_wgs84]
    return f"W{w}_S{s}_E{e}_N{n}".replace("-", "m")


def _tile_key(bbox_wgs84, tile_size_deg=None):
    """
    Phase 14.5: Quantised tile cache key — snaps bbox to a fixed grid.
    Ensures floating-point boundary jitter does not generate different cache keys
    for functionally identical bounding boxes.
    """
    ts = tile_size_deg or CACHE_TILE_SIZE_DEG
    w, s, e, n = bbox_wgs84
    tw = math.floor(w / ts) * ts
    ts_ = math.floor(s / ts) * ts
    te = math.ceil(e / ts) * ts
    tn = math.ceil(n / ts) * ts
    return f"T_W{tw}_S{ts_}_E{te}_N{tn}".replace("-", "m").replace(".", "p")


def _resolve_cache_key(bbox_wgs84):
    """
    Phase 14.5: Returns the best cache key, with backward compatibility.
    Tries the new quantised _tile_key first; if a legacy _bbox_key cached file
    exists for a given prefix, it will be found by the caller's os.path.exists check.
    """
    return _tile_key(bbox_wgs84)


def _cache_fingerprint(*inputs):
    """
    Phase 14.5: Content-addressed fingerprint for intermediate raster caching.
    Hashes source file mtimes + config values to detect staleness.
    """
    h = hashlib.md5()
    for inp in inputs:
        h.update(str(inp).encode())
    return h.hexdigest()[:12]


# ── Phase 14.5: Intermediate Raster Cache ─────────────────────────────────────

class RasterCache:
    """
    Content-addressed GeoTIFF cache for expensive vector-to-raster products.
    
    Each product is keyed by (name, fingerprint). If the fingerprint matches,
    the cached file is loaded directly (~<1s) instead of recomputing (~minutes).
    """

    def __init__(self, cache_dir=None):
        self._dir = cache_dir or DATA_DIR
        os.makedirs(self._dir, exist_ok=True)

    def get(self, name, fingerprint, shape=None):
        """Load cached raster if fingerprint matches. Returns (array, hit_bool)."""
        if not ENABLE_RASTER_CACHE:
            return None, False
        path = os.path.join(self._dir, f"rcache_{name}_{fingerprint}.tif")
        if os.path.exists(path):
            try:
                with rasterio.open(path) as src:
                    arr = src.read(1)
                    if shape is not None and arr.shape != shape:
                        log.warning(
                            f"RasterCache shape mismatch for {name}: "
                            f"cached={arr.shape}, expected={shape}. Recomputing."
                        )
                        return None, False
                    log.info(f"RasterCache HIT: {name} (fp={fingerprint})")
                    return arr, True
            except Exception as exc:
                log.warning(f"RasterCache read error for {name} ({exc}). Recomputing.")
                return None, False
        return None, False

    def put(self, name, fingerprint, array, transform, crs_epsg=UTM_EPSG):
        """Persist raster to disk with fingerprint."""
        if not ENABLE_RASTER_CACHE:
            return
        path = os.path.join(self._dir, f"rcache_{name}_{fingerprint}.tif")
        rows, cols = array.shape
        try:
            with rasterio.open(
                path, "w", driver="GTiff",
                height=rows, width=cols, count=1,
                dtype=array.dtype,
                crs=f"EPSG:{crs_epsg}",
                transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(array, 1)
            log.info(f"RasterCache STORE: {name} → {path}")
        except Exception as exc:
            log.warning(f"RasterCache write failed for {name} ({exc}). Continuing without cache.")


def _mock_dem(bbox_wgs84, resolution_m):
    """Deterministic synthetic DEM for offline testing (seeded)."""
    west, south, east, north = bbox_wgs84
    x0, y0 = wgs84_to_utm(west, south)
    x1, y1 = wgs84_to_utm(east, north)
    cols = max(10, int((x1 - x0) / resolution_m))
    rows = max(10, int((y1 - y0) / resolution_m))

    log.warning(f"Using MOCK DEM: {rows}×{cols} @ {resolution_m} m/px — results are NOT for real use")

    rng = np.random.default_rng(42)
    base = np.linspace(50, 200, rows)[:, None] * np.ones((1, cols))
    noise = rng.normal(0, 8, (rows, cols))
    for _ in range(5):
        r = rng.integers(rows // 4, 3 * rows // 4)
        c = rng.integers(cols // 4, 3 * cols // 4)
        Y, X = np.ogrid[:rows, :cols]
        bump = 80 * np.exp(-((Y - r) ** 2 + (X - c) ** 2) / (2 * (rows // 8) ** 2))
        base += bump
    dem = (base + noise).astype(np.float32)

    transform = from_bounds(x0, y0, x1, y1, cols, rows)
    return dem, transform, "MOCK"


def _reproject_to_utm(dem_wgs, wgs_tf, wgs_crs, wgs_w, wgs_h,
                      bbox_wgs84, cache_utm):
    """Reproject a WGS-84 DEM array to UTM and cache the result."""
    west, south, east, north = bbox_wgs84
    dst_crs = rasterio.crs.CRS.from_epsg(UTM_EPSG)
    utm_tf, utm_w, utm_h = calculate_default_transform(
        wgs_crs, dst_crs, wgs_w, wgs_h,
        left=west, bottom=south, right=east, top=north,
    )
    dem_utm = np.full((utm_h, utm_w), DEM_NODATA_SENTINEL, dtype=np.float32)
    reproject(
        source=dem_wgs, destination=dem_utm,
        src_transform=wgs_tf, src_crs=wgs_crs,
        dst_transform=utm_tf, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=DEM_NODATA_SENTINEL,
        dst_nodata=DEM_NODATA_SENTINEL,
    )

    # Phase 14.5 hardened: void-fill the UTM DEM before writing to cache.
    # Uses exact sentinel comparison instead of threshold range.
    # Bilinear reprojection can leave edge sentinel cells that corrupt
    # slope, earthwork-proxy, and floodplain calculations downstream.
    # EDT nearest-neighbour fill replaces all sentinel cells with the nearest
    # valid elevation value, producing a seamless surface for all consumers.
    nodata_mask = (dem_utm == DEM_NODATA_SENTINEL)
    # Also catch near-sentinel values from bilinear interpolation at edges
    nodata_mask |= (dem_utm < -500)
    if nodata_mask.any():
        from scipy.ndimage import distance_transform_edt
        valid_count = int((~nodata_mask).sum())
        if valid_count == 0:
            log.error("DEM void-fill: entire raster is NoData — cannot fill.")
            raise RuntimeError("DEM reprojection produced all-NoData raster.")
        _, idx = distance_transform_edt(nodata_mask, return_indices=True)
        dem_utm[nodata_mask] = dem_utm[idx[0][nodata_mask], idx[1][nodata_mask]]
        n_filled = int(nodata_mask.sum())
        pct_filled = 100.0 * n_filled / dem_utm.size
        log.info(
            f"DEM void-fill: {n_filled:,} nodata cells filled by EDT nearest-neighbour "
            f"({pct_filled:.1f}% of raster)."
        )

    # Post-fill assertion: no sentinels should remain
    remaining_nodata = int(np.sum(dem_utm == DEM_NODATA_SENTINEL))
    if remaining_nodata > 0:
        log.error(
            f"DEM void-fill FAILED: {remaining_nodata:,} sentinel cells remain after fill."
        )
        raise RuntimeError(
            f"DEM void-fill incomplete: {remaining_nodata} cells still at sentinel value."
        )

    with rasterio.open(
        cache_utm, "w", driver="GTiff",
        height=utm_h, width=utm_w, count=1,
        dtype=dem_utm.dtype,
        crs=dst_crs,
        transform=utm_tf,
        nodata=DEM_NODATA_SENTINEL,
    ) as dst:
        dst.write(dem_utm, 1)
    return dem_utm, utm_tf


def _try_download_dem(dem_type, bbox_wgs84, cache_wgs, cache_utm):
    """
    Attempt to download one DEM type from OpenTopography.
    Returns (dem_utm, utm_tf) on success, raises on failure.
    """
    west, south, east, north = bbox_wgs84
    params = {
        "demtype":      dem_type,
        "west":         float(west),
        "south":        float(south),
        "east":         float(east),
        "north":        float(north),
        "outputFormat": "GTiff",
        "API_Key":      os.getenv("OPENTOPOGRAPHY_API_KEY", ""),
    }
    log.info(f"Downloading {dem_type} DEM from OpenTopography …")
    resp = requests.get(OPENTOPO_URL, params=params, timeout=180, stream=True)
    resp.raise_for_status()

    with open(cache_wgs, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    with rasterio.open(cache_wgs) as src:
        dem_wgs = src.read(1).astype(np.float32)
        # Replace true NoData (SRTM uses -32768; OpenTopo variants use various values)
        native_nodata = src.nodata
        if native_nodata is not None:
            dem_wgs[dem_wgs == native_nodata] = DEM_NODATA_SENTINEL
        # Clip any remaining ocean/bad values below plausible terrain floor
        # Myanmar lowest point is sea level; no valid land below -10 m
        dem_wgs[(dem_wgs < -10) & (dem_wgs != DEM_NODATA_SENTINEL)] = DEM_NODATA_SENTINEL
        wgs_tf  = src.transform
        wgs_crs = src.crs
        wgs_w   = src.width
        wgs_h   = src.height

    return _reproject_to_utm(dem_wgs, wgs_tf, wgs_crs, wgs_w, wgs_h, bbox_wgs84, cache_utm)


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_dem(bbox_wgs84, resolution_m=RESOLUTION):
    """
    Fetch DEM using priority chain defined by DEM_PREFERENCE.
    Returns (dem_utm_array, utm_affine_transform, dem_source_label).
    """
    _ensure_data_dir()
    tile_key = _tile_key(bbox_wgs84)
    legacy_key = _bbox_key(bbox_wgs84)

    # Check for any cached UTM DEM (try new tile key first, then legacy)
    for key in [tile_key, legacy_key]:
        for dem_type in DEM_PREFERENCE:
            cache_utm = os.path.join(DATA_DIR, f"dem_utm_{dem_type}_{key}.tif")
            if os.path.exists(cache_utm):
                log.info(f"Loading cached {dem_type} UTM DEM: {cache_utm}")
                with rasterio.open(cache_utm) as src:
                    dem_utm = src.read(1).astype(np.float32)
                    utm_tf  = src.transform
                return dem_utm, utm_tf, dem_type

    # Try to download in preference order (use new tile key for caching)
    for dem_type in DEM_PREFERENCE:
        cache_wgs = os.path.join(DATA_DIR, f"dem_wgs84_{dem_type}_{tile_key}.tif")
        cache_utm = os.path.join(DATA_DIR, f"dem_utm_{dem_type}_{tile_key}.tif")
        try:
            dem_utm, utm_tf = _try_download_dem(dem_type, bbox_wgs84, cache_wgs, cache_utm)
            log.info(f"DEM acquired: {dem_type} @ 30 m")
            return dem_utm, utm_tf, dem_type
        except Exception as exc:
            log.warning(f"{dem_type} download failed ({exc}). Trying next source …")

    # All remote sources failed — use mock DEM
    log.warning("All DEM sources failed. Falling back to synthetic mock DEM.")
    dem, tf, label = _mock_dem(bbox_wgs84, resolution_m)
    return dem, tf, label


# ── Phase 3: D8 stream network from DEM ───────────────────────────────────────

def _fill_depressions(dem, nodata):
    """
    Iterative pit-filling: raise cells that are lower than ALL 8 neighbours
    to the minimum-neighbour elevation. Repeats until no pits remain (max 50 passes).
    Fast approximation sufficient for flow-accumulation stream derivation.
    """
    from scipy.ndimage import generic_filter
    # 3×3 footprint with centre masked out → only 8 true neighbours
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False

    valid = dem != nodata
    filled = dem.copy()
    for _ in range(50):
        # Minimum of the 8 neighbours (centre excluded)
        neighbour_min = generic_filter(
            filled, np.min, footprint=footprint, mode='nearest'
        )
        # A pit is a valid cell strictly below all its neighbours
        pit = valid & (filled < neighbour_min)
        if not np.any(pit):
            break
        # Raise pits to the lowest exit (minimum neighbour)
        filled = np.where(pit, neighbour_min, filled)
    return filled


def _d8_flow_direction(dem_filled):
    """
    D8 (deterministic 8-direction) flow direction.
    Each cell drains to the steepest of its 8 neighbours.
    Returns an array of neighbour offset indices 0–7 (N, NE, E, SE, S, SW, W, NW),
    or -1 for flat/edge cells.

    Offsets (dr, dc):
      0=N(-1,0) 1=NE(-1,+1) 2=E(0,+1) 3=SE(+1,+1)
      4=S(+1,0) 5=SW(+1,-1) 6=W(0,-1) 7=NW(-1,-1)
    """
    # Diagonal cells are 1/sqrt(2) further, so raw slope needs normalisation
    DIAG_SCALE = 1.0 / np.sqrt(2.0)
    DR = np.array([-1, -1,  0,  1, 1,  1,  0, -1], dtype=np.int8)
    DC = np.array([ 0,  1,  1,  1, 0, -1, -1, -1], dtype=np.int8)
    SCALE = np.array([1., DIAG_SCALE, 1., DIAG_SCALE,
                      1., DIAG_SCALE, 1., DIAG_SCALE])

    rows, cols = dem_filled.shape
    fdir = np.full((rows, cols), -1, dtype=np.int8)

    for k in range(8):
        dr, dc = int(DR[k]), int(DC[k])
        r0 = max(0, -dr);  r1 = rows - max(0, dr)
        c0 = max(0, -dc);  c1 = cols - max(0, dc)
        # Drop from each cell to its neighbour in direction k
        drop = (dem_filled[r0:r1, c0:c1] - dem_filled[r0+dr:r1+dr, c0+dc:c1+dc]) * SCALE[k]
        # Update fdir where this direction is steepest so far
        src_r = slice(r0, r1); src_c = slice(c0, c1)
        best = np.where(fdir[src_r, src_c] == -1, -np.inf,
                        (dem_filled[src_r, src_c] -
                         dem_filled[src_r, src_c]) * SCALE[k])   # placeholder
        # We need the max-drop direction — rebuild with a running max
        # Use vectorised approach: for each (k), update wherever drop > current_max
        pass   # handled in _flow_accumulation via the drop array

    # Vectorised approach: compute all 8 drops simultaneously
    drops = np.full((8, rows, cols), -np.inf, dtype=np.float32)
    for k in range(8):
        dr, dc = int(DR[k]), int(DC[k])
        r0 = max(0, -dr);  r1 = rows - max(0, dr)
        c0 = max(0, -dc);  c1 = cols - max(0, dc)
        drops[k, r0:r1, c0:c1] = (
            (dem_filled[r0:r1, c0:c1] - dem_filled[r0+dr:r1+dr, c0+dc:c1+dc]) * SCALE[k]
        )
    fdir = np.argmax(drops, axis=0).astype(np.int8)
    # Flat or edge cells with no positive drop -> -1
    fdir[drops.max(axis=0) <= 0] = -1
    return fdir


def _flow_accumulation(fdir):
    """
    Count upstream contributing cells for each grid cell (= flow accumulation).

    Phase 5 upgrade: Level-batch vectorised algorithm.
    Instead of processing one cell at a time with a Python deque,
    all cells at the same ‘tree depth’ (0 = headwaters, n = cells with n
    upstream segments) are processed simultaneously as numpy array ops.

    Complexity: O(max_depth × n) numpy ops vs O(n) Python iterations.
    For typical DEMs max_depth ≈ 30–100, so Python call overhead drops
    from ~n to ~100 — a 100–1000× reduction in interpreter round-trips.

    Returns float32 array of upstream cell counts (1 = headwater).
    """
    rows, cols = fdir.shape
    DR = np.array([-1, -1,  0,  1, 1,  1,  0, -1], dtype=np.int32)
    DC = np.array([ 0,  1,  1,  1, 0, -1, -1, -1], dtype=np.int32)

    accum = np.ones((rows, cols), dtype=np.float32)

    # ── Build flat receiver index array ─────────────────────────────────────────
    # recv_flat[i] = flat index of cell i’s downstream neighbour, or -1
    rr, cc = np.mgrid[0:rows, 0:cols]
    rr_flat = rr.ravel(); cc_flat = cc.ravel()
    fdir_flat = fdir.ravel()

    valid_mask = fdir_flat >= 0
    nr_flat = np.where(valid_mask, rr_flat + DR[np.maximum(fdir_flat, 0)], -1)
    nc_flat = np.where(valid_mask, cc_flat + DC[np.maximum(fdir_flat, 0)], -1)

    in_bounds = ((nr_flat >= 0) & (nr_flat < rows) &
                 (nc_flat >= 0) & (nc_flat < cols))
    valid_recv = valid_mask & in_bounds

    recv_flat = np.full(rows * cols, -1, dtype=np.int32)
    recv_flat[valid_recv] = (nr_flat[valid_recv] * cols + nc_flat[valid_recv]).astype(np.int32)

    # ── Compute in-degree for each cell ─────────────────────────────────────────
    in_deg = np.zeros(rows * cols, dtype=np.int32)
    valid_recv_idx = recv_flat[recv_flat >= 0]
    np.add.at(in_deg, valid_recv_idx, 1)

    # ── Level-batch processing ─────────────────────────────────────────────────
    # level_0: headwaters (in_degree == 0)
    # Each iteration, all “ready” cells propagate their accumulation
    # downstream in one vectorised np.add.at call, then their receivers’
    # in-degree is decremented.  Proceed until no ready cells remain.
    accum_flat = accum.ravel()   # view, shares memory with accum
    remaining = in_deg.copy()

    for level in range(rows * cols):   # upper bound on tree depth
        ready = np.where(remaining == 0)[0]   # all current headwaters
        if len(ready) == 0:
            break
        # Propagate accumulation to receivers
        has_recv = recv_flat[ready] >= 0
        src = ready[has_recv]
        dst = recv_flat[src]
        if len(src) > 0:
            np.add.at(accum_flat, dst, accum_flat[src])
            np.subtract.at(remaining, dst, 1)   # decrement downstream in-degree
        remaining[ready] = -1   # mark as processed (never picked again)

    return accum_flat.reshape(rows, cols)



def derive_stream_network(dem, transform, resolution_m,
                           threshold_km2=STREAM_ACCUM_THRESHOLD_KM2):
    """
    Derive a binary stream network from DEM elevation data.

    Steps:
      1. Fill depressions (pit-filling).
      2. D8 flow direction (each cell drains to steepest neighbour).
      3. Flow accumulation (upstream cell count per cell).
      4. Threshold: cells with upstream area ≥ threshold_km2 are streams.

    Returns a float32 raster with values in [0, 1] where 1 = stream channel.
    """
    nodata = DEM_NODATA_SENTINEL
    rows, cols = dem.shape
    cell_area_km2 = (resolution_m ** 2) / 1e6
    threshold_cells = threshold_km2 / cell_area_km2

    log.info(
        f"Deriving stream network from DEM: {rows}×{cols}, "
        f"threshold={threshold_km2} km² ({threshold_cells:.0f} cells)"
    )

    # Use nodata-safe filled DEM
    dem_safe = np.where(dem == nodata, np.nanmedian(dem[dem != nodata]), dem)
    filled = _fill_depressions(dem_safe, nodata)
    fdir   = _d8_flow_direction(filled)
    accum  = _flow_accumulation(fdir)

    stream_mask = (accum >= threshold_cells).astype(np.float32)
    n_stream = int(stream_mask.sum())
    log.info(
        f"Stream network derived: {n_stream:,} stream cells "
        f"({n_stream*cell_area_km2:.1f} km² of channels)"
    )
    return stream_mask


def fetch_osm_layers(bbox_wgs84):
    """
    Fetch buildings, water, existing roads (highway), and land cover (landuse/natural) 
    from OSM via Overpass.
    Returns (buildings_gdf, water_gdf, roads_gdf, lulc_gdf, osm_stats_dict).

    Phase 14.5: Retry with exponential backoff. Raises RuntimeError on permanent
    failure instead of silently returning empty data.
    """
    _ensure_data_dir()
    tile_key = _tile_key(bbox_wgs84)
    legacy_key = _bbox_key(bbox_wgs84)

    # Phase 14.5: backward-compatible cache file resolution
    def _resolve_cache(prefix):
        """Try new tile key first, then fall back to legacy bbox key."""
        new_path = os.path.join(DATA_DIR, f"{prefix}_{tile_key}.gpkg")
        if os.path.exists(new_path):
            return new_path
        legacy_path = os.path.join(DATA_DIR, f"{prefix}_{legacy_key}.gpkg")
        if os.path.exists(legacy_path):
            return legacy_path
        return new_path  # default to new key for fresh downloads

    cache_buildings = _resolve_cache("buildings")
    cache_water     = _resolve_cache("water")
    cache_roads     = _resolve_cache("roads")
    cache_lulc      = _resolve_cache("lulc")

    west  = float(bbox_wgs84[0])
    south = float(bbox_wgs84[1])
    east  = float(bbox_wgs84[2])
    north = float(bbox_wgs84[3])

    stats = {
        "buildings": 0, "water": 0, "roads": 0, "lulc": 0,
        "buildings_from_cache": False, "water_from_cache": False,
        "roads_from_cache": False, "lulc_from_cache": False
    }

    def _load_or_fetch(cache_path, tags, label, stat_key, from_cache_key, **kwargs):
        if os.path.exists(cache_path):
            try:
                gdf = gpd.read_file(cache_path)
                stats[stat_key] = len(gdf)
                stats[from_cache_key] = True
                log.info(f"Loaded cached OSM {label}: {len(gdf)} features")
                return gdf
            except Exception as exc:
                log.warning(f"Cache read failed ({exc}), re-downloading {label}.")

        # Phase 14.5: Retry with exponential backoff, fail-hard on exhaustion
        max_retries = OVERPASS_MAX_RETRIES
        backoff_base = OVERPASS_BACKOFF_BASE_S
        last_exc = None

        for attempt in range(1, max_retries + 1):
            log.info(f"Fetching OSM {label} from Overpass (attempt {attempt}/{max_retries}) …")
            try:
                gdf = ox.features_from_bbox(
                    bbox=(west, south, east, north),
                    tags=tags,
                )
                # Retain specific tags if requested, otherwise just geometry
                if "retain_tags" in kwargs:
                    cols_to_keep = ["geometry"] + [t for t in kwargs["retain_tags"] if t in gdf.columns]
                    gdf = gdf[cols_to_keep].copy().reset_index(drop=True)
                else:
                    gdf = gdf[["geometry"]].copy().reset_index(drop=True)

                gdf.to_file(cache_path, driver="GPKG")
                stats[stat_key] = len(gdf)
                log.info(f"OSM {label}: {len(gdf)} features fetched and cached.")
                return gdf
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff_base * (2 ** (attempt - 1))
                    log.warning(
                        f"OSM {label} attempt {attempt}/{max_retries} failed ({exc}). "
                        f"Retrying in {wait:.0f}s …"
                    )
                    _time.sleep(wait)

        # All retries exhausted — fail hard
        raise RuntimeError(
            f"OSM {label} fetch failed after {max_retries} attempts. "
            f"Last error: {last_exc}. "
            f"Cannot proceed — empty data would cause routing through buildings/water."
        ) from last_exc

    buildings = _load_or_fetch(
        cache_buildings,
        {"building": True},
        "buildings", "buildings", "buildings_from_cache",
    )
    water = _load_or_fetch(
        cache_water,
        {"natural": ["water", "wetland"], "waterway": True},
        "water / waterways", "water", "water_from_cache",
        retain_tags=["name", "waterway", "natural"]
    )
    roads = _load_or_fetch(
        cache_roads,
        # Phase 5.4: Full highway class list including trunk/primary, which are
        # the strongest attractors for alignment reuse.  'path' is intentionally
        # excluded — it maps footpaths/hiking trails that have no road formation
        # or ROW relevance, and add noise in Myanmar's rural areas.
        #
        # Myanmar OSM note: major corridors (trunk/primary) are reasonably mapped;
        # secondary/track coverage is variable.  Presence = reliable; absence
        # does not imply road-free terrain.
        {
            "highway": [
                "motorway", "trunk", "primary",        # grade-separated / paved national
                "secondary", "tertiary",               # regional / collector
                "unclassified", "track",               # minor / earthwork (no 'path')
            ]
        },
        "roads / tracks", "roads", "roads_from_cache",
        retain_tags=["highway"]
    )
    lulc = _load_or_fetch(
        cache_lulc,
        # Phase 5.4: Expanded LULC tag set for Myanmar context.
        # Added: orchard, vineyard, rubber, meadow (agriculture compensation);
        #        mud, reef (coastal/tidal subgrade);
        #        boundary=protected_area, leisure=nature_reserve (legal barriers).
        #
        # Myanmar OSM note: LULC coverage is severely incomplete — many real
        # forest and wetland areas have NO OSM polygon.  The pipeline applies
        # LULC_UNMAPPED_BASE as a background multiplier to partially compensate.
        # OSM_LULC_WARN_THRESHOLD triggers a WARNING if coverage looks thin.
        {
            "landuse": [
                "forest", "farmland", "conservation", "wood",
                "orchard", "vineyard", "rubber", "meadow",
                "national_park",
            ],
            "natural":  ["wetland", "wood", "scrub", "bare_rock", "mud", "reef",
                         "mangrove"],
            "leisure":  ["nature_reserve"],
            "boundary": ["protected_area", "national_park"],
        },
        "land use cover", "lulc", "lulc_from_cache",
        retain_tags=["landuse", "natural", "leisure", "boundary"]
    )

    # Phase 5.4: LULC data-coverage warning.
    # If very few polygons returned, the LULC layer is likely severely
    # under-representing ground truth — flag this prominently so reviewers
    # do not mistake a quiet cost surface for a fully-mapped one.
    if stats["lulc"] < OSM_LULC_WARN_THRESHOLD:
        log.warning(
            f"OSM LULC coverage: only {stats['lulc']} polygon(s) returned for "
            f"this bounding box (threshold: {OSM_LULC_WARN_THRESHOLD}).  "
            f"LULC penalties will be SIGNIFICANTLY underrepresented — "
            f"many real forests, wetlands, and protected areas are likely "
            f"unmapped in OSM for this part of Myanmar.  "
            f"LULC_UNMAPPED_BASE background multiplier partially compensates."
        )

    # ── Phase 3: DEM stream fallback ─────────────────────────────────────
    # If OSM water coverage is too sparse, a dem_stream_fallback key is set in
    # stats so the caller (main.py) knows to use the DEM-derived raster
    # instead of/in addition to the empty OSM water layer.
    stats["dem_stream_fallback"] = False
    if stats["water"] < OSM_WATER_FALLBACK_TRIGGER:
        log.warning(
            f"OSM water features: {stats['water']} (threshold: {OSM_WATER_FALLBACK_TRIGGER}). "
            f"DEM-derived stream network will be used instead."
        )
        stats["dem_stream_fallback"] = True

    return buildings, water, roads, lulc, stats


def derive_stream_mask_utm(dem_utm, transform_utm, resolution_m=30):
    """
    Convenience wrapper: derive stream network and return as a float32 raster
    aligned to the UTM DEM grid. Used by main.py when dem_stream_fallback=True.
    """
    stream = derive_stream_network(dem_utm, transform_utm, resolution_m)
    log.info("DEM-derived stream mask applied as water_mask substitute.")
    return stream


# ── Phase 11: ESA WorldCover 10m ─────────────────────────────────────────────

def fetch_worldcover(bbox_wgs84):
    """
    Fetch ESA WorldCover 10m land cover from Microsoft Planetary Computer.

    Returns (lulc_array, utm_transform) where lulc_array is a uint8 raster
    with ESA class values (10=Tree, 20=Shrub, 30=Grass, 40=Crop, 50=Built,
    60=Bare, 70=Snow, 80=Water, 90=Wetland, 95=Mangrove, 100=Moss).

    Returns (None, None) if the fetch fails (caller should fall back to OSM).
    """
    _ensure_data_dir()
    tile_key = _tile_key(bbox_wgs84)
    legacy_key = _bbox_key(bbox_wgs84)

    # Phase 14.5: backward-compatible cache lookup
    cache_utm = os.path.join(DATA_DIR, f"worldcover_utm_{tile_key}.tif")
    legacy_cache = os.path.join(DATA_DIR, f"worldcover_utm_{legacy_key}.tif")
    if os.path.exists(legacy_cache) and not os.path.exists(cache_utm):
        cache_utm = legacy_cache

    # Check cache first
    if os.path.exists(cache_utm):
        log.info(f"Loading cached WorldCover: {cache_utm}")
        with rasterio.open(cache_utm) as src:
            return src.read(1), src.transform

    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError:
        try:
            from pystac_client import Client
        except ImportError:
            log.warning("pystac-client not installed — cannot fetch WorldCover.")
            return None, None
        pc = None  # planetary_computer is optional for signing

    west, south, east, north = [float(v) for v in bbox_wgs84]

    try:
        log.info("Querying Planetary Computer for ESA WorldCover 10m …")
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )

        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=[west, south, east, north],
            query={"esa_worldcover:product_version": {"eq": "v200"}},
        )

        items = list(search.items())
        if not items:
            # Try without version filter
            search = catalog.search(
                collections=["esa-worldcover"],
                bbox=[west, south, east, north],
            )
            items = list(search.items())

        if not items:
            log.warning("No WorldCover tiles found for this bounding box.")
            return None, None

        log.info(f"WorldCover: found {len(items)} tile(s). Reading …")

        # Sign URLs if planetary_computer is available
        if pc is not None:
            items = [pc.sign(item) for item in items]

        # Read and mosaic tiles using rioxarray
        try:
            import rioxarray  # noqa: F811
            import xarray as xr

            datasets = []
            for item in items:
                asset = item.assets.get("map") or list(item.assets.values())[0]
                href = asset.href
                ds = rioxarray.open_rasterio(href, chunks="auto")
                datasets.append(ds)

            # Fix IV: use merge_arrays for a proper spatial mosaic.
            # xr.concat(..., dim="band") stacked tiles along the band axis
            # instead of mosaicking them spatially — producing wrong results
            # for any multi-tile bounding box.
            if len(datasets) == 1:
                merged = datasets[0]
            else:
                merged = rioxarray.merge_arrays(datasets, nodata=255)

            # Phase 14.5: Explicit cleanup — release individual tile memory
            # before clipping/reprojection to avoid OOM on large tile requests.
            for ds in datasets:
                try:
                    ds.close()
                except Exception:
                    pass
            del datasets
            gc.collect()

            # Clip to bounding box (WGS-84)
            clipped = merged.rio.clip_box(
                minx=west, miny=south, maxx=east, maxy=north
            )
            del merged

            # Reproject to UTM
            from config import UTM_EPSG
            reprojected = clipped.rio.reproject(f"EPSG:{UTM_EPSG}")
            del clipped

            # Extract numpy array and transform
            lulc_arr = reprojected.values
            if lulc_arr.ndim == 3:
                lulc_arr = lulc_arr[0]  # (band, row, col) → (row, col)
            lulc_arr = lulc_arr.astype(np.uint8)
            utm_tf = reprojected.rio.transform()
            del reprojected
            gc.collect()

            # Cache to disk
            rows_wc, cols_wc = lulc_arr.shape
            with rasterio.open(
                cache_utm, "w", driver="GTiff",
                height=rows_wc, width=cols_wc, count=1,
                dtype="uint8",
                crs=f"EPSG:{UTM_EPSG}",
                transform=utm_tf,
                compress="deflate",
            ) as dst:
                dst.write(lulc_arr, 1)

            log.info(
                f"WorldCover: {rows_wc}×{cols_wc} raster at 10m cached → {cache_utm}"
            )
            return lulc_arr, utm_tf

        except Exception as exc:
            log.warning(f"rioxarray WorldCover read failed ({exc}). Trying rasterio fallback …")
            # Fallback: direct rasterio windowed read
            item = items[0]
            asset = item.assets.get("map") or list(item.assets.values())[0]
            href = asset.href

            with rasterio.open(href) as src:
                # Compute the window for our bounding box
                from rasterio.windows import from_bounds as window_from_bounds
                window = window_from_bounds(west, south, east, north, src.transform)
                lulc_wgs = src.read(1, window=window).astype(np.uint8)
                wgs_tf = src.window_transform(window)
                wgs_crs = src.crs
                wgs_h, wgs_w = lulc_wgs.shape

            # Reproject to UTM
            from config import UTM_EPSG
            dst_crs = rasterio.crs.CRS.from_epsg(UTM_EPSG)
            from rasterio.warp import calculate_default_transform as calc_tf
            utm_tf, utm_w, utm_h = calc_tf(
                wgs_crs, dst_crs, wgs_w, wgs_h,
                left=west, bottom=south, right=east, top=north,
            )
            lulc_utm = np.zeros((utm_h, utm_w), dtype=np.uint8)
            reproject(
                source=lulc_wgs, destination=lulc_utm,
                src_transform=wgs_tf, src_crs=wgs_crs,
                dst_transform=utm_tf, dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

            with rasterio.open(
                cache_utm, "w", driver="GTiff",
                height=utm_h, width=utm_w, count=1,
                dtype="uint8", crs=dst_crs, transform=utm_tf,
                compress="deflate",
            ) as dst:
                dst.write(lulc_utm, 1)

            log.info(f"WorldCover (rasterio fallback): {utm_h}×{utm_w} cached → {cache_utm}")
            return lulc_utm, utm_tf

    except Exception as exc:
        log.warning(f"WorldCover fetch failed ({exc}). Falling back to OSM LULC.")
        return None, None


# ── Phase 11: Overture Maps Buildings ─────────────────────────────────────────

def fetch_overture_buildings(bbox_wgs84):
    """
    Fetch building footprints from Overture Maps (Meta/Microsoft ML-derived).

    Returns a GeoDataFrame in WGS-84 CRS, or an empty GeoDataFrame on failure.
    These have dramatically better rural Myanmar coverage than OSM alone.
    """
    _ensure_data_dir()
    tile_key = _tile_key(bbox_wgs84)
    legacy_key = _bbox_key(bbox_wgs84)

    # Phase 14.5: backward-compatible cache lookup
    cache_path = os.path.join(DATA_DIR, f"overture_buildings_{tile_key}.gpkg")
    legacy_path = os.path.join(DATA_DIR, f"overture_buildings_{legacy_key}.gpkg")
    if os.path.exists(legacy_path) and not os.path.exists(cache_path):
        cache_path = legacy_path

    if os.path.exists(cache_path):
        try:
            gdf = gpd.read_file(cache_path)
            log.info(f"Loaded cached Overture buildings: {len(gdf)} footprints")
            return gdf
        except Exception as exc:
            log.warning(f"Overture cache read failed ({exc}), re-downloading.")

    try:
        import overturemaps
    except ImportError:
        log.warning("overturemaps not installed — cannot fetch Overture data.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    west, south, east, north = [float(v) for v in bbox_wgs84]
    bbox = (west, south, east, north)

    # Overture S3 buckets are public — ensure unsigned requests
    import os as _os
    _os.environ.setdefault("AWS_NO_SIGN_REQUEST", "true")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"Fetching Overture Maps buildings from S3 (attempt {attempt}/{max_retries}) …")
            reader = overturemaps.record_batch_reader("building", bbox=bbox)
            table = reader.read_all()

            # Convert Arrow table → GeoDataFrame
            import pyarrow as pa  # bundled with overturemaps

            # Overture schema has a 'geometry' column in WKB format
            df = table.to_pandas()
            if "geometry" in df.columns:
                from shapely import wkb
                df["geometry"] = df["geometry"].apply(
                    lambda g: wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g
                )
                gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            else:
                log.warning("Overture buildings table has no geometry column.")
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

            # Keep only relevant columns
            keep_cols = ["geometry"]
            for c in ["height", "num_floors", "class", "subtype"]:
                if c in gdf.columns:
                    keep_cols.append(c)
            gdf = gdf[keep_cols].copy().reset_index(drop=True)

            gdf.to_file(cache_path, driver="GPKG")
            log.info(f"Overture buildings: {len(gdf)} footprints fetched and cached.")
            return gdf

        except Exception as exc:
            if attempt < max_retries:
                wait = 2 ** attempt  # 2, 4 seconds
                log.warning(
                    f"Overture attempt {attempt} failed ({exc}). "
                    f"Retrying in {wait}s …"
                )
                import time
                time.sleep(wait)
            else:
                log.warning(f"Overture buildings fetch failed after {max_retries} attempts ({exc}). Using OSM only.")
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def merge_building_sources(osm_buildings, overture_buildings, dedup_radius_m=15.0):
    """
    Merge OSM and Overture building footprints with spatial deduplication.

    Overture buildings whose centroid is within dedup_radius_m of any OSM
    building centroid are dropped (assumed duplicate).  Remaining Overture
    buildings are appended.

    Args:
        osm_buildings:      GeoDataFrame in UTM CRS
        overture_buildings: GeoDataFrame in WGS-84 CRS (will be reprojected)
        dedup_radius_m:     deduplication radius in metres

    Returns:
        Merged GeoDataFrame in UTM CRS.
    """
    if overture_buildings is None or len(overture_buildings) == 0:
        log.info("No Overture buildings to merge — using OSM only.")
        return osm_buildings

    if osm_buildings is None or len(osm_buildings) == 0:
        log.info("No OSM buildings — using Overture only.")
        from config import UTM_EPSG
        return overture_buildings.to_crs(epsg=UTM_EPSG)

    from config import UTM_EPSG

    # Reproject Overture to UTM
    ov_utm = overture_buildings.to_crs(epsg=UTM_EPSG)

    # Compute centroids for spatial deduplication
    osm_centroids = osm_buildings.geometry.centroid
    ov_centroids = ov_utm.geometry.centroid

    # Fix VIII + Phase 14.5: Overture dedup via geometry intersection (buffered sjoin).
    # Chunked processing to bound peak memory on large datasets.
    DEDUP_CHUNK_SIZE = 50_000
    try:
        ov_buffered = ov_utm.copy()
        ov_buffered["geometry"] = ov_utm.geometry.buffer(10.0)

        # Phase 14.5: Chunked sjoin to cap memory on large tiles
        dup_indices = set()
        n_chunks = max(1, (len(ov_buffered) + DEDUP_CHUNK_SIZE - 1) // DEDUP_CHUNK_SIZE)
        log.info(f"Overture dedup: {len(ov_buffered)} buildings in {n_chunks} chunk(s)")
        for start in range(0, len(ov_buffered), DEDUP_CHUNK_SIZE):
            chunk = ov_buffered.iloc[start:start + DEDUP_CHUNK_SIZE]
            duplicates = gpd.sjoin(
                chunk, osm_buildings[["geometry"]],
                how="inner", predicate="intersects"
            )
            dup_indices.update(duplicates.index.unique())
        ov_new = ov_utm.drop(index=list(dup_indices)).copy()
        n_dedup = len(ov_utm) - len(ov_new)
        del ov_buffered
        gc.collect()
    except Exception as exc:
        # Fallback to centroid-proximity if sjoin fails (e.g. empty GDF)
        log.warning(f"Overture sjoin dedup failed ({exc}), falling back to centroid proximity.")
        keep_mask = np.ones(len(ov_utm), dtype=bool)
        from shapely import STRtree
        tree = STRtree(osm_centroids.values)
        for i, ov_c in enumerate(ov_centroids):
            nearest_idx = tree.nearest(ov_c)
            nearest_osm = osm_centroids.iloc[nearest_idx]
            dist = ov_c.distance(nearest_osm)
            if dist < dedup_radius_m:
                keep_mask[i] = False
        n_dedup = int((~keep_mask).sum())
        ov_new = ov_utm[keep_mask].copy()

    # Ensure schema compatibility — keep only geometry column
    ov_new = ov_new[["geometry"]].copy()
    osm_clean = osm_buildings[["geometry"]].copy() if "geometry" in osm_buildings.columns else osm_buildings

    import pandas as pd
    merged = gpd.GeoDataFrame(
        pd.concat([osm_clean, ov_new], ignore_index=True),
        crs=f"EPSG:{UTM_EPSG}"
    )

    log.info(
        f"Buildings merged: {len(osm_buildings)} OSM + {len(ov_new)} Overture "
        f"({n_dedup} duplicates removed) = {len(merged)} total"
    )
    return merged


# ── Phase 14.2: Custom Water Dataset ──────────────────────────────────────────

def fetch_custom_water(bbox_wgs84):
    """
    Fetch and clip custom water polygons from a local GeoPackage file.
    Returns a GeoDataFrame in WGS-84 CRS, or an empty GeoDataFrame if disabled/failed.
    """
    from config import USE_CUSTOM_WATER, CUSTOM_WATER_GPKG, CUSTOM_WATER_FILTER_COLUMN, CUSTOM_WATER_TARGETS
    if not USE_CUSTOM_WATER:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if not os.path.exists(CUSTOM_WATER_GPKG):
        log.warning(f"Custom water GeoPackage not found at {CUSTOM_WATER_GPKG}. Falling back to OSM only.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    try:
        log.info(f"Loading custom water data from {CUSTOM_WATER_GPKG} ...")
        gdf = gpd.read_file(CUSTOM_WATER_GPKG)
        
        # Filter features based on targets
        if CUSTOM_WATER_FILTER_COLUMN and CUSTOM_WATER_FILTER_COLUMN in gdf.columns:
            filtered_gdf = gdf[gdf[CUSTOM_WATER_FILTER_COLUMN].isin(CUSTOM_WATER_TARGETS)].copy()
            log.info(f"Filtered {len(gdf)} down to {len(filtered_gdf)} features matching {CUSTOM_WATER_TARGETS}.")
        else:
            filtered_gdf = gdf.copy()

        # Clip to bounding box
        west, south, east, north = [float(v) for v in bbox_wgs84]
        clipped = gpd.clip(filtered_gdf, box(west, south, east, north))
        
        log.info(f"Custom water: {len(clipped)} features retained after clipping to bounding box.")
        return clipped.reset_index(drop=True)

    except Exception as exc:
        log.warning(f"Failed to load/clip custom water dataset ({exc}). Falling back to OSM only.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        

def fetch_dam_lake(bbox_wgs84):
    """
    Fetch and clip Dam & Lake polygons from a local GeoPackage file.
    These are used as absolute NO-GO avoidance zones.
    Returns a GeoDataFrame in WGS-84 CRS.
    """
    from config import USE_DAM_LAKE_AVOIDANCE, DAM_LAKE_GPKG
    if not USE_DAM_LAKE_AVOIDANCE:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
    if not os.path.exists(DAM_LAKE_GPKG):
        log.warning(f"Dam/Lake GeoPackage not found at {DAM_LAKE_GPKG}. Dam avoidance skipped.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
    try:
        log.info(f"Loading Dam & Lake avoidance zones from {DAM_LAKE_GPKG} ...")
        gdf = gpd.read_file(DAM_LAKE_GPKG)
        
        # Clip to bounding box
        west, south, east, north = [float(v) for v in bbox_wgs84]
        clipped = gpd.clip(gdf, box(west, south, east, north))
        
        log.info(f"Dam/Lake: {len(clipped)} features retained strictly for avoidance after clipping.")
        return clipped.reset_index(drop=True)
        
    except Exception as exc:
        log.warning(f"Failed to load/clip Dam & Lake dataset ({exc}). Avoidance skipped.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

