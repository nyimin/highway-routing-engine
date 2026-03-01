"""
highway_alignment.py
====================
Preliminary Highway Alignment Generator
Myanmar Expressway Standards

Pipeline:
  1. Bounding box + 20% margin
  2. DEM acquisition (SRTM via `elevation` lib, or mock fallback)
  3. OSM vector data  (buildings, water) via osmnx
  4. Reproject everything to local UTM
  5. Slope surface from DEM
  6. Rasterise exclusion zones
  7. Cost-surface assembly
  8. Dijkstra pathfinding (skimage.graph.route_through_array)
  9. LineString + B-spline smoothing
  10. 440m curve-radius verification
  11. 61m RoW setback re-verification
  12. Export  preliminary_route.geojson  (EPSG:4326)

Usage:
  python highway_alignment.py
  # Edit POINT_A / POINT_B below before running.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  USER CONFIGURATION  ← Edit these coordinates
# ─────────────────────────────────────────────────────────────────────────────
# WGS-84 (lon, lat).
# Point A: 17.17894837308733358°N, 94.59079006576709503°E  (Ayeyarwaddy Region, Myanmar)
# Point B: 17.486777°N, 96.457154°E  (Bago Region, Myanmar)
POINT_A = (94.59079006576709503, 17.17894837308733358)   # (longitude, latitude) of start
POINT_B = (96.457154, 17.486777)                         # (longitude, latitude) of end

OUTPUT_FILE = "preliminary_route.geojson"

# UTM Zone selection:
#   Point A @ 94.35°E → Zone 46N (EPSG:32646)
#   Point B @ 96.46°E → straddles Zone 47N boundary
#   Common CRS: EPSG:32646 (midpoint ~95.4°E – max distortion < 0.1 %)
UTM_EPSG = 32646

# DEM resolution used for cost raster (metres per pixel).
# Larger = faster but less accurate.  SRTM native ≈ 30 m.
RESOLUTION = 30        # metres

# Engineering parameters
ROW_BUFFER_M      = 61      # half of the 122 m total RoW
SLOPE_OPTIMAL_PCT = 3.0     # ruling gradient
SLOPE_MAX_PCT     = 6.0     # absolute maximum
MIN_CURVE_RADIUS  = 440     # metres
WATER_PENALTY     = 5000    # high flat multiplier to guarantee perpendicular crossing

# Cache directory – all downloaded layers are stored here
DATA_DIR = "data"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import sys
import warnings
import json
import logging
import math
import traceback
import io
import zipfile
import netrc
import os

# Load .env from project root (keeps secrets out of source code)
from dotenv import load_dotenv
load_dotenv()  # reads .env in cwd

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.merge import merge as rasterio_merge
import osmnx as ox
from shapely.geometry import (
    LineString, Point, MultiLineString, mapping
)
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.graph import route_through_array
from pyproj import Transformer
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("highway_alignment")

# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_data_dir():
    """Create data/ directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)

def _bbox_key(bbox_wgs84):
    """
    A short, filesystem-safe string that uniquely identifies a bounding box.
    Rounded to 4 decimal places (~11 m precision) to avoid float noise.
    """
    w, s, e, n = [round(v, 4) for v in bbox_wgs84]
    return f"W{w}_S{s}_E{e}_N{n}".replace("-", "m")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bbox_with_margin(pt_a, pt_b, margin=0.20):
    """Return (west, south, east, north) with a % margin."""
    lons = [pt_a[0], pt_b[0]]
    lats = [pt_a[1], pt_b[1]]
    dlon = (max(lons) - min(lons)) * margin
    dlat = (max(lats) - min(lats)) * margin
    # Ensure a minimum margin so small routes get enough context
    dlon = max(dlon, 0.05)
    dlat = max(dlat, 0.05)
    return (
        min(lons) - dlon,   # west
        min(lats) - dlat,   # south
        max(lons) + dlon,   # east
        max(lats) + dlat,   # north
    )


def wgs84_to_utm(lon, lat, epsg=UTM_EPSG):
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return t.transform(lon, lat)


def utm_to_wgs84(x, y, epsg=UTM_EPSG):
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return t.transform(x, y)


def xy_to_rowcol(x, y, transform):
    """Convert UTM (x, y) → (row, col) given a rasterio Affine transform."""
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def rowcol_to_xy(row, col, transform):
    """Convert (row, col) → UTM (x, y) cell-centre."""
    x = transform.c + (col + 0.5) * transform.a
    y = transform.f + (row + 0.5) * transform.e
    return x, y


def curve_radius_at_point(pt_prev, pt_curr, pt_next):
    """
    Menger curvature formula → radius (metres).
    Returns ∞ if the three points are collinear.
    """
    ax, ay = pt_prev
    bx, by = pt_curr
    cx, cy = pt_next
    ab = math.hypot(bx - ax, by - ay)
    bc = math.hypot(cx - bx, cy - by)
    ac = math.hypot(cx - ax, cy - ay)
    area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0
    if area == 0:
        return float("inf")
    return (ab * bc * ac) / (4.0 * area)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEM ACQUISITION  (with disk cache)
# ─────────────────────────────────────────────────────────────────────────────

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

def fetch_dem(bbox_wgs84, resolution_m=RESOLUTION):
    """
    Download SRTM GL3 (90 m) via OpenTopography and cache the result.

    Cache logic:
      - On first run  → download, reproject to UTM, save to data/dem_<key>.tif
      - On later runs → load directly from data/dem_<key>.tif  (no network call)
      - Cache is invalidated automatically when the bounding box changes.

    Falls back to synthetic mock DEM on any network/API failure.
    """
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    import tempfile

    _ensure_data_dir()
    key      = _bbox_key(bbox_wgs84)
    cache_wgs = os.path.join(DATA_DIR, f"dem_wgs84_{key}.tif")   # raw WGS-84 download
    cache_utm = os.path.join(DATA_DIR, f"dem_utm_{key}.tif")      # reprojected UTM

    west, south, east, north = bbox_wgs84

    # ── Load from cache if available ────────────────────────────────────────
    if os.path.exists(cache_utm):
        log.info(f"Loading cached UTM DEM: {cache_utm}")
        with rasterio.open(cache_utm) as src:
            dem_utm = src.read(1).astype(np.float32)
            utm_tf  = src.transform
        log.info(
            f"Cached DEM: shape={dem_utm.shape}, "
            f"min={dem_utm.min():.1f} m, max={dem_utm.max():.1f} m"
        )
        return dem_utm, utm_tf

    # ── Download fresh ───────────────────────────────────────────────────────
    params = {
        "demtype":      "SRTMGL3",
        "west":         float(west),
        "south":        float(south),
        "east":         float(east),
        "north":        float(north),
        "outputFormat": "GTiff",
        "API_Key":      os.getenv("OPENTOPOGRAPHY_API_KEY", ""),
    }
    log.info("Downloading SRTM DEM from OpenTopography ...")
    try:
        resp = requests.get(OPENTOPO_URL, params=params, timeout=180, stream=True)
        resp.raise_for_status()

        # Save raw WGS-84 GeoTIFF to cache
        with open(cache_wgs, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        log.info(f"Raw DEM saved: {cache_wgs}")

        with rasterio.open(cache_wgs) as src:
            dem_wgs = src.read(1).astype(np.float32)
            dem_wgs[dem_wgs < -100] = 0
            wgs_tf  = src.transform
            wgs_crs = src.crs
            wgs_w   = src.width
            wgs_h   = src.height

        log.info(
            f"SRTM DEM: shape=({wgs_h},{wgs_w}), "
            f"min={dem_wgs.min():.1f} m, max={dem_wgs.max():.1f} m"
        )

        # Reproject to UTM
        dst_crs = rasterio.crs.CRS.from_epsg(UTM_EPSG)
        utm_tf, utm_w, utm_h = calculate_default_transform(
            wgs_crs, dst_crs, wgs_w, wgs_h,
            left=west, bottom=south, right=east, top=north,
        )
        dem_utm = np.zeros((utm_h, utm_w), dtype=np.float32)
        reproject(
            source=dem_wgs, destination=dem_utm,
            src_transform=wgs_tf, src_crs=wgs_crs,
            dst_transform=utm_tf, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

        # Save cached UTM GeoTIFF
        with rasterio.open(
            cache_utm, "w", driver="GTiff",
            height=utm_h, width=utm_w, count=1,
            dtype=dem_utm.dtype,
            crs=rasterio.crs.CRS.from_epsg(UTM_EPSG),
            transform=utm_tf,
        ) as dst:
            dst.write(dem_utm, 1)
        log.info(
            f"UTM DEM cached: {cache_utm}  "
            f"shape={dem_utm.shape}, min={dem_utm.min():.1f} m, max={dem_utm.max():.1f} m"
        )
        return dem_utm, utm_tf

    except Exception as exc:
        log.warning(f"OpenTopography DEM download failed ({exc}). Using mock DEM.")
        return _mock_dem(bbox_wgs84, resolution_m)




def _mock_dem(bbox_wgs84, resolution_m):
    """
    Synthetic 30-m DEM with gentle north-south gradient + random hills,
    so the routing cost surface is non-trivial.
    """
    west, south, east, north = bbox_wgs84
    x0, y0 = wgs84_to_utm(west, south)
    x1, y1 = wgs84_to_utm(east, north)
    cols = max(10, int((x1 - x0) / resolution_m))
    rows = max(10, int((y1 - y0) / resolution_m))

    log.info(f"Mock DEM: {rows} rows × {cols} cols @ {resolution_m} m/px")

    rng = np.random.default_rng(42)
    # Base elevation 50 m + N-S gradient (2 m / km) + Gaussian bumps
    base = np.linspace(50, 200, rows)[:, None] * np.ones((1, cols))
    noise = rng.normal(0, 8, (rows, cols))
    # A few ridge features
    for _ in range(5):
        r = rng.integers(rows // 4, 3 * rows // 4)
        c = rng.integers(cols // 4, 3 * cols // 4)
        Y, X = np.ogrid[:rows, :cols]
        bump = 80 * np.exp(-((Y - r) ** 2 + (X - c) ** 2) / (2 * (rows // 8) ** 2))
        base += bump
    dem = (base + noise).astype(np.float32)

    transform = from_bounds(x0, y0, x1, y1, cols, rows)
    return dem, transform


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OSM VECTOR DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_osm_layers(bbox_wgs84):
    """
    Download OSM buildings and water bodies, caching each as a GeoPackage.

    Cache logic:
      - On first run  → query Overpass API, save to data/buildings_<key>.gpkg
                        and data/water_<key>.gpkg
      - On later runs → load directly from the GeoPackage files
      - Cache is keyed by bounding box; changes to bbox auto-invalidate.
    """
    _ensure_data_dir()
    key            = _bbox_key(bbox_wgs84)
    cache_buildings = os.path.join(DATA_DIR, f"buildings_{key}.gpkg")
    cache_water     = os.path.join(DATA_DIR, f"water_{key}.gpkg")

    # Cast to plain Python floats to avoid osmnx NaN-to-int crash
    west  = float(bbox_wgs84[0])
    south = float(bbox_wgs84[1])
    east  = float(bbox_wgs84[2])
    north = float(bbox_wgs84[3])

    def _load_or_fetch(cache_path, tags, label):
        if os.path.exists(cache_path):
            log.info(f"Loading cached OSM {label}: {cache_path}")
            try:
                return gpd.read_file(cache_path)
            except Exception as exc:
                log.warning(f"Cache read failed ({exc}), re-downloading {label}.")

        log.info(f"Fetching OSM {label} from Overpass ...")
        try:
            # osmnx 2.x: bbox=(west, south, east, north)
            gdf = ox.features_from_bbox(
                bbox=(west, south, east, north),
                tags=tags,
            )
            gdf = gdf[["geometry"]].copy().reset_index(drop=True)
            log.info(f"OSM {label}: {len(gdf)} features -> saving to {cache_path}")
            gdf.to_file(cache_path, driver="GPKG")
            return gdf
        except Exception as exc:
            log.warning(f"OSM {label} fetch failed ({exc}). Returning empty layer.")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    buildings = _load_or_fetch(
        cache_buildings,
        {"building": True},
        "buildings",
    )
    water = _load_or_fetch(
        cache_water,
        {"natural": ["water", "wetland"], "waterway": True},
        "water / waterways",
    )
    return buildings, water


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SLOPE CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope(dem, resolution_m):
    """
    Compute slope in PERCENT from an elevation array.
    Uses numpy gradient (central differences).
    Zero-valued border cells (UTM reprojection nodata) are masked out
    before differencing to avoid phantom extreme-slope artefacts.
    """
    # Replace nodata zeros at the border with the nearest valid value
    # so the gradient doesn't spike at the edge
    dem_work = dem.copy()
    nodata_mask = (dem_work == 0)
    if nodata_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(nodata_mask, return_indices=True)
        dem_work[nodata_mask] = dem_work[idx[0][nodata_mask], idx[1][nodata_mask]]

    dy, dx = np.gradient(dem_work, resolution_m)   # rise / run
    slope_pct = np.sqrt(dx**2 + dy**2) * 100.0     # %
    slope_pct[nodata_mask] = 0.0                    # restore nodata to 0 cost
    return slope_pct.astype(np.float32), nodata_mask


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RASTERISE VECTOR LAYERS
# ─────────────────────────────────────────────────────────────────────────────

def rasterise_layer(gdf_utm, transform, shape, value=1.0):
    """
    Burn a GeoDataFrame (UTM) into a raster matching (shape, transform).
    Returns float32 array with 0 (clear) or `value` (occupied).
    """
    if gdf_utm is None or len(gdf_utm) == 0:
        return np.zeros(shape, dtype=np.float32)
    geoms = [g for g in gdf_utm.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=np.float32)
    burned = rasterize(
        [(g, value) for g in geoms],
        out_shape=shape,
        transform=transform,
        fill=0.0,
        dtype=np.float32,
    )
    return burned


# ─────────────────────────────────────────────────────────────────────────────
# 7.  COST SURFACE
# ─────────────────────────────────────────────────────────────────────────────

IMPASSABLE   = 1e9   # sentinel cost for forbidden cells
BORDER_CELLS = 20    # pixel-width of hard impassable border ring (prevents
                     # the path from escaping through DEM nodata edges)


def build_cost_surface(slope_pct, building_mask, water_mask, nodata_mask=None):
    """
    Combine slope, building exclusions, water, nodata, and a hard border ring
    into a single cost array.

    Slope cost:
      0-3 %   ->  1.0  (optimal)
      3-6 %   ->  exponential 1 ... 10
      > 6 %   ->  IMPASSABLE

    Building buffer (61 m already applied): -> IMPASSABLE
    Water:                                  -> WATER_PENALTY multiplier
    Nodata / zero-elevation cells:          -> IMPASSABLE
    BORDER_CELLS-wide edge ring:            -> IMPASSABLE  (keeps path inside DEM)
    """
    rows, cols = slope_pct.shape
    cost = np.ones((rows, cols), dtype=np.float64)

    # Slope contribution
    mask_optimal = slope_pct <= SLOPE_OPTIMAL_PCT
    mask_penalty = (slope_pct > SLOPE_OPTIMAL_PCT) & (slope_pct <= SLOPE_MAX_PCT)
    mask_impass  = slope_pct > SLOPE_MAX_PCT

    # Normalised position within [3 %, 6 %] -> exponential cost 1 ... 10
    slope_norm = np.where(
        mask_penalty,
        (slope_pct - SLOPE_OPTIMAL_PCT) / (SLOPE_MAX_PCT - SLOPE_OPTIMAL_PCT),
        0.0,
    )
    cost = np.where(mask_optimal, 1.0, cost)
    cost = np.where(mask_penalty, np.exp(slope_norm * math.log(10)), cost)
    cost = np.where(mask_impass, IMPASSABLE, cost)

    # -------------------------------------------------------------------------
    # River Crossing Engineering Logic
    # -------------------------------------------------------------------------
    from scipy.ndimage import binary_closing, binary_dilation

    if np.any(water_mask > 0):
        # 1. Fill small alluvial islands/sandbars
        # A morphological closing merges small gaps. Structural element radius = 4 pixels (~120m)
        radius = 4
        y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        disk = x_grid**2 + y_grid**2 <= radius**2
        water_closed = binary_closing(water_mask > 0, structure=disk)

        # 2. Extremely High Flat Penalty
        # By making EVERY pixel of water equally and massively expensive (e.g. 5000x cost),
        # the pathfinder mathematically MUST minimise the total number of water pixels it touches.
        # Minimising water pixels guarantees the path crosses at the narrowest possible point,
        # and crosses exactly perpendicularly (a straight line across).
        # We removed the shore-distance penalty because it inadvertently created a "cheap"
        # shore-hugging lane which caused the road to bend inside the river.
        water_cost = np.where(water_closed, cost * WATER_PENALTY, 0)
        cost = np.where(water_closed, cost + water_cost, cost)

        # 3. Bridge Abutment Zone (Ignore Steep River Banks)
        # Create a 3-pixel (~90m) buffer around the edge of the water mask.
        radius_abut = 3
        y_grid_abut, x_grid_abut = np.ogrid[-radius_abut : radius_abut + 1, -radius_abut : radius_abut + 1]
        disk_abut = x_grid_abut**2 + y_grid_abut**2 <= radius_abut**2
        
        # Dilate water to find the banks, then subtract the water itself to get just the dry shore strip
        water_dilated = binary_dilation(water_closed, structure=disk_abut)
        abutment_zone = water_dilated & ~water_closed
        
        # Where the DEM sees a cliff (>6% slope) right on the river bank, and previously
        # marked it IMPASSABLE (1e9), we override it to a passable cost (100.0).
        # This allows the pathfinder to "climb out" of the river instead of being trapped
        # and swimming downstream.
        mask_steep_abutment = abutment_zone & mask_impass
        cost = np.where(mask_steep_abutment, 100.0, cost)
        
        log.info(f"Bridge abutment zones recovered {np.sum(mask_steep_abutment)} steep bank cells")

    # Building exclusion (must come before border / nodata – still overrides)
    cost = np.where(building_mask > 0, IMPASSABLE, cost)

    # Nodata cells (UTM reprojection artefacts) -> IMPASSABLE
    if nodata_mask is not None:
        cost[nodata_mask] = IMPASSABLE

    # Hard border ring -> IMPASSABLE  (prevents routing along DEM edges)
    b = max(1, BORDER_CELLS)
    cost[:b,  :]  = IMPASSABLE   # top
    cost[-b:, :]  = IMPASSABLE   # bottom
    cost[:,  :b]  = IMPASSABLE   # left
    cost[:, -b:]  = IMPASSABLE   # right

    log.info(
        f"Cost surface: min={cost[cost < IMPASSABLE].min():.2f}, "
        f"max={cost[cost < IMPASSABLE].max():.2f}, "
        f"impassable cells={np.sum(cost >= IMPASSABLE)}"
    )
    return cost.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PATHFINDING
# ─────────────────────────────────────────────────────────────────────────────

def find_path(cost, start_rc, end_rc):
    """
    Run Dijkstra / MCP via skimage.graph.route_through_array.
    start_rc / end_rc: (row, col) tuples.
    Returns list of (row, col) tuples.
    """
    # route_through_array expects fully-connected by default (use_start_value=False)
    path_indices, cost_val = route_through_array(
        cost,
        start_rc,
        end_rc,
        fully_connected=True,   # 8-connected grid
        geometric=True,         # diagonal steps cost √2 more
    )
    log.info(f"Pathfinding: {len(path_indices)} waypoints, "
             f"total cost={cost_val:.1f}")
    return path_indices


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SMOOTHING + CURVE-RADIUS VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def smooth_path(coords_utm, n_points=500, smoothing=None):
    """
    Fit a B-spline through the raw path coordinates.
    coords_utm: list of (x, y) in metres.
    Returns smoothed list of (x, y).
    """
    if len(coords_utm) < 4:
        return coords_utm

    xs = np.array([c[0] for c in coords_utm])
    ys = np.array([c[1] for c in coords_utm])

    # Parametric B-spline (k=3 → cubic)
    if smoothing is None:
        # Heuristic: allow ~2 m average deviation per point
        smoothing = len(coords_utm) * 4.0

    # Apply Douglas-Peucker simplification before B-spline to remove raster zig-zags
    from shapely.geometry import LineString
    raw_ls = LineString(coords_utm)
    # Simplify with a tolerance of 15m. Points that deviate less than 15m from the line 
    # connecting their neighbors are dropped, creating longer straight tangents.
    simplified_ls = raw_ls.simplify(tolerance=15.0, preserve_topology=True)
    simp_path = list(simplified_ls.coords)
    
    if len(simp_path) < 4:
         log.warning(f"Douglas-Peucker simplification over-simplified the path ({len(simp_path)} points). Reverting to raw.")
         simp_path = coords_utm
    else:
         log.info(f"Path simplification: {len(coords_utm)} -> {len(simp_path)} points")

    xs = np.array([c[0] for c in simp_path])
    ys = np.array([c[1] for c in simp_path])

    try:
        tck, u = splprep([xs, ys], s=smoothing, k=3)
        u_fine = np.linspace(0, 1, n_points)
        x_sm, y_sm = splev(u_fine, tck)
        return list(zip(x_sm.tolist(), y_sm.tolist()))
    except Exception as exc:
        log.warning(f"B-spline smoothing failed ({exc}). Returning simplified path.")
        return simp_path


def verify_curve_radius(coords_utm, min_radius=MIN_CURVE_RADIUS):
    """
    Walk the smoothed path and find the minimum curve radius.
    Returns (min_r, [bottleneck coords]) and logs all violations.
    """
    pts = coords_utm
    n = len(pts)
    if n < 3:
        return float("inf"), []

    min_r = float("inf")
    violations = []
    for i in range(1, n - 1):
        r = curve_radius_at_point(pts[i - 1], pts[i], pts[i + 1])
        if r < min_r:
            min_r = r
        if r < min_radius:
            violations.append((pts[i], r))

    if violations:
        log.warning(
            f"Curve-radius violations: {len(violations)} segments "
            f"below {min_radius} m (minimum found: {min_r:.1f} m)"
        )
        for coord, r in violations[:5]:
            log.warning(f"  Bottleneck at UTM {coord[0]:.1f}, {coord[1]:.1f}  r={r:.1f} m")
    else:
        log.info(f"Curve radius OK: minimum = {min_r:.1f} m (≥ {min_radius} m)")

    return min_r, violations


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SETBACK RE-VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_row_setback(line_utm, buildings_utm, buffer_m=ROW_BUFFER_M):
    """
    Check that the smoothed LineString maintains ≥ buffer_m from buildings.
    """
    if buildings_utm is None or len(buildings_utm) == 0:
        log.info("No building data; RoW setback check skipped.")
        return True

    all_buildings = unary_union(buildings_utm.geometry.tolist())
    min_dist = line_utm.distance(all_buildings)

    if min_dist < buffer_m:
        log.warning(
            f"RoW setback VIOLATED: closest building is {min_dist:.1f} m "
            f"from centreline (minimum is {buffer_m} m)"
        )
        return False
    else:
        log.info(f"RoW setback OK: {min_dist:.1f} m ≥ {buffer_m} m")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 11.  METADATA
# ─────────────────────────────────────────────────────────────────────────────

def compute_metadata(line_utm, slope_pct, transform):
    """
    Length of the route (m) and maximum slope along the centreline.
    """
    length_m = line_utm.length

    # Sample slope at each vertex
    max_slope = 0.0
    rows, cols = slope_pct.shape
    for x, y in line_utm.coords:
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        if 0 <= row < rows and 0 <= col < cols:
            s = float(slope_pct[row, col])
            if s > max_slope:
                max_slope = s

    return {
        "total_length_m": round(length_m, 1),
        "total_length_km": round(length_m / 1000, 3),
        "max_slope_pct": round(max_slope, 2),
        "utm_epsg": UTM_EPSG,
        "row_buffer_m": ROW_BUFFER_M,
        "min_curve_radius_m": MIN_CURVE_RADIUS,
        "point_a_lonlat": list(POINT_A),
        "point_b_lonlat": list(POINT_B),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 12.  EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_geojson(line_wgs84, metadata, output_path):
    feature = {
        "type": "Feature",
        "geometry": mapping(line_wgs84),
        "properties": metadata,
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    log.info(f"Exported: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("═" * 65)
    log.info("  Highway Alignment Generator  –  Myanmar Expressway Standards")
    log.info("═" * 65)
    log.info(f"Point A : lon={POINT_A[0]}, lat={POINT_A[1]}")
    log.info(f"Point B : lon={POINT_B[0]}, lat={POINT_B[1]}")
    log.info(f"UTM CRS : EPSG:{UTM_EPSG}")

    # ── Step 1: Bounding box ─────────────────────────────────────────────────
    bbox = bbox_with_margin(POINT_A, POINT_B)
    west, south, east, north = bbox
    log.info(f"BBox (WGS-84): W={west:.4f} S={south:.4f} E={east:.4f} N={north:.4f}")

    # ── Step 2: DEM ──────────────────────────────────────────────────────────
    dem, transform = fetch_dem(bbox)
    rows, cols = dem.shape

    # ── Step 3: OSM vector data ──────────────────────────────────────────────
    log.info("Fetching OSM layers …")
    buildings_wgs, water_wgs = fetch_osm_layers(bbox)

    # ── Step 4: Reproject to UTM ─────────────────────────────────────────────
    def to_utm(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        return gdf.to_crs(epsg=UTM_EPSG)

    buildings_utm = to_utm(buildings_wgs)
    water_utm     = to_utm(water_wgs)

    # Apply 61-m RoW buffer to buildings
    if buildings_utm is not None and len(buildings_utm) > 0:
        log.info(f"Buffering buildings by {ROW_BUFFER_M} m …")
        buildings_utm = buildings_utm.copy()
        buildings_utm["geometry"] = buildings_utm.geometry.buffer(ROW_BUFFER_M)

    # ── Step 5: Slope ────────────────────────────────────────────────────────
    log.info("Computing slope ...")
    slope_pct, nodata_mask = compute_slope(dem, RESOLUTION)
    log.info(f"Slope: max={slope_pct.max():.1f}%, "
             f"cells>{SLOPE_MAX_PCT}%: {np.sum(slope_pct > SLOPE_MAX_PCT)}")

    # ── Step 6: Rasterise exclusion zones ────────────────────────────────────
    log.info("Rasterising exclusion zones …")
    building_mask = rasterise_layer(buildings_utm, transform, (rows, cols))
    water_mask    = rasterise_layer(water_utm,     transform, (rows, cols))

    # ── Step 7: Cost surface ─────────────────────────────────────────────────
    log.info("Building cost surface ...")
    cost = build_cost_surface(slope_pct, building_mask, water_mask, nodata_mask)

    # ── Step 8: Convert points to grid indices ───────────────────────────────
    xa, ya = wgs84_to_utm(*POINT_A)
    xb, yb = wgs84_to_utm(*POINT_B)
    start_rc = xy_to_rowcol(xa, ya, transform)
    end_rc   = xy_to_rowcol(xb, yb, transform)

    # Clamp to valid grid range, keeping BORDER_CELLS away from edges
    b = BORDER_CELLS
    start_rc = (
        max(b, min(rows - 1 - b, start_rc[0])),
        max(b, min(cols - 1 - b, start_rc[1])),
    )
    end_rc = (
        max(b, min(rows - 1 - b, end_rc[0])),
        max(b, min(cols - 1 - b, end_rc[1])),
    )
    log.info(f"Grid indices  A={start_rc}  B={end_rc}")

    # Sanity: start/end must not be impassable
    for label, rc in [("A", start_rc), ("B", end_rc)]:
        if cost[rc] >= IMPASSABLE:
            log.warning(
                f"Point {label} at {rc} lands on an impassable cell. "
                f"Relaxing local 3×3 neighbourhood to cost=1."
            )
            r, c = rc
            cost[
                max(0, r-1):min(rows, r+2),
                max(0, c-1):min(cols, c+2),
            ] = 1.0

    # ── Step 9: Pathfinding ──────────────────────────────────────────────────
    log.info("Building cost pyramid and running multi-scale routing (MS-LCP) …")
    from cost_surface import build_cost_pyramid
    from routing import multi_scale_lcp
    from config import PYRAMID_LEVELS, DOWNSAMPLE_RATIO, DOWNSAMPLE_METHOD

    cost_pyramid = build_cost_pyramid(cost, PYRAMID_LEVELS, DOWNSAMPLE_RATIO, DOWNSAMPLE_METHOD)
    path_indices = multi_scale_lcp(
        cost_pyramid, start_rc, end_rc, water_mask, transform,
        resolution_m=RESOLUTION, dem=dem
    )

    # Convert indices → UTM coordinates
    path_utm = [rowcol_to_xy(r, c, transform) for r, c in path_indices]

    # ── Step 10: Smoothing ───────────────────────────────────────────────────
    log.info("Smoothing path (B-spline) …")
    smooth_utm = smooth_path(path_utm)

    # ── Step 11: Curve-radius check ──────────────────────────────────────────
    log.info("Verifying curve radii …")
    min_radius, violations = verify_curve_radius(smooth_utm)

    if violations:
        # Attempt tighter smoothing to reduce sharp bends
        log.info("Re-smoothing with increased smoothing factor …")
        smooth_utm = smooth_path(path_utm, n_points=1000,
                                  smoothing=len(path_utm) * 20.0)
        min_radius, violations = verify_curve_radius(smooth_utm)
        if violations:
            log.warning(
                "Could not fully satisfy 440 m curve-radius constraint. "
                "Outputting best-effort geometry. See warnings above for bottlenecks."
            )

    # Build final LineString
    line_utm = LineString(smooth_utm)

    # ── Step 12: RoW setback post-check ─────────────────────────────────────
    log.info("Verifying RoW setback …")
    verify_row_setback(line_utm, buildings_utm, ROW_BUFFER_M)

    # ── Step 13: Reproject to WGS-84 ────────────────────────────────────────
    log.info("Reprojecting to WGS-84 …")
    wgs84_coords = [utm_to_wgs84(x, y) for x, y in smooth_utm]
    line_wgs84 = LineString(wgs84_coords)

    # ── Step 14: Metadata + Export ───────────────────────────────────────────
    meta = compute_metadata(line_utm, slope_pct, transform)
    log.info(
        f"Route: {meta['total_length_km']} km, "
        f"max slope {meta['max_slope_pct']} %"
    )

    meta["min_curve_radius_achieved_m"] = round(min_radius, 1)
    meta["curve_radius_violations"] = len(violations)

    export_geojson(line_wgs84, meta, OUTPUT_FILE)

    log.info("═" * 65)
    log.info("  COMPLETE – Output: %s", OUTPUT_FILE)
    log.info(f"  Total length  : {meta['total_length_km']} km")
    log.info(f"  Max slope     : {meta['max_slope_pct']} %")
    log.info(f"  Min curve r.  : {meta['min_curve_radius_achieved_m']} m")
    log.info(f"  RoW violations: {meta['curve_radius_violations']}")
    log.info("═" * 65)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("Fatal error in pipeline:")
        traceback.print_exc()
        sys.exit(1)
