"""
cost_surface.py — Raster cost model for Myanmar highway alignment
=================================================================
Phase 4 additions:
  5. compute_aspect() — DEM slope direction (0–360°, N=0, clockwise).
  6. compute_sidehill_penalty() — cross-slope instability multiplier
     based on angle between travel bearing (A→B) and slope aspect.
  7. build_cost_surface() now accepts bearing_deg to apply sidehill layer.

Phase 1/2 layers retained:
  1. Nonlinear 4-zone slope cost.
  2. 5-tier river hierarchy penalty.
  3. Floodplain mask (DEM-derived).
  4. Landslide susceptibility proxy (slope + curvature).
"""
import math
import logging
import numpy as np
from scipy.ndimage import (
    binary_closing, binary_dilation,
    label as nd_label, distance_transform_edt,
)
from rasterio.features import rasterize

from config import (
    SLOPE_OPTIMAL_PCT, SLOPE_MODERATE_PCT, SLOPE_MAX_PCT, SLOPE_CLIFF_PCT,
    IMPASSABLE, WATER_PENALTY_TIERS, BORDER_CELLS, DEM_NODATA_SENTINEL,
)

log = logging.getLogger("highway_alignment")


# ── Slope ────────────────────────────────────────────────────────────────────

def compute_slope(dem, resolution_m):
    """
    Returns (slope_pct, nodata_mask, curvature).
    nodata_mask: True where DEM has no valid data (sentinel -9999 or equivalent).
    curvature: second derivative (positive = convex hill, negative = concave valley).
    """
    dem_work = dem.copy().astype(np.float32)
    nodata_mask = (dem_work <= DEM_NODATA_SENTINEL + 1)   # sentinel ± tolerance

    # Fill nodata by nearest-valid-neighbor before gradient computation
    if nodata_mask.any():
        _, idx = distance_transform_edt(nodata_mask, return_indices=True)
        dem_work[nodata_mask] = dem_work[idx[0][nodata_mask], idx[1][nodata_mask]]

    dy, dx = np.gradient(dem_work, resolution_m)
    slope_pct = np.sqrt(dx ** 2 + dy ** 2) * 100.0
    slope_pct[nodata_mask] = 0.0

    # Profile curvature (second derivative along direction of steepest descent)
    d2y, _ = np.gradient(dy, resolution_m)
    _, d2x = np.gradient(dx, resolution_m)
    curvature = d2x + d2y   # Laplacian — positive = convex (hilltop), negative = concave

    return slope_pct.astype(np.float32), nodata_mask, curvature.astype(np.float32)


# ── Slope cost (4-zone nonlinear) ────────────────────────────────────────────

def _slope_cost_array(slope_pct):
    """
    4-zone slope cost function calibrated for Myanmar rural trunk standard.

    Zone 1 (≤ s_opt):   gentle ramp  1.0 → 1.5   (preferred terrain)
    Zone 2 (s_opt–s_mod): exponential 1.5 → 10    (acceptable, increasing earthwork)
    Zone 3 (s_mod–s_max): exponential 10 → 200    (near-limit; switchbacks required)
    Zone 4 (s_max–s_cliff): 200 → near-impassable  (engineering feasible but very expensive)
    Zone 5 (≥ s_cliff):  IMPASSABLE                 (cliff; tunnelling only)

    All thresholds are read from config (scenario-specific).
    """
    s_opt   = SLOPE_OPTIMAL_PCT
    s_mod   = SLOPE_MODERATE_PCT
    s_max   = SLOPE_MAX_PCT
    s_cliff = SLOPE_CLIFF_PCT

    cost = np.ones_like(slope_pct, dtype=np.float64)

    # Zone 1 — gentle
    m1 = slope_pct <= s_opt
    cost[m1] = 1.0 + (slope_pct[m1] / max(s_opt, 0.01)) * 0.5

    # Zone 2 — moderate (exponential, 1.5 → 10)
    m2 = (slope_pct > s_opt) & (slope_pct <= s_mod)
    t2 = (slope_pct[m2] - s_opt) / max(s_mod - s_opt, 0.01)
    cost[m2] = 1.5 * np.exp(t2 * math.log(10 / 1.5))

    # Zone 3 — steep (exponential, 10 → 200; switchback zone)
    m3 = (slope_pct > s_mod) & (slope_pct <= s_max)
    t3 = (slope_pct[m3] - s_mod) / max(s_max - s_mod, 0.01)
    cost[m3] = 10.0 * np.exp(t3 * math.log(20.0))

    # Zone 4 — near-cliff (200 → ~4000; tunnelling/major cut required)
    m4 = (slope_pct > s_max) & (slope_pct < s_cliff)
    t4 = (slope_pct[m4] - s_max) / max(s_cliff - s_max, 0.01)
    cost[m4] = 200.0 * np.exp(t4 * math.log(20.0))

    # Zone 5 — cliff
    cost[slope_pct >= s_cliff] = IMPASSABLE

    return cost


# ── Rasterise vector layer ───────────────────────────────────────────────────

def rasterise_layer(gdf_utm, transform, shape, value=1.0):
    if gdf_utm is None or len(gdf_utm) == 0:
        return np.zeros(shape, dtype=np.float32)
    geoms = [g for g in gdf_utm.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=np.float32)
        
    burned = np.zeros(shape, dtype=np.float32)
    chunk_size = 50000
    for i in range(0, len(geoms), chunk_size):
        chunk = geoms[i : i + chunk_size]
        chunk_burned = rasterize(
            [(g, value) for g in chunk],
            out_shape=shape,
            transform=transform,
            fill=0.0,
            dtype=np.float32,
        )
        np.maximum(burned, chunk_burned, out=burned)
        
    return burned


# ── River hierarchy ──────────────────────────────────────────────────────────

def _river_hierarchy_penalties(water_mask, resolution_m):
    """
    Classify connected water-body components by approximate width and assign
    a tier-based crossing penalty.

    Tiers (approximate width → penalty multiplier):
      0: < 10 m  → culvert/ford          ×5
      1: 10–50 m → small bridge          ×50
      2: 50–200 m → medium bridge        ×500
      3: 200–500 m → major bridge        ×5 000
      4: > 500 m → Ayeyarwady-scale      ×50 000
    """
    tiers = WATER_PENALTY_TIERS   # list of 5 multipliers from config
    binary = (water_mask > 0).astype(np.uint8)
    labeled, n_components = nd_label(binary)

    if n_components == 0:
        return np.zeros_like(water_mask, dtype=np.float64)

    penalty_map = np.zeros_like(water_mask, dtype=np.float64)

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        pixel_count = comp_mask.sum()
        # Approximate "short axis" width using pixel count and bounding-box aspect
        rows_idx = np.any(comp_mask, axis=1)
        cols_idx = np.any(comp_mask, axis=0)
        height_px = rows_idx.sum()
        width_px  = cols_idx.sum()
        # Short axis ≈ min dimension as a rough width proxy
        short_axis_m = min(height_px, width_px) * resolution_m

        if   short_axis_m < 10:    tier = 0
        elif short_axis_m < 50:    tier = 1
        elif short_axis_m < 200:   tier = 2
        elif short_axis_m < 500:   tier = 3
        else:                      tier = 4

        penalty_map[comp_mask] = tiers[tier]
        if tier >= 2:
            log.info(f"  River component {comp_id}: ~{short_axis_m:.0f} m wide → tier {tier} (×{tiers[tier]:,})")

    return penalty_map


# ── Floodplain mask (DEM-derived) ────────────────────────────────────────────

def compute_floodplain_mask(dem, water_mask, resolution_m,
                             buffer_m=3_000, elev_buffer_m=5.0):
    """
    Identify cells likely to be seasonally inundated.

    A cell is flagged as floodplain if:
      • Within buffer_m horizontal distance of a permanent water body (OSM / DEM-derived), AND
      • Its elevation is within elev_buffer_m metres above the nearest water surface.

    This approximates the annual flood pool without requiring a hydrodynamic model.
    Myanmar rivers commonly flood 3–15 km laterally in the Ayeyarwady valley.
    """
    if not np.any(water_mask > 0):
        return np.zeros(dem.shape, dtype=bool)

    buffer_px = max(1, int(buffer_m / resolution_m))
    water_binary = (water_mask > 0)

    # Distance from nearest permanent water
    water_dist_px, (nearest_r, nearest_c) = distance_transform_edt(
        ~water_binary, return_indices=True
    )
    within_buffer = (water_dist_px * resolution_m) < buffer_m

    # Elevation above nearest water surface
    rows, cols = dem.shape
    valid_r = np.clip(nearest_r, 0, rows - 1)
    valid_c = np.clip(nearest_c, 0, cols - 1)
    water_surface_elev = dem[valid_r, valid_c]

    low_lying = (dem - water_surface_elev) <= elev_buffer_m

    floodplain = within_buffer & low_lying & ~water_binary  # exclude the river itself
    log.info(f"Floodplain mask: {floodplain.sum():,} cells flagged "
             f"({floodplain.sum() * resolution_m**2 / 1e6:.1f} km²)")
    return floodplain


# ── Landslide susceptibility proxy ──────────────────────────────────────────

def compute_landslide_susceptibility(slope_pct, curvature, resolution_m):
    """
    DEM-derived landslide susceptibility multiplier.

    High susceptibility when:
      • Slope > 25% (approx 14°) — threshold for shallow rotational failures
      • Curvature > 0 (convex hillslope — material free to move downslope)
      • Combined indicator: amplified cost in high-risk zones

    Returns a multiplier array (1.0 = no amplification, up to 5.0 for severe zones).
    This does NOT replace formal geotechnical assessment; it biases routing away
    from historically active landslide terrain using only DEM proxies.
    """
    multiplier = np.ones_like(slope_pct, dtype=np.float64)

    # Mild susceptibility: slope 20–30%
    m_mild = (slope_pct >= 20) & (slope_pct < 30)
    multiplier[m_mild] = 1.5

    # Moderate: slope >25% with convex curvature (classic failure geometry)
    m_mod = (slope_pct >= 25) & (curvature > 0)
    t_mod = np.clip((slope_pct[m_mod] - 25) / 15.0, 0, 1)
    multiplier[m_mod] = 1.5 + t_mod * 2.5   # 1.5 → 4.0

    # Severe: slope >35% (near-cliff)
    m_sev = slope_pct >= 35
    multiplier[m_sev] = 5.0

    n_flagged = np.sum(multiplier > 1.0)
    if n_flagged > 0:
        log.info(f"Landslide proxy: {n_flagged:,} cells with susceptibility > 1× "
                 f"(max multiplier applied: {multiplier[multiplier > 1.0].max():.1f}×)")
    return multiplier

# ── Phase 4: Aspect and sidehill penalty ──────────────────────────────────────

def compute_aspect(dem, resolution_m):
    """
    Compute terrain aspect (direction of steepest descent) in degrees.
    Convention: 0° = North, 90° = East, 180° = South, 270° = West (clockwise).
    Returns array in range [0, 360).
    """
    dy, dx = np.gradient(dem.astype(np.float64), resolution_m)
    # atan2(dx, dy) gives bearing of steepest descent in geographic convention
    aspect_rad = np.arctan2(dx, dy)   # note: dx=east, dy=north
    aspect_deg = np.degrees(aspect_rad) % 360.0
    return aspect_deg.astype(np.float32)


def compute_sidehill_penalty(slope_pct, aspect_deg, bearing_deg):
    """
    Anisotropic sidehill instability proxy.

    A highway at bearing `bearing_deg` (A→B azimuth) traversing a hillside at
    90° to the slope direction creates:
      • Higher cut/fill volumes than contouring or ridge-following
      • Lateral drainage pressure on pavement
      • Instability at the cut face

    Cross-slope angle = deviation of travel direction from the slope’s
    downhill direction, clamped to [0, 90°].

    Multiplier schedule (only applied where slope > SLOPE_MODERATE_PCT):
      0–30° cross-slope  → 1.0×  (following slope direction, i.e. ridge/valley)
      30–60°             → 1.0–1.5× (partial sidehill)
      60–90°             → 1.5–2.5× (full sidehill traverse — most expensive)

    Note: this is a global approximation using the main A→B bearing.
    A fully anisotropic implementation would require a directed graph.
    """
    # Angular difference between travel bearing and slope aspect,
    # normalised to [0, 90°] (symmetric: going up or down slope equally preferred
    # over sidehill)
    diff = np.abs(aspect_deg - bearing_deg) % 360.0
    # Fold: 0–180 (uphill/downhill range), then mirror to 0–90
    diff = np.where(diff > 180.0, 360.0 - diff, diff)   # 0–180
    cross_slope_deg = np.where(diff > 90.0, 180.0 - diff, diff)  # 0–90

    # Only penalise cells where slope is significant
    significant = slope_pct >= SLOPE_MODERATE_PCT

    # Multiplier linearly interpolated from angle
    # 0° → 1.0,  90° → 2.5
    raw_mult = 1.0 + (cross_slope_deg / 90.0) * 1.5
    # Extra amplification only when BOTH steep AND significantly sidehill (>30°)
    # This prevents penalising following-slope traverses on steep terrain.
    very_steep_sidehill = (slope_pct >= SLOPE_MAX_PCT) & (cross_slope_deg > 30.0)
    raw_mult = np.where(very_steep_sidehill, raw_mult * 1.5, raw_mult)   # capped ≤ 3.75×

    multiplier = np.where(significant, raw_mult, 1.0)

    flagged = np.sum(multiplier > 1.2)
    if flagged > 0:
        log.info(
            f"Sidehill penalty: {flagged:,} cells with multiplier >1.2× "
            f"(bearing={bearing_deg:.1f}°, moderate slope threshold={SLOPE_MODERATE_PCT}%)"
        )
    return multiplier.astype(np.float64)




def build_cost_surface(slope_pct, building_mask, water_mask, nodata_mask=None,
                       dem=None, curvature=None, resolution_m=30, bearing_deg=None):
    """
    Assemble the full cost surface.

    Layers in order (multiplicative unless stated):
      1. Base slope cost (4-zone nonlinear)
      2. Landslide susceptibility multiplier (if curvature available)
      3. Sidehill instability multiplier (Phase 4 — if bearing_deg provided)
      4. River hierarchy penalty (additive)
      5. Bridge abutment zone recovery
      6. Floodplain amplification (3×)
      7. Building exclusion (IMPASSABLE)
      8. NoData exclusion  (IMPASSABLE)
      9. Border exclusion  (IMPASSABLE)
    """
    log.info("Building cost surface …")
    rows, cols = slope_pct.shape

    # ── 1. Slope cost ─────────────────────────────────────────────────────
    cost = _slope_cost_array(slope_pct)

    # ── 2. Landslide susceptibility ───────────────────────────────────────
    if curvature is not None:
        landslide_mult = compute_landslide_susceptibility(slope_pct, curvature, resolution_m)
        cost *= landslide_mult

    # ── 3. Sidehill instability (Phase 4) ─────────────────────────────────
    # Penalises cells where the travel direction (A→B bearing) cuts across the
    # slope at 60–90°.  Only applied when dem available and bearing is known.
    if bearing_deg is not None and dem is not None:
        aspect = compute_aspect(dem, resolution_m)
        sidehill_mult = compute_sidehill_penalty(slope_pct, aspect, bearing_deg)
        cost = np.where(cost < IMPASSABLE, cost * sidehill_mult, cost)

    # ── 4. River hierarchy penalty ────────────────────────────────────────
    if np.any(water_mask > 0):
        # Morphologically close small gaps in water mask (bridge one-pixel breaks)
        radius = 4
        y_g, x_g = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = x_g ** 2 + y_g ** 2 <= radius ** 2
        water_closed = binary_closing(water_mask > 0, structure=disk)

        hierarchy_penalty = _river_hierarchy_penalties(
            water_closed.astype(np.float32), resolution_m
        )
        # Add penalty on top of existing cost (not replace) so slope cost still matters
        cost = np.where(water_closed, cost + hierarchy_penalty, cost)

        # ── 4. Bridge abutment zone recovery ─────────────────────────────
        # Relax steep-bank cells immediately adjacent to water to encourage
        # approach routes that can realistically reach the bank.
        radius_abut = 3
        y_a, x_a = np.ogrid[-radius_abut:radius_abut + 1, -radius_abut:radius_abut + 1]
        disk_abut = x_a ** 2 + y_a ** 2 <= radius_abut ** 2
        water_dilated = binary_dilation(water_closed, structure=disk_abut)
        abutment_zone = water_dilated & ~water_closed

        # Only recover cells that are steep (would otherwise be IMPASSABLE)
        mask_steep_abutment = abutment_zone & (cost >= IMPASSABLE)
        # Set to a high-but-passable cost representing difficult embankment work
        cost = np.where(mask_steep_abutment, 300.0, cost)
        log.info(f"Bridge abutment zones: {np.sum(mask_steep_abutment):,} steep bank cells recovered.")

    # ── 5. Floodplain amplification ───────────────────────────────────────
    if dem is not None:
        floodplain = compute_floodplain_mask(dem, water_mask, resolution_m)
        # 3× amplification — passable but costly (drainage structures, embankment)
        cost = np.where(floodplain & (cost < IMPASSABLE), cost * 3.0, cost)

    # ── 6. Building exclusion ─────────────────────────────────────────────
    cost = np.where(building_mask > 0, IMPASSABLE, cost)

    # ── 7. NoData exclusion ───────────────────────────────────────────────
    if nodata_mask is not None:
        cost[nodata_mask] = IMPASSABLE

    # ── 8. Border exclusion ───────────────────────────────────────────────
    b = max(1, BORDER_CELLS)
    cost[:b,  :] = IMPASSABLE
    cost[-b:, :] = IMPASSABLE
    cost[:,  :b] = IMPASSABLE
    cost[:, -b:] = IMPASSABLE

    valid = cost[cost < IMPASSABLE]
    log.info(
        f"Cost surface built: shape={cost.shape}, "
        f"valid cells={len(valid):,}, "
        f"min={valid.min():.2f}, max={valid.max():.2f}, "
        f"impassable={np.sum(cost >= IMPASSABLE):,}"
    )
    return cost.astype(np.float64)
