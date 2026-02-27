"""
cost_surface.py — Raster cost model for Myanmar highway alignment
=================================================================
Phase 5.1 additions:
  5. Existing Road Discount — encourages route to follow tracks.
  6. LULC Penalty — multiplicative cost for environmental barriers 
     (wetlands, forests) derived from OSM natural/landuse.

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
    ROAD_CLASS_DISCOUNTS, ROW_CORRIDOR_BUFFER_M, ROW_CORRIDOR_BUFFER_DISCOUNT,
    LULC_PENALTIES, LULC_UNMAPPED_BASE, SLOPE_LULC_INTERACT, LULC_EDGE_DECAY_M,
    BUILDING_BASE_PENALTY, BUILDING_AREA_MULT, BUILDING_MAX_PENALTY, ROW_BUFFER_M
)
import rasterio
from rasterio.warp import reproject, Resampling

log = logging.getLogger("highway_alignment")


# ── Phase 12: Earthwork Proxy Cost ───────────────────────────────────────────

def compute_earthwork_proxy(dem, slope_pct, resolution_m, weight=0.30, floodplain_mask=None):
    """
    Estimate earthwork difficulty from local terrain relief (inspired by 3D-CHA*).

    Concept: how much cut/fill would a road at design grade require at each cell?
    Instead of exact cross-sections (too slow for cost surface), we approximate:

      1. Compute local terrain curvature proxy: the difference between the
         cell elevation and the mean of its 5×5 neighbourhood = local relief.
      2. Convert relief to approximate cross-section volume using formation width
         and standard batter slopes.
      3. Normalize to a cost multiplier [1.0 … ~3.0].

    Args:
        dem:             float32 DEM array (UTM, metres)
        slope_pct:       float32 slope-percent array
        resolution_m:    pixel size in metres
        weight:          blending weight (0 = disabled, 0.3 = default, 1.0 = dominant)
        floodplain_mask: optional boolean mask denoting floodplain cells

    Returns:
        float32 multiplier array (values ≥ 1.0), same shape as dem.
    """
    from config import (
        FORMATION_WIDTH_M, SCENARIO_PROFILE,
        CUT_BATTER_HV, FILL_BATTER_HV, GRADE_MAX_PCT,
    )

    rows, cols = dem.shape
    if rows < 5 or cols < 5:
        return np.ones_like(dem, dtype=np.float32)

    # Get profile-specific formation width and max grade
    fw = FORMATION_WIDTH_M[SCENARIO_PROFILE]
    max_g = GRADE_MAX_PCT[SCENARIO_PROFILE] / 100.0  # fraction

    try:
        from config import FLOODPLAIN_MIN_FILL_M
    except ImportError:
        FLOODPLAIN_MIN_FILL_M = 2.5

    # ── Step 1: local elevation relief ────────────────────────────────────
    # Difference between cell and local mean = how much earthwork is needed
    # to level the road through this terrain.
    from scipy.ndimage import uniform_filter
    dem_smooth = uniform_filter(dem.astype(np.float64), size=5, mode='reflect')
    relief = np.abs(dem - dem_smooth).astype(np.float32)  # metres
    
    if floodplain_mask is not None:
        # Increase relief in floodplains to simulate mandatory high embankments
        relief = np.where(floodplain_mask, relief + FLOODPLAIN_MIN_FILL_M, relief)

    # ── Step 2: approximate cross-section volume per metre of road ────────
    # For a relief of h metres:
    #   cut volume  ≈ ½ × h × (fw + h × cut_batter) per metre of road
    #   fill volume ≈ ½ × h × (fw + h × fill_batter) per metre of road
    # Average batter
    avg_batter = (CUT_BATTER_HV + FILL_BATTER_HV) / 2.0
    volume_per_m = 0.5 * relief * (fw + relief * avg_batter)

    # ── Step 3: slope-grade interaction ───────────────────────────────────
    # If terrain slope exceeds max design grade, extra earthwork is needed
    # to bring the grade down to the design limit.
    grade_excess = np.maximum(slope_pct / 100.0 - max_g, 0.0)  # fraction
    # Extra volume from grade reduction: proportional to excess × pixel length
    grade_volume = grade_excess * resolution_m * fw * 0.5
    volume_per_m += grade_volume

    # ── Step 4: normalize to multiplier ───────────────────────────────────
    # Typical Myanmar rural cut: 500–2000 m³/km = 0.5–2.0 m³/m
    # We want volume_per_m of ~2 m³/m to produce a multiplier of ~2.0
    # Scale: 1.0 + weight × volume_per_m / reference_volume
    ref_volume = fw * 0.5  # half-formation width as reference
    multiplier = 1.0 + weight * np.clip(volume_per_m / max(ref_volume, 1.0), 0.0, 5.0)

    n_penalised = int(np.sum(multiplier > 1.1))
    log.info(
        f"Earthwork proxy: {n_penalised:,} cells above 1.1× "
        f"(max={multiplier.max():.2f}×, weight={weight})"
    )
    return multiplier.astype(np.float32)


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

# ── Building Expropriation Penalties ─────────────────────────────────────────

def _apply_building_penalties(buildings_gdf, transform, shape, resolution_m=30):
    """
    Generate an area-based penalty map for buildings with a concentric distance decay.
    """
    from scipy.ndimage import distance_transform_edt
    log.info("Building penalties: starting …")
    penalty_map = np.zeros(shape, dtype=np.float32)
    
    if buildings_gdf is None or len(buildings_gdf) == 0:
        return penalty_map

    buildings = buildings_gdf.copy()
    buildings['area_sqm'] = buildings.geometry.area
    buildings['peak_penalty'] = (
        BUILDING_BASE_PENALTY + buildings['area_sqm'] * BUILDING_AREA_MULT
    ).clip(upper=BUILDING_MAX_PENALTY)

    log.info("Building penalties: peak penalties calculated.")

    # Use `.envelope` to reduce complex OSM polygons to simple bounding boxes.
    # At 30m resolution, this approximation adds minimal error but radically speeds up `rasterize`.
    geoms = buildings.geometry.envelope.tolist()
    chunk_size = 50000
    binary_mask = np.zeros(shape, dtype=np.float32)
    
    log.info(f"Building penalties: rasterizing {len(geoms)} footprints in chunks …")
    for i in range(0, len(geoms), chunk_size):
        chunk = geoms[i : i + chunk_size]
        log.info(f"Building penalties: rasterizing chunk {i}–{i+len(chunk)} …")
        chunk_mask = rasterize(
            chunk,
            out_shape=shape,
            transform=transform,
            fill=0.0,
            default_value=1.0,
            dtype=np.float32,
            all_touched=True
        )
        np.maximum(binary_mask, chunk_mask, out=binary_mask)
        
    log.info("Building penalties: computing base penalty raster …")
    mean_peak_penalty = buildings['peak_penalty'].mean()
    base_penalty_raster = binary_mask * mean_peak_penalty

    empty_mask = (base_penalty_raster == 0)
    
    if not np.any(empty_mask):
         return base_penalty_raster

    if np.all(empty_mask):
         return penalty_map
         
    log.info("Building penalties: computing EDT distance decay …")
    distances_px, indices = distance_transform_edt(empty_mask, return_indices=True)
    distances_m = distances_px * resolution_m
    
    log.info("Building penalties: extracting nearest peak penalties …")
    nearest_peak_penalty = base_penalty_raster[tuple(indices)]
    
    log.info("Building penalties: computing decay weights …")
    weights = np.zeros(shape, dtype=np.float32)
    weights[distances_m == 0] = 1.0
    
    mask1 = (distances_m > 0) & (distances_m <= ROW_BUFFER_M * 0.33)
    weights[mask1] = 0.6
    
    mask2 = (distances_m > ROW_BUFFER_M * 0.33) & (distances_m <= ROW_BUFFER_M * 0.66)
    weights[mask2] = 0.3
    
    mask3 = (distances_m > ROW_BUFFER_M * 0.66) & (distances_m <= ROW_BUFFER_M)
    weights[mask3] = 0.1
    
    penalty_map = nearest_peak_penalty * weights
    
    log.info(f"Building penalties: complete. "
             f"Cells with penalty > 0: {np.sum(penalty_map > 0):,}, "
             f"max penalty: {penalty_map.max():.0f}")
    return penalty_map


# ── Phase 11: WorldCover 10m → LULC penalty raster ──────────────────────────

def worldcover_to_lulc_raster(wc_array, wc_transform, target_shape,
                                target_transform, slope_pct=None):
    """
    Convert ESA WorldCover 10m class raster into a float32 LULC penalty
    multiplier raster aligned to the DEM grid.

    Steps:
      1. Map class values → WORLDCOVER_PENALTIES multipliers (10m resolution)
      2. Resample from 10m to 30m (or whatever the DEM grid is) via bilinear
      3. Apply SLOPE_LULC_INTERACT amplification (if slope_pct provided)
      4. Cells with unmapped/NoData class → LULC_UNMAPPED_BASE

    Args:
        wc_array:         uint8 WorldCover raster (10m, UTM)
        wc_transform:     rasterio affine transform for wc_array
        target_shape:     (rows, cols) of the DEM/cost grid
        target_transform: rasterio affine transform of the DEM/cost grid
        slope_pct:        optional float32 slope raster for interaction

    Returns:
        float32 array, shape == target_shape, values >= 1.0
    """
    from config import (
        WORLDCOVER_PENALTIES, LULC_UNMAPPED_BASE,
        SLOPE_LULC_INTERACT, SLOPE_MODERATE_PCT, UTM_EPSG,
    )

    rows_wc, cols_wc = wc_array.shape
    rows, cols = target_shape

    # Step 1: map class values → penalty multipliers at 10m
    penalty_10m = np.full((rows_wc, cols_wc), LULC_UNMAPPED_BASE, dtype=np.float32)
    for class_val, mult in WORLDCOVER_PENALTIES.items():
        penalty_10m[wc_array == class_val] = mult

    # Step 2: resample 10m → target grid (30m) using bilinear
    dst_crs = rasterio.crs.CRS.from_epsg(UTM_EPSG)
    penalty_resampled = np.full((rows, cols), LULC_UNMAPPED_BASE, dtype=np.float32)

    reproject(
        source=penalty_10m,
        destination=penalty_resampled,
        src_transform=wc_transform,
        src_crs=dst_crs,
        dst_transform=target_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    # Step 3: apply slope × LULC interaction (same as OSM path)
    if slope_pct is not None:
        t = np.clip(slope_pct / SLOPE_MODERATE_PCT, 0.0, 1.0)
        penalty_resampled *= (1.0 + SLOPE_LULC_INTERACT * t)

    # Floor at 1.0 (no discount)
    np.maximum(penalty_resampled, 1.0, out=penalty_resampled)

    n_above = int(np.sum(penalty_resampled > LULC_UNMAPPED_BASE))
    log.info(
        f"WorldCover LULC: {n_above:,} cells above background "
        f"(max penalty: {penalty_resampled.max():.1f}×)"
    )
    return penalty_resampled


# ── Phase 5.4: LULC Penalties (slope interaction + EDT boundary decay) ────────

def _apply_lulc_penalties(lulc_gdf, transform, shape, slope_pct=None):
    """
    Build a float32 LULC cost-multiplier raster with three layers of refinement:

    1. Per-class penalty rasterisation
       Each geometry is paired with its LULC_PENALTIES multiplier (max wins
       on overlaps).  Tag precedence: natural > landuse > leisure > boundary.

    2. Background base multiplier (LULC_UNMAPPED_BASE)
       Cells outside every LULC polygon receive LULC_UNMAPPED_BASE (default
       1.15) instead of 1.0.  Rationale: in Myanmar, unmapped terrain is far
       more likely to be secondary forest or scrub than clear open farmland.
       OSM LULC coverage is severely incomplete; this prevents the router from
       treating unmapped jungle as penalty-free greenfield.

    3. Slope × LULC interaction (SLOPE_LULC_INTERACT)
       Multiplies each cell's LULC value by (1 + SLOPE_LULC_INTERACT * t),
       where t = clamp(slope_pct / SLOPE_MODERATE_PCT, 0, 1).  Steep terrain
       amplifies LULC cost because access roads, drainage, and slope-stability
       risk all grow non-linearly with gradient in vegetated terrain.

    4. EDT boundary soft transition (LULC_EDGE_DECAY_M)
       Hard polygon edges create 1-cell cost cliffs that can cause the router
       to hug LULC boundaries (cheapest cell just outside the penalty zone).
       An EDT-based linear ramp over LULC_EDGE_DECAY_M smoothly blends the
       penalty down to LULC_UNMAPPED_BASE at the transition distance, removing
       boundary-hugging artefacts.

    Args:
        lulc_gdf   : GeoDataFrame with LULC polygons (UTM CRS)
        transform  : rasterio affine transform
        shape      : (rows, cols) tuple
        slope_pct  : float32 slope-percent raster; pass None to skip step 3

    Returns:
        float32 array, shape == shape, values >= LULC_UNMAPPED_BASE
    """
    rows, cols = shape

    # Step 1: Rasterise per-class penalties
    # Start with background = LULC_UNMAPPED_BASE (not 1.0) so unmapped terrain
    # is nudged above the fully-open-land baseline.
    burned = np.full(shape, LULC_UNMAPPED_BASE, dtype=np.float32)

    if lulc_gdf is None or len(lulc_gdf) == 0:
        log.warning(
            "LULC GeoDataFrame is empty — no penalties applied beyond "
            f"LULC_UNMAPPED_BASE={LULC_UNMAPPED_BASE:.2f}. "
            "This is expected if OSM returned zero LULC polygons; see "
            "OSM_LULC_WARN_THRESHOLD in config.py."
        )
        return burned

    shapes = []
    for _, row in lulc_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Tag precedence: natural > landuse > leisure > boundary.
        # Higher specificity tags override more general ones.
        penalty = LULC_UNMAPPED_BASE  # floor
        for tag_key in ("natural", "landuse", "leisure", "boundary"):
            val = row.get(tag_key)
            if val and val in LULC_PENALTIES:
                penalty = max(penalty, LULC_PENALTIES[val])

        if penalty > LULC_UNMAPPED_BASE:
            shapes.append((geom, penalty))

    if shapes:
        chunk_size = 50_000
        for i in range(0, len(shapes), chunk_size):
            chunk = shapes[i : i + chunk_size]
            chunk_burned = rasterize(
                chunk,
                out_shape=shape,
                transform=transform,
                fill=LULC_UNMAPPED_BASE,  # background = base, not 1.0
                dtype=np.float32,
            )
            np.maximum(burned, chunk_burned, out=burned)  # keep highest penalty

    n_penalised = int(np.sum(burned > LULC_UNMAPPED_BASE))
    log.info(
        f"LULC: {n_penalised:,} cells above background ({LULC_UNMAPPED_BASE}x), "
        f"max multiplier={burned.max():.1f}x  [{len(shapes)} polygons rasterised]"
    )

    # Step 2: Slope × LULC interaction
    # lulc_eff = lulc * (1 + SLOPE_LULC_INTERACT * t)  where t in [0,1]
    if slope_pct is not None and SLOPE_LULC_INTERACT > 0:
        t = np.clip(
            slope_pct.astype(np.float32) / max(SLOPE_MODERATE_PCT, 0.01),
            0.0, 1.0
        )
        interaction = 1.0 + SLOPE_LULC_INTERACT * t
        burned = burned * interaction
        max_interact = float(burned.max())
        log.info(
            f"LULC: slope×LULC interaction applied "
            f"(SLOPE_LULC_INTERACT={SLOPE_LULC_INTERACT}); "
            f"new max={max_interact:.1f}x"
        )

    # Step 3: EDT boundary soft transition
    # Cells outside all polygons but within LULC_EDGE_DECAY_M of a polygon
    # receive a linearly interpolated penalty between the polygon value and
    # LULC_UNMAPPED_BASE, eliminating the hard 1-cell cost cliff at borders.
    decay_px = max(1, int(LULC_EDGE_DECAY_M / 30))  # 30 m pixel size
    if decay_px > 0:
        # Binary mask: 1 where any LULC polygon was burned
        poly_mask = (burned > (LULC_UNMAPPED_BASE * 1.001)).astype(np.uint8)
        if poly_mask.any():
            dist_px, nearest_idx = distance_transform_edt(
                ~poly_mask.astype(bool), return_indices=True
            )
            in_decay = (dist_px > 0) & (dist_px <= decay_px)
            if np.any(in_decay):
                # Linearly interpolate: weight=1 at polygon edge, 0 at decay_px
                w = 1.0 - (dist_px[in_decay] / decay_px)
                nearest_val = burned[
                    nearest_idx[0][in_decay],
                    nearest_idx[1][in_decay]
                ]
                # Apply weighted blend toward nearest polygon penalty
                decayed = LULC_UNMAPPED_BASE + w * (nearest_val - LULC_UNMAPPED_BASE)
                # Only raise, never lower (the background is already UNMAPPED_BASE)
                burned[in_decay] = np.maximum(burned[in_decay], decayed.astype(np.float32))
            log.info(
                f"LULC: EDT boundary decay applied over {LULC_EDGE_DECAY_M} m "
                f"({decay_px} px) — {np.sum(in_decay):,} transition cells smoothed."
            )

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

    Performance: uses ndimage.find_objects for O(1) per-component bounding box
    instead of per-component full-array scan. ~500× faster on large grids.
    """
    tiers = WATER_PENALTY_TIERS   # list of 5 multipliers from config
    binary = (water_mask > 0).astype(np.uint8)
    labeled, n_components = nd_label(binary)

    if n_components == 0:
        return np.zeros_like(water_mask, dtype=np.float64)

    # ── Vectorized: get bounding box slices for all components at once ────
    slices = nd_label.__module__  # just to reference scipy.ndimage
    from scipy.ndimage import find_objects
    obj_slices = find_objects(labeled)  # list of (row_slice, col_slice) per component

    # Width breakpoints in metres: <10, <50, <200, <500, ≥500
    breakpoints = np.array([10.0, 50.0, 200.0, 500.0])

    # Build a per-component penalty lookup: comp_penalty[comp_id] = penalty
    # Index 0 is unused (background), indices 1..n_components are components
    comp_penalty = np.zeros(n_components + 1, dtype=np.float64)
    n_tier2_plus = 0

    for comp_id, sl in enumerate(obj_slices, start=1):
        if sl is None:
            continue
        row_sl, col_sl = sl
        height_px = row_sl.stop - row_sl.start
        width_px = col_sl.stop - col_sl.start
        short_axis_m = min(height_px, width_px) * resolution_m

        tier = int(np.searchsorted(breakpoints, short_axis_m))
        comp_penalty[comp_id] = tiers[tier]
        if tier >= 2:
            n_tier2_plus += 1

    # ── Single vectorized assignment: map labeled→penalty in one pass ────
    penalty_map = comp_penalty[labeled]

    log.info(
        f"River hierarchy: {n_components:,} water components classified "
        f"({n_tier2_plus:,} at tier ≥ 2) in one vectorised pass."
    )
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

# ── Phase 5.4: Class-based road discount + ROW corridor buffer ────────────────

def _apply_road_discounts(roads_gdf, roads_mask, cost, transform, shape, resolution_m=30):
    """
    Apply a two-tier road cost discount:

    Tier 1 — Per-class multiplier on road centreline cells
        Each cell in roads_mask is multiplied by the discount for its OSM
        highway= class (from ROAD_CLASS_DISCOUNTS).  Where multiple road
        classes overlap a cell, the best (lowest) multiplier wins.
        Cells at IMPASSABLE (cliff) are never discounted — a misaligned GPS
        trace on a cliff face does not make the cliff passable.

    Tier 2 — ROW corridor buffer on adjacent greenfield cells
        Cells within ROW_CORRIDOR_BUFFER_M of *any* road but not on the
        centreline itself receive ROW_CORRIDOR_BUFFER_DISCOUNT (default 0.88×).
        This reflects that land inside an established right-of-way strip has
        already been surveyed and may be partly cleared / accessible.

        The buffer discount is intentionally conservative (0.88 rather than
        0.80) because OSM centreline positions in Myanmar can drift 20–50 m;
        aggressive discounting of adjacent cells risks rewarding GPS error
        rather than real corridor access.

    Args:
        roads_gdf   : GeoDataFrame with 'highway' column (UTM CRS).  May be
                      None if roads were loaded without retain_tags.
        roads_mask  : float32 raster, > 0 where any road centreline exists
        cost        : float64 cost surface array (modified in-place)
        transform   : rasterio affine transform
        shape       : (rows, cols)
        resolution_m: pixel size in metres (used for buffer distance)
    """
    if roads_mask is None or not np.any(roads_mask > 0):
        return cost

    # --- Tier 1: per-class centreline discount ---
    # Start with the default for all road cells, then override per class.
    class_discount = np.ones(shape, dtype=np.float32)
    on_road = (roads_mask > 0)

    # Default discount for any road cell (catches unmapped class= values)
    class_discount[on_road] = ROAD_CLASS_DISCOUNTS["default"]

    if roads_gdf is not None and len(roads_gdf) > 0 and "highway" in roads_gdf.columns:
        # Build a per-class raster by rasterising each highway value separately.
        # Use the lowest multiplier where polygons overlap.
        for hw_class, multiplier in ROAD_CLASS_DISCOUNTS.items():
            if hw_class == "default":
                continue
            subset = roads_gdf[roads_gdf["highway"] == hw_class]
            if len(subset) == 0:
                continue
            geoms = [g for g in subset.geometry if g is not None and not g.is_empty]
            if not geoms:
                continue
            class_raster = rasterize(
                [(g, multiplier) for g in geoms],
                out_shape=shape,
                transform=transform,
                fill=1.0,   # non-road background = no discount
                dtype=np.float32,
            )
            # np.minimum keeps the strongest discount (lowest multiplier)
            np.minimum(class_discount, class_raster + (~on_road).astype(np.float32),
                       out=class_discount)
        log.info(
            f"Road discounts: per-class multipliers applied to {on_road.sum():,} "
            f"centreline cells  (classes: "
            + ", ".join(
                f"{k}={v}x" for k, v in ROAD_CLASS_DISCOUNTS.items() if k != "default"
            ) + ")"
        )
    else:
        log.info(
            "Road discounts: no 'highway' column in GDF — using default "
            f"discount {ROAD_CLASS_DISCOUNTS['default']}x for all road cells."
        )

    discounted = cost * class_discount
    # Cliffs stay impassable regardless of road presence
    cost = np.where(on_road, np.where(cost >= IMPASSABLE, cost, np.maximum(0.3, discounted)), cost)

    # --- Tier 2: ROW corridor buffer ---
    buffer_px = max(1, int(ROW_CORRIDOR_BUFFER_M / resolution_m))
    if buffer_px > 0:
        y_b, x_b = np.ogrid[-buffer_px:buffer_px + 1, -buffer_px:buffer_px + 1]
        disk_b = (x_b ** 2 + y_b ** 2) <= buffer_px ** 2
        from scipy.ndimage import binary_dilation
        road_dilated = binary_dilation(on_road, structure=disk_b)
        buffer_zone = road_dilated & ~on_road  # adjacent cells only, not centreline
        # Apply partial discount only to greenfield (passable) buffer cells
        passable_buffer = buffer_zone & (cost < IMPASSABLE)
        cost = np.where(
            passable_buffer,
            cost * ROW_CORRIDOR_BUFFER_DISCOUNT,
            cost
        )
        log.info(
            f"Road ROW buffer: {buffer_px} px ({ROW_CORRIDOR_BUFFER_M} m), "
            f"{ROW_CORRIDOR_BUFFER_DISCOUNT}x discount on "
            f"{passable_buffer.sum():,} adjacent greenfield cells."
        )

    return cost


def build_cost_surface(slope_pct, building_penalty_map, water_mask, roads_mask=None,
                       roads_gdf=None, lulc_penalty_map=None, nodata_mask=None,
                       dem=None, curvature=None, resolution_m=30, transform=None):
    """
    Assemble the full cost surface.

    Layers in order (multiplicative unless stated):
      1. Base slope cost (4-zone nonlinear)
      2. Landslide susceptibility multiplier (if curvature available)
      3. Phase 5.4: Class-based Road Discount + ROW corridor buffer
      4. Phase 5.4: LULC Environmental Penalty (incl. background base,
         slope×interaction, EDT boundary decay)
      5. River hierarchy penalty (additive)
      6. Bridge abutment zone recovery
      7. Floodplain amplification (3×)
      8. Building exclusion/expropriation penalty (additive)
      9. NoData / Border exclusion  (IMPASSABLE)

    Args:
        transform: rasterio affine transform — required for per-class road
                   rasterisation (Tier 1 of road discounts).  If None, falls
                   back to default discount for all road cells.
    """
    log.info("Building cost surface …")
    rows, cols = slope_pct.shape

    # ── 1. Slope cost ─────────────────────────────────────────────────────
    cost = _slope_cost_array(slope_pct)

    # ── 1a. Precompute Floodplain Mask ────────────────────────────────────
    floodplain = None
    if dem is not None:
        # compute_floodplain_mask must be defined earlier in the file or imported
        floodplain = compute_floodplain_mask(dem, water_mask, resolution_m)

    # ── 1b. Phase 12: Earthwork proxy (3D-CHA* inspired) ─────────────────
    # Multiplicative layer that penalises cells requiring large cut/fill.
    if dem is not None:
        from config import EARTHWORK_PROXY_WEIGHT
        if EARTHWORK_PROXY_WEIGHT > 0:
            earthwork_mult = compute_earthwork_proxy(
                dem, slope_pct, resolution_m, weight=EARTHWORK_PROXY_WEIGHT, floodplain_mask=floodplain
            )
            cost *= earthwork_mult

    # ── 2. Landslide susceptibility ───────────────────────────────────────
    if curvature is not None:
        landslide_mult = compute_landslide_susceptibility(slope_pct, curvature, resolution_m)
        cost *= landslide_mult

    # ── 3. Phase 5.4: Class-based Road Discount + ROW Corridor Buffer ────
    # Per-class multipliers + conservative 50 m buffer around all road corridors.
    # See _apply_road_discounts() docstring for Myanmar OSM data-quality notes.
    cost = _apply_road_discounts(
        roads_gdf, roads_mask, cost, transform,
        cost.shape, resolution_m
    )

    # ── 4. Phase 5.4: LULC Environmental Penalty ──────────────────────────
    # lulc_penalty_map already includes: background base, slope×interaction,
    # and EDT boundary decay (all applied in _apply_lulc_penalties).
    if lulc_penalty_map is not None:
        cost *= lulc_penalty_map
        log.info(
            f"LULC: multiplied into cost surface — "
            f"{np.sum(lulc_penalty_map > LULC_UNMAPPED_BASE):,} cells above background."
        )

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
    if floodplain is not None:
        # 3× amplification — passable but costly (drainage structures, embankment)
        cost = np.where(floodplain & (cost < IMPASSABLE), cost * 3.0, cost)

    # ── 6. Building exclusion/expropriation ───────────────────────────────
    # Phase 5.2: Buildings use an area-based concentric penalty map instead of a flat penalty
    if building_penalty_map is not None:
        cost = cost + building_penalty_map

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


# ── Phase 4: Multi-Resolution Routing Pyramid (Tang & Dou 2023) ──────────────

def build_cost_pyramid(fine_cost, levels=3, ratio=2, method="average"):
    """
    Generate a progressive multi-resolution cost pyramid (Tang & Dou, 2023).

    Args:
        fine_cost (np.ndarray): The base high-resolution cost surface (Level 0).
        levels (int): The number of downsampled levels to create. 
                      Total pyramid depth = levels + 1.
        ratio (int): The downsampling factor between consecutive levels.
        method (str): 'average' or 'maximum' aggregation method.

    Returns:
        list of np.ndarray: Cost pyramid from highest resolution (Level 0) to 
                            lowest resolution (Level N).
                            pyramid[0] = fine_cost
                            pyramid[-1] = coarsest_cost
    """
    from skimage.measure import block_reduce
    
    func = np.mean if method == "average" else np.max
    pyramid = [fine_cost]
    current_cost = fine_cost
    
    for lvl in range(1, levels + 1):
        # Pad dimensions to be a multiple of ratio before downsampling
        pad_rows = (ratio - (current_cost.shape[0] % ratio)) % ratio
        pad_cols = (ratio - (current_cost.shape[1] % ratio)) % ratio
        
        if pad_rows > 0 or pad_cols > 0:
            current_cost = np.pad(
                current_cost, 
                ((0, pad_rows), (0, pad_cols)), 
                mode='constant', 
                constant_values=IMPASSABLE
            )
            
        downsampled = block_reduce(current_cost, block_size=(ratio, ratio), func=func, cval=IMPASSABLE)
        
        # Ensure extremely high costs (e.g. averaged IMPASSABLE) remain clamped to IMPASSABLE
        downsampled[downsampled >= (IMPASSABLE / (ratio**2))] = IMPASSABLE
        
        pyramid.append(downsampled)
        current_cost = downsampled
        
        valid_cells = downsampled[downsampled < IMPASSABLE]
        max_cost = valid_cells.max() if len(valid_cells) > 0 else 0
        log.info(
            f"Pyramid Level {lvl} (1:{ratio**lvl}): shape={downsampled.shape}, "
            f"max_valid_cost={max_cost:.1f}"
        )
        
    return pyramid

