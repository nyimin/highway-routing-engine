"""
main.py — Highway alignment pipeline orchestrator
==================================================
Changes vs original:
  - Passes dem + curvature to build_cost_surface (floodplain + landslide layers).
  - Passes dem to routing (bank stability in bridge siting).
  - Passes slope_pct + transform to smooth_path (segment-aware smoothing).
  - Collects all log warnings and adds them to output metadata.
  - Runs sustained grade check and clothoid transition reports.
  - Phase 3: DEM-derived stream fallback when OSM water is sparse.
  - Phase 3: CheckpointManager — saves stage state to JSON so crashed runs restart
    from the last completed stage, not from scratch.
"""
import json
import logging
import math
import os
import time
import traceback
import tracemalloc
import sys
import numpy as np
from shapely.geometry import LineString

from config import (
    WAYPOINTS, UTM_EPSG, ROW_BUFFER_M, SLOPE_MAX_PCT,
    BORDER_CELLS, IMPASSABLE, RESOLUTION, OUTPUT_FILE,
    CHECKPOINT_FILE, FORCE_RESTART,
    MEMORY_WARN_GB, TILE_ROUTING_THRESHOLD_KM, PERF_TIMING_ENABLED,
    ENABLE_TILE_ROUTING, TILE_LENGTH_KM, TILE_OVERLAP_KM, TILE_LATERAL_MARGIN_DEG,
    COARSE_FACTOR, CORRIDOR_BAND_KM,
    EXPORT_INTERMEDIATES, GENERATE_VISUALIZATIONS,
    SLOPE_OPTIMAL_PCT, SLOPE_MODERATE_PCT, SLOPE_CLIFF_PCT,
    ROAD_CLASS_DISCOUNTS,
    # Phase 6
    EXPORT_3D_GEOJSON, OUTPUT_FILE_3D,
    VC_K_VALUES, GRADE_MAX_PCT, MIN_VC_LENGTH_M, MIN_VPI_SPACING_M,
    SCENARIO_PROFILE, DESIGN_SPEED_KMPH,
    # Phase 7
    FORMATION_WIDTH_M, CUT_BATTER_HV, FILL_BATTER_HV, SWELL_FACTOR,
    OUTPUT_EARTHWORK_CSV,
    # Phase 8
    BRIDGE_FREEBOARD_M, BRIDGE_COST_PER_M2_USD, BRIDGE_DECK_WIDTH_M,
    OUTPUT_STRUCTURES_CSV,
    # Phase 9
    EARTHWORK_CUT_RATE_USD_M3, EARTHWORK_FILL_RATE_USD_M3, PAVEMENT_RATE_USD_M2,
    CORRIDOR_WIDTH_M, LAND_ACQ_DEFAULT_USD_PER_HA, LAND_ACQ_RATES,
    ENV_MITIGATION_FACTOR, CONTINGENCY_FACTOR, ENGINEERING_FACTOR,
    OUTPUT_COST_CSV,
    # Phase 10
    OUTPUT_REPORT_HTML, OUTPUT_REPORT_PDF,
    # Phase 11
    USE_WORLDCOVER_LULC, USE_OVERTURE_BUILDINGS, OVERTURE_DEDUP_RADIUS_M,
    # Phase 14.5
    BUILDING_BASE_PENALTY, BUILDING_AREA_MULT, ROW_BUFFER_M as ROW_BUFFER_M_CFG,
    ENABLE_RASTER_CACHE,
)
from geometry_utils import (
    bbox_with_margin, wgs84_to_utm, utm_to_wgs84,
    xy_to_rowcol, rowcol_to_xy,
    smooth_path, verify_curve_radius, verify_row_setback, compute_metadata,
    verify_design_lengths,
    export_geojson, export_geojson_3d,
    extract_longitudinal_profile, check_sustained_grade,
    compute_bearing, compute_clothoid_transitions,
)
from data_fetch import (
    fetch_dem, fetch_osm_layers, derive_stream_mask_utm,
    fetch_worldcover, fetch_overture_buildings, merge_building_sources,
    fetch_custom_water, fetch_dam_lake,
    RasterCache, _cache_fingerprint,  # Phase 14.5
)
from cost_surface import (
    compute_slope, rasterise_layer, build_cost_surface,
    _slope_cost_array, _river_hierarchy_penalties,
    worldcover_to_lulc_raster, build_cost_pyramid,
)
from routing import multi_scale_lcp
from config import PYRAMID_LEVELS, DOWNSAMPLE_RATIO, DOWNSAMPLE_METHOD
from structures import filter_bridge_worthy_water

log = logging.getLogger("highway_alignment")


# ── Warning collector ─────────────────────────────────────────────────────────

class _WarningCapture(logging.Handler):
    """Captures warning-level log messages so they can be stored in metadata."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.warnings = []

    def emit(self, record):
        self.warnings.append(self.format(record))


# ── Phase 5: Stage timer ───────────────────────────────────────────────────────

class _StageTimer:
    """Lightweight wall-clock timer for pipeline stages."""

    def __init__(self, enabled=True):
        self._enabled = enabled
        self._stages  = {}       # stage_name -> elapsed_seconds
        self._t0      = None
        self._current = None

    def start(self, stage_name):
        if not self._enabled:
            return
        self._current = stage_name
        self._t0      = time.monotonic()

    def stop(self):
        if not self._enabled or self._t0 is None:
            return
        elapsed = time.monotonic() - self._t0
        self._stages[self._current] = round(elapsed, 2)
        self._t0 = None

    def as_dict(self):
        return dict(self._stages)

    def log_summary(self):
        if not self._enabled or not self._stages:
            return
        total = sum(self._stages.values())
        log.info("Stage timing:")
        for name, secs in self._stages.items():
            log.info(f"  {name:<30s} {secs:6.1f}s")
        log.info(f"  {'TOTAL':<30s} {total:6.1f}s")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = _WarningCapture()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger("highway_alignment").addHandler(handler)
    return handler


# ── Stage checkpoint ────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Persists pipeline stage results to a JSON file so that if the process
    crashes (e.g. OOM mid-routing), a re-run can skip completed stages.

    Usage:
        ckpt = CheckpointManager()                 # loads existing checkpoint
        dem, tf, src = ckpt.get('dem') or (...)    # returns None if not cached
        ckpt.save('dem', {'source': src})           # mark stage complete

    The checkpoint stores lightweight metadata only (stage names, config
    hashes, cached file paths). Large arrays like DEM are re-read from disk.
    """

    def __init__(self, path=CHECKPOINT_FILE, force_restart=FORCE_RESTART):
        self.path = path
        self._data = {}
        if not force_restart and os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    self._data = json.load(f)
                log.info(f"Checkpoint loaded: {list(self._data.keys())}")
            except Exception as exc:
                log.warning(f"Could not load checkpoint ({exc}). Starting fresh.")
                self._data = {}
        elif force_restart and os.path.exists(path):
            os.remove(path)
            log.info("FORCE_RESTART: existing checkpoint deleted.")

    def get(self, stage):
        """Return saved stage data or None if not yet completed."""
        return self._data.get(stage)

    def save(self, stage, data):
        """Mark a stage as complete and persist."""
        self._data[stage] = data
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            log.warning(f"Checkpoint write failed ({exc}).")

    def clear(self):
        """Remove the checkpoint file (called on successful pipeline completion)."""
        if os.path.exists(self.path):
            os.remove(self.path)
            log.info("Checkpoint cleared (pipeline completed successfully).")


def run_leg_monolithic(pt_a, pt_b, bbox, timer, ckpt, warn_handler):
    """
    Rounds out Phase 16 refactoring by isolating the monolithic pipeline 
    so it can be easily called iteratively per leg.
    Returns path_utm.
    """
    from data_fetch import (
        fetch_dem, fetch_osm_layers, fetch_custom_water, fetch_dam_lake,
        fetch_overture_buildings, merge_building_sources, RasterCache,
        derive_stream_mask_utm, _cache_fingerprint,
    )
    from cost_surface import (
        compute_slope, rasterise_layer, build_cost_surface,
        worldcover_to_lulc_raster, build_cost_pyramid,
    )
    from routing import multi_scale_lcp
    from structures import filter_bridge_worthy_water
    from config import USE_CUSTOM_WATER, USE_DAM_LAKE_AVOIDANCE, DAM_LAKE_BUFFER_M

    # ── 2. DEM
    dem, transform, dem_source = fetch_dem(bbox)
    rows, cols = dem.shape

    # ── 3. OSM layers
    buildings_wgs, water_wgs, roads_wgs, lulc_wgs, osm_stats = fetch_osm_layers(bbox)

    def to_utm(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        return gdf.to_crs(epsg=UTM_EPSG)

    buildings_utm = to_utm(buildings_wgs)
    water_utm     = to_utm(water_wgs)
    roads_utm     = to_utm(roads_wgs)
    lulc_utm      = to_utm(lulc_wgs)

    if USE_CUSTOM_WATER:
        custom_water_wgs = fetch_custom_water(bbox)
        if custom_water_wgs is not None and len(custom_water_wgs) > 0:
            custom_water_utm = to_utm(custom_water_wgs)
            import pandas as pd
            if "natural" not in custom_water_utm.columns:
                custom_water_utm["natural"] = "water"
            else:
                custom_water_utm["natural"] = custom_water_utm["natural"].fillna("water")
            if water_utm is None or len(water_utm) == 0:
                water_utm = custom_water_utm
            else:
                water_utm = pd.concat([water_utm, custom_water_utm], ignore_index=True)

    exclusion_gdf_utm = None
    if USE_DAM_LAKE_AVOIDANCE:
        dam_lake_wgs = fetch_dam_lake(bbox)
        if dam_lake_wgs is not None and len(dam_lake_wgs) > 0:
            dam_lake_utm = to_utm(dam_lake_wgs)
            if DAM_LAKE_BUFFER_M > 0:
                dam_lake_utm["geometry"] = dam_lake_utm.geometry.buffer(DAM_LAKE_BUFFER_M)
            exclusion_gdf_utm = dam_lake_utm

    if water_utm is not None and len(water_utm) > 0:
        water_utm = filter_bridge_worthy_water(water_utm)

    if USE_OVERTURE_BUILDINGS:
        overture_wgs = fetch_overture_buildings(bbox)
        if overture_wgs is not None and len(overture_wgs) > 0:
            buildings_utm = merge_building_sources(
                buildings_utm, overture_wgs,
                dedup_radius_m=OVERTURE_DEDUP_RADIUS_M,
            )

    slope_pct, nodata_mask, curvature = compute_slope(dem, RESOLUTION)

    rcache = RasterCache()
    building_penalty_map = np.zeros((rows, cols), dtype=np.float32)
    if buildings_utm is not None and len(buildings_utm) > 0:
        from config import ROW_BUFFER_M as ROW_BUFFER_M_CFG
        bldg_fp = _cache_fingerprint(
            len(buildings_utm), (rows, cols),
            BUILDING_BASE_PENALTY, BUILDING_AREA_MULT, ROW_BUFFER_M_CFG, RESOLUTION,
        )
        cached_bldg, bldg_hit = rcache.get("building_penalty", bldg_fp, shape=(rows, cols))
        if bldg_hit:
            building_penalty_map = cached_bldg.astype(np.float32)
        else:
            from cost_surface import _apply_building_penalties
            building_penalty_map = _apply_building_penalties(buildings_utm, transform, (rows, cols), RESOLUTION)
            rcache.put("building_penalty", bldg_fp, building_penalty_map, transform)

    water_fp = _cache_fingerprint(len(water_utm) if water_utm is not None else 0, (rows, cols))
    cached_water, water_hit = rcache.get("water_mask", water_fp, shape=(rows, cols))
    if water_hit:
        water_mask = cached_water.astype(np.float32)
    else:
        water_mask = rasterise_layer(water_utm, transform, (rows, cols))
        rcache.put("water_mask", water_fp, water_mask, transform)

    roads_fp = _cache_fingerprint(len(roads_utm) if roads_utm is not None else 0, (rows, cols))
    cached_roads, roads_hit = rcache.get("roads_mask", roads_fp, shape=(rows, cols))
    if roads_hit:
        roads_mask = cached_roads.astype(np.float32)
    else:
        roads_mask = rasterise_layer(roads_utm, transform, (rows, cols))
        rcache.put("roads_mask", roads_fp, roads_mask, transform)

    from cost_surface import _apply_lulc_penalties
    lulc_penalty_map = np.full((rows, cols), 1.0, dtype=np.float32)
    if USE_WORLDCOVER_LULC:
        wc_array, wc_transform = fetch_worldcover(bbox)
        if wc_array is not None:
            lulc_penalty_map = worldcover_to_lulc_raster(
                wc_array, wc_transform,
                target_shape=(rows, cols),
                target_transform=transform,
                slope_pct=slope_pct,
            )
    if lulc_utm is not None and len(lulc_utm) > 0:
        osm_penalty_map = _apply_lulc_penalties(lulc_utm, transform, (rows, cols), slope_pct=slope_pct)
        np.maximum(lulc_penalty_map, osm_penalty_map, out=lulc_penalty_map)

    if osm_stats.get('dem_stream_fallback'):
        dem_streams = derive_stream_mask_utm(dem, transform, resolution_m=RESOLUTION)
        water_mask = np.maximum(water_mask, dem_streams).astype(np.float32)
        from rasterio.features import shapes
        from shapely.geometry import shape
        import geopandas as gpd
        vectors = []
        for geom, val in shapes((dem_streams > 0).astype(np.int32), mask=(dem_streams > 0), transform=transform):
            poly = shape(geom)
            vectors.append(poly)
        if vectors:
            dem_water_gdf = gpd.GeoDataFrame({"waterway": ["dem_stream"] * len(vectors)}, geometry=vectors, crs=f"EPSG:{UTM_EPSG}")
            if water_utm is None or len(water_utm) == 0:
                water_utm = dem_water_gdf
            else:
                import pandas as pd
                water_utm = pd.concat([water_utm, dem_water_gdf], ignore_index=True)

    cost = build_cost_surface(
        slope_pct, building_penalty_map, water_mask,
        roads_mask=roads_mask, roads_gdf=roads_utm,
        lulc_penalty_map=lulc_penalty_map,
        nodata_mask=nodata_mask, dem=dem, curvature=curvature,
        resolution_m=RESOLUTION, transform=transform,
        exclusion_gdf=exclusion_gdf_utm
    )

    xa, ya = wgs84_to_utm(*pt_a)
    xb, yb = wgs84_to_utm(*pt_b)
    start_rc = xy_to_rowcol(xa, ya, transform)
    end_rc   = xy_to_rowcol(xb, yb, transform)

    b = BORDER_CELLS
    start_rc = (max(b, min(rows - 1 - b, start_rc[0])), max(b, min(cols - 1 - b, start_rc[1])))
    end_rc = (max(b, min(rows - 1 - b, end_rc[0])), max(b, min(cols - 1 - b, end_rc[1])))

    for label, rc in [("A", start_rc), ("B", end_rc)]:
        if cost[rc] >= IMPASSABLE:
            r, c = rc
            r0, r1 = max(0, r - 1), min(rows, r + 2)
            c0, c1 = max(0, c - 1), min(cols, c + 2)
            neighbourhood = cost[r0:r1, c0:c1].copy()
            valid_costs = neighbourhood[neighbourhood < IMPASSABLE]
            floor_cost = float(np.percentile(valid_costs, 5)) if len(valid_costs) > 0 else 1.0
            floor_cost = max(floor_cost, 1.0)
            cost[r0:r1, c0:c1] = np.minimum(cost[r0:r1, c0:c1], floor_cost)

    cost_pyramid = build_cost_pyramid(cost, PYRAMID_LEVELS, DOWNSAMPLE_RATIO, DOWNSAMPLE_METHOD)
    path_indices = multi_scale_lcp(
        cost_pyramid, start_rc, end_rc, water_mask, transform,
        resolution_m=RESOLUTION, dem=dem
    )

    if not path_indices:
        return []

    return [rowcol_to_xy(r, c, transform) for r, c in path_indices]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    warn_handler = _setup_logging()
    timer = _StageTimer(enabled=PERF_TIMING_ENABLED)
    tracemalloc.start()
    ckpt  = CheckpointManager()

    log.info("═" * 65)
    log.info("  Highway Alignment Generator  –  Myanmar Preliminary Feasibility")
    log.info("═" * 65)
    for i, wp in enumerate(WAYPOINTS):
        log.info(f"Waypoint {i+1} : lon={wp[0]}, lat={wp[1]}")
    log.info(f"UTM CRS : EPSG:{UTM_EPSG}")

    # ── 1. Bounding box ────────────────────────────────────────────────────
    # ── 1. Bounding box ────────────────────────────────────────────────────
    timer.start("1_bbox")
    bbox = bbox_with_margin(WAYPOINTS)
    west, south, east, north = bbox
    log.info(f"BBox (WGS-84): W={west:.4f} S={south:.4f} E={east:.4f} N={north:.4f}")
    timer.stop()

    # ── Phase 16: Multi-Waypoint Routing Dispatch ──────────────────────────
    full_path_utm = []
    segment_indices = []

    for leg_idx in range(len(WAYPOINTS) - 1):
        pt_a = WAYPOINTS[leg_idx]
        pt_b = WAYPOINTS[leg_idx + 1]
        log.info(f"\n{'═'*20} ROUTING LEG {leg_idx + 1}/{len(WAYPOINTS)-1} {'═'*20}")
        log.info(f"  From: (lon={pt_a[0]:.4f}, lat={pt_a[1]:.4f})")
        log.info(f"  To:   (lon={pt_b[0]:.4f}, lat={pt_b[1]:.4f})")

        ax, ay = wgs84_to_utm(*pt_a)
        bx, by = wgs84_to_utm(*pt_b)
        leg_km = ((bx - ax)**2 + (by - ay)**2)**0.5 / 1000.0
        log.info(f"  Leg {leg_idx + 1} straight-line: {leg_km:.1f} km")

        use_tile_routing = (
            leg_km > TILE_ROUTING_THRESHOLD_KM
            and ENABLE_TILE_ROUTING
            and not FAST_MODE
        )

        leg_path_utm = []

        if use_tile_routing:
            log.info(f"  Activating TILE ROUTING for Leg {leg_idx + 1}")
            from tile_routing import TilePartitioner, run_tiled_pipeline
            timer.start(f"leg_{leg_idx+1}_tile_routing")
            tiles = TilePartitioner(
                [pt_a, pt_b],
                tile_length_km=TILE_LENGTH_KM,
                overlap_km=TILE_OVERLAP_KM,
                margin_deg=TILE_LATERAL_MARGIN_DEG,
            ).partition()
            
            # Reset global_tile_idx override back to 0 for just this leg
            for i, t in enumerate(tiles):
                tiles[i] = t._replace(tile_index=i)

            leg_path_utm = run_tiled_pipeline(tiles, timer=timer, ckpt=ckpt)
            timer.stop()

            if not leg_path_utm:
                log.error(f"Tile routing failed for Leg {leg_idx + 1}.")
                sys.exit(1)
        else:
            log.info(f"  Activating MONOLITHIC PIPELINE for Leg {leg_idx + 1}")
            leg_path_utm = run_leg_monolithic(pt_a, pt_b, bbox, timer, ckpt, warn_handler)
            if not leg_path_utm:
                log.error(f"Monolithic routing failed for Leg {leg_idx + 1}.")
                sys.exit(1)

        # Merge paths
        if leg_idx > 0 and len(full_path_utm) > 0 and len(leg_path_utm) > 0:
            # Drop the first point of the next leg if it's identical/very close to the last point
            last_x, last_y = full_path_utm[-1]
            first_x, first_y = leg_path_utm[0]
            if math.hypot(last_x - first_x, last_y - first_y) < 1.0:
                leg_path_utm = leg_path_utm[1:]

        full_path_utm.extend(leg_path_utm)
        segment_indices.extend([leg_idx] * len(leg_path_utm))

    path_utm = full_path_utm
    log.info(f"\nAll {len(WAYPOINTS)-1} legs routed successfully. Total waypoints: {len(path_utm)}\n")

    # For post-processing (smoothing, structures, viz), we still need the
    # full-corridor DEM, water, and buildings. Fetch them using the
    # monolithic bbox (these are cached from tile fetches via _tile_key).
    log.info("Tile routing complete. Loading full-corridor data for post-processing …")
    dem, transform, dem_source = fetch_dem(bbox)
    rows, cols = dem.shape

    buildings_wgs, water_wgs, roads_wgs, lulc_wgs, osm_stats = fetch_osm_layers(bbox)

    def to_utm(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        return gdf.to_crs(epsg=UTM_EPSG)

    buildings_utm = to_utm(buildings_wgs)
    water_utm = to_utm(water_wgs)
    roads_utm = to_utm(roads_wgs)
    lulc_utm  = to_utm(lulc_wgs)

    from config import USE_CUSTOM_WATER
    if USE_CUSTOM_WATER:
        custom_water_wgs = fetch_custom_water(bbox)
        if custom_water_wgs is not None and len(custom_water_wgs) > 0:
            custom_water_utm = to_utm(custom_water_wgs)
            import pandas as pd
        
            # Add a natural="water" tag so it passes filter_bridge_worthy_water
            if "natural" not in custom_water_utm.columns:
                custom_water_utm["natural"] = "water"
            else:
                custom_water_utm["natural"] = custom_water_utm["natural"].fillna("water")
            
            if water_utm is None or len(water_utm) == 0:
                water_utm = custom_water_utm
            else:
                water_utm = pd.concat([water_utm, custom_water_utm], ignore_index=True)
            log.info(f"Custom water merged: total water features now {len(water_utm)}.")
        else:
            log.info("Custom water: none fetched — using OSM only.")

    # ── 3d. Phase 14.3: Dam & Lake Avoidance Zones ──────────────────────────
    from config import USE_DAM_LAKE_AVOIDANCE, DAM_LAKE_BUFFER_M
    exclusion_gdf_utm = None
    if USE_DAM_LAKE_AVOIDANCE:
        dam_lake_wgs = fetch_dam_lake(bbox)
        if dam_lake_wgs is not None and len(dam_lake_wgs) > 0:
            dam_lake_utm = to_utm(dam_lake_wgs)
        
            # Apply geometric buffer for structural avoidance
            if DAM_LAKE_BUFFER_M > 0:
                # Store original for export or visuals if needed, buffer the active geometry
                dam_lake_utm["geometry"] = dam_lake_utm.geometry.buffer(DAM_LAKE_BUFFER_M)
                log.info(f"Dam/Lake: Buffered exclusion zones by {DAM_LAKE_BUFFER_M} m.")
        
            exclusion_gdf_utm = dam_lake_utm
            log.info(f"Dam/Lake: Registered {len(exclusion_gdf_utm)} strictly avoided geometries.")
        else:
            log.info("Dam/Lake: none fetched — no absolute avoidance zones registered.")

    # Harmonize: Pre-filter water polygons to only bridge-worthy rivers
    # so the routing engine and the structure detector see the exact same rivers.
    if water_utm is not None and len(water_utm) > 0:
        from structures import filter_bridge_worthy_water
        water_utm = filter_bridge_worthy_water(water_utm)

    if buildings_utm is not None and len(buildings_utm) > 0:
        log.info(f"Loaded {len(buildings_utm)} OSM buildings.")

    if USE_OVERTURE_BUILDINGS:
        overture_wgs = fetch_overture_buildings(bbox)
        if overture_wgs is not None and len(overture_wgs) > 0:
            buildings_utm = merge_building_sources(
                buildings_utm, overture_wgs,
                dedup_radius_m=OVERTURE_DEDUP_RADIUS_M,
            )
        else:
            log.info("Overture buildings: none fetched — using OSM only.")

    if buildings_utm is not None and len(buildings_utm) > 0:
        log.info(f"Total buildings for penalty layer: {len(buildings_utm)}")

    # ── 4. Slope + curvature ───────────────────────────────────────────────
    log.info("Computing slope and curvature …")
    slope_pct, nodata_mask, curvature = compute_slope(dem, RESOLUTION)
    log.info(
        f"Slope: max={slope_pct.max():.1f}%, "
        f"cells>{SLOPE_MAX_PCT}%: {np.sum(slope_pct > SLOPE_MAX_PCT):,}"
    )
    def to_utm(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        return gdf.to_crs(epsg=UTM_EPSG)

    buildings_utm = to_utm(buildings_wgs)
    water_utm     = to_utm(water_wgs)
    roads_utm     = to_utm(roads_wgs)
    lulc_utm      = to_utm(lulc_wgs)

    # ── 3c. Phase 14.2: Custom Water Dataset (supplement OSM) ─────────────
    from config import USE_CUSTOM_WATER
    if USE_CUSTOM_WATER:
        custom_water_wgs = fetch_custom_water(bbox)
        if custom_water_wgs is not None and len(custom_water_wgs) > 0:
            custom_water_utm = to_utm(custom_water_wgs)
            import pandas as pd
        
            # Add a natural="water" tag so it passes filter_bridge_worthy_water
            if "natural" not in custom_water_utm.columns:
                custom_water_utm["natural"] = "water"
            else:
                custom_water_utm["natural"] = custom_water_utm["natural"].fillna("water")
            
            if water_utm is None or len(water_utm) == 0:
                water_utm = custom_water_utm
            else:
                water_utm = pd.concat([water_utm, custom_water_utm], ignore_index=True)
            log.info(f"Custom water merged: total water features now {len(water_utm)}.")
        else:
            log.info("Custom water: none fetched — using OSM only.")

    # ── 3d. Phase 14.3: Dam & Lake Avoidance Zones ──────────────────────────
    from config import USE_DAM_LAKE_AVOIDANCE, DAM_LAKE_BUFFER_M
    exclusion_gdf_utm = None
    if USE_DAM_LAKE_AVOIDANCE:
        dam_lake_wgs = fetch_dam_lake(bbox)
        if dam_lake_wgs is not None and len(dam_lake_wgs) > 0:
            dam_lake_utm = to_utm(dam_lake_wgs)
        
            # Apply geometric buffer for structural avoidance
            if DAM_LAKE_BUFFER_M > 0:
                # Store original for export or visuals if needed, buffer the active geometry
                dam_lake_utm["geometry"] = dam_lake_utm.geometry.buffer(DAM_LAKE_BUFFER_M)
                log.info(f"Dam/Lake: Buffered exclusion zones by {DAM_LAKE_BUFFER_M} m.")
        
            exclusion_gdf_utm = dam_lake_utm
            log.info(f"Dam/Lake: Registered {len(exclusion_gdf_utm)} strictly avoided geometries.")
        else:
            log.info("Dam/Lake: none fetched — no absolute avoidance zones registered.")

    # Harmonize: Pre-filter water polygons to only bridge-worthy rivers
    # so the routing engine and the structure detector see the exact same rivers.
    if water_utm is not None and len(water_utm) > 0:
        water_utm = filter_bridge_worthy_water(water_utm)

    if buildings_utm is not None and len(buildings_utm) > 0:
        log.info(f"Loaded {len(buildings_utm)} OSM buildings.")

    # ── 3b. Phase 11: Overture Maps buildings (supplement OSM) ────────────
    if USE_OVERTURE_BUILDINGS:
        overture_wgs = fetch_overture_buildings(bbox)
        if overture_wgs is not None and len(overture_wgs) > 0:
            buildings_utm = merge_building_sources(
                buildings_utm, overture_wgs,
                dedup_radius_m=OVERTURE_DEDUP_RADIUS_M,
            )
        else:
            log.info("Overture buildings: none fetched — using OSM only.")

    if buildings_utm is not None and len(buildings_utm) > 0:
        log.info(f"Total buildings for penalty layer: {len(buildings_utm)}")

    # ── 4. Slope + curvature ───────────────────────────────────────────────
    log.info("Computing slope and curvature …")
    slope_pct, nodata_mask, curvature = compute_slope(dem, RESOLUTION)
    log.info(
        f"Slope: max={slope_pct.max():.1f}%, "
        f"cells>{SLOPE_MAX_PCT}%: {np.sum(slope_pct > SLOPE_MAX_PCT):,}"
    )

    # ── 4b. A→B bearing (for anisotropic sidehill cost) ───────────────────
    # Removed global bearing sidehill logic as per Phase 5.1 audit.

    # ── 5. Rasterise vector layers — merge with DEM stream fallback if needed ─
    log.info("Rasterising exclusion zones and environmental layers …")

    # Phase 14.5: Intermediate raster cache for expensive products
    rcache = RasterCache()

    # Phase 5.2: Buildings now use an area-based concentric penalty map
    building_penalty_map = np.zeros((rows, cols), dtype=np.float32)
    if buildings_utm is not None and len(buildings_utm) > 0:
        log.info(f"Applying building penalties to {len(buildings_utm)} buildings...")
        # Phase 14.5: Check raster cache first
        bldg_fp = _cache_fingerprint(
            len(buildings_utm), (rows, cols),
            BUILDING_BASE_PENALTY, BUILDING_AREA_MULT, ROW_BUFFER_M_CFG, RESOLUTION,
        )
        cached_bldg, bldg_hit = rcache.get("building_penalty", bldg_fp, shape=(rows, cols))
        if bldg_hit:
            building_penalty_map = cached_bldg.astype(np.float32)
        else:
            from cost_surface import _apply_building_penalties
            building_penalty_map = _apply_building_penalties(buildings_utm, transform, (rows, cols), RESOLUTION)
            rcache.put("building_penalty", bldg_fp, building_penalty_map, transform)

    # Phase 14.5: Cache water and roads masks too
    water_fp = _cache_fingerprint(
        len(water_utm) if water_utm is not None else 0, (rows, cols)
    )
    cached_water, water_hit = rcache.get("water_mask", water_fp, shape=(rows, cols))
    if water_hit:
        log.info("Water mask loaded from raster cache.")
        water_mask = cached_water.astype(np.float32)
    else:
        log.info("Rasterising water_mask...")
        water_mask = rasterise_layer(water_utm, transform, (rows, cols))
        rcache.put("water_mask", water_fp, water_mask, transform)

    roads_fp = _cache_fingerprint(
        len(roads_utm) if roads_utm is not None else 0, (rows, cols)
    )
    cached_roads, roads_hit = rcache.get("roads_mask", roads_fp, shape=(rows, cols))
    if roads_hit:
        log.info("Roads mask loaded from raster cache.")
        roads_mask = cached_roads.astype(np.float32)
    else:
        log.info("Rasterising roads_mask...")
        roads_mask = rasterise_layer(roads_utm, transform, (rows, cols))
        rcache.put("roads_mask", roads_fp, roads_mask, transform)
    
    # ── 5b. Phase 11: LULC source selection (WorldCover + OSM blend) ──────
    # Fix IX: WorldCover is satellite-derived (wall-to-wall vegetation class)
    # but does NOT encode legal barriers (protected areas, national parks).
    # OSM polygons DO capture government-designated legal zones with high accuracy.
    # Strategy: compute both, take element-wise maximum — WorldCover for terrain
    # cost, OSM for legal penalty uplift.  Falls back gracefully if either fails.
    from cost_surface import _apply_lulc_penalties

    lulc_penalty_map = np.full((rows, cols), 1.0, dtype=np.float32)
    lulc_source = "none"

    if USE_WORLDCOVER_LULC:
        log.info("Phase 11: Fetching ESA WorldCover 10m for LULC …")
        wc_array, wc_transform = fetch_worldcover(bbox)
        if wc_array is not None:
            lulc_penalty_map = worldcover_to_lulc_raster(
                wc_array, wc_transform,
                target_shape=(rows, cols),
                target_transform=transform,
                slope_pct=slope_pct,
            )
            lulc_source = "WorldCover 10m"
            log.info("LULC source: ESA WorldCover 10m (satellite-derived, wall-to-wall)")
        else:
            log.warning("WorldCover fetch failed — falling back to OSM LULC only.")

    # Always blend OSM LULC on top (maximum).  Even when WorldCover succeeds,
    # OSM-mapped protected areas / national parks carry legal penalty information
    # that WorldCover cannot encode.  np.maximum preserves the higher penalty
    # from either source at each cell.
    if lulc_utm is not None and len(lulc_utm) > 0:
        log.info("Blending OSM LULC polygons (protected areas / legal barriers) …")
        osm_penalty_map = _apply_lulc_penalties(
            lulc_utm, transform, (rows, cols), slope_pct=slope_pct
        )
        # Blend: take the worse (higher) penalty at each cell
        np.maximum(lulc_penalty_map, osm_penalty_map, out=lulc_penalty_map)
        if lulc_source == "none":
            lulc_source = "OSM polygons"
        else:
            lulc_source = "WorldCover 10m + OSM blended"
        log.info(
            f"LULC blend complete: max penalty={lulc_penalty_map.max():.1f}×, "
            f"cells above 1.0: {np.sum(lulc_penalty_map > 1.0):,}"
        )
    elif lulc_source == "none":
        lulc_source = "none (unmapped base 1.0)"

    log.info(f"LULC penalty source: {lulc_source}")

    # Phase 3: if OSM water is sparse, derive stream network from DEM
    if osm_stats.get('dem_stream_fallback'):
        log.info("Deriving stream network from DEM (OSM water fallback active) …")
        dem_streams = derive_stream_mask_utm(dem, transform, resolution_m=RESOLUTION)
        # Merge: union of OSM (sparse) and DEM-derived streams
        water_mask = np.maximum(water_mask, dem_streams).astype(np.float32)
        n_stream_cells = int((water_mask > 0).sum())
        log.info(f"Combined water mask: {n_stream_cells:,} cells "
                 f"(OSM={osm_stats['water']} + DEM-derived)")
             
        # Fix 17: Removed dead-code duplicate vectorisation loop.
        # Only the polygon-based approach (below) successfully appends to dem_water_gdf.
        from rasterio.features import shapes
        from shapely.geometry import shape
        import geopandas as gpd

        vectors = []
        for geom, val in shapes((dem_streams > 0).astype(np.int32), mask=(dem_streams > 0), transform=transform):
            poly = shape(geom)
            vectors.append(poly)

        if vectors:
            dem_water_gdf = gpd.GeoDataFrame(
                {"waterway": ["dem_stream"] * len(vectors)},
                geometry=vectors,
                crs=f"EPSG:{UTM_EPSG}"
            )
            if water_utm is None or len(water_utm) == 0:
                water_utm = dem_water_gdf
            else:
                import pandas as pd
                water_utm = pd.concat([water_utm, dem_water_gdf], ignore_index=True)
            log.info(f"Appended {len(vectors)} DEM-derived stream polygons to water_utm.")

        warn_handler.warnings.append(
            f"DEM stream fallback active: OSM water returned only "
            f"{osm_stats['water']} features. Stream network derived "
            f"from DEM flow accumulation."
        )

    # ── 6. Cost surface (all layers) ───────────────────────────────────────
    log.info("Building cost surface …")

    # Pre-compute individual layer snapshots for the layer decomposition panel
    slope_cost_layer = _slope_cost_array(slope_pct)

    # Water hierarchy layer (for visualization)
    water_layer = np.zeros((rows, cols), dtype=np.float64)
    if np.any(water_mask > 0):
        from scipy.ndimage import binary_closing
        radius = 4
        y_g, x_g = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = x_g ** 2 + y_g ** 2 <= radius ** 2
        water_closed_viz = binary_closing(water_mask > 0, structure=disk)
        water_layer = _river_hierarchy_penalties(
            water_closed_viz.astype(np.float32), RESOLUTION
        )

    # Phase 5.4: Road discount viz — render per-class multiplier values so the
    # layer decomposition panel shows a gradient, not a uniform flat blob.
    road_layer = np.ones((rows, cols), dtype=np.float32)
    if roads_mask is not None and np.any(roads_mask > 0) and roads_utm is not None:
        if len(roads_utm) > 0 and "highway" in roads_utm.columns:
            from rasterio.features import rasterize as _rasterize
            for hw_class, multiplier in ROAD_CLASS_DISCOUNTS.items():
                if hw_class == "default":
                    continue
                subset = roads_utm[roads_utm["highway"] == hw_class]
                if len(subset) == 0:
                    continue
                geoms = [g for g in subset.geometry if g is not None and not g.is_empty]
                if not geoms:
                    continue
                class_r = _rasterize(
                    [(g, multiplier) for g in geoms],
                    out_shape=(rows, cols), transform=transform,
                    fill=1.0, dtype=np.float32,
                )
                # Keep strongest discount (lowest value) per cell
                np.minimum(road_layer, class_r, out=road_layer)
        else:
            # No class info — fall back to default discount
            road_layer = np.where(
                roads_mask > 0,
                ROAD_CLASS_DISCOUNTS["default"],
                1.0
            ).astype(np.float32)

    cost = build_cost_surface(
        slope_pct, building_penalty_map, water_mask,
        roads_mask=roads_mask,
        roads_gdf=roads_utm,          # Phase 5.4: for per-class discount
        lulc_penalty_map=lulc_penalty_map,
        nodata_mask=nodata_mask,
        dem=dem,
        curvature=curvature,
        resolution_m=RESOLUTION,
        transform=transform,          # Phase 5.4: required for class rasterisation
        exclusion_gdf=exclusion_gdf_utm # Phase 14.3: Dam/Lake avoidance
    )

    # ── 6b. Export intermediate rasters (optional) ────────────────────────
    if EXPORT_INTERMEDIATES:
        os.makedirs("output", exist_ok=True)
        try:
            import rasterio
            profile = {
                'driver': 'GTiff', 'dtype': 'float32',
                'width': cols, 'height': rows, 'count': 1,
                'crs': f'EPSG:{UTM_EPSG}', 'transform': transform,
                'compress': 'deflate',
            }
            with rasterio.open('output/cost_surface.tif', 'w', **profile) as dst:
                dst.write(cost.astype(np.float32), 1)
            with rasterio.open('output/building_penalty.tif', 'w', **profile) as dst:
                dst.write(building_penalty_map.astype(np.float32), 1)
            log.info("Intermediate GeoTIFFs saved to output/")
        except Exception as exc:
            log.warning(f"Could not save intermediate GeoTIFFs: {exc}")

    # ── 7. Map path_utm back to path_indices for downstream functions ──────
    from geometry_utils import xy_to_rowcol
    path_indices = [xy_to_rowcol(x, y, transform) for x, y in path_utm]

    # ── 9. Adaptive smoothing ──────────────────────────────────────────────
    log.info("Smoothing path (segment-aware B-spline) …")
    smooth_utm = smooth_path(
        path_utm,
        slope_pct=slope_pct,
        transform=transform,
    )

    # ── 10. Curve radius verification (with re-smooth attempt) ─────────────
    log.info("Verifying curve radii …")
    min_radius, violations = verify_curve_radius(smooth_utm)

    if violations:
        log.info("Re-smoothing with increased factor …")
        smooth_utm = smooth_path(
            path_utm, 
            smoothing=len(path_utm) * 20.0,
            slope_pct=slope_pct, transform=transform,
        )
        min_radius, violations = verify_curve_radius(smooth_utm)
        if violations:
            log.warning(
                f"Could not fully satisfy {min_radius:.0f} m curve-radius constraint. "
                "Outputting best-effort geometry."
            )

    # ── 10a. Map smoothed points back to leg segments via Nearest Neighbor ─
    from scipy.spatial import cKDTree
    tree = cKDTree(path_utm)
    _, idx = tree.query(smooth_utm)
    smooth_segment_indices = [segment_indices[i] for i in idx]

    # ── 10b. Phase 12: minimum tangent/curve length check ─────────────────
    design_lengths = verify_design_lengths(smooth_utm)
    n_dl_violations = (
        len(design_lengths["tangent_violations"])
        + len(design_lengths["curve_violations"])
    )
    if n_dl_violations > 0:
        warn_handler.warnings.append(
            f"Design length violations: {n_dl_violations} segments below minimum"
        )

    line_utm = LineString(smooth_utm)

    # ── 11. RoW setback ────────────────────────────────────────────────────
    log.info("Verifying RoW setback …")
    verify_row_setback(line_utm, buildings_utm, ROW_BUFFER_M)

    # ── 12. Longitudinal profile and sustained grade check ─────────────────
    log.info("Checking sustained grade (heavy truck standard) …")
    dists_m, elevs_m = extract_longitudinal_profile(smooth_utm, dem, transform)
    grade_violations = check_sustained_grade(elevs_m, dists_m, max_grade=0.08, window_m=3_000)

    log.info("Computing clothoid transition requirements …")
    clothoid_transitions = compute_clothoid_transitions(smooth_utm)
    infeasible_spirals = sum(1 for t in clothoid_transitions if not t["feasible"])

    # ── 12b. Phase 6: Vertical alignment design (grade-clipping) ──────────
    va_result = None
    if EXPORT_3D_GEOJSON:
        timer.start("12b_vertical_alignment")
        try:
            from vertical_alignment import build_vertical_alignment
            k_crest, k_sag = VC_K_VALUES.get(
                int(DESIGN_SPEED_KMPH),
                VC_K_VALUES[min(VC_K_VALUES, key=lambda v: abs(v - DESIGN_SPEED_KMPH))]
            )
            g_max = GRADE_MAX_PCT.get(SCENARIO_PROFILE, 8.0)

            va_result = build_vertical_alignment(
                dists_m, elevs_m,
                design_speed_kmph=DESIGN_SPEED_KMPH,
                max_grade_pct=g_max,
                k_crest=k_crest,
                k_sag=k_sag,
                min_vc_length_m=MIN_VC_LENGTH_M,
                min_vpi_spacing_m=MIN_VPI_SPACING_M,
            )
            log.info(
                f"Vertical alignment: {len(va_result.vertical_curves)} VCs  "
                f"max_grade={va_result.max_grade_pct:.2f}%  "
                f"grade_violations={len(va_result.grade_violations)}  "
                f"SSD_violations={len(va_result.ssd_violations)}  "
                f"max_fill={va_result.cut_fill_m.max():.1f} m  "
                f"max_cut={-va_result.cut_fill_m.min():.1f} m"
            )
        except Exception as exc:
            log.warning(f"Vertical alignment failed ({exc}) — 3D export skipped.")
            import traceback as _tb
            _tb.print_exc()
        finally:
            timer.stop()

    # ── 12c. Phase 7: Earthwork volumes (cut/fill + mass-haul) ───────────
    ew_result = None
    if va_result is not None:
        timer.start("12c_earthwork")
        try:
            from earthwork import compute_earthwork, export_earthwork_csv
            fw = FORMATION_WIDTH_M.get(SCENARIO_PROFILE, 11.0)
            ew_result = compute_earthwork(
                va_result.distances_m,
                va_result.cut_fill_m,
                formation_width_m=fw,
                cut_batter_HV=CUT_BATTER_HV,
                fill_batter_HV=FILL_BATTER_HV,
                swell_factor=SWELL_FACTOR,
            )
            export_earthwork_csv(ew_result, OUTPUT_EARTHWORK_CSV)
        except Exception as exc:
            log.warning(f"Earthwork computation failed ({exc}) — skipping.")
            import traceback as _tb
            _tb.print_exc()
        finally:
            timer.stop()

    # ── 12d. Phase 8: Bridge and culvert inventory ───────────────────
    si_result = None
    if va_result is not None:
        timer.start("12d_structures")
        try:
            from structures import build_structure_inventory, export_structures_csv
            # flow_accum may or may not be in scope depending on whether Phase 3
            # D8 stream derivation ran; default to None gracefully.
            _flow_accum = locals().get("flow_accum", None)
            si_result = build_structure_inventory(
                smooth_utm=smooth_utm,
                va_result=va_result,
                water_utm=water_utm,
                water_mask=water_mask,
                flow_accum=_flow_accum,
                transform=transform,
                path_indices=path_indices,
                bridge_freeboard_m=BRIDGE_FREEBOARD_M,
                bridge_cost_per_m2_usd=BRIDGE_COST_PER_M2_USD,
                bridge_width_m=BRIDGE_DECK_WIDTH_M,
                resolution_m=RESOLUTION,
            )
            export_structures_csv(si_result, OUTPUT_STRUCTURES_CSV)
        except Exception as exc:
            log.warning(f"Structure inventory failed ({exc}) — skipping.")
            import traceback as _tb
            _tb.print_exc()
        finally:
            timer.stop()

    # ── 12e placeholder — cost model runs after meta is built (below) ───
    cost_result = None

    # ── 13. Reproject and export ───────────────────────────────────────────
    log.info("Reprojecting to WGS-84 …")
    wgs84_coords = [utm_to_wgs84(x, y) for x, y in smooth_utm]
    line_wgs84 = LineString(wgs84_coords)

    meta = compute_metadata(
        line_utm, slope_pct, transform,
        dem=dem,
        dem_source=dem_source,
        osm_stats=osm_stats,
        warnings_list=warn_handler.warnings,
    )
    meta['min_curve_radius_achieved_m'] = round(min_radius, 1)
    meta['curve_radius_violations']     = len(violations)
    meta['sustained_grade_violations']  = len(grade_violations)
    meta['clothoid_curves_checked']     = len(clothoid_transitions)
    meta['clothoid_infeasible']         = infeasible_spirals
    # Limit to first 50 transitions to keep GeoJSON readable
    meta['clothoid_transitions']        = clothoid_transitions[:50]

    # Phase 6 vertical alignment metadata
    if va_result is not None:
        meta['vertical_design_version']    = 'Phase6'
        meta['vc_count']                   = len(va_result.vertical_curves)
        meta['vertical_max_grade_pct']     = round(va_result.max_grade_pct, 2)
        meta['vertical_grade_violations']  = len(va_result.grade_violations)
        meta['vertical_ssd_violations']    = len(va_result.ssd_violations)
        meta['max_fill_m']                 = round(float(va_result.cut_fill_m.max()), 1)
        meta['max_cut_m']                  = round(float(-va_result.cut_fill_m.min()), 1)

    # Phase 7 earthwork metadata
    if ew_result is not None:
        meta['total_cut_Mm3']       = round(ew_result.total_cut_m3  / 1e6, 3)
        meta['total_fill_Mm3']      = round(ew_result.total_fill_m3 / 1e6, 3)
        meta['net_import_Mm3']      = round(ew_result.net_import_m3 / 1e6, 3)
        meta['balance_points']      = len(ew_result.balance_stations_m)
        meta['formation_width_m']   = ew_result.formation_width_m

    # Phase 8 structure metadata
    if si_result is not None:
        meta['bridge_count']              = si_result.bridge_count
        meta['total_bridge_length_m']     = round(si_result.total_bridge_length_m, 1)
        meta['total_bridge_cost_USD']     = round(si_result.total_bridge_cost_usd, 0)

    # ── 12e. Phase 9: Parametric cost model ─────────────────────────────────
    # Fix 19: Moved to before export_geojson so cost_result fields appear in
    # the 2D GeoJSON properties. The old position (after export) meant the
    # cost fields were only generated for the 3D GeoJSON, not the 2D one.
    cost_result = None
    timer.start("12e_cost_model")
    try:
        from cost_model import compute_cost_model, export_cost_csv
        cost_result = compute_cost_model(
            meta=meta,
            ew_result=ew_result,
            si_result=si_result,
            scenario_profile=SCENARIO_PROFILE,
            lulc_wgs=lulc_wgs,
            cut_rate_usd_m3=EARTHWORK_CUT_RATE_USD_M3,
            fill_rate_usd_m3=EARTHWORK_FILL_RATE_USD_M3,
            pavement_rate_m2=PAVEMENT_RATE_USD_M2.get(SCENARIO_PROFILE, 120.0) if isinstance(PAVEMENT_RATE_USD_M2, dict) else PAVEMENT_RATE_USD_M2,
            corridor_width_m=CORRIDOR_WIDTH_M,
            land_acq_default=LAND_ACQ_DEFAULT_USD_PER_HA,
            land_acq_rates=LAND_ACQ_RATES,
            env_factor=ENV_MITIGATION_FACTOR,
            contingency_factor=CONTINGENCY_FACTOR,
            engineering_factor=ENGINEERING_FACTOR,
        )
        export_cost_csv(cost_result, OUTPUT_COST_CSV)
    except Exception as exc:
        log.warning(f"Cost model failed ({exc}) — skipping.")
        import traceback as _tb
        _tb.print_exc()
    finally:
        timer.stop()

    # Phase 9 cost model metadata (populated before GeoJSON so fields are included)
    if cost_result is not None:
        meta['cost_model_version']         = 'Phase9'
        meta['total_project_cost_USD']     = round(cost_result.total_project_cost_usd, 0)
        meta['cost_per_km_USD']            = round(cost_result.cost_per_km_usd, 0)
        meta['civil_subtotal_USD']         = round(cost_result.civil_subtotal_usd, 0)
        meta['contingency_USD']            = round(cost_result.contingency_usd, 0)
        meta['engineering_USD']            = round(cost_result.engineering_usd, 0)
        meta['land_acquisition_ha']        = round(cost_result.land_acquisition_ha, 1)
        meta['pavement_area_m2']           = round(cost_result.pavement_area_m2, 0)

    # Phase 5: peak memory and stage timing
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(peak_mem / 1e6, 1)
    meta['peak_memory_mb'] = peak_mem_mb
    meta['stage_timing_s'] = timer.as_dict()

    export_geojson(line_wgs84, meta, OUTPUT_FILE)

    # Phase 6: 3D GeoJSON export
    if va_result is not None and EXPORT_3D_GEOJSON:
        # Pair design elevations with WGS-84 coordinates.
        # Both are derived from smooth_utm (same length: 500 points).
        wgs84_coords_3d = [
            (lon, lat, float(va_result.z_design[i]))
            for i, (lon, lat) in enumerate(wgs84_coords)
        ]
        export_geojson_3d(wgs84_coords_3d, meta, OUTPUT_FILE_3D)


    # ── 12f. Phase 10: Feasibility report ────────────────────────────────
    timer.start("12f_report")
    try:
        from report import generate_report
        generate_report(
            meta=meta,
            va_result=va_result,
            ew_result=ew_result,
            si_result=si_result,
            cost_result=cost_result,
            output_html=OUTPUT_REPORT_HTML,
            output_pdf=OUTPUT_REPORT_PDF,
            waypoints=WAYPOINTS,
            segment_indices=smooth_segment_indices,
        )
    except ImportError as exc:
        log.warning(f"Report generation skipped — {exc}")
    except Exception as exc:
        log.warning(f"Report generation failed ({exc}) — continuing.")
        import traceback as _tb
        _tb.print_exc()
    finally:
        timer.stop()

    ckpt.clear()   # successful run — remove checkpoint

    log.info("═" * 65)
    log.info("  COMPLETE — Output: %s", OUTPUT_FILE)
    log.info(f"  Total length       : {meta['total_length_km']} km")
    log.info(f"  Max slope          : {meta['max_slope_pct']} %")
    log.info(f"  Min curve radius   : {meta['min_curve_radius_achieved_m']} m")
    log.info(f"  Radius violations  : {meta['curve_radius_violations']}")
    log.info(f"  Grade violations   : {meta['sustained_grade_violations']}")
    log.info(f"  Clothoid curves    : {meta['clothoid_curves_checked']} checked, "
             f"{meta['clothoid_infeasible']} infeasible")
    if va_result is not None:
        log.info(f"  Vertical curves    : {meta['vc_count']} VCs  "
                 f"max grade={meta['vertical_max_grade_pct']}%")
        log.info(f"  Vert grade viol.   : {meta['vertical_grade_violations']}")
        log.info(f"  SSD violations     : {meta['vertical_ssd_violations']}")
        log.info(f"  Max fill / cut     : +{meta['max_fill_m']} m / -{meta['max_cut_m']} m")
        log.info(f"  3D GeoJSON         : {OUTPUT_FILE_3D}")
    if ew_result is not None:
        net_lbl = 'import' if ew_result.net_import_m3 > 0 else 'spoil'
        log.info(f"  Earthwork cut      : {meta['total_cut_Mm3']} Mm³")
        log.info(f"  Earthwork fill     : {meta['total_fill_Mm3']} Mm³")
        log.info(f"  Net {net_lbl:<6}         : {abs(meta['net_import_Mm3']):.3f} Mm³")
        log.info(f"  Balance points     : {meta['balance_points']}")
        log.info(f"  Earthwork CSV      : {OUTPUT_EARTHWORK_CSV}")
    if si_result is not None:
        log.info(f"  Bridges            : {meta['bridge_count']} "
                 f"({meta['total_bridge_length_m']:.0f} m span, "
                 f"USD {meta['total_bridge_cost_USD']/1e6:.2f} M)")
        log.info(f"  Structures CSV     : {OUTPUT_STRUCTURES_CSV}")
    log.info(f"  Data confidence    : {meta['data_confidence']}")
    log.info(f"  DEM source         : {meta['dem_source']}")
    log.info(f"  Peak memory        : {peak_mem_mb} MB")
    log.info("═" * 65)
    timer.log_summary()

    if cost_result is not None:
        log.info(f"  Total project cost : USD {cost_result.total_project_cost_usd/1e6:.2f} M")
        log.info(f"  Cost per km        : USD {cost_result.cost_per_km_usd/1e6:.2f} M/km")
        log.info(f"  Pavement (m²)      : {cost_result.pavement_area_m2:,.0f} m²")
        log.info(f"  Land acquisition   : {cost_result.land_acquisition_ha:.1f} ha")
        log.info(f"  Cost CSV           : {OUTPUT_COST_CSV}")
        log.info(f"  Report HTML        : {OUTPUT_REPORT_HTML}")
    if meta["data_warnings"]:
        log.warning("Data quality issues:")
        for w in meta["data_warnings"]:
            log.warning(f"  • {w}")

    # ── 14. Visualization suite (Phase 5.3) ──────────────────────────────
    if GENERATE_VISUALIZATIONS:
        timer.start("14_visualization")
        try:
            from visualize_route import generate_all_visuals

            # Sample slope along route_rc (raw path) — must match route_rc length
            slope_along = np.array([
                float(slope_pct[
                    max(0, min(rows - 1, r)),
                    max(0, min(cols - 1, c))
                ])
                for r, c in path_indices
            ])

            # Compute distances along route_rc to match slope_along length.
            # (distances_m/elevations_m came from smooth_utm which has a
            # different number of points — can't mix them with route_rc.)
            rc_xs = np.array([c for _, c in path_indices], dtype=np.float64) * RESOLUTION
            rc_ys = np.array([r for r, _ in path_indices], dtype=np.float64) * RESOLUTION
            rc_segs = np.sqrt(np.diff(rc_xs)**2 + np.diff(rc_ys)**2)
            rc_dists = np.concatenate(([0.0], np.cumsum(rc_segs)))

            viz_data = {
                'cost': cost,
                'building_penalty': building_penalty_map,
                'layers': {
                    'slope': slope_cost_layer,
                    'lulc': lulc_penalty_map,
                    'water': water_layer,
                    'building': building_penalty_map,
                    'road': road_layer,
                    'unified': cost,
                },
                'dem': dem,
                'route_rc': path_indices,
                'route_utm': smooth_utm,
                'route_wgs84': wgs84_coords,
                'segment_indices': smooth_segment_indices,
                'waypoints': WAYPOINTS,
                'distances_m': dists_m,
                'elevations_m': elevs_m,
                'rc_distances_m': rc_dists,      # same length as route_rc
                'grade_violations': grade_violations,
                'slope_along': slope_along,
                'slope_thresholds': {
                    's_opt': SLOPE_OPTIMAL_PCT,
                    's_mod': SLOPE_MODERATE_PCT,
                    's_max': SLOPE_MAX_PCT,
                    's_cliff': SLOPE_CLIFF_PCT,
                },
                'meta': meta,
                'transform': transform,
                'resolution_m': RESOLUTION,
                'buildings_wgs': buildings_wgs,
                'water_wgs': water_wgs,
                # Phase 6
                'va_result': va_result,
                # Phase 7
                'ew_result': ew_result,
                # Phase 8
                'si_result': si_result,
            }
            generate_all_visuals(viz_data)

            # ── Auto-launch route_map.html in browser ─────────────────────
            # serve.py handles port-already-in-use gracefully (just opens browser).
            # On a fresh run it starts a no-cache server so the browser always
            # gets the latest generated file.
            try:
                import subprocess, webbrowser, time as _time, socket as _sock
                PORT = 8765

                def _port_in_use(p):
                    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
                        s.settimeout(0.3)
                        return s.connect_ex(("127.0.0.1", p)) == 0

                if not _port_in_use(PORT):
                    # Start a new detached serve.py process
                    kwargs = {"creationflags": 0x00000008} if sys.platform == "win32" else {"start_new_session": True}
                    subprocess.Popen(
                        [sys.executable, "serve.py", "--bg"],
                        **kwargs
                    )
                    _time.sleep(0.8)  # allow server to bind

                webbrowser.open(f"http://127.0.0.1:{PORT}/route_map.html")
                log.info(f"Route map opened in browser: http://127.0.0.1:{PORT}/route_map.html")
            except Exception as _e:
                log.warning(f"Could not auto-open route map: {_e}")

        except Exception as exc:
            log.warning(f"Visualization suite failed: {exc}")
            import traceback as _tb
            _tb.print_exc()
        finally:
            timer.stop()



if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("highway_alignment").error("Fatal error in pipeline:")
        traceback.print_exc()
        sys.exit(1)
