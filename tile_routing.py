"""
tile_routing.py — Phase 15: Tile-based corridor routing
=========================================================
Partitions long corridors into overlapping tiles, processes each tile
independently (fetch → rasterize → route), then stitches path segments
across tile boundaries.

Activated automatically when corridor_km > TILE_ROUTING_THRESHOLD_KM.
Short corridors bypass this module entirely (monolithic pipeline).
"""
import gc
import logging
import math
import numpy as np
from collections import namedtuple

from config import (
    DATA_DIR, UTM_EPSG, RESOLUTION,
    TILE_LENGTH_KM, TILE_OVERLAP_KM, TILE_LATERAL_MARGIN_DEG,
    TILE_ROUTING_THRESHOLD_KM,
    USE_WORLDCOVER_LULC, USE_OVERTURE_BUILDINGS, OVERTURE_DEDUP_RADIUS_M,
    SLOPE_MAX_PCT, BORDER_CELLS, IMPASSABLE,
    PYRAMID_LEVELS, DOWNSAMPLE_RATIO, DOWNSAMPLE_METHOD,
    BUILDING_BASE_PENALTY, BUILDING_AREA_MULT, ROW_BUFFER_M,
    EXPORT_INTERMEDIATES, ROAD_CLASS_DISCOUNTS,
)
from geometry_utils import wgs84_to_utm, utm_to_wgs84, bbox_with_margin, xy_to_rowcol, rowcol_to_xy

log = logging.getLogger("highway_alignment")

# ── Data types ────────────────────────────────────────────────────────────────

TileBBox = namedtuple("TileBBox", [
    "bbox_wgs84",      # (west, south, east, north)
    "entry_lonlat",    # (lon, lat) corridor enters this tile
    "exit_lonlat",     # (lon, lat) corridor exits this tile
    "tile_index",      # sequential index (0 .. N-1)
    "is_first",        # True for Tile 0
    "is_last",         # True for Tile N-1
])

TileResult = namedtuple("TileResult", [
    "tile_index",
    "path_utm",        # list of (easting, northing)
    "entry_utm",       # (easting, northing) of entry point
    "exit_utm",        # (easting, northing) of exit point
])


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tile Partitioner
# ══════════════════════════════════════════════════════════════════════════════

class TilePartitioner:
    """
    Split a long corridor into overlapping tiles along the A→B bearing.

    Each tile is a rectangular WGS-84 bounding box aligned to the corridor
    axis with configurable length, overlap, and lateral margin.
    """

    def __init__(self, waypoints,
                 tile_length_km=None, overlap_km=None, margin_deg=None):
        self.waypoints = waypoints  # list of (lon, lat)
        self.tile_length_km = tile_length_km or TILE_LENGTH_KM
        self.overlap_km = overlap_km or TILE_OVERLAP_KM
        self.margin_deg = margin_deg or TILE_LATERAL_MARGIN_DEG

    def partition(self):
        """
        Returns a list of TileBBox namedtuples covering all waypoints.
        """
        tiles = []
        global_tile_idx = 0

        for leg_idx in range(len(self.waypoints) - 1):
            pt_a = self.waypoints[leg_idx]
            pt_b = self.waypoints[leg_idx + 1]

            ax, ay = wgs84_to_utm(*pt_a)
            bx, by = wgs84_to_utm(*pt_b)
            corridor_m = math.hypot(bx - ax, by - ay)
            corridor_km = corridor_m / 1000.0

            # Corridor bearing (UTM)
            bearing_rad = math.atan2(bx - ax, by - ay)  # from A toward B

            # Step size along corridor (with overlap)
            step_km = self.tile_length_km - self.overlap_km
            step_m = step_km * 1000.0
            tile_half_m = (self.tile_length_km * 1000.0) / 2.0

            # Generate tile centres along the corridor
            pos_m = 0.0

            while pos_m < corridor_m:
                # Centre of this tile along the corridor line
                tile_end_m = min(pos_m + self.tile_length_km * 1000.0, corridor_m)

                # Entry/exit points along the corridor line
                entry_m = pos_m
                exit_m = tile_end_m

                # Convert corridor positions to UTM coords
                entry_x = ax + entry_m * math.sin(bearing_rad)
                entry_y = ay + entry_m * math.cos(bearing_rad)
                exit_x = ax + exit_m * math.sin(bearing_rad)
                exit_y = ay + exit_m * math.cos(bearing_rad)

                # Convert entry/exit to WGS-84
                entry_lonlat = utm_to_wgs84(entry_x, entry_y)
                exit_lonlat = utm_to_wgs84(exit_x, exit_y)

                # Build bbox: use corridor endpoints + lateral margin
                # Collect corner coords and add margin
                lons = [entry_lonlat[0], exit_lonlat[0]]
                lats = [entry_lonlat[1], exit_lonlat[1]]
                margin = self.margin_deg

                bbox_wgs84 = (
                    min(lons) - margin,
                    min(lats) - margin,
                    max(lons) + margin,
                    max(lats) + margin,
                )

                # For first tile in the entire journey
                is_first = (global_tile_idx == 0)

                # Check if this tile reaches the end of the current leg
                is_leg_last = (exit_m >= corridor_m - 1.0)
                is_last_overall = is_leg_last and (leg_idx == len(self.waypoints) - 2)

                if pos_m == 0.0:
                    entry_lonlat = pt_a
                if is_leg_last:
                    exit_lonlat = pt_b
                    # Expand bbox to include point_b
                    bbox_wgs84 = (
                        min(bbox_wgs84[0], pt_b[0] - margin),
                        min(bbox_wgs84[1], pt_b[1] - margin),
                        max(bbox_wgs84[2], pt_b[0] + margin),
                        max(bbox_wgs84[3], pt_b[1] + margin),
                    )

                tiles.append(TileBBox(
                    bbox_wgs84=bbox_wgs84,
                    entry_lonlat=entry_lonlat,
                    exit_lonlat=exit_lonlat,
                    tile_index=global_tile_idx,
                    is_first=is_first,
                    is_last=is_last_overall,
                ))

                if is_leg_last:
                    break

                pos_m += step_m
                global_tile_idx += 1
            
            # Ensure the next leg's first tile gets the incremented index
            global_tile_idx += 1

        log.info(
            f"TilePartitioner: {len(tiles)} tile(s) covering {len(self.waypoints)-1} legs "
            f"(tile={self.tile_length_km:.0f} km, overlap={self.overlap_km:.0f} km)"
        )
        return tiles


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-Tile Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _process_single_tile(tile, timer=None):
    """
    Fetch data, build cost surface, and route for a single tile.

    This is the full per-tile pipeline: DEM → OSM → Overture → slope →
    rasterize → cost surface → MS-LCP routing.

    Returns a TileResult with the path in UTM coordinates.
    """
    from data_fetch import (
        fetch_dem, fetch_osm_layers, derive_stream_mask_utm,
        fetch_worldcover, fetch_overture_buildings, merge_building_sources,
        fetch_custom_water, fetch_dam_lake,
        RasterCache, _cache_fingerprint,
    )
    from cost_surface import (
        compute_slope, rasterise_layer, build_cost_surface,
        worldcover_to_lulc_raster, build_cost_pyramid,
        _apply_building_penalties, _apply_lulc_penalties,
    )
    from routing import multi_scale_lcp
    from config import USE_CUSTOM_WATER, USE_DAM_LAKE_AVOIDANCE, DAM_LAKE_BUFFER_M

    idx = tile.tile_index
    bbox = tile.bbox_wgs84
    log.info(f"═══ Tile {idx}: Processing bbox W={bbox[0]:.4f} S={bbox[1]:.4f} "
             f"E={bbox[2]:.4f} N={bbox[3]:.4f} ═══")

    # ── 2a. DEM ────────────────────────────────────────────────────────────
    dem, transform, dem_source = fetch_dem(bbox)
    rows, cols = dem.shape
    log.info(f"Tile {idx}: DEM {rows}×{cols}, source={dem_source}")

    # ── 2b. OSM layers ─────────────────────────────────────────────────────
    buildings_wgs, water_wgs, roads_wgs, lulc_wgs, osm_stats = fetch_osm_layers(bbox)

    def to_utm(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        return gdf.to_crs(epsg=UTM_EPSG)

    buildings_utm = to_utm(buildings_wgs)
    water_utm = to_utm(water_wgs)
    roads_utm = to_utm(roads_wgs)
    lulc_utm = to_utm(lulc_wgs)

    # Custom water
    if USE_CUSTOM_WATER:
        custom_water_wgs = fetch_custom_water(bbox)
        if custom_water_wgs is not None and len(custom_water_wgs) > 0:
            custom_water_utm = to_utm(custom_water_wgs)
            if "natural" not in custom_water_utm.columns:
                custom_water_utm["natural"] = "water"
            else:
                custom_water_utm["natural"] = custom_water_utm["natural"].fillna("water")
            import pandas as pd
            if water_utm is None or len(water_utm) == 0:
                water_utm = custom_water_utm
            else:
                water_utm = pd.concat([water_utm, custom_water_utm], ignore_index=True)

    # Dam/lake avoidance
    exclusion_gdf_utm = None
    if USE_DAM_LAKE_AVOIDANCE:
        dam_lake_wgs = fetch_dam_lake(bbox)
        if dam_lake_wgs is not None and len(dam_lake_wgs) > 0:
            dam_lake_utm = to_utm(dam_lake_wgs)
            if DAM_LAKE_BUFFER_M > 0:
                dam_lake_utm["geometry"] = dam_lake_utm.geometry.buffer(DAM_LAKE_BUFFER_M)
            exclusion_gdf_utm = dam_lake_utm

    # Filter water
    from structures import filter_bridge_worthy_water
    if water_utm is not None and len(water_utm) > 0:
        water_utm = filter_bridge_worthy_water(water_utm)

    # ── 2c. Overture buildings ─────────────────────────────────────────────
    if USE_OVERTURE_BUILDINGS:
        overture_wgs = fetch_overture_buildings(bbox)
        if overture_wgs is not None and len(overture_wgs) > 0:
            buildings_utm = merge_building_sources(
                buildings_utm, overture_wgs,
                dedup_radius_m=OVERTURE_DEDUP_RADIUS_M,
            )

    # ── 2d. Slope & curvature ──────────────────────────────────────────────
    slope_pct, nodata_mask, curvature = compute_slope(dem, RESOLUTION)

    # ── 2e. Rasterise layers (with RasterCache) ───────────────────────────
    rcache = RasterCache()

    # Building penalty
    building_penalty_map = np.zeros((rows, cols), dtype=np.float32)
    if buildings_utm is not None and len(buildings_utm) > 0:
        bldg_fp = _cache_fingerprint(
            len(buildings_utm), (rows, cols),
            BUILDING_BASE_PENALTY, BUILDING_AREA_MULT, ROW_BUFFER_M, RESOLUTION,
        )
        cached_bldg, bldg_hit = rcache.get("building_penalty", bldg_fp, shape=(rows, cols))
        if bldg_hit:
            building_penalty_map = cached_bldg.astype(np.float32)
        else:
            building_penalty_map = _apply_building_penalties(
                buildings_utm, transform, (rows, cols), RESOLUTION
            )
            rcache.put("building_penalty", bldg_fp, building_penalty_map, transform)

    # Water mask
    water_fp = _cache_fingerprint(
        len(water_utm) if water_utm is not None else 0, (rows, cols)
    )
    cached_water, water_hit = rcache.get("water_mask", water_fp, shape=(rows, cols))
    if water_hit:
        water_mask = cached_water.astype(np.float32)
    else:
        water_mask = rasterise_layer(water_utm, transform, (rows, cols))
        rcache.put("water_mask", water_fp, water_mask, transform)

    # Roads mask
    roads_fp = _cache_fingerprint(
        len(roads_utm) if roads_utm is not None else 0, (rows, cols)
    )
    cached_roads, roads_hit = rcache.get("roads_mask", roads_fp, shape=(rows, cols))
    if roads_hit:
        roads_mask = cached_roads.astype(np.float32)
    else:
        roads_mask = rasterise_layer(roads_utm, transform, (rows, cols))
        rcache.put("roads_mask", roads_fp, roads_mask, transform)

    # LULC
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
        osm_penalty = _apply_lulc_penalties(
            lulc_utm, transform, (rows, cols), slope_pct=slope_pct
        )
        np.maximum(lulc_penalty_map, osm_penalty, out=lulc_penalty_map)

    # DEM stream fallback
    if osm_stats.get('dem_stream_fallback'):
        dem_streams = derive_stream_mask_utm(dem, transform, resolution_m=RESOLUTION)
        water_mask = np.maximum(water_mask, dem_streams).astype(np.float32)

    # ── 2f. Cost surface ───────────────────────────────────────────────────
    cost = build_cost_surface(
        slope_pct, building_penalty_map, water_mask,
        roads_mask=roads_mask,
        roads_gdf=roads_utm,
        lulc_penalty_map=lulc_penalty_map,
        nodata_mask=nodata_mask,
        dem=dem,
        curvature=curvature,
        resolution_m=RESOLUTION,
        transform=transform,
        exclusion_gdf=exclusion_gdf_utm,
    )

    # ── 2g. Entry/exit grid points ─────────────────────────────────────────
    entry_x, entry_y = wgs84_to_utm(*tile.entry_lonlat)
    exit_x, exit_y = wgs84_to_utm(*tile.exit_lonlat)
    start_rc = xy_to_rowcol(entry_x, entry_y, transform)
    end_rc = xy_to_rowcol(exit_x, exit_y, transform)

    b = BORDER_CELLS
    start_rc = (
        max(b, min(rows - 1 - b, start_rc[0])),
        max(b, min(cols - 1 - b, start_rc[1])),
    )
    end_rc = (
        max(b, min(rows - 1 - b, end_rc[0])),
        max(b, min(cols - 1 - b, end_rc[1])),
    )

    # Relax impassable endpoints
    for label, rc in [("entry", start_rc), ("exit", end_rc)]:
        if cost[rc] >= IMPASSABLE:
            r, c = rc
            r0, r1 = max(0, r - 1), min(rows, r + 2)
            c0, c1 = max(0, c - 1), min(cols, c + 2)
            neighbourhood = cost[r0:r1, c0:c1].copy()
            valid_costs = neighbourhood[neighbourhood < IMPASSABLE]
            floor_cost = float(np.percentile(valid_costs, 5)) if len(valid_costs) > 0 else 1.0
            floor_cost = max(floor_cost, 1.0)
            cost[r0:r1, c0:c1] = np.minimum(cost[r0:r1, c0:c1], floor_cost)

    # ── 2h. Route with MS-LCP ──────────────────────────────────────────────
    cost_pyramid = build_cost_pyramid(cost, PYRAMID_LEVELS, DOWNSAMPLE_RATIO, DOWNSAMPLE_METHOD)
    path_indices = multi_scale_lcp(
        cost_pyramid, start_rc, end_rc, water_mask, transform,
        resolution_m=RESOLUTION, dem=dem,
    )

    if not path_indices:
        log.error(f"Tile {idx}: Routing failed — empty path!")
        return TileResult(
            tile_index=idx,
            path_utm=[],
            entry_utm=(entry_x, entry_y),
            exit_utm=(exit_x, exit_y),
        )

    path_utm = [rowcol_to_xy(r, c, transform) for r, c in path_indices]

    log.info(f"Tile {idx}: Routed {len(path_utm)} waypoints.")

    # Free tile memory
    del cost, cost_pyramid, dem, slope_pct, building_penalty_map
    del water_mask, roads_mask, lulc_penalty_map
    gc.collect()

    return TileResult(
        tile_index=idx,
        path_utm=path_utm,
        entry_utm=(entry_x, entry_y),
        exit_utm=(exit_x, exit_y),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cross-Tile Path Stitching
# ══════════════════════════════════════════════════════════════════════════════

def stitch_tile_paths(tile_results):
    """
    Merge per-tile path segments into a single continuous path.

    For each pair of adjacent tiles, the overlap zone is where both paths
    have valid coverage. We find the best splice point by:
    1. Walking backward along Tile[i] path to find the overlap zone
    2. Walking forward along Tile[i+1] path to find the overlap zone
    3. Finding the point pair with minimum Euclidean distance
    4. Splicing: Tile[i] up to splice point, Tile[i+1] from splice point onward

    Returns: unified path_utm list of (easting, northing) tuples.
    """
    if len(tile_results) == 0:
        return []

    if len(tile_results) == 1:
        return tile_results[0].path_utm

    # Sort by tile index
    sorted_results = sorted(tile_results, key=lambda t: t.tile_index)

    # Start with the first tile's full path
    unified_path = list(sorted_results[0].path_utm)

    for i in range(1, len(sorted_results)):
        prev_result = sorted_results[i - 1]
        curr_result = sorted_results[i]

        if not curr_result.path_utm:
            log.warning(f"Tile {curr_result.tile_index}: Empty path — skipping.")
            continue

        prev_path = prev_result.path_utm
        curr_path = curr_result.path_utm

        # Find the splice point: closest approach between the tail of
        # the previous path and the head of the current path
        overlap_m = TILE_OVERLAP_KM * 1000.0

        # Search the last portion of prev_path and first portion of curr_path
        # Use distance from the exit point of prev_tile as the overlap marker
        exit_x, exit_y = prev_result.exit_utm

        # Find index in unified_path where we're within overlap zone of the exit
        splice_idx_prev = len(unified_path) - 1
        for j in range(len(unified_path) - 1, max(0, len(unified_path) - 5000), -1):
            px, py = unified_path[j]
            dist = math.hypot(px - exit_x, py - exit_y)
            if dist < overlap_m:
                splice_idx_prev = j
                break

        # Find index in curr_path where it enters past the overlap zone
        entry_x, entry_y = curr_result.entry_utm
        splice_idx_curr = 0
        for j in range(min(5000, len(curr_path))):
            px, py = curr_path[j]
            dist = math.hypot(px - entry_x, py - entry_y)
            if dist > overlap_m * 0.5:
                splice_idx_curr = j
                break

        # Now find the closest pair of points in the overlap zone
        # between unified_path[splice_idx_prev:] and curr_path[:splice_idx_curr+500]
        best_dist = float('inf')
        best_j_prev = splice_idx_prev
        best_j_curr = splice_idx_curr

        search_prev_start = max(0, splice_idx_prev - 500)
        search_curr_end = min(len(curr_path), splice_idx_curr + 500)

        # Sample every 10th point for efficiency
        for j_prev in range(search_prev_start, len(unified_path), 10):
            px, py = unified_path[j_prev]
            for j_curr in range(0, search_curr_end, 10):
                cx, cy = curr_path[j_curr]
                d = math.hypot(px - cx, py - cy)
                if d < best_dist:
                    best_dist = d
                    best_j_prev = j_prev
                    best_j_curr = j_curr

        # Refine around the best match
        for j_prev in range(max(0, best_j_prev - 15), min(len(unified_path), best_j_prev + 15)):
            px, py = unified_path[j_prev]
            for j_curr in range(max(0, best_j_curr - 15), min(len(curr_path), best_j_curr + 15)):
                cx, cy = curr_path[j_curr]
                d = math.hypot(px - cx, py - cy)
                if d < best_dist:
                    best_dist = d
                    best_j_prev = j_prev
                    best_j_curr = j_curr

        log.info(
            f"Tile stitch {i-1}→{i}: splice at unified[{best_j_prev}], "
            f"curr[{best_j_curr}], gap={best_dist:.1f} m"
        )

        # Splice: keep unified_path up to best_j_prev, then append curr_path from best_j_curr
        unified_path = unified_path[:best_j_prev + 1] + curr_path[best_j_curr:]

    log.info(f"Stitched path: {len(unified_path)} total waypoints from {len(sorted_results)} tiles.")
    return unified_path


# ══════════════════════════════════════════════════════════════════════════════
# 4. Tiled Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_tiled_pipeline(tiles, timer=None, ckpt=None):
    """
    Process all tiles sequentially and stitch results.

    Parameters
    ----------
    tiles : list of TileBBox
    timer : _StageTimer (optional)
    ckpt  : CheckpointManager (optional)

    Returns
    -------
    path_utm : list of (easting, northing) tuples — the unified route
    """
    log.info(f"╔═══ TILE ROUTING: {len(tiles)} tile(s) ═══╗")

    tile_results = []

    for tile in tiles:
        log.info(f"╠═ Tile {tile.tile_index + 1}/{len(tiles)} {'(first)' if tile.is_first else '(last)' if tile.is_last else ''} ═╣")

        if timer:
            timer.start(f"tile_{tile.tile_index}")

        result = _process_single_tile(tile, timer=timer)
        tile_results.append(result)

        if timer:
            timer.stop()

        if not result.path_utm:
            log.error(f"Tile {tile.tile_index}: Routing failed! Pipeline may produce incomplete results.")

    log.info("╠═ Stitching tile paths ═╣")
    unified_path = stitch_tile_paths(tile_results)

    log.info(f"╚═══ TILE ROUTING COMPLETE: {len(unified_path)} waypoints ═══╝")

    return unified_path
