"""
structures.py — Phase 8: Bridge and Culvert Inventory
=======================================================
Detects where the highway alignment crosses water bodies and small streams,
estimates bridge and culvert dimensions from geometry and DEM data, assigns
preliminary unit costs, and outputs a structure inventory for use in Phase 9
(parametric cost model) and Phase 10 (feasibility report).

Approach
--------
BRIDGES
  1. Intersect the smoothed UTM alignment LineString with the OSM water
     polygon / line GeoDataFrame to find crossing segments.
  2. Each continuous stretch of water crossing is one bridge.
  3. Bridge length  = arc length of route inside the water polygon (m).
  4. Bridge width   = perpendicular span across water body from DEM transect
                      (fallback: 1.3 × route-segment length).
  5. Deck elevation = FGL z_design at crossing mid-station + BRIDGE_FREEBOARD_M.
  6. Preliminary cost = BRIDGE_COST_PER_M_DECK × bridge_length ×
                        BRIDGE_WIDTH_FACTOR (accounts for deck area not just span).

CULVERTS
  1. Identify DEM low points along the vertical alignment where z_design is
     a local minimum and flow_accum > MIN_CULVERT_ACCUM_CELLS.
  2. Each qualifying low point that is NOT already a bridge is a culvert site.
  3. Culvert cost = CULVERT_UNIT_COST_USD (lump-sum per structure).

Public API
----------
    build_structure_inventory(
        smooth_utm, va_result, water_utm, flow_accum,
        transform, dem, utm_epsg, ...
    ) → StructureInventory
"""

from __future__ import annotations

import csv
import logging
import os
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("highway_alignment")

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Structure:
    """A single bridge or culvert along the alignment."""
    structure_id: int
    structure_type: str          # 'bridge' or 'culvert'
    segment_index: int           # The multi-waypoint leg index (0=A->B, 1=B->C)
    chainage_m: float            # mid-point chainage along alignment
    chainage_start_m: float      # start of structure
    chainage_end_m: float        # end of structure
    length_m: float              # span/crossing length (m)
    deck_elevation_m: float      # design deck elevation (m above MSL)
    freeboard_m: float           # headroom above HWL (bridges only)
    estimated_cost_usd: float    # preliminary cost estimate
    water_name: str              # OSM name if available, else 'unnamed'
    lon: float                   # WGS-84 longitude of mid-point
    lat: float                   # WGS-84 latitude of mid-point


@dataclass
class StructureInventory:
    """Full output of build_structure_inventory()."""
    structures: list[Structure]
    total_bridge_length_m: float
    total_bridge_cost_usd: float
    total_bridge_cost_usd: float
    bridge_count: int


# ── Step 0: Filter water features to bridge-worthy subset ─────────────────────

def filter_bridge_worthy_water(water_utm):
    """
    Filter water_utm GeoDataFrame to retain only features that could
    realistically require a bridge.

    LineString/MultiLineString waterways are buffered by 5.0m to convert 
    them into Polygon representations of the river channel.

    Exclusions (Myanmar low-quality OSM context):
      1. natural=wetland (seasonally saturated land, not a waterway).
      2. Non-crossable waterway tags (dams, weirs, fish ponds, docks, ditches).
      3. Features smaller than BRIDGE_MIN_WATER_AREA_M2.

    Retained features:
      - Polygons/MultiPolygons and LineStrings/MultiLineStrings
      - natural=water polygons (rivers, lakes, reservoirs)
      - Polygons with waterway in BRIDGE_WORTHY_WATERWAY_TAGS
    """
    if water_utm is None or len(water_utm) == 0:
        return water_utm

    try:
        from config import (
            BRIDGE_WORTHY_WATERWAY_TAGS, BRIDGE_EXCLUDE_WATERWAY_TAGS,
            BRIDGE_EXCLUDE_NATURAL_TAGS, BRIDGE_MIN_WATER_AREA_M2,
        )
    except ImportError:
        log.warning("Structures: could not import bridge filter config — using unfiltered water.")
        return water_utm

    n_before = len(water_utm)

    # 1. Keep Polygon/MultiPolygon AND LineString/MultiLineString geometries
    mask_geom = water_utm.geometry.geom_type.isin([
        "Polygon", "MultiPolygon", "LineString", "MultiLineString"
    ])

    # 2. Exclude natural=wetland
    mask_natural_ok = True  # default: keep
    if "natural" in water_utm.columns:
        nat_vals = water_utm["natural"].fillna("")
        mask_natural_ok = ~nat_vals.isin(BRIDGE_EXCLUDE_NATURAL_TAGS)

    # 3. Require bridge-worthy waterway tags, OR natural=water if it's explicitly a river.
    mask_water_ok = False
    if "waterway" in water_utm.columns:
        ww_vals = water_utm["waterway"].fillna("")
        mask_water_ok = ww_vals.isin(BRIDGE_WORTHY_WATERWAY_TAGS)
    
    # Also include natural=water if it's tagged as water=river or water=canal
    if not mask_water_ok is True and "water" in water_utm.columns:
        w_vals = water_utm["water"].fillna("")
        mask_water_ok = mask_water_ok | w_vals.isin(["river", "canal"])

    # If neither waterway nor water tags are present/matched, we still might have a generic large natural=water polygon (e.g., lake/reservoir).
    # We will let those through if they are large enough (checked in step 4), but the primary filter is now positive enforcement, not just negative exclusion.
    
    # Combined mask
    mask = mask_geom & mask_natural_ok & mask_water_ok
    filtered = water_utm[mask].copy()

    # Convert lines to polygons by buffering (simulate 10m-wide channel)
    if len(filtered) > 0:
        is_line = filtered.geometry.geom_type.isin(["LineString", "MultiLineString"])
        if is_line.any():
            filtered.loc[is_line, "geometry"] = filtered.loc[is_line, "geometry"].buffer(5.0)

    # 4. Minimum polygon area filter
    if len(filtered) > 0:
        areas = filtered.geometry.area
        filtered = filtered[areas >= BRIDGE_MIN_WATER_AREA_M2].copy()

    n_after = len(filtered)
    log.info(
        f"Structures: bridge-worthy water filter: {n_before} → {n_after} features "
        f"(buffered lines, removed wetlands/dams/small ponds)"
    )
    return filtered.reset_index(drop=True)


# ── Step 0b: Crossing angle validation ────────────────────────────────────────

def _validate_crossing(route_ls, seg, water_geom, mid_pt, local_width_m=None):
    """
    Validate if a crossing is a true transverse crossing rather than
    the route skimming the bank or running longitudinally inside the river.
    
    Uses ratio between crossing span (seg.length) and local river width if available.
    If span > 3 * local_width, it's a longitudinal run, not a crossing.

    Returns (is_valid, local_width, span_length)
    """
    span_len = seg.length
    if span_len < 1.0:
        return False, 0.0, span_len

    # Find the midpoint of the route segment inside the water
    mid_pt = seg.interpolate(0.5, normalized=True)
    
    # If we already have a raster-derived local width, use it
    if local_width_m is not None and local_width_m > 0:
        local_width = local_width_m
    else:
        # Calculate route direction vector at midpoint
        eps = min(5.0, span_len / 4.0)
        try:
            mid_s = route_ls.project(mid_pt)
            p1 = route_ls.interpolate(max(0.0, mid_s - eps))
            p2 = route_ls.interpolate(min(route_ls.length, mid_s + eps))
            dr = p2.x - p1.x
            dc = p2.y - p1.y
            norm = math.hypot(dr, dc)
            if norm < 1e-6:
                return True, span_len, span_len
                
            dx, dy = dr/norm, dc/norm
            
            # Perpendicular vector to measure river width
            px, py = -dy, dx
            
            # Measure local river width by throwing a transect line across the midpoint
            from shapely.geometry import LineString
            t_start = (mid_pt.x - px * 1000.0, mid_pt.y - py * 1000.0)
            t_end   = (mid_pt.x + px * 1000.0, mid_pt.y + py * 1000.0)
            transect = LineString([t_start, t_end])
            
            width_inter = transect.intersection(water_geom)
            local_width = width_inter.length if not width_inter.is_empty else span_len
            
        except Exception as e:
            log.debug(f"Structures: crossing validation exception: {e}")
            return True, span_len, span_len

    # If the route span is more than 3x the local width, it's running parallel/longitudinally
    if span_len > max(local_width * 3.0, 50.0):
        return False, local_width, span_len
        
    return True, local_width, span_len


# ── Step 1: Find water crossings using route–water intersection ───────────────

def _find_water_crossings(smooth_utm, water_utm, va_result, water_mask=None, transform=None, resolution_m=30.0):
    """
    Intersect the alignment LineString with bridge-worthy water body polygons.

    Pre-filters water_utm to real rivers/canals only (excludes wetlands,
    LineString waterways, dams, fish ponds, tiny polygons).
    Validates crossing angle to reject false positives from parallel overlaps.

    Returns list of dicts:
        {start_m, end_m, mid_m, length_m, water_name}
    where x_m are chainages along the alignment.
    """
    if water_utm is None:
        log.info("Structures: no water data — skipping bridge detection.")
        return []
    try:
        n_water = len(water_utm)
    except TypeError:
        n_water = 0
    if n_water == 0:
        log.info("Structures: water layer is empty — skipping bridge detection.")
        return []

    # Phase 8b: filter to bridge-worthy water features (real rivers only)
    water_utm = filter_bridge_worthy_water(water_utm)
    if water_utm is None or len(water_utm) == 0:
        log.info("Structures: no bridge-worthy water after filtering — 0 bridges.")
        return []

    try:
        from shapely.geometry import LineString as SLine
        from shapely.ops import unary_union
    except ImportError:
        log.warning("Structures: shapely not available — skipping bridge detection.")
        return []

    # Build route as Shapely LineString in UTM
    route_ls = SLine([(x, y) for x, y in smooth_utm])
    route_len = route_ls.length

    # Optional: Calculate distance transform on water mask for robust width detection
    edt_width_m = None
    if water_mask is not None and transform is not None:
        try:
            from scipy.ndimage import distance_transform_edt
            binary = (water_mask > 0).astype(np.uint8)
            edt = distance_transform_edt(binary)
            edt_width_m = edt * 2.0 * resolution_m
            log.info("Structures: Computed raster-based hydraulic width map for crossings.")
        except Exception as e:
            log.warning(f"Structures: Failed to compute water distance transform: {e}")

    # Helper: chainage of a point on the route (projected distance)
    def pt_to_chainage(pt):
        frac = route_ls.project(pt) / route_len
        return float(frac * va_result.distances_m[-1])

    def pt_to_width(pt):
        """Sample the EDT width raster at a given coordinate."""
        if edt_width_m is None or transform is None:
            return None
        col, row = ~transform * (pt.x, pt.y)
        col, row = int(col), int(row)
        if 0 <= row < edt_width_m.shape[0] and 0 <= col < edt_width_m.shape[1]:
            val = edt_width_m[row, col]
            return float(val) if val > 0 else None
        return None

    crossings = []
    for _, row in water_utm.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            inter = route_ls.intersection(geom)
        except Exception:
            continue
        if inter.is_empty:
            continue

        # Collect sub-segments of route inside the water body
        from shapely.geometry import MultiLineString, GeometryCollection
        segs = []
        if inter.geom_type == "LineString":
            segs = [inter]
        elif inter.geom_type in ("MultiLineString", "GeometryCollection"):
            segs = [g for g in inter.geoms if g.geom_type == "LineString"]

        # Phase 8b: import crossing angle threshold
        try:
            from config import BRIDGE_MIN_CROSSING_ANGLE_DEG
        except ImportError:
            BRIDGE_MIN_CROSSING_ANGLE_DEG = 30.0  # Increased to prevent false positives

        for seg in segs:
            if seg.length < 1.0:
                continue

            mid_pt = seg.interpolate(0.5, normalized=True)
            local_width_m = pt_to_width(mid_pt)

            # Phase 8b: crossing validation (replaces angle check)
            is_valid, local_width, span = _validate_crossing(route_ls, seg, geom, mid_pt, local_width_m=local_width_m)
            if not is_valid:
                log.debug(
                    f"Structures: rejected crossing (span {span:.1f}m > 3x width {local_width:.1f}m "
                    f"— route parallel to water body)"
                )
                continue

            start_pt = seg.interpolate(0)
            end_pt   = seg.interpolate(seg.length)
            s_start  = pt_to_chainage(start_pt)
            s_end    = pt_to_chainage(end_pt)
            if s_end < s_start:
                s_start, s_end = s_end, s_start
            span = s_end - s_start
            
            # Use raster width if it's much larger than intersecting span (e.g. LineStrings buffered to 10m)
            reported_span = span
            if local_width_m is not None and local_width_m > min(span * 1.5, span + 10.0):
                reported_span = local_width_m

            if reported_span < 5.0:  # Ignore tiny clips on the edge
                continue
            crossings.append({
                "start_m":    s_start,
                "end_m":      s_end,
                "mid_m":      (s_start + s_end) / 2.0,
                "length_m":   reported_span,
                "water_name": str(row.get("name", "unnamed")) if hasattr(row, "get") else "unnamed",
            })

    # Merge crossings that are very close (within 50 m)
    crossings.sort(key=lambda c: c["start_m"])
    merged = []
    for c in crossings:
        if merged and c["start_m"] - merged[-1]["end_m"] < 50.0:
            prev = merged[-1]
            prev["end_m"]    = max(prev["end_m"], c["end_m"])
            # Preserve raster-derived widths if they were larger
            prev["length_m"] = max(prev["length_m"], c["length_m"], prev["end_m"] - prev["start_m"])
            prev["mid_m"]    = (prev["start_m"] + prev["end_m"]) / 2.0
            if prev["water_name"] == "unnamed":
                prev["water_name"] = c["water_name"]
        else:
            merged.append(dict(c))

    log.info(f"Structures: {len(merged)} bridge crossing(s) detected.")
    return merged


# ── Step 2: Find culvert sites using flow accumulation + vertical low points ─

def _find_culvert_sites(va_result, flow_accum, transform,
                        path_indices,
                        bridge_chainages: set,
                        bridge_ranges: list | None = None,
                        min_accum_cells: int = 200,
                        min_spacing_m: float = 100.0):
    """
    Identify culvert locations as local vertical-alignment minima that:
      a) have D8 flow accumulation > min_accum_cells (real drainage), AND
      b) are NOT already inside a bridge crossing span.

    Fix 13: bridge_ranges is a list of (start_m, end_m) tuples; culverts inside
    any bridge span [start_m-20, end_m+20] are suppressed.

    Returns list of dicts: {chainage_m, flow_accum_cells}
    """
    if flow_accum is None:
        log.info("Structures: no flow_accum — skipping culvert detection.")
        return []

    dists = va_result.distances_m
    z     = va_result.z_design
    n     = len(dists)
    if n < 3 or len(path_indices) == 0:
        return []

    # Detect local minima in z_design (sag points where water accumulates)
    is_min = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if z[i] < z[i - 1] and z[i] < z[i + 1]:
            is_min[i] = True

    fa_rows, fa_cols = flow_accum.shape
    n_path = len(path_indices)

    culverts = []
    last_chainage = -999.0

    for i in range(n):
        if not is_min[i]:
            continue
        s = float(dists[i])

        # Skip if too close to previous culvert
        if s - last_chainage < min_spacing_m:
            continue

        # Fix 13: Skip if inside any bridge span interval (not just midpoint proximity)
        s_inside_bridge = False
        if bridge_ranges:
            for (br_start, br_end) in bridge_ranges:
                if br_start - 20.0 <= s <= br_end + 20.0:
                    s_inside_bridge = True
                    break
        if s_inside_bridge:
            continue
        # Legacy midpoint check (kept for scenarios without bridge_ranges)
        if not bridge_ranges and any(abs(s - bc) < 20.0 for bc in bridge_chainages):
            continue

        # Map station fraction → path_indices index
        frac = s / dists[-1] if dists[-1] > 0 else 0.0
        pi   = int(frac * (n_path - 1))
        pi   = max(0, min(n_path - 1, pi))
        pr, pc = path_indices[pi]
        pr = max(0, min(fa_rows - 1, int(pr)))
        pc = max(0, min(fa_cols - 1, int(pc)))

        accum = int(flow_accum[pr, pc])
        if accum < min_accum_cells:
            continue

        culverts.append({"chainage_m": s, "flow_accum_cells": accum})
        last_chainage = s

    log.info(f"Structures: {len(culverts)} culvert site(s) detected "
             f"(min_accum={min_accum_cells} cells).")
    return culverts


# ── Step 3: Interpolate z_design at a chainage ────────────────────────────────

def _z_at(chainage_m: float, va_result) -> float:
    """Linear interpolation of z_design at a given chainage."""
    d = va_result.distances_m
    z = va_result.z_design
    if chainage_m <= d[0]:
        return float(z[0])
    if chainage_m >= d[-1]:
        return float(z[-1])
    idx = np.searchsorted(d, chainage_m)
    idx = max(1, min(len(d) - 1, idx))
    t   = (chainage_m - d[idx - 1]) / (d[idx] - d[idx - 1])
    return float(z[idx - 1] + t * (z[idx] - z[idx - 1]))


# ── Step 4: Convert UTM mid-point to WGS-84 ──────────────────────────────────

def _utm_chainage_to_wgs84(chainage_m: float, smooth_utm, va_result):
    """
    Return WGS-84 (lon, lat) for a given chainage along the alignment.
    Uses linear interpolation along smooth_utm coordinates.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        return (0.0, 0.0)

    d    = va_result.distances_m
    frac = chainage_m / d[-1] if d[-1] > 0 else 0.0
    frac = max(0.0, min(1.0, frac))
    idx  = frac * (len(smooth_utm) - 1)
    i0   = int(idx)
    i1   = min(i0 + 1, len(smooth_utm) - 1)
    t    = idx - i0
    x    = smooth_utm[i0][0] + t * (smooth_utm[i1][0] - smooth_utm[i0][0])
    y    = smooth_utm[i0][1] + t * (smooth_utm[i1][1] - smooth_utm[i0][1])

    try:
        from geometry_utils import utm_to_wgs84
        lon, lat = utm_to_wgs84(x, y)
        return (lon, lat)
    except Exception:
        return (0.0, 0.0)


# ── CSV export ────────────────────────────────────────────────────────────────

def export_structures_csv(inventory: StructureInventory, output_path: str) -> None:
    """Write structure list to CSV."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "chainage_m",
            "chainage_start_m", "chainage_end_m",
            "length_m", "deck_elev_m", "freeboard_m",
            "cost_usd", "water_name", "lon", "lat",
        ])
        for s in inventory.structures:
            writer.writerow([
                s.structure_id,
                round(s.chainage_m, 1),
                round(s.chainage_start_m, 1),
                round(s.chainage_end_m, 1),
                round(s.length_m, 1),
                round(s.deck_elevation_m, 1),
                round(s.freeboard_m, 1),
                round(s.estimated_cost_usd, 0),
                s.water_name,
                round(s.lon, 6),
                round(s.lat, 6),
            ])
    log.info(f"Structures CSV exported: {output_path}  ({len(inventory.structures)} structures)")


# ── Public entry point ────────────────────────────────────────────────────────

def build_structure_inventory(
    smooth_utm,
    va_result,
    water_utm,
    water_mask=None,
    flow_accum=None,
    transform=None,
    path_indices=None,
    smooth_segment_indices=None,
    bridge_freeboard_m: float     = 1.5,
    bridge_cost_per_m2_usd: float = 3_500.0,
    bridge_width_m: float         = 12.0,
    resolution_m: float           = 30.0,
) -> StructureInventory:
    """
    Detect bridges along the 3D alignment and estimate costs.

    Parameters
    ----------
    smooth_utm : list of (x, y) tuples
        Smoothed alignment in UTM coordinates.
    va_result : VerticalAlignmentResult
        Phase 6 result (distances_m, z_design, cut_fill_m).
    water_utm : GeoDataFrame or None
        OSM water bodies in UTM projection.
    flow_accum : ndarray or None
        D8 flow accumulation raster (cells × upstream area).
    transform : rasterio.Affine
        Geotransform for the DEM/flow_accum grid.
    path_indices : list of (row, col)
        Raw raster path from routing.
    bridge_freeboard_m : float
        Clearance above design high water level (m). Myanmar DRD uses 1.5 m.
    bridge_cost_per_m2_usd : float
        Composite unit rate for superstructure + substructure (USD/m²).
        World Bank Myanmar roads sector: USD 3,000–4,500/m². Default 3,500.
    bridge_width_m : float
        Assumed bridge deck width (m) = carriageway + parapets.
        For rural_trunk 2-lane: 11 m + 1 m parapets = 12 m.

    Returns
    -------
    StructureInventory
    """
    # Fetch config parameters inside the function to avoid circular imports during testing
    try:
        from config import BRIDGE_BANK_SETBACK_M 
    except ImportError:
        BRIDGE_BANK_SETBACK_M = 15.0

    # Step 1 — bridges
    crossings = _find_water_crossings(
        smooth_utm, water_utm, va_result,
        water_mask=water_mask, transform=transform, resolution_m=resolution_m
    )

    structures: list[Structure] = []
    # Fix 13: Store [start_m, end_m] ranges instead of only midpoints so that
    # _find_culvert_sites can suppress culverts anywhere inside a bridge span,
    # not just within 20 m of the midpoint (which misses long bridge approaches).
    bridge_chainages_set: set[float] = set()      # midpoints (kept for legacy compat)
    bridge_ranges: list[tuple[float, float]] = [] # (start_m, end_m) inclusive

    # Extract minimum bridge length from WATER_PENALTY_TIERS via config if possible, else default 10m
    try:
        from config import BRIDGE_MIN_SPAN_M
    except ImportError:
        BRIDGE_MIN_SPAN_M = 10.0

    for i, cr in enumerate(crossings):
        mid_s     = cr["mid_m"]
        # Freeboard is enforced in Phase 6 z_design. Deck elevation IS z_design.
        deck_elev = _z_at(mid_s, va_result) 
        lon, lat  = _utm_chainage_to_wgs84(mid_s, smooth_utm, va_result)

        seg_idx = 0
        if smooth_segment_indices is not None and va_result is not None:
            s_idx = np.searchsorted(va_result.distances_m, mid_s)
            s_idx = min(len(va_result.distances_m) - 1, s_idx)
            seg_idx = smooth_segment_indices[s_idx]
        
        # Re-classify minor crossings as culverts based on RAW span
        if cr["length_m"] < BRIDGE_MIN_SPAN_M:
            structures.append(Structure(
                structure_id      = i + 1,
                structure_type    = "culvert", # Major Box Culvert
                segment_index     = seg_idx,
                chainage_m        = mid_s,
                chainage_start_m  = mid_s, # Culverts don't have span representations in output
                chainage_end_m    = mid_s,
                length_m          = cr["length_m"], # Setbacks don't apply to culverts
                deck_elevation_m  = deck_elev,
                freeboard_m       = 0.0,
                estimated_cost_usd= CULVERT_MAJOR_COST_USD,
                water_name        = cr["water_name"],
                lon               = lon,
                lat               = lat,
            ))
            bridge_chainages_set.add(mid_s)  # Still prevent another culvert here
            bridge_ranges.append((mid_s, mid_s))  # zero-span entry for this culvert
            continue

        # If it's a bridge, calculate total length including bank setbacks
        total_bridge_len_m = cr["length_m"] + (2 * BRIDGE_BANK_SETBACK_M)

        cost      = bridge_cost_per_m2_usd * total_bridge_len_m * bridge_width_m
        
        structures.append(Structure(
            structure_id      = i + 1,
            structure_type    = "bridge",
            segment_index     = seg_idx,
            chainage_m        = mid_s,
            chainage_start_m  = cr["start_m"],
            chainage_end_m    = cr["end_m"],
            length_m          = total_bridge_len_m,
            deck_elevation_m  = deck_elev,
            freeboard_m       = bridge_freeboard_m,
            estimated_cost_usd= cost,
            water_name        = cr["water_name"],
            lon               = lon,
            lat               = lat,
        ))
        bridge_chainages_set.add(mid_s)
        bridge_ranges.append((cr["start_m"], cr["end_m"]))

    bridge_cnt = len(structures)

    structures.sort(key=lambda s: s.chainage_m)

    total_bridge_len  = sum(s.length_m for s in structures if s.structure_type == "bridge")
    total_bridge_cost = sum(s.estimated_cost_usd for s in structures if s.structure_type == "bridge")

    log.info(
        f"Structure inventory: "
        f"{bridge_cnt} bridge(s) ({total_bridge_len:.0f} m total span, "
        f"USD {total_bridge_cost/1e6:.2f} M)"
    )

    return StructureInventory(
        structures             = structures,
        total_bridge_length_m  = total_bridge_len,
        total_bridge_cost_usd  = total_bridge_cost,
        bridge_count           = bridge_cnt,
    )
