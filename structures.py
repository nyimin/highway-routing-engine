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
    chainage_m: float            # mid-point chainage along alignment
    chainage_start_m: float      # start of structure (bridges only, else = chainage)
    chainage_end_m: float        # end of structure (bridges only, else = chainage)
    length_m: float              # span/crossing length (m)
    deck_elevation_m: float      # design deck elevation (m above MSL); 0 for culverts
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
    total_culvert_cost_usd: float
    total_structure_cost_usd: float
    bridge_count: int
    culvert_count: int


# ── Step 1: Find water crossings using route–water intersection ───────────────

def _find_water_crossings(smooth_utm, water_utm, va_result):
    """
    Intersect the alignment LineString with water body geometries.

    Returns list of dicts:
        {start_m, end_m, mid_m, length_m, water_name}
    where x_m are chainages along the alignment.

    Falls back to an empty list if Shapely or GeoPandas are unavailable
    or if water_utm is None/empty.
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

    try:
        from shapely.geometry import LineString as SLine
        from shapely.ops import unary_union
    except ImportError:
        log.warning("Structures: shapely not available — skipping bridge detection.")
        return []

    # Build route as Shapely LineString in UTM
    route_ls = SLine([(x, y) for x, y in smooth_utm])
    route_len = route_ls.length

    # Helper: chainage of a point on the route (projected distance)
    def pt_to_chainage(pt):
        frac = route_ls.project(pt) / route_len
        return float(frac * va_result.distances_m[-1])

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

        for seg in segs:
            if seg.length < 1.0:
                continue
            start_pt = seg.interpolate(0)
            end_pt   = seg.interpolate(seg.length)
            s_start  = pt_to_chainage(start_pt)
            s_end    = pt_to_chainage(end_pt)
            if s_end < s_start:
                s_start, s_end = s_end, s_start
            span = s_end - s_start
            if span < 2.0:
                continue
            crossings.append({
                "start_m":    s_start,
                "end_m":      s_end,
                "mid_m":      (s_start + s_end) / 2.0,
                "length_m":   span,
                "water_name": str(row.get("name", "unnamed")) if hasattr(row, "get") else "unnamed",
            })

    # Merge crossings that are very close (within 50 m)
    crossings.sort(key=lambda c: c["start_m"])
    merged = []
    for c in crossings:
        if merged and c["start_m"] - merged[-1]["end_m"] < 50.0:
            prev = merged[-1]
            prev["end_m"]    = max(prev["end_m"], c["end_m"])
            prev["length_m"] = prev["end_m"] - prev["start_m"]
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
                        min_accum_cells: int = 200,
                        min_spacing_m: float = 100.0):
    """
    Identify culvert locations as local vertical-alignment minima that:
      a) have D8 flow accumulation > min_accum_cells (real drainage), AND
      b) are NOT already inside a bridge crossing.

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

        # Skip if inside a bridge
        if any(abs(s - bc) < 20.0 for bc in bridge_chainages):
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
            "id", "type", "chainage_m",
            "chainage_start_m", "chainage_end_m",
            "length_m", "deck_elev_m", "freeboard_m",
            "cost_usd", "water_name", "lon", "lat",
        ])
        for s in inventory.structures:
            writer.writerow([
                s.structure_id, s.structure_type,
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
    flow_accum,
    transform,
    path_indices,
    bridge_freeboard_m: float     = 1.5,
    bridge_cost_per_m2_usd: float = 3_500.0,
    bridge_width_m: float         = 12.0,
    culvert_unit_cost_usd: float  = 15_000.0,
    min_culvert_accum_cells: int  = 200,
) -> StructureInventory:
    """
    Detect bridges and culverts along the 3D alignment and estimate costs.

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
    culvert_unit_cost_usd : float
        Lump sum per box culvert or pipe culvert (USD). Default 15,000.
    min_culvert_accum_cells : int
        Minimum upstream cells for culvert siting. Cells × RESOLUTION² ≈ drainage
        area. Default 200 → at 30 m resolution ≈ 0.18 km² catchment.

    Returns
    -------
    StructureInventory
    """
    # Step 1 — bridges
    crossings = _find_water_crossings(smooth_utm, water_utm, va_result)

    structures: list[Structure] = []
    bridge_mid_chainages = set()

    for i, cr in enumerate(crossings):
        mid_s     = cr["mid_m"]
        deck_elev = _z_at(mid_s, va_result) + bridge_freeboard_m
        cost      = bridge_cost_per_m2_usd * cr["length_m"] * bridge_width_m
        lon, lat  = _utm_chainage_to_wgs84(mid_s, smooth_utm, va_result)

        structures.append(Structure(
            structure_id      = i + 1,
            structure_type    = "bridge",
            chainage_m        = mid_s,
            chainage_start_m  = cr["start_m"],
            chainage_end_m    = cr["end_m"],
            length_m          = cr["length_m"],
            deck_elevation_m  = deck_elev,
            freeboard_m       = bridge_freeboard_m,
            estimated_cost_usd= cost,
            water_name        = cr["water_name"],
            lon               = lon,
            lat               = lat,
        ))
        bridge_mid_chainages.add(mid_s)

    bridge_cnt = len(structures)

    # Step 2 — culverts
    culvert_sites = _find_culvert_sites(
        va_result, flow_accum, transform, path_indices,
        bridge_chainages=bridge_mid_chainages,
        min_accum_cells=min_culvert_accum_cells,
    )

    for j, site in enumerate(culvert_sites):
        s        = site["chainage_m"]
        lon, lat = _utm_chainage_to_wgs84(s, smooth_utm, va_result)
        structures.append(Structure(
            structure_id      = bridge_cnt + j + 1,
            structure_type    = "culvert",
            chainage_m        = s,
            chainage_start_m  = s,
            chainage_end_m    = s,
            length_m          = 0.0,
            deck_elevation_m  = _z_at(s, va_result),
            freeboard_m       = 0.0,
            estimated_cost_usd= culvert_unit_cost_usd,
            water_name        = "drainage",
            lon               = lon,
            lat               = lat,
        ))

    structures.sort(key=lambda s: s.chainage_m)

    total_bridge_len  = sum(s.length_m for s in structures if s.structure_type == "bridge")
    total_bridge_cost = sum(s.estimated_cost_usd for s in structures if s.structure_type == "bridge")
    total_culv_cost   = sum(s.estimated_cost_usd for s in structures if s.structure_type == "culvert")

    log.info(
        f"Structure inventory: "
        f"{bridge_cnt} bridge(s) ({total_bridge_len:.0f} m total span, "
        f"USD {total_bridge_cost/1e6:.2f} M)  "
        f"{len(culvert_sites)} culvert(s) (USD {total_culv_cost/1e3:.0f} K)"
    )

    return StructureInventory(
        structures             = structures,
        total_bridge_length_m  = total_bridge_len,
        total_bridge_cost_usd  = total_bridge_cost,
        total_culvert_cost_usd = total_culv_cost,
        total_structure_cost_usd = total_bridge_cost + total_culv_cost,
        bridge_count           = bridge_cnt,
        culvert_count          = len(culvert_sites),
    )
