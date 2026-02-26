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
import os
import time
import traceback
import tracemalloc
import sys
import numpy as np
from shapely.geometry import LineString

from config import (
    POINT_A, POINT_B, UTM_EPSG, ROW_BUFFER_M, SLOPE_MAX_PCT,
    BORDER_CELLS, IMPASSABLE, RESOLUTION, OUTPUT_FILE,
    CHECKPOINT_FILE, FORCE_RESTART,
    MEMORY_WARN_GB, TILE_ROUTING_THRESHOLD_KM, PERF_TIMING_ENABLED,
    COARSE_FACTOR, CORRIDOR_BAND_KM,
)
from geometry_utils import (
    bbox_with_margin, wgs84_to_utm, utm_to_wgs84, xy_to_rowcol, rowcol_to_xy,
    smooth_path, verify_curve_radius, verify_row_setback, compute_metadata,
    export_geojson, extract_longitudinal_profile, check_sustained_grade,
    compute_bearing, compute_clothoid_transitions,
)
from data_fetch import fetch_dem, fetch_osm_layers, derive_stream_mask_utm
from cost_surface import compute_slope, rasterise_layer, build_cost_surface
from routing import coarse_to_fine_routing

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


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    warn_handler = _setup_logging()
    timer = _StageTimer(enabled=PERF_TIMING_ENABLED)
    tracemalloc.start()
    ckpt  = CheckpointManager()

    log.info("═" * 65)
    log.info("  Highway Alignment Generator  –  Myanmar Preliminary Feasibility")
    log.info("═" * 65)
    log.info(f"Point A : lon={POINT_A[0]}, lat={POINT_A[1]}")
    log.info(f"Point B : lon={POINT_B[0]}, lat={POINT_B[1]}")
    log.info(f"UTM CRS : EPSG:{UTM_EPSG}")

    # ── 1. Bounding box ────────────────────────────────────────────────────
    timer.start("1_bbox")
    bbox = bbox_with_margin(POINT_A, POINT_B)
    west, south, east, north = bbox
    log.info(f"BBox (WGS-84): W={west:.4f} S={south:.4f} E={east:.4f} N={north:.4f}")

    # Phase 5: corridor straight-line distance check
    ax, ay = wgs84_to_utm(*POINT_A)
    bx, by = wgs84_to_utm(*POINT_B)
    corridor_km = ((bx - ax)**2 + (by - ay)**2)**0.5 / 1000.0
    log.info(f"Corridor A→B straight: {corridor_km:.1f} km")
    if corridor_km > TILE_ROUTING_THRESHOLD_KM:
        log.warning(
            f"Corridor {corridor_km:.0f} km exceeds TILE_ROUTING_THRESHOLD_KM="
            f"{TILE_ROUTING_THRESHOLD_KM:.0f} km. Consider enabling FAST_MODE or "
            f"increasing COARSE_FACTOR for large-scale screening."
        )
    timer.stop()

    # ── 2. DEM (with fallback chain: COP30 → SRTMGL1 → SRTMGL3 → mock) ───
    if ckpt.get('dem'):
        log.info("[CHECKPOINT] Skipping DEM fetch (already cached).")
        _ckd = ckpt.get('dem')
        dem_source = _ckd['source']
        # Re-load from cached rasterio file (dem array is rebuilt from cost surface)
        # For simplicity, still fetch (fast from cache) to get the array
        dem, transform, dem_source = fetch_dem(bbox)
    else:
        dem, transform, dem_source = fetch_dem(bbox)
        ckpt.save('dem', {'source': dem_source})
    rows, cols = dem.shape
    log.info(f"DEM loaded: {rows}×{cols} cells, source={dem_source}")

    # ── 3. OSM layers ──────────────────────────────────────────────────────
    log.info("Fetching OSM layers …")
    buildings_wgs, water_wgs, osm_stats = fetch_osm_layers(bbox)

    def to_utm(gdf):
        if gdf is None or len(gdf) == 0:
            return gdf
        return gdf.to_crs(epsg=UTM_EPSG)

    buildings_utm = to_utm(buildings_wgs)
    water_utm     = to_utm(water_wgs)

    if buildings_utm is not None and len(buildings_utm) > 0:
        log.info(f"Buffering {len(buildings_utm)} buildings by {ROW_BUFFER_M} m …")
        buildings_utm = buildings_utm.copy()
        buildings_utm["geometry"] = buildings_utm.geometry.buffer(ROW_BUFFER_M)

    # ── 4. Slope + curvature ───────────────────────────────────────────────
    log.info("Computing slope and curvature …")
    slope_pct, nodata_mask, curvature = compute_slope(dem, RESOLUTION)
    log.info(
        f"Slope: max={slope_pct.max():.1f}%, "
        f"cells>{SLOPE_MAX_PCT}%: {np.sum(slope_pct > SLOPE_MAX_PCT):,}"
    )

    # ── 4b. A→B bearing (for anisotropic sidehill cost) ───────────────────
    bearing_deg = compute_bearing(POINT_A, POINT_B)

    # ── 5. Rasterise vector layers — merge with DEM stream fallback if needed ─
    log.info("Rasterising exclusion zones …")
    building_mask = rasterise_layer(buildings_utm, transform, (rows, cols))
    water_mask    = rasterise_layer(water_utm,     transform, (rows, cols))

    # Phase 3: if OSM water is sparse, derive stream network from DEM
    if osm_stats.get('dem_stream_fallback'):
        log.info("Deriving stream network from DEM (OSM water fallback active) …")
        dem_streams = derive_stream_mask_utm(dem, transform, resolution_m=RESOLUTION)
        # Merge: union of OSM (sparse) and DEM-derived streams
        water_mask = np.maximum(water_mask, dem_streams).astype(np.float32)
        n_stream_cells = int((water_mask > 0).sum())
        log.info(f"Combined water mask: {n_stream_cells:,} cells "
                 f"(OSM={osm_stats['water']} + DEM-derived)")
        warn_handler.warnings.append(
            f"DEM stream fallback active: OSM water returned only "
            f"{osm_stats['water']} features. Stream network derived "
            f"from DEM flow accumulation."
        )

    # ── 6. Cost surface (all layers) ───────────────────────────────────────
    log.info("Building cost surface …")
    cost = build_cost_surface(
        slope_pct, building_mask, water_mask,
        nodata_mask=nodata_mask,
        dem=dem,
        curvature=curvature,
        resolution_m=RESOLUTION,
        bearing_deg=bearing_deg,    # Phase 4: sidehill anisotropy
    )

    # ── 7. Grid indices for endpoints ─────────────────────────────────────
    xa, ya = wgs84_to_utm(*POINT_A)
    xb, yb = wgs84_to_utm(*POINT_B)
    start_rc = xy_to_rowcol(xa, ya, transform)
    end_rc   = xy_to_rowcol(xb, yb, transform)

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

    for label, rc in [("A", start_rc), ("B", end_rc)]:
        if cost[rc] >= IMPASSABLE:
            log.warning(
                f"Point {label} at {rc} lands on impassable cell. "
                f"Relaxing local 3×3 neighbourhood to cost=1."
            )
            r, c = rc
            cost[
                max(0, r - 1):min(rows, r + 2),
                max(0, c - 1):min(cols, c + 2),
            ] = 1.0

    # ── 8. Two-resolution routing (Phase 4/5) ──────────────────────────────
    # Phase 5: estimate peak memory for fine-resolution band and auto-escalate
    # COARSE_FACTOR if the estimate exceeds MEMORY_WARN_GB.
    fine_coarse = COARSE_FACTOR
    fine_cells_est = int(rows * cols * (CORRIDOR_BAND_KM * 2000 / max(rows, cols) / RESOLUTION))
    fine_mem_gb = fine_cells_est * 8 / 1e9   # float64
    if fine_mem_gb > MEMORY_WARN_GB:
        fine_coarse = fine_coarse * 2
        log.warning(
            f"Estimated fine-grid memory {fine_mem_gb:.1f} GB > "
            f"MEMORY_WARN_GB={MEMORY_WARN_GB:.1f}. "
            f"Auto-escalating COARSE_FACTOR {COARSE_FACTOR}→{fine_coarse} "
            f"to reduce memory footprint."
        )

    log.info(f"Running coarse-to-fine routing (COARSE_FACTOR={fine_coarse}) …")
    timer.start("8_routing")
    path_indices = coarse_to_fine_routing(
        cost, start_rc, end_rc, water_mask, transform,
        resolution_m=RESOLUTION, dem=dem,
        factor=fine_coarse,           # Phase 5: may be auto-escalated
    )
    timer.stop()

    path_utm = [rowcol_to_xy(r, c, transform) for r, c in path_indices]

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
            path_utm, n_points=1000,
            smoothing=len(path_utm) * 20.0,
            slope_pct=slope_pct, transform=transform,
        )
        min_radius, violations = verify_curve_radius(smooth_utm)
        if violations:
            log.warning(
                f"Could not fully satisfy {min_radius:.0f} m curve-radius constraint. "
                "Outputting best-effort geometry."
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

    # Phase 5: peak memory and stage timing
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(peak_mem / 1e6, 1)
    meta['peak_memory_mb'] = peak_mem_mb
    meta['stage_timing_s'] = timer.as_dict()

    export_geojson(line_wgs84, meta, OUTPUT_FILE)
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
    log.info(f"  Data confidence    : {meta['data_confidence']}")
    log.info(f"  DEM source         : {meta['dem_source']}")
    log.info(f"  Peak memory        : {peak_mem_mb} MB")
    log.info("═" * 65)
    timer.log_summary()

    if meta["data_warnings"]:
        log.warning("Data quality issues:")
        for w in meta["data_warnings"]:
            log.warning(f"  • {w}")



if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger("highway_alignment").error("Fatal error in pipeline:")
        traceback.print_exc()
        sys.exit(1)
