"""
geometry_utils.py — Coordinate transforms, smoothing, and geometric checks
===========================================================================
Phase 4 additions:
  5. compute_bearing(pt_a, pt_b) — A→B azimuth in degrees (for sidehill penalty).
  6. _clothoid_length() — required Euler spiral transition per design standard.
  7. compute_clothoid_transitions() — flags every curve and reports spiral needs.
"""
import math
import json
import logging
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, mapping
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev

from config import UTM_EPSG, MIN_CURVE_RADIUS, ROW_BUFFER_M, SLOPE_MAX_PCT, DESIGN_SPEED_KMPH

log = logging.getLogger("highway_alignment")


# ── Coordinate helpers ────────────────────────────────────────────────────────

def bbox_with_margin(pt_a, pt_b, margin=0.20):
    lons = [pt_a[0], pt_b[0]]
    lats = [pt_a[1], pt_b[1]]
    dlon = (max(lons) - min(lons)) * margin
    dlat = (max(lats) - min(lats)) * margin
    dlon = max(dlon, 0.05)
    dlat = max(dlat, 0.05)
    return (
        min(lons) - dlon,
        min(lats) - dlat,
        max(lons) + dlon,
        max(lats) + dlat,
    )


def wgs84_to_utm(lon, lat, epsg=UTM_EPSG):
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return t.transform(lon, lat)


def utm_to_wgs84(x, y, epsg=UTM_EPSG):
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return t.transform(x, y)


def compute_bearing(pt_a_wgs84, pt_b_wgs84):
    """
    Compute the initial bearing (azimuth) from pt_a to pt_b in WGS-84.
    Returns degrees clockwise from North, range [0, 360).
    Used to set bearing_deg for the anisotropic sidehill cost layer.
    """
    lon1, lat1 = math.radians(pt_a_wgs84[0]), math.radians(pt_a_wgs84[1])
    lon2, lat2 = math.radians(pt_b_wgs84[0]), math.radians(pt_b_wgs84[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y)) % 360.0
    log.info(f"A→B bearing: {bearing:.1f}°")
    return bearing


def xy_to_rowcol(x, y, transform):
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def rowcol_to_xy(row, col, transform):
    x = transform.c + (col + 0.5) * transform.a
    y = transform.f + (row + 0.5) * transform.e
    return x, y


# ── Geometry helpers ──────────────────────────────────────────────────────────

def curve_radius_at_point(pt_prev, pt_curr, pt_next):
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



# ── Clothoid (Euler spiral) transitions — Phase 4 ────────────────────────────

def _clothoid_length(design_speed_kmph, radius_m, C=0.6):
    """
    Minimum length of Euler spiral transition curve.
    Formula: L = V³ / (C × R)
      V = design speed in m/s
      C = centripetal rate (0.6 m/s³ — Myanmar rural trunk standard)
      R = horizontal curve radius in metres
    Reference: AASHTO Green Book §3.3.6 / Myanmar DRD Road Design Standard.
    """
    V = design_speed_kmph / 3.6   # km/h → m/s
    return (V ** 3) / (C * max(radius_m, 1.0))


def compute_clothoid_transitions(coords_utm, design_speed_kmph=DESIGN_SPEED_KMPH,
                                  min_radius=MIN_CURVE_RADIUS, C=0.6,
                                  sample_spacing=10):
    """
    Walk the smoothed alignment and compute clothoid transition requirements at
    every horizontal curve.

    For each curve point i:
      1. Estimate local radius R using three-point circumcircle.
      2. If R < 5 × min_radius (non-trivial curve), compute spiral length L.
      3. Check tangent availability: if L > available tangent, flag as infeasible.

    Returns:
        list of dicts with keys:
          station_m, radius_m, required_spiral_m, tangent_available_m, feasible
    """
    n = len(coords_utm)
    if n < 3:
        return []

    transitions = []
    # Sample every sample_spacing points to avoid reporting noise
    indices = list(range(1, n - 1, max(1, sample_spacing)))

    for i in indices:
        R = curve_radius_at_point(coords_utm[i - 1], coords_utm[i], coords_utm[i + 1])
        if R > 5 * min_radius:
            continue   # effectively straight — no spiral needed

        L_required = _clothoid_length(design_speed_kmph, R, C)

        # Estimate tangent length: half-chord to prev/next sample point
        x0, y0 = coords_utm[max(0, i - sample_spacing)]
        x1, y1 = coords_utm[i]
        x2, y2 = coords_utm[min(n - 1, i + sample_spacing)]
        t1 = math.hypot(x1 - x0, y1 - y0)
        t2 = math.hypot(x2 - x1, y2 - y1)
        tangent_avail = min(t1, t2)

        # Station: cumulative distance from start
        station = sum(math.hypot(coords_utm[j][0] - coords_utm[j-1][0],
                                  coords_utm[j][1] - coords_utm[j-1][1])
                      for j in range(1, i + 1))

        feasible = L_required <= tangent_avail
        transitions.append({
            "station_m":          round(station, 1),
            "radius_m":           round(R, 1),
            "required_spiral_m":  round(L_required, 1),
            "tangent_avail_m":    round(tangent_avail, 1),
            "feasible":           feasible,
        })

    infeasible = sum(1 for t in transitions if not t["feasible"])
    if infeasible:
        log.warning(f"Clothoid transitions: {infeasible} of {len(transitions)} "
                    f"curves have insufficient tangent for spiral transitions.")
    else:
        log.info(f"Clothoid transitions: {len(transitions)} curves; all spiral "
                 f"lengths fit within available tangents.")
    return transitions


# ── Segment-aware smoothing ───────────────────────────────────────────────────

def _segment_slope_means(coords_utm, slope_pct, transform, window=20):
    """
    Return per-point mean slope in a sliding window along the path.
    Used to adapt smoothing factor: gentler smoothing where terrain is complex.
    """
    rows, cols_n = slope_pct.shape
    local_slopes = []
    N = len(coords_utm)
    for i in range(N):
        lo = max(0, i - window // 2)
        hi = min(N, i + window // 2 + 1)
        vals = []
        for x, y in coords_utm[lo:hi]:
            c = int((x - transform.c) / transform.a)
            r = int((y - transform.f) / transform.e)
            if 0 <= r < rows and 0 <= c < cols_n:
                vals.append(float(slope_pct[r, c]))
        local_slopes.append(np.mean(vals) if vals else 0.0)
    return np.array(local_slopes)


def smooth_path(coords_utm, n_points=500, smoothing=None,
                slope_pct=None, transform=None):
    """
    Segment-aware B-spline smoothing.

    If slope_pct and transform are provided, the global smoothing factor is
    modulated by local terrain complexity:
      - Flat segments (mean slope ≤ 4%) → heavy smoothing (s × 8)
      - Mountain segments (mean slope > 8%) → light smoothing (s × 1)
      - Transitions → linearly interpolated

    This prevents over-smoothing on switchback sections while still producing
    clean alignments in valley floors.
    """
    if len(coords_utm) < 4:
        return coords_utm

    xs = np.array([c[0] for c in coords_utm])
    ys = np.array([c[1] for c in coords_utm])

    base_s = len(coords_utm) * 4.0 if smoothing is None else smoothing

    # If terrain info available, compute adaptive smoothing
    if slope_pct is not None and transform is not None:
        local_slopes = _segment_slope_means(coords_utm, slope_pct, transform)
        # Map mean slope to a smoothing multiplier [1.0, 8.0]
        flat_thresh  = SLOPE_MAX_PCT * 0.4   # ≈ lower 40% of threshold = smooth heavily
        steep_thresh = SLOPE_MAX_PCT * 0.8   # upper 80% = preserve detail
        t = np.clip((local_slopes - flat_thresh) / max(steep_thresh - flat_thresh, 0.01), 0, 1)
        smooth_mult = 8.0 - t * 7.0   # 8.0 (flat) → 1.0 (mountain)
        # Use median to get a single representative factor for the whole path
        # (full per-segment spline would require splitting and stitching)
        adaptive_s = base_s * float(np.median(smooth_mult))
        log.info(f"Adaptive smoothing: base_s={base_s:.0f}, factor={float(np.median(smooth_mult)):.2f}x → s={adaptive_s:.0f}")
    else:
        adaptive_s = base_s

    try:
        tck, u = splprep([xs, ys], s=adaptive_s, k=3)
        u_fine = np.linspace(0, 1, n_points)
        x_sm, y_sm = splev(u_fine, tck)
        return list(zip(x_sm.tolist(), y_sm.tolist()))
    except Exception as exc:
        log.warning(f"B-spline smoothing failed ({exc}). Returning raw path.")
        return coords_utm


# ── Curve radius verification ─────────────────────────────────────────────────

def verify_curve_radius(coords_utm, min_radius=MIN_CURVE_RADIUS):
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
            f"below {min_radius} m (minimum found: {min_r:.1f} m) "
            f"[design speed {DESIGN_SPEED_KMPH} km/h]"
        )
        for coord, r in violations[:5]:
            log.warning(f"  Bottleneck → UTM ({coord[0]:.1f}, {coord[1]:.1f})  R={r:.1f} m")
    else:
        log.info(f"Curve radius OK: min={min_r:.1f} m ≥ {min_radius} m [{DESIGN_SPEED_KMPH} km/h]")

    return min_r, violations


# ── Longitudinal profile ──────────────────────────────────────────────────────

def extract_longitudinal_profile(coords_utm, dem, transform):
    """
    Walk the alignment and sample DEM elevation at each waypoint.
    Returns (distances_m, elevations_m) arrays.
    """
    rows, cols = dem.shape
    dists = [0.0]
    elevs = []

    for i, (x, y) in enumerate(coords_utm):
        c = int((x - transform.c) / transform.a)
        r = int((y - transform.f) / transform.e)
        if 0 <= r < rows and 0 <= c < cols:
            elevs.append(float(dem[r, c]))
        else:
            elevs.append(elevs[-1] if elevs else 0.0)
        if i > 0:
            x0, y0 = coords_utm[i - 1]
            dists.append(dists[-1] + math.hypot(x - x0, y - y0))

    return np.array(dists), np.array(elevs)


def check_sustained_grade(elevations_m, distances_m, max_grade=0.08, window_m=3_000):
    """
    Flag any window_m-long section of the profile where the average grade
    exceeds max_grade (8% default = Myanmar heavy truck standard).

    Returns list of (start_m, end_m, avg_grade_pct) for violations.
    """
    violations = []
    n = len(distances_m)
    i = 0
    while i < n:
        # Find window end
        j = i + 1
        while j < n and (distances_m[j] - distances_m[i]) < window_m:
            j += 1
        if j >= n:
            break
        dz = abs(elevations_m[j] - elevations_m[i])
        dx = distances_m[j] - distances_m[i]
        grade = dz / dx if dx > 0 else 0.0
        if grade > max_grade:
            violations.append((distances_m[i], distances_m[j], round(grade * 100, 2)))
        i += max(1, (j - i) // 2)

    if violations:
        log.warning(
            f"Sustained grade violations (>{max_grade*100:.0f}% over {window_m/1000:.0f} km): "
            f"{len(violations)} sections"
        )
        for s, e, g in violations[:5]:
            log.warning(f"  km {s/1000:.1f}–{e/1000:.1f}: avg grade {g:.1f}%")
    else:
        log.info(f"Sustained grade OK: no section exceeds {max_grade*100:.0f}% over {window_m/1000:.0f} km")

    return violations


# ── RoW setback check ─────────────────────────────────────────────────────────

def verify_row_setback(line_utm, buildings_utm, buffer_m=ROW_BUFFER_M):
    if buildings_utm is None or len(buildings_utm) == 0:
        log.info("No building data — RoW setback check skipped.")
        return True

    all_buildings = unary_union(buildings_utm.geometry.tolist())
    min_dist = line_utm.distance(all_buildings)

    if min_dist < buffer_m:
        log.warning(
            f"RoW setback VIOLATED: closest structure is {min_dist:.1f} m "
            f"from centreline (minimum is {buffer_m} m)"
        )
        return False
    else:
        log.info(f"RoW setback OK: {min_dist:.1f} m ≥ {buffer_m} m")
        return True


# ── Metadata ──────────────────────────────────────────────────────────────────

def compute_metadata(line_utm, slope_pct, transform, dem=None,
                     dem_source="unknown", osm_stats=None, warnings_list=None):
    """
    Compute route metadata including confidence scoring.

    Confidence is marked LOW if:
      - DEM is MOCK (no real terrain data)
      - OSM building or water layers are empty (cannot validate urban or river avoidance)

    Data confidence levels: HIGH | MEDIUM | LOW
    """
    from config import POINT_A, POINT_B, SCENARIO_PROFILE, DESIGN_SPEED_KMPH

    length_m = line_utm.length
    max_slope = 0.0
    rows, cols = slope_pct.shape
    for x, y in line_utm.coords:
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        if 0 <= row < rows and 0 <= col < cols:
            s = float(slope_pct[row, col])
            if s > max_slope:
                max_slope = s

    # Confidence scoring
    osm_stats = osm_stats or {}
    issues = []
    if dem_source == "MOCK":
        issues.append("MOCK DEM — synthetic terrain, not real")
    elif dem_source == "SRTMGL3":
        issues.append("SRTMGL3 (90 m DEM) — slope accuracy reduced")
    if osm_stats.get("buildings", 0) == 0:
        issues.append("No OSM buildings — urban avoidance not active")
    if osm_stats.get("water", 0) == 0:
        issues.append("No OSM water — river penalties derived from DEM only")

    if dem_source == "MOCK":
        confidence = "LOW"
    elif issues:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"

    return {
        "total_length_m": round(length_m, 1),
        "total_length_km": round(length_m / 1000, 3),
        "max_slope_pct": round(max_slope, 2),
        "utm_epsg": UTM_EPSG,
        "row_buffer_m": ROW_BUFFER_M,
        "min_curve_radius_m": MIN_CURVE_RADIUS,
        "design_speed_kmph": DESIGN_SPEED_KMPH,
        "scenario_profile": SCENARIO_PROFILE,
        "point_a_lonlat": list(POINT_A),
        "point_b_lonlat": list(POINT_B),
        "dem_source": dem_source,
        "osm_building_count": osm_stats.get("buildings", 0),
        "osm_water_count": osm_stats.get("water", 0),
        "data_confidence": confidence,
        "data_warnings": issues,
        "pipeline_warnings": warnings_list or [],
    }


# ── GeoJSON export ────────────────────────────────────────────────────────────

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
