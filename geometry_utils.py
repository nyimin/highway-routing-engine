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

from config import (
    UTM_EPSG, MIN_CURVE_RADIUS, ROW_BUFFER_M, SLOPE_MAX_PCT, DESIGN_SPEED_KMPH,
    MIN_TANGENT_LENGTH_M, MIN_CURVE_LENGTH_M,
)

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


def smooth_path(coords_utm, n_points=None, smoothing=None,
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

    # Dynamically determine point density (1 point per 10m)
    if n_points is None or n_points <= 1000:
        dist = sum(math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) for i in range(1, len(xs)))
        n_points = max(500, int(dist / 10.0))

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


# ── Phase 12: Design length verification (3D-CHA* inspired) ────────────────

def verify_design_lengths(coords_utm, min_radius=MIN_CURVE_RADIUS,
                          min_tangent_m=MIN_TANGENT_LENGTH_M,
                          min_curve_m=MIN_CURVE_LENGTH_M):
    """
    Check that tangent and curve segments meet minimum design lengths.

    Inspired by 3D-CHA* which enforces:
      - MIN_LEN_TAN: shortest acceptable straight segment
      - MIN_LEN_CURV: shortest acceptable circular curve

    Short tangents between reverse curves create dangerous reverse-superelevation
    transitions. Short curves give drivers insufficient time to perceive the
    change in direction and adjust steering.

    Classification:
      - Curve segment: R < 5 × min_radius  (non-trivial curvature)
      - Tangent segment: R ≥ 5 × min_radius (effectively straight)

    Args:
        coords_utm:     smoothed alignment as list of (x, y) tuples
        min_radius:     minimum curve radius for classifying curves
        min_tangent_m:  minimum tangent length (from config profile)
        min_curve_m:    minimum curve length (from config profile)

    Returns:
        dict with:
          tangent_violations: list of (start_station_m, length_m)
          curve_violations:   list of (start_station_m, length_m, avg_radius_m)
          n_tangents: total tangent segments
          n_curves:   total curve segments
    """
    pts = coords_utm
    n = len(pts)
    if n < 3:
        return {"tangent_violations": [], "curve_violations": [],
                "n_tangents": 0, "n_curves": 0}

    curve_threshold = 5.0 * min_radius

    # classify each segment as curve or tangent
    seg_types = []      # 'T' or 'C'
    seg_radii = []      # radius for curves
    seg_lengths = []    # segment length in metres
    stations = [0.0]    # cumulative station for each point

    for i in range(1, n):
        d = math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
        stations.append(stations[-1] + d)

    for i in range(1, n - 1):
        r = curve_radius_at_point(pts[i-1], pts[i], pts[i+1])
        seg_len = stations[i+1] - stations[i-1]  # length over the triplet
        if r < curve_threshold:
            seg_types.append('C')
            seg_radii.append(r)
        else:
            seg_types.append('T')
            seg_radii.append(float('inf'))
        seg_lengths.append(seg_len / 2.0)  # approximate per-point contribution

    # group consecutive same-type segments
    groups = []  # (type, start_station, length, [radii])
    if not seg_types:
        return {"tangent_violations": [], "curve_violations": [],
                "n_tangents": 0, "n_curves": 0}

    curr_type = seg_types[0]
    curr_start = stations[1]
    curr_len = seg_lengths[0]
    curr_radii = [seg_radii[0]]

    for j in range(1, len(seg_types)):
        if seg_types[j] == curr_type:
            curr_len += seg_lengths[j]
            curr_radii.append(seg_radii[j])
        else:
            groups.append((curr_type, curr_start, curr_len, curr_radii))
            curr_type = seg_types[j]
            curr_start = stations[j + 1]
            curr_len = seg_lengths[j]
            curr_radii = [seg_radii[j]]
    groups.append((curr_type, curr_start, curr_len, curr_radii))

    # check violations
    tangent_violations = []
    curve_violations = []
    n_tangents = 0
    n_curves = 0

    for gtype, gstart, glen, gradii in groups:
        if gtype == 'T':
            n_tangents += 1
            if glen < min_tangent_m:
                tangent_violations.append((round(gstart, 1), round(glen, 1)))
        else:
            n_curves += 1
            avg_r = np.mean([r for r in gradii if r < float('inf')])
            if glen < min_curve_m:
                curve_violations.append((round(gstart, 1), round(glen, 1), round(avg_r, 1)))

    # logging
    if tangent_violations:
        log.warning(
            f"Design lengths: {len(tangent_violations)} tangent(s) below "
            f"{min_tangent_m:.0f} m (min found: {min(v[1] for v in tangent_violations):.0f} m)"
        )
        for s, l in tangent_violations[:5]:
            log.warning(f"  Station {s:.0f} m: tangent length {l:.0f} m")
    else:
        log.info(f"Tangent lengths OK: all {n_tangents} tangents ≥ {min_tangent_m:.0f} m")

    if curve_violations:
        log.warning(
            f"Design lengths: {len(curve_violations)} curve(s) below "
            f"{min_curve_m:.0f} m (min found: {min(v[1] for v in curve_violations):.0f} m)"
        )
        for s, l, r in curve_violations[:5]:
            log.warning(f"  Station {s:.0f} m: curve length {l:.0f} m (R={r:.0f} m)")
    else:
        log.info(f"Curve lengths OK: all {n_curves} curves ≥ {min_curve_m:.0f} m")

    return {
        "tangent_violations": tangent_violations,
        "curve_violations": curve_violations,
        "n_tangents": n_tangents,
        "n_curves": n_curves,
    }


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

    # Avoid unary_union on 500k+ buildings (causes silent GEOS crash/OOM)
    # Avoid single nearest() query with a 200km line which causes O(N) worst-case bounding box checks.
    # Instead, segment the line, find nearest building per segment, then find exact min distance among those candidates.
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString

    coords = list(line_utm.coords)
    segments_gs = gpd.GeoSeries([LineString([coords[i], coords[i+1]]) for i in range(len(coords)-1)])
    
    nearest_idx = buildings_utm.sindex.nearest(segments_gs)
    if nearest_idx is None or len(nearest_idx) == 0 or len(nearest_idx[1]) == 0:
        log.info("No buildings found near the alignment.")
        return True
    
    candidate_bldg_idx = np.unique(nearest_idx[1])
    candidate_bldgs = buildings_utm.iloc[candidate_bldg_idx].geometry
    min_dist = candidate_bldgs.distance(line_utm).min()

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


def export_geojson_3d(coords_wgs84_3d, metadata, output_path):
    """
    Export a 3D LineString GeoJSON with coordinates as [lon, lat, z_design].

    Parameters
    ----------
    coords_wgs84_3d : list of (lon, lat, z) tuples
        WGS-84 longitude, latitude, and design elevation (m) at each station.
    metadata : dict
        Feature properties — same as for export_geojson().
    output_path : str
        Destination file path (e.g. 'preliminary_route_3d.geojson').

    Notes
    -----
    GeoJSON RFC 7946 explicitly supports Z coordinates in coordinate arrays.
    Most GIS tools (QGIS, kepler.gl, Mapbox) render the Z component for 3D
    visualisation.  The CRS is always WGS-84 geographic (EPSG:4326).
    """
    coords = [[float(lon), float(lat), float(z)] for lon, lat, z in coords_wgs84_3d]
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": metadata,
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    log.info(f"Exported 3D GeoJSON: {output_path}  ({len(coords)} vertices)")

