"""
routing.py — Least-cost pathfinding with two-resolution strategy
=================================================================
Phase 4 additions:
  1. find_path() is now a dispatcher: FMM engine if scikit-fmm installed,
     else Dijkstra via skimage (fully transparent fallback).
  2. _downsample_cost() / _build_fine_band_mask() — coarse-to-fine infrastructure.
  3. coarse_to_fine_routing() — new top-level entry point used by main.py.
     Runs a fast 300 m Dijkstra first, carves an 8 km corridor band, then
     runs full two-pass rubber-band routing within that band only.
     Memory reduction: ~COARSE_FACTOR² (order-of-magnitude for long corridors).
"""
import math
import logging
import numpy as np
from scipy.ndimage import binary_dilation, label as nd_label

from config import (
    RUBBER_BAND_MACRO_W, RUBBER_BAND_MICRO_W,
    MIN_BRIDGE_SPACING_M, IMPASSABLE,
    COARSE_FACTOR, CORRIDOR_BAND_KM, FAST_MODE, ROUTING_ENGINE,
    TURNING_ANGLE_FILTER_DEG,
)

log = logging.getLogger("highway_alignment")

# ── FMM availability check (one-time import attempt) ─────────────────────────
_FMM_AVAILABLE = False
try:
    import skfmm as _skfmm
    _FMM_AVAILABLE = True
    log.info("scikit-fmm detected — FMM engine available.")
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Core path-finding engines
# ═══════════════════════════════════════════════════════════════════════════════

def _find_path_dijkstra(cost, start_rc, end_rc):
    """8-connected Dijkstra via scikit-image (always available)."""
    from skimage.graph import route_through_array
    path_indices, cost_val = route_through_array(
        cost, start_rc, end_rc,
        fully_connected=True, geometric=True,
    )
    log.info(f"Dijkstra: {len(path_indices)} waypoints, total cost={cost_val:.1f}")
    return list(path_indices)


def _find_path_fmm(cost, start_rc, end_rc):
    """
    Fast Marching Method path extraction via scikit-fmm.

    FMM propagates a wavefront from start_rc through speed=1/cost.
    Path back-traces the travel-time gradient from end_rc to start_rc,
    giving smoother trajectories on continuous terrain than 8-connected Dijkstra.

    Falls back to Dijkstra on any numerical failure.
    """
    try:
        rows, cols = cost.shape
        # Speed = reciprocal of cost (clamp to avoid divide-by-zero at IMPASSABLE cells)
        safe_cost = np.where(cost >= IMPASSABLE, IMPASSABLE, np.maximum(cost, 1e-6))
        speed = 1.0 / safe_cost

        # Source mask: phi < 0 at start point, > 0 elsewhere
        phi = np.ones((rows, cols), dtype=np.float64)
        phi[start_rc[0], start_rc[1]] = -1.0

        travel_time = _skfmm.travel_time(phi, speed, order=2)

        # Back-trace: gradient descent from end to start on travel_time surface
        path = _gradient_descent_path(travel_time, start_rc, end_rc)
        log.info(f"FMM: {len(path)} waypoints extracted")
        return path
    except Exception as exc:
        log.warning(f"FMM failed ({exc}), falling back to Dijkstra.")
        return _find_path_dijkstra(cost, start_rc, end_rc)


def _gradient_descent_path(travel_time, start_rc, end_rc, max_steps=100_000):
    """
    Trace a path from end_rc back to start_rc using sub-pixel gradient descent
    on the travel_time surface.

    Uses np.gradient() for smooth derivative estimation, then rounds to the
    nearest integer cell for the final path. The sub-pixel stepping produces
    smoother trajectories than pure 8-connected integer stepping, eliminating
    stair-step artifacts on continuous FMM travel-time fields.
    """
    rows, cols = travel_time.shape
    # Pre-compute gradients for the whole field (cheap, one-time)
    grad_r, grad_c = np.gradient(travel_time)

    r, c = float(end_rc[0]), float(end_rc[1])
    path = [(end_rc[0], end_rc[1])]
    visited = {(end_rc[0], end_rc[1])}
    step_size = 0.8  # sub-pixel step (< 1.0 avoids overshooting)
    stall = 0        # consecutive steps without visiting a new cell
    max_stall = 50   # break if stuck oscillating in one cell

    for _ in range(max_steps):
        ri, ci = int(round(r)), int(round(c))
        if (ri, ci) == tuple(start_rc):
            break

        # Clamp to grid bounds
        ri_s = max(0, min(rows - 1, ri))
        ci_s = max(0, min(cols - 1, ci))

        # Get gradient at current position
        gr = grad_r[ri_s, ci_s]
        gc = grad_c[ri_s, ci_s]
        mag = math.sqrt(gr * gr + gc * gc)
        if mag < 1e-12:
            # Gradient vanished — fall back to 8-connected neighbor search
            best_r, best_c = ri_s, ci_s
            best_t = travel_time[ri_s, ci_s]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = ri_s + dr, ci_s + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        if travel_time[nr, nc] < best_t:
                            best_t = travel_time[nr, nc]
                            best_r, best_c = nr, nc
            if (best_r, best_c) == (ri_s, ci_s):
                break
            r, c = float(best_r), float(best_c)
        else:
            # Step in negative gradient direction (toward lower travel time)
            r -= step_size * gr / mag
            c -= step_size * gc / mag

        # Clamp to grid
        r = max(0.0, min(float(rows - 1), r))
        c = max(0.0, min(float(cols - 1), c))

        cell = (int(round(r)), int(round(c)))
        if cell not in visited:
            path.append(cell)
            visited.add(cell)
            stall = 0
        else:
            stall += 1
            if stall >= max_stall:
                log.debug(f"Gradient descent stalled at {cell} after {max_stall} "
                          f"steps — breaking early.")
                break

    # Ensure start is included
    if path[-1] != tuple(start_rc):
        path.append(tuple(start_rc))

    path.reverse()  # start → end
    return path


def _filter_sharp_reversals(path, max_turn_deg=TURNING_ANGLE_FILTER_DEG):
    """
    Remove waypoints that cause near-180° turns (stutter-step artifacts).

    Walks the path and measures the interior angle at every triplet
    (P[i-1], P[i], P[i+1]).  If the angle exceeds max_turn_deg (meaning
    an almost-reversal), P[i] is dropped.

    Preserves start and end points unconditionally.
    """
    if len(path) < 3:
        return path

    cos_threshold = math.cos(math.radians(max_turn_deg))
    filtered = [path[0]]

    for i in range(1, len(path) - 1):
        r0, c0 = filtered[-1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        v1r, v1c = r1 - r0, c1 - c0
        v2r, v2c = r2 - r1, c2 - c1
        mag1 = math.sqrt(v1r * v1r + v1c * v1c)
        mag2 = math.sqrt(v2r * v2r + v2c * v2c)
        if mag1 < 1e-9 or mag2 < 1e-9:
            continue  # zero-length segment — skip
        cos_angle = (v1r * v2r + v1c * v2c) / (mag1 * mag2)
        # cos_angle close to -1 → reversal; if angle > threshold, drop it
        if cos_angle < cos_threshold:
            continue  # skip this waypoint (reversal)
        filtered.append(path[i])

    filtered.append(path[-1])
    n_removed = len(path) - len(filtered)
    if n_removed > 0:
        log.info(f"Turning-angle filter: removed {n_removed} reversal points "
                 f"(threshold {max_turn_deg}°)")
    return filtered


def find_path(cost, start_rc, end_rc):
    """
    Dispatcher: uses FMM if available and ROUTING_ENGINE allows it,
    otherwise Dijkstra. Applies reversal filter to raw path.
    """
    use_fmm = _FMM_AVAILABLE and ROUTING_ENGINE in ('auto', 'fmm')
    if use_fmm:
        raw = _find_path_fmm(cost, start_rc, end_rc)
    else:
        raw = _find_path_dijkstra(cost, start_rc, end_rc)
    return _filter_sharp_reversals(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Rubber-band penalty (Phase 1 — normalised, numerically stable)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_rubber_band_penalty(cost, start_rc, end_rc, weight, reference_mask=None):
    """
    Multiplicative penalty pushing the router toward a reference line.
    If reference_mask is provided (boolean array, True = reference path),
    the penalty is based on the distance transform to that mask.
    Otherwise, it defaults to the straight-line chord between start/end.
    
    Max exponent capped at 3 (exp(3) ≈ 20×).
    """
    if weight <= 0:
        return cost

    rows, cols = cost.shape
    r0, c0 = start_rc
    r1, c1 = end_rc
    len_sq = float((r1 - r0) ** 2 + (c1 - c0) ** 2)

    if reference_mask is not None:
        # Distance transform to the reference mask (0 on the mask, >0 elsewhere)
        from scipy.ndimage import distance_transform_edt
        perp_dist_px = distance_transform_edt(~reference_mask)
    else:
        # Fallback to straight-line A->B distance
        y_grid, x_grid = np.mgrid[0:rows, 0:cols]
        if len_sq == 0:
            return cost
        t = ((y_grid - r0) * (r1 - r0) + (x_grid - c0) * (c1 - c0)) / len_sq
        t = np.clip(t, 0.0, 1.0)
        proj_r = r0 + t * (r1 - r0)
        proj_c = c0 + t * (c1 - c0)
        perp_dist_px = np.sqrt((y_grid - proj_r) ** 2 + (x_grid - proj_c) ** 2)

    # Use bounding-box diagonal of the start/end extent rather than full grid
    # diagonal. This prevents over-penalization on narrow corridors where the
    # grid is much larger than the actual routing extent.
    bb_diag = math.sqrt(float((r1 - r0) ** 2 + (c1 - c0) ** 2))
    norm_diag = max(bb_diag, max(rows, cols) * 0.1, 1.0)  # floor to avoid near-zero
    normalized_dist = perp_dist_px / norm_diag
    exponent = np.clip(normalized_dist * weight, 0.0, 3.0)
    return cost * np.exp(exponent)


# ═══════════════════════════════════════════════════════════════════════════════
# Bridge crossing logic (Phase 1 — narrowest perpendicular crossing)
# ═══════════════════════════════════════════════════════════════════════════════

def _measure_channel_width(water_mask, r, c, direction_rc, resolution_m, max_scan_px=300):
    dr, dc = direction_rc
    perp_r, perp_c = -dc, dr
    norm = math.sqrt(perp_r ** 2 + perp_c ** 2)
    if norm == 0:
        return 0.0
    perp_r /= norm
    perp_c /= norm

    rows, cols = water_mask.shape
    count = 0
    for sign in (1, -1):
        for step in range(1, max_scan_px + 1):
            nr = int(round(r + sign * step * perp_r))
            nc = int(round(c + sign * step * perp_c))
            if not (0 <= nr < rows and 0 <= nc < cols):
                break
            if water_mask[nr, nc] == 0:
                break
            count += 1
    return (count + 1) * resolution_m


def _bank_stability_score(water_mask, r, c, dem, resolution_m, band_px=5):
    rows, cols = water_mask.shape
    slopes = []
    for dr in range(-band_px, band_px + 1):
        for dc in range(-band_px, band_px + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and water_mask[nr, nc] == 0:
                if dem is not None:
                    try:
                        dz_r = float(dem[min(nr+1,rows-1),nc] - dem[max(nr-1,0),nc])
                        dz_c = float(dem[nr,min(nc+1,cols-1)] - dem[nr,max(nc-1,0)])
                        slope_pct = math.sqrt(dz_r**2 + dz_c**2) / (2*resolution_m) * 100
                        slopes.append(slope_pct)
                    except Exception:
                        pass
    if not slopes:
        return 1.0
    return 1.0 + (np.mean(slopes) / 100.0) + (np.std(slopes) / 50.0)


def find_optimal_crossings(water_mask, macro_path, transform, resolution_m,
                             dem=None, search_window_px=200):
    rows, cols = water_mask.shape
    water_hits = [(i, r, c) for i, (r, c) in enumerate(macro_path) if water_mask[r, c] > 0]
    if not water_hits:
        log.info("Macro path did not cross water — no bridge needed.")
        return []

    segments, seg = [], [water_hits[0]]
    for prev, curr in zip(water_hits, water_hits[1:]):
        if curr[0] - prev[0] <= 3:
            seg.append(curr)
        else:
            segments.append(seg)
            seg = [curr]
    segments.append(seg)
    log.info(f"Water-body segments intersected: {len(segments)}")

    bridges_rc = []
    for seg in segments:
        seg_centre_idx = seg[len(seg) // 2][0]
        lo = max(0, seg_centre_idx - search_window_px)
        hi = min(len(macro_path) - 1, seg_centre_idx + search_window_px)
        candidates = [(macro_path[i][0], macro_path[i][1])
                      for i in range(lo, hi + 1)
                      if water_mask[macro_path[i][0], macro_path[i][1]] > 0]
        if not candidates:
            candidates = [(seg[len(seg)//2][1], seg[len(seg)//2][2])]

        best_rc_for_seg, best_score_for_seg = None, float("inf")
        for r, c in candidates:
            path_idx = next((i for i, (pr,pc) in enumerate(macro_path) if pr==r and pc==c), None)
            if path_idx is None or path_idx == 0 or path_idx >= len(macro_path)-1:
                direction = (1, 0)
            else:
                pr0, pc0 = macro_path[path_idx-1]
                pr1, pc1 = macro_path[path_idx+1]
                dr, dc = pr1-pr0, pc1-pc0
                norm = math.sqrt(dr**2+dc**2) or 1.0
                direction = (dr/norm, dc/norm)

            width_m = _measure_channel_width(water_mask, r, c, direction, resolution_m)

            braiding = 1.0
            labeled_water, _ = nd_label(water_mask > 0)
            comp_id = labeled_water[r, c]
            if comp_id > 0:
                comp = labeled_water == comp_id
                aspect = max(np.any(comp,axis=1).sum(), np.any(comp,axis=0).sum()) / \
                         max(min(np.any(comp,axis=1).sum(), np.any(comp,axis=0).sum()), 1)
                if aspect > 5 and width_m > 200 * resolution_m:
                    braiding = 3.0

            score = width_m * _bank_stability_score(water_mask, r, c, dem, resolution_m) * braiding
            if score < best_score_for_seg:
                best_score_for_seg, best_rc_for_seg = score, (r, c)

        if best_rc_for_seg:
            bridges_rc.append(best_rc_for_seg)
            bx, by = transform.c + (best_rc_for_seg[1]+0.5) * transform.a, transform.f + (best_rc_for_seg[0]+0.5) * transform.e
            log.info(f"Bridge sited at grid={best_rc_for_seg}, UTM=({bx:.0f},{by:.0f}) score={best_score_for_seg:.1f}")

    return bridges_rc


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-pass rubber-band micro-alignment (within a corridor)
# ═══════════════════════════════════════════════════════════════════════════════

def multi_pass_routing(cost, start_rc, end_rc, water_mask, transform,
                       resolution_m=30, dem=None, reference_mask=None):
    """
    Macro rubber-band → bridge siting → N micro-alignment passes.
    Called by coarse_to_fine_routing on the band-masked fine cost grid.
    If reference_mask is provided, rubber-banding pulls toward that mask
    (e.g. the coarse centerline) instead of the global straight line.
    """
    log.info("Macro rubber-band routing …")
    macro_cost = apply_rubber_band_penalty(
        cost, start_rc, end_rc, weight=RUBBER_BAND_MACRO_W, reference_mask=reference_mask
    )
    macro_path = find_path(macro_cost, start_rc, end_rc)

    bridges_rc = find_optimal_crossings(water_mask, macro_path, transform, resolution_m, dem=dem)

    if not bridges_rc:
        log.info("No water crossing — single micro-alignment pass.")
        micro_cost = apply_rubber_band_penalty(
            cost, start_rc, end_rc, weight=RUBBER_BAND_MICRO_W, reference_mask=reference_mask
        )
        return find_path(micro_cost, start_rc, end_rc)

    # Dedup sequential identical bridges just in case
    clean_bridges = []
    for br in bridges_rc:
        if not clean_bridges or br != clean_bridges[-1]:
            clean_bridges.append(br)

    # Build sequence of waypoints: Start -> Bridge 1 -> Bridge 2 -> ... -> End
    waypoints = [start_rc] + clean_bridges + [end_rc]
    full_path = []

    for i in range(len(waypoints) - 1):
        wp_from = waypoints[i]
        wp_to = waypoints[i + 1]
        log.info(f"Micro alignment {i + 1}/{len(waypoints) - 1}: {wp_from} → {wp_to} …")
        mc_seg = apply_rubber_band_penalty(
            cost, wp_from, wp_to, weight=RUBBER_BAND_MICRO_W, reference_mask=reference_mask
        )
        seg_path = find_path(mc_seg, wp_from, wp_to)
        if i > 0:
            full_path += seg_path[1:]
        else:
            full_path += seg_path

    log.info(f"Combined path: {len(full_path)} waypoints")
    return full_path


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Multi-Scale Segmented LCP (Tang & Dou 2023)
# ═══════════════════════════════════════════════════════════════════════════════
import concurrent.futures

def _clamp_rc(rc, shape):
    return (max(0, min(shape[0]-1, int(rc[0]))), max(0, min(shape[1]-1, int(rc[1]))))


def _extract_directional_waypoints(path, angle_thresh_deg, max_dist_px=200):
    """
    Extract key directional points from a path (Tang & Dou 2023).
    A point P[i] is a directional point if the angle between vector(P[i-1], P[i]) 
    and vector(P[i], P[i+1]) is <= angle_thresh_deg.
    Additionally retains points if distance from last waypoint exceeds max_dist_px.
    """
    if len(path) < 3:
        return path

    cos_threshold = math.cos(math.radians(angle_thresh_deg))
    waypoints = [path[0]]
    last_wp_idx = 0

    for i in range(1, len(path) - 1):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        
        v1r, v1c = r1 - r0, c1 - c0
        v2r, v2c = r2 - r1, c2 - c1
        
        mag1 = math.sqrt(v1r * v1r + v1c * v1c)
        mag2 = math.sqrt(v2r * v2r + v2c * v2c)
        
        dist_from_last = math.sqrt((r1 - path[last_wp_idx][0])**2 + (c1 - path[last_wp_idx][1])**2)
        
        if mag1 < 1e-9 or mag2 < 1e-9:
            continue
            
        cos_angle = (v1r * v2r + v1c * v2c) / (mag1 * mag2)
        
        # cos_angle <= cos_threshold means the turn is sharper than the threshold angle
        if cos_angle <= cos_threshold or dist_from_last >= max_dist_px:
            waypoints.append(path[i])
            last_wp_idx = i
            
    waypoints.append(path[-1])
    return waypoints


def _map_waypoints_to_high_res(waypoints_low, high_cost, ratio):
    """
    Project low-resolution waypoints to high-resolution grid.
    Searches the corresponding ratio x ratio block and selects the cell with minimum cost.
    """
    mapped = []
    rows, cols = high_cost.shape
    for (r, c) in waypoints_low:
        r_hi = r * ratio
        c_hi = c * ratio
        
        # Search block
        r_end = min(r_hi + ratio, rows)
        c_end = min(c_hi + ratio, cols)
        
        block = high_cost[r_hi:r_end, c_hi:c_end]
        if block.size == 0:
            mapped.append((r_hi, c_hi))
            continue
            
        # Find min cost in block
        min_idx = np.unravel_index(np.argmin(block, axis=None), block.shape)
        mapped.append((r_hi + min_idx[0], c_hi + min_idx[1]))
        
    return mapped


def _route_segment_worker(args):
    """
    Worker for parallel routing between two waypoints.
    args: (cost_grid, wp_from, wp_to)
    Note: passes a cropped view of cost_grid to save memory/compute.
    """
    cost_grid, wp_from, wp_to = args
    r0, c0 = wp_from
    r1, c1 = wp_to
    
    # Margin for local bounding box (dynamic based on distance, min 50px)
    dist = math.sqrt((r1 - r0)**2 + (c1 - c0)**2)
    margin = max(50, int(dist * 0.3))
    
    rmin = max(0, min(r0, r1) - margin)
    rmax = min(cost_grid.shape[0], max(r0, r1) + margin + 1)
    cmin = max(0, min(c0, c1) - margin)
    cmax = min(cost_grid.shape[1], max(c0, c1) + margin + 1)
    
    local_cost = cost_grid[rmin:rmax, cmin:cmax]
    
    local_start = _clamp_rc((r0 - rmin, c0 - cmin), local_cost.shape)
    local_end = _clamp_rc((r1 - rmin, c1 - cmin), local_cost.shape)
    
    local_path = find_path(local_cost, local_start, local_end)
    
    # Re-translate to global coordinates
    global_path = [(r + rmin, c + cmin) for r, c in local_path]
    return global_path


def multi_scale_lcp(cost_pyramid, start_rc, end_rc, water_mask, transform, resolution_m=30, dem=None):
    """
    Progressive MS-LCP strategy (Tang & Dou 2023):
    1. Route on coarsest level.
    2. Extract waypoints, project to N-1, route between waypoints in parallel.
    3. Repeat until Level 0.
    4. Execute final micro-alignment pass with highway constraints.
    """
    from config import DOWNSAMPLE_RATIO, WAYPOINT_ANGLE_THRESH_DEG, PARALLEL_WAYPOINT_THRESH, FAST_MODE

    levels = len(cost_pyramid) - 1
    log.info(f"MS-LCP: Starting multi-scale routing over {levels} downsampled levels.")
    
    # 1. Base route on the coarsest level
    coarsest_ratio = DOWNSAMPLE_RATIO ** levels
    curr_start = _clamp_rc((start_rc[0] // coarsest_ratio, start_rc[1] // coarsest_ratio), cost_pyramid[-1].shape)
    curr_end = _clamp_rc((end_rc[0] // coarsest_ratio, end_rc[1] // coarsest_ratio), cost_pyramid[-1].shape)
    
    log.info(f"MS-LCP: Pass 0 (Level {levels}) - Routing on {cost_pyramid[-1].shape} grid.")
    curr_path = find_path(cost_pyramid[-1], curr_start, curr_end)
    
    if FAST_MODE and levels > 0:
        log.info("FAST_MODE active — returning upsampled coarse path.")
        fine_path = [
            _clamp_rc((r * coarsest_ratio + coarsest_ratio // 2, c * coarsest_ratio + coarsest_ratio // 2), cost_pyramid[0].shape)
            for r, c in curr_path
        ]
        return fine_path

    # 2. Iterate up the pyramid
    for lvl in range(levels - 1, -1, -1):
        high_res_cost = cost_pyramid[lvl]
        log.info(f"MS-LCP: Pass {levels - lvl} (Level {lvl}) - Projecting to {high_res_cost.shape} grid.")
        
        # Extract waypoints
        waypoints_low = _extract_directional_waypoints(curr_path, WAYPOINT_ANGLE_THRESH_DEG)
        log.info(f"MS-LCP: Extracted {len(waypoints_low)} waypoints from coarse path.")
        
        # Map to high res
        waypoints_hi = _map_waypoints_to_high_res(waypoints_low, high_res_cost, DOWNSAMPLE_RATIO)
        
        # Ensure exact start/end at Level 0
        if lvl == 0:
            waypoints_hi[0] = start_rc
            waypoints_hi[-1] = end_rc
            
        # Segmented routing
        segments = []
        tasks = []
        
        # Prepare tasks
        for i in range(len(waypoints_hi) - 1):
            tasks.append((high_res_cost, waypoints_hi[i], waypoints_hi[i+1]))
            
        use_parallel = len(tasks) >= PARALLEL_WAYPOINT_THRESH
        
        if use_parallel:
            # Note: ThreadPoolExecutor is used because `find_path` underlying C-extensions (scikit-image/skfmm) release the GIL.
            # This avoids heavy IPC memory serialization of the large cost_grid that ProcessPoolExecutor would incur.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(_route_segment_worker, tasks))
                for i, res in enumerate(results):
                    if i > 0 and len(res) > 0:
                        segments.append(res[1:])
                    else:
                        segments.append(res)
        else:
            for task in tasks:
                res = _route_segment_worker(task)
                if len(segments) > 0 and len(res) > 0:
                    segments.append(res[1:])
                else:
                    segments.append(res)
                    
        curr_path = [pt for seg in segments for pt in seg]
        log.info(f"MS-LCP: Level {lvl} path stitched, total waypoints={len(curr_path)}")
        
    # 3. Final Highway Engineering Pass (Level 0)
    log.info("MS-LCP: Executing final micro-alignment (highway constraints) on Level 0.")
    
    # Build a mask of the stitched line for the final rubber-band pull
    fine_centerline = np.zeros(cost_pyramid[0].shape, dtype=bool)
    for r, c in curr_path:
        fine_centerline[_clamp_rc((r, c), fine_centerline.shape)] = True
        
    struct = np.ones((5, 5), dtype=bool)
    fine_centerline = binary_dilation(fine_centerline, structure=struct)
    
    final_path = multi_pass_routing(
        cost_pyramid[0], start_rc, end_rc, water_mask, transform,
        resolution_m=resolution_m, dem=dem, reference_mask=fine_centerline
    )
    
    return final_path
