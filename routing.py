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


def find_optimal_crossing(water_mask, macro_path, transform, resolution_m,
                           dem=None, search_window_px=200):
    rows, cols = water_mask.shape
    water_hits = [(i, r, c) for i, (r, c) in enumerate(macro_path) if water_mask[r, c] > 0]
    if not water_hits:
        log.info("Macro path did not cross water — no bridge needed.")
        return None

    segments, seg = [], [water_hits[0]]
    for prev, curr in zip(water_hits, water_hits[1:]):
        if curr[0] - prev[0] <= 3:
            seg.append(curr)
        else:
            segments.append(seg)
            seg = [curr]
    segments.append(seg)
    log.info(f"Water-body segments intersected: {len(segments)}")

    best_rc, best_score = None, float("inf")
    for seg in segments:
        seg_centre_idx = seg[len(seg) // 2][0]
        lo = max(0, seg_centre_idx - search_window_px)
        hi = min(len(macro_path) - 1, seg_centre_idx + search_window_px)
        candidates = [(macro_path[i][0], macro_path[i][1])
                      for i in range(lo, hi + 1)
                      if water_mask[macro_path[i][0], macro_path[i][1]] > 0]
        if not candidates:
            candidates = [(seg[len(seg)//2][1], seg[len(seg)//2][2])]

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
            if score < best_score:
                best_score, best_rc = score, (r, c)

    if best_rc:
        r, c = best_rc
        x = transform.c + (c+0.5) * transform.a
        y = transform.f + (r+0.5) * transform.e
        log.info(f"Bridge siting: grid={best_rc}, UTM=({x:.0f},{y:.0f}), score={best_score:.1f}")
    return best_rc


# ═══════════════════════════════════════════════════════════════════════════════
# Two-pass rubber-band micro-alignment (within a corridor)
# ═══════════════════════════════════════════════════════════════════════════════

def two_pass_routing(cost, start_rc, end_rc, water_mask, transform,
                      resolution_m=30, dem=None, reference_mask=None):
    """
    Macro rubber-band → bridge siting → two micro-alignment passes.
    Called by coarse_to_fine_routing on the band-masked fine cost grid.
    If reference_mask is provided, rubber-banding pulls toward that mask
    (e.g. the coarse centerline) instead of the global straight line.
    """
    log.info("Macro rubber-band routing …")
    macro_cost = apply_rubber_band_penalty(
        cost, start_rc, end_rc, weight=RUBBER_BAND_MACRO_W, reference_mask=reference_mask
    )
    macro_path = find_path(macro_cost, start_rc, end_rc)

    bridge_rc = find_optimal_crossing(water_mask, macro_path, transform, resolution_m, dem=dem)

    if bridge_rc is None:
        log.info("No water crossing — single micro-alignment pass.")
        micro_cost = apply_rubber_band_penalty(
            cost, start_rc, end_rc, weight=RUBBER_BAND_MICRO_W, reference_mask=reference_mask
        )
        return find_path(micro_cost, start_rc, end_rc)

    log.info("Micro alignment 2a: Origin → Bridge …")
    mc1 = apply_rubber_band_penalty(
        cost, start_rc, bridge_rc, weight=RUBBER_BAND_MICRO_W, reference_mask=reference_mask
    )
    path1 = find_path(mc1, start_rc, bridge_rc)

    log.info("Micro alignment 2b: Bridge → Destination …")
    mc2 = apply_rubber_band_penalty(
        cost, bridge_rc, end_rc, weight=RUBBER_BAND_MICRO_W, reference_mask=reference_mask
    )
    path2 = find_path(mc2, bridge_rc, end_rc)

    full_path = path1 + path2[1:]
    log.info(f"Combined path: {len(full_path)} waypoints")
    return full_path


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Two-resolution routing  (coarse corridor → fine band)
# ═══════════════════════════════════════════════════════════════════════════════

def _downsample_cost(cost, factor):
    """
    Block-reduce cost grid by `factor` using np.max (conservative: worst cell wins).
    Ensures impassable features are preserved on the coarse grid.
    Trims to a multiple of `factor` before reshaping.
    """
    rows, cols = cost.shape
    rows_t = (rows // factor) * factor
    cols_t = (cols // factor) * factor
    trimmed = cost[:rows_t, :cols_t]
    # Reshape to (coarse_rows, factor, coarse_cols, factor) then max over blocks
    coarse = trimmed.reshape(rows_t // factor, factor, cols_t // factor, factor)
    return coarse.max(axis=(1, 3))


def _build_fine_band_mask(coarse_path, coarse_shape, fine_shape, factor, band_km, resolution_m):
    """
    1. Mark coarse path cells on a coarse boolean grid.
    2. Dilate by band_cells_coarse (= band_km * 1000 / coarse_resolution_m).
    3. Upsample to fine resolution by repeating each cell `factor` times.
    4. Clip/pad to match fine_shape exactly.
    Returns a boolean mask: True = inside corridor band.
    """
    coarse_resolution_m = factor * resolution_m
    band_cells_coarse = max(1, int(band_km * 1000.0 / coarse_resolution_m))

    coarse_mask = np.zeros(coarse_shape, dtype=bool)
    cr, cc = coarse_shape
    for r, c in coarse_path:
        if 0 <= r < cr and 0 <= c < cc:
            coarse_mask[r, c] = True

    struct = np.ones((2 * band_cells_coarse + 1, 2 * band_cells_coarse + 1), dtype=bool)
    coarse_band = binary_dilation(coarse_mask, structure=struct)

    # Upsample: each coarse cell → factor × factor fine cells
    fine_band = np.repeat(np.repeat(coarse_band, factor, axis=0), factor, axis=1)
    
    # Also create a mask for JUST the upsampled central path (for rubber-banding)
    fine_centerline = np.repeat(np.repeat(coarse_mask, factor, axis=0), factor, axis=1)

    # Crop or pad to exactly match fine_shape
    fr, fc = fine_shape
    if fine_band.shape[0] >= fr and fine_band.shape[1] >= fc:
        fine_band = fine_band[:fr, :fc]
        fine_centerline = fine_centerline[:fr, :fc]
    else:
        padded = np.zeros(fine_shape, dtype=bool)
        padded_center = np.zeros(fine_shape, dtype=bool)
        h = min(fine_band.shape[0], fr)
        w = min(fine_band.shape[1], fc)
        padded[:h, :w] = fine_band[:h, :w]
        padded_center[:h, :w] = fine_centerline[:h, :w]
        fine_band = padded
        fine_centerline = padded_center

    # Guarantee endpoints are always inside the band (safety margin)
    return fine_band, fine_centerline


def _clamp_rc(rc, shape):
    return (max(0, min(shape[0]-1, rc[0])), max(0, min(shape[1]-1, rc[1])))


def coarse_to_fine_routing(cost, start_rc, end_rc, water_mask, transform,
                            resolution_m=30, dem=None,
                            factor=COARSE_FACTOR, band_km=CORRIDOR_BAND_KM):
    """
    Two-resolution routing strategy:

    ┌─────────────────────────────────────────────────────────────────┐
    │ Pass 0 (Coarse, ~300 m): Dijkstra on 10× downsampled grid      │
    │   → identifies the macro corridor in seconds                    │
    │   → memory: ~1/100 of full fine grid                           │
    │                                                                 │
    │ Band masking: 8 km corridor carved around coarse path           │
    │   → reduces active fine-routing area to ~10–20% of total grid  │
    │                                                                 │
    │ Pass 1+2 (Fine, 30 m): Full two-pass rubber-band routing        │
    │   within the band-masked cost grid                              │
    └─────────────────────────────────────────────────────────────────┘

    FAST_MODE (config): skips the fine pass entirely — coarse path only.
    Useful for rapid scenario screening before committing to a full run.
    """
    rows, cols = cost.shape

    # ── Coarse pass ────────────────────────────────────────────────────────
    log.info(f"Coarse routing: downsampling {factor}× ({rows}×{cols} → "
             f"{rows//factor}×{cols//factor}) …")
    coarse_cost = _downsample_cost(cost, factor)
    coarse_shape = coarse_cost.shape

    start_coarse = _clamp_rc((start_rc[0]//factor, start_rc[1]//factor), coarse_shape)
    end_coarse   = _clamp_rc((end_rc[0]//factor,   end_rc[1]//factor),   coarse_shape)

    coarse_path = find_path(coarse_cost, start_coarse, end_coarse)
    log.info(f"Coarse path: {len(coarse_path)} waypoints "
             f"(~{len(coarse_path)*factor*resolution_m/1000:.1f} km equivalent)")

    # ── FAST_MODE early exit ────────────────────────────────────────────────
    if FAST_MODE:
        log.info("FAST_MODE active — returning coarse path (upsampled to fine grid).")
        # Map coarse coordinates back to fine grid centre points
        fine_path = [(r*factor + factor//2, c*factor + factor//2)
                     for r, c in coarse_path]
        # Clamp to fine grid bounds
        fine_path = [_clamp_rc(rc, (rows, cols)) for rc in fine_path]
        return fine_path

    # ── Build band mask & centerline mask ─────────────────────────────────
    fine_band, fine_centerline = _build_fine_band_mask(
        coarse_path, coarse_shape, (rows, cols), factor, band_km, resolution_m
    )
    # Always include the exact endpoint cells
    fine_band[start_rc[0], start_rc[1]] = True
    fine_band[end_rc[0],   end_rc[1]]   = True
    fine_centerline[start_rc[0], start_rc[1]] = True
    fine_centerline[end_rc[0],   end_rc[1]]   = True

    coverage_pct = fine_band.sum() / fine_band.size * 100
    log.info(f"Corridor band: {fine_band.sum():,} cells ({coverage_pct:.1f}% of fine grid), "
             f"band_km={band_km} km each side")

    # ── Apply band to fine cost ────────────────────────────────────────────
    banded_cost = np.where(fine_band, cost, IMPASSABLE)
    banded_water = np.where(fine_band, water_mask, 0).astype(water_mask.dtype)

    # ── Fine two-pass routing within band ─────────────────────────────────
    log.info("Fine routing within corridor band (with centerline rubber-banding) …")
    return two_pass_routing(
        banded_cost, start_rc, end_rc, banded_water, transform,
        resolution_m=resolution_m, dem=dem, reference_mask=fine_centerline
    )
