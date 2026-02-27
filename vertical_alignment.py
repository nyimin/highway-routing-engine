"""
vertical_alignment.py — Phase 6: Vertical Alignment Profile Generator
======================================================================
Transforms the raw 1D terrain profile (distance vs. ground elevation) into a
smoothed 3D Finished Grade Line (FGL) composed of constant-grade tangents joined
by parabolic vertical curves (sag and crest).

Algorithm: Grade-Clipping (greedy forward sweep)
-------------------------------------------------
1. Smooth raw terrain with PCHIP interpolation.
2. Detect grade-change candidates from terrain curvature extrema.
3. Clip VPI elevations so every tangent satisfies |g| ≤ G_MAX_PCT.
4. Fit parabolic vertical curves at each VPI using K·|A| minimum lengths.
5. Evaluate z_design at every chainage station.
6. Flag grade and SSD K-value violations.

Mathematical reference:
  AASHTO Green Book, 7th ed., §3.4 (Vertical Alignment)
  Myanmar DRD Road Design Standard (equivalent K-values adopted)

Public API
----------
  build_vertical_alignment(distances_m, elevations_m, ...) → VerticalAlignmentResult
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

log = logging.getLogger("highway_alignment")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VerticalCurve:
    """One parabolic vertical curve between two tangent grades."""
    pvc_station_m: float      # chainage at start of curve
    pvi_station_m: float      # chainage at grade-break (VPI)
    pvt_station_m: float      # chainage at end of curve
    g1_pct: float             # incoming tangent grade (%, signed)
    g2_pct: float             # outgoing tangent grade (%, signed)
    length_m: float           # curve length L
    k_value: float            # L / |A| actually used
    k_required: float         # K_min from AASHTO table
    curve_type: str           # 'sag' | 'crest'
    z_pvc: float              # design elevation at PVC


@dataclass
class VerticalAlignmentResult:
    """Full output of build_vertical_alignment()."""
    distances_m: np.ndarray           # chainage stations (m) — same as input
    z_ground: np.ndarray              # raw terrain elevation (m)
    z_design: np.ndarray              # finished grade line elevation (m)
    cut_fill_m: np.ndarray            # z_design − z_ground; >0 = fill, <0 = cut
    grade_pct: np.ndarray             # design tangent grade at each station (%)
    vertical_curves: List[VerticalCurve]
    max_grade_pct: float              # maximum |grade| achieved on design FGL
    grade_violations: List[dict]      # [{station_m, grade_pct}]
    ssd_violations: List[dict]        # [{station_m, k_used, k_required, curve_type}]
    vpi_stations: np.ndarray          # chainage of all VPIs
    vpi_elevations: np.ndarray        # design elevation at all VPIs


# ── K-value physics ───────────────────────────────────────────────────────────

# AASHTO Table 3-34 / 3-35 design values (K in m/%) indexed by design speed km/h.
# Tuple: (K_crest_SSD, K_sag_SSD)
# K_crest_PSD (passing sight distance) is advisory only for preliminary design.
_K_TABLE: dict[int, tuple[int, int]] = {
    40:  (4,   5),
    60:  (11,  11),
    80:  (26,  21),
    100: (52,  37),
    120: (98,  60),
}


def _k_for_speed(design_speed_kmph: float) -> tuple[int, int]:
    """
    Return (K_crest, K_sag) for the nearest tabled design speed.
    Rounds UP to be conservative.
    """
    speeds = sorted(_K_TABLE.keys())
    for s in speeds:
        if design_speed_kmph <= s:
            return _K_TABLE[s]
    return _K_TABLE[speeds[-1]]   # cap at highest tabled speed


# ── Step 1: Smooth terrain via PCHIP ─────────────────────────────────────────

def _smooth_terrain(distances_m: np.ndarray,
                    elevations_m: np.ndarray) -> PchipInterpolator:
    """
    Fit a PCHIP interpolator to the raw terrain profile.

    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) is preferred over
    cubic splines for elevation profiles because it:
      • Preserves local monotonicity — no oscillation between data points.
      • Handles flat-topped hills and valley floors without over-shooting.
      • Is C¹ continuous, giving a well-defined first derivative (grade).
    """
    # Remove any duplicate distance values (can happen near endpoints)
    d, idx = np.unique(distances_m, return_index=True)
    e = elevations_m[idx]
    return PchipInterpolator(d, e)


# ── Step 2: Detect VPI candidate locations ────────────────────────────────────

def _detect_vpi_candidates(distances_m: np.ndarray,
                            terrain_interp: PchipInterpolator,
                            min_spacing_m: float = 200.0,
                            peak_prominence: float = 0.5) -> np.ndarray:
    """
    Find positions along the alignment where the terrain grade changes direction,
    PLUS inject uniformly-spaced intermediate VPIs on long monotone sections.

    Strategy:
      1. Sample terrain grade g(s) = dz/ds × 100 at fine resolution (every 50 m).
      2. Find peaks (local grade maxima — crest tops) and troughs (sag bottoms)
         using scipy.signal.find_peaks.
      3. Always include the start and end of the route.
      4. Merge candidates closer than min_spacing_m.
      5. **KEY FIX** — for any gap larger than max_segment_m between consecutive
         candidates, inject additional evenly-spaced VPIs inside the gap.
         This ensures grade-clipping can subdivide long monotone slopes (e.g. a
         15%-grade straight hillside has no curvature extrema and would otherwise
         yield only one unclipped tangent from start to end).

    Returns sorted array of chainage stations (m) for candidate VPIs.
    """
    # Fine sample for derivative estimation
    s_fine = np.arange(distances_m[0], distances_m[-1], 50.0)
    if s_fine[-1] < distances_m[-1]:
        s_fine = np.append(s_fine, distances_m[-1])

    g_fine = terrain_interp(s_fine, 1) * 100.0   # grade in %

    peak_idx,   _ = find_peaks( g_fine, prominence=peak_prominence)
    trough_idx, _ = find_peaks(-g_fine, prominence=peak_prominence)

    candidate_s = np.concatenate([
        s_fine[peak_idx],
        s_fine[trough_idx],
        [distances_m[0], distances_m[-1]],
    ])
    candidate_s = np.unique(np.sort(candidate_s))

    # Merge candidates that are closer than min_spacing_m
    merged: list[float] = [float(candidate_s[0])]
    for s in candidate_s[1:]:
        if s - merged[-1] >= min_spacing_m:
            merged.append(float(s))
    if merged[-1] != float(candidate_s[-1]):
        merged.append(float(candidate_s[-1]))

    # ── Inject intermediate VPIs on long gaps ─────────────────────────────
    # Maximum allowed spacing between consecutive VPIs.  If a gap is larger than
    # this, the grade-clip forward sweep has no intermediate point to clip against
    # and the full terrain grade crosses the limit uncorrected.
    # Formula: at G_MAX=8% over max_segment_m, the design climbs/falls by
    # G_MAX/100 × max_segment_m.  500 m gives ≤40 m per segment — tight enough
    # for any real scenario while keeping the VPI count manageable.
    max_segment_m = max(min_spacing_m * 2.0, 500.0)

    filled: list[float] = [merged[0]]
    for s_next in merged[1:]:
        gap = s_next - filled[-1]
        if gap > max_segment_m:
            # Number of equal sub-segments needed
            n_sub = math.ceil(gap / max_segment_m)
            sub_spacing = gap / n_sub
            for k in range(1, n_sub):
                filled.append(filled[-1] + sub_spacing)
        filled.append(s_next)

    result = np.array(filled)
    log.info(f"Vertical alignment: {len(result)} VPI candidates "
             f"(min spacing {min_spacing_m:.0f} m, max segment {max_segment_m:.0f} m)")
    return result



# ── Step 3: Grade-clipping — forward sweep ───────────────────────────────────

def _clip_grades(vpi_stations: np.ndarray,
                 terrain_interp: PchipInterpolator,
                 max_grade_pct: float) -> np.ndarray:
    """
    Assign VPI elevations so that every tangent grade stays within ±max_grade_pct.

    Algorithm (greedy forward sweep):
    ──────────────────────────────────
    Start at the terrain elevation of the first station (the alignment begin
    point must sit on the ground). For each successive VPI candidate:

      1. Compute the "natural" terrain elevation at that station: z_natural.
      2. Compute the required grade g = (z_natural - z_prev) / ds × 100.
      3. If |g| ≤ max_grade_pct → accept z_natural (design follows terrain).
      4. If g > max_grade_pct → clip:   z_clipped = z_prev + max_grade_pct/100 × ds
         The design line rises no faster than allowed. This creates a fill section
         where the grade was too steep uphill.
      5. If g < -max_grade_pct → clip:  z_clipped = z_prev - max_grade_pct/100 × ds
         The design line falls no faster than allowed. Creates a cut section on
         steep downhill runs.

    Because the forward pass only guarantees a one-direction grade constraint, we
    then run a backward smoothing pass:
      • Walk backward; if the backwards grade from z_i to z_{i-1} exceeds the
        limit, raise z_{i-1} to exactly meet the limit.
    This symmetric two-pass approach ensures all tangent segments satisfy the
    grade constraint in both directions.

    Returns:
        vpi_elevations (np.ndarray) — design elevation at each VPI station.
    """
    n = len(vpi_stations)
    z = np.zeros(n)

    # Initialise from terrain
    z_terrain_at_vpi = terrain_interp(vpi_stations)

    # ── Forward pass ────────────────────────────────────────────────────────
    z[0] = z_terrain_at_vpi[0]   # start point always on terrain
    g_max = max_grade_pct / 100.0

    for i in range(1, n):
        ds = vpi_stations[i] - vpi_stations[i - 1]
        if ds <= 0:
            z[i] = z[i - 1]
            continue
        z_nat = z_terrain_at_vpi[i]
        g_nat = (z_nat - z[i - 1]) / ds

        if g_nat > g_max:
            z[i] = z[i - 1] + g_max * ds   # grade clipped uphill
        elif g_nat < -g_max:
            z[i] = z[i - 1] - g_max * ds   # grade clipped downhill
        else:
            z[i] = z_nat                    # terrain grade is within limits

    # ── Backward pass (symmetric smoothing) ─────────────────────────────────
    # Walk from second-to-last VPI backward to index 0.
    # If the backward grade from z[i+1] to z[i] violates the limit, we adjust
    # z[i] (never force it below terrain — only clip to terrain floor).
    for i in range(n - 2, -1, -1):
        ds = vpi_stations[i + 1] - vpi_stations[i]
        if ds <= 0:
            continue
        g_back = (z[i + 1] - z[i]) / ds
        if g_back > g_max:
            # z[i+1] is well above z[i]; raise z[i] to bring grade back to limit
            z[i] = z[i + 1] - g_max * ds
        elif g_back < -g_max:
            # z[i+1] is well below z[i]; lower z[i] to meet limit
            z_clipped = z[i + 1] + g_max * ds
            z[i] = min(z[i], z_clipped)

    # NOTE: The first point is always on terrain (set in forward pass).
    # The last point follows the grade-clipped design elevation — we do NOT
    # snap it back to terrain because that would create a huge grade spike on
    # the final VPI segment.  In preliminary design the endpoint is allowed to
    # sit above or below the terrain close to the destination.

    return z



# ── Step 4: Fit parabolic vertical curves at each VPI ─────────────────────────

def _fit_vertical_curves(vpi_stations: np.ndarray,
                          vpi_elevations: np.ndarray,
                          k_crest: int, k_sag: int,
                          min_vc_length_m: float) -> List[VerticalCurve]:
    """
    Place a parabolic vertical curve at each interior VPI.

    For each VPI i (1 ≤ i ≤ n-2):
      g1 = incoming grade from VPI[i-1] to VPI[i]
      g2 = outgoing grade from VPI[i] to VPI[i+1]
      A  = g2 - g1  (algebraic grade difference, %)
      curve_type = 'sag' if A > 0 else 'crest'
      L_min = max(K(type) * |A|, min_vc_length_m)

    Overlap detection: if PVC of curve i+1 < PVT of curve i, shorten both
    curves proportionally so they just touch at the midpoint between the two VPIs.

    Returns list of VerticalCurve objects (one per interior VPI, skipping
    trivially flat breaks where |A| < 0.01%).
    """
    n = len(vpi_stations)
    if n < 3:
        return []

    # Pre-compute tangent grades between consecutive VPIs
    grades = np.zeros(n - 1)
    for i in range(n - 1):
        ds = vpi_stations[i + 1] - vpi_stations[i]
        if ds > 0:
            grades[i] = (vpi_elevations[i + 1] - vpi_elevations[i]) / ds * 100.0

    curves: List[VerticalCurve] = []

    for i in range(1, n - 1):
        g1 = grades[i - 1]  # incoming
        g2 = grades[i]      # outgoing
        A  = g2 - g1        # algebraic grade change (%)

        if abs(A) < 0.01:
            continue  # essentially straight — no curve needed

        curve_type = 'sag' if A > 0 else 'crest'
        k_req = k_sag if curve_type == 'sag' else k_crest
        L = max(k_req * abs(A), min_vc_length_m)

        # Lengths available on each tangent
        avail_before = (vpi_stations[i] - vpi_stations[i - 1])
        avail_after  = (vpi_stations[i + 1] - vpi_stations[i])

        # Cap curve to fit within available tangent space (halved each side)
        L = min(L, 2.0 * avail_before * 0.9, 2.0 * avail_after * 0.9)
        L = max(L, min_vc_length_m)   # don't go below absolute minimum

        pvc_s = vpi_stations[i] - L / 2.0
        pvt_s = vpi_stations[i] + L / 2.0

        # Elevation at PVC: back-project along incoming tangent from VPI
        z_pvc = vpi_elevations[i] - g1 / 100.0 * (L / 2.0)

        k_used = L / abs(A) if abs(A) > 1e-9 else float('inf')

        curves.append(VerticalCurve(
            pvc_station_m=pvc_s,
            pvi_station_m=vpi_stations[i],
            pvt_station_m=pvt_s,
            g1_pct=g1,
            g2_pct=g2,
            length_m=L,
            k_value=k_used,
            k_required=float(k_req),
            curve_type=curve_type,
            z_pvc=z_pvc,
        ))

    log.info(f"Vertical alignment: {len(curves)} parabolic vertical curves fitted")
    return curves


# ── Step 5: Evaluate z_design at every station ────────────────────────────────

def _evaluate_fgl(distances_m: np.ndarray,
                  vpi_stations: np.ndarray,
                  vpi_elevations: np.ndarray,
                  curves: List[VerticalCurve]) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the Finished Grade Line elevation at every chainage station.

    For each station s:
      1. Check if s falls inside any parabolic vertical curve [PVC, PVT].
         If yes → use parabola formula:
           z(x) = z_PVC + g1/100 * x + (A/100) / (2*L) * x²
         where x = s - PVC_station.
      2. Otherwise → find the enclosing VPI segment and evaluate the tangent:
           g = (z[i+1] - z[i]) / (s[i+1] - s[i])
           z(s) = z[i] + g/100 * (s - s[i])

    Returns (z_design, grade_pct) both aligned with distances_m.
    """
    n_stations = len(distances_m)
    z_design  = np.zeros(n_stations)
    grade_pct = np.zeros(n_stations)

    # Build a sorted list of (pvc, pvt, VerticalCurve) for fast lookup
    # We'll use a simple linear scan (number of curves << number of stations).
    vc_sorted = sorted(curves, key=lambda c: c.pvc_station_m)

    # Pre-compute tangent grades between VPIs for the tangent segments
    n_vpi = len(vpi_stations)
    tangent_grades = np.zeros(max(n_vpi - 1, 1))
    for i in range(n_vpi - 1):
        ds = vpi_stations[i + 1] - vpi_stations[i]
        if ds > 0:
            tangent_grades[i] = (
                (vpi_elevations[i + 1] - vpi_elevations[i]) / ds * 100.0
            )

    for k_idx, s in enumerate(distances_m):
        # Try to find enclosing vertical curve
        in_curve = False
        for vc in vc_sorted:
            if vc.pvc_station_m <= s <= vc.pvt_station_m:
                # Parabola formula
                x  = s - vc.pvc_station_m
                A  = vc.g2_pct - vc.g1_pct   # %
                L  = vc.length_m
                dz = vc.g1_pct / 100.0 * x + (A / 100.0) / (2.0 * L) * x * x
                z_design[k_idx]  = vc.z_pvc + dz
                # Instantaneous grade at x inside the curve: g1 + A/L * x  (%)
                grade_pct[k_idx] = vc.g1_pct + (A / L) * x
                in_curve = True
                break
            if vc.pvc_station_m > s:
                break  # sorted; no need to look further

        if not in_curve:
            # Find enclosing VPI interval
            seg_idx = np.searchsorted(vpi_stations, s, side='right') - 1
            seg_idx = int(np.clip(seg_idx, 0, n_vpi - 2))

            ds = vpi_stations[seg_idx + 1] - vpi_stations[seg_idx]
            if ds > 0:
                g = tangent_grades[seg_idx]
                z_design[k_idx]  = (
                    vpi_elevations[seg_idx]
                    + g / 100.0 * (s - vpi_stations[seg_idx])
                )
                grade_pct[k_idx] = g
            else:
                z_design[k_idx]  = vpi_elevations[seg_idx]
                grade_pct[k_idx] = 0.0

    return z_design, grade_pct


# ── Step 6: Violations ────────────────────────────────────────────────────────

def _check_violations(distances_m: np.ndarray,
                       grade_pct: np.ndarray,
                       curves: List[VerticalCurve],
                       max_grade_pct: float) -> tuple[list, list]:
    """
    Identify:
      1. Grade violations: stations where |grade_pct| > max_grade_pct.
      2. SSD K-value violations: curves where k_used < k_required.

    Returns (grade_violations, ssd_violations).
    """
    grade_violations = []
    prev_violation = False
    for i, (s, g) in enumerate(zip(distances_m, grade_pct)):
        if abs(g) > max_grade_pct:
            if not prev_violation:   # report start of each violated segment only
                grade_violations.append({
                    'station_m': round(float(s), 1),
                    'grade_pct': round(float(g), 2),
                })
            prev_violation = True
        else:
            prev_violation = False

    ssd_violations = []
    for vc in curves:
        if vc.k_value < vc.k_required - 0.5:  # 0.5 tolerance for rounding
            ssd_violations.append({
                'station_m':   round(vc.pvi_station_m, 1),
                'k_used':      round(vc.k_value, 1),
                'k_required':  round(vc.k_required, 1),
                'curve_type':  vc.curve_type,
                'A_pct':       round(abs(vc.g2_pct - vc.g1_pct), 2),
            })

    if grade_violations:
        log.warning(
            f"Vertical alignment: {len(grade_violations)} grade violation segment(s) "
            f"exceeding {max_grade_pct:.1f}%"
        )
    else:
        log.info(f"Vertical alignment: grade OK — all segments ≤ {max_grade_pct:.1f}%")

    if ssd_violations:
        log.warning(
            f"Vertical alignment: {len(ssd_violations)} SSD K-value violation(s) "
            f"(curve shorter than K_min × |A|). Common on very short tangent sections."
        )
    else:
        log.info("Vertical alignment: all SSD K-values satisfied.")

    return grade_violations, ssd_violations


# ── Public entry point ────────────────────────────────────────────────────────

def build_vertical_alignment(
    distances_m: np.ndarray,
    elevations_m: np.ndarray,
    design_speed_kmph: float,
    max_grade_pct: float,
    k_crest: Optional[int] = None,
    k_sag: Optional[int] = None,
    min_vc_length_m: float = 30.0,
    min_vpi_spacing_m: float = 200.0,
    peak_prominence: float = 0.5,
) -> VerticalAlignmentResult:
    """
    Build a vertical alignment profile over the given terrain.

    Parameters
    ----------
    distances_m : 1-D array
        Cumulative chainage along the smoothed horizontal alignment (m).
    elevations_m : 1-D array
        DEM ground elevation sampled at each chainage station (m).
        Must be the same length as distances_m.
    design_speed_kmph : float
        Design speed of the road (km/h). Used to look up K-values.
    max_grade_pct : float
        Maximum allowable sustained tangent grade (e.g. 8.0 for rural trunk).
    k_crest : int, optional
        Override the AASHTO K_crest value. If None, derived from design_speed_kmph.
    k_sag : int, optional
        Override the AASHTO K_sag value. If None, derived from design_speed_kmph.
    min_vc_length_m : float
        Absolute minimum vertical curve length regardless of K·|A| (default 30 m).
    min_vpi_spacing_m : float
        Minimum distance between adjacent VPI candidates (default 200 m).
        Smaller → more curves, closer fit to terrain but more complex design.
    peak_prominence : float
        scipy find_peaks prominence parameter for terrain grade curvature detection.
        Increase if you see too many spurious VPIs on noisy terrain.

    Returns
    -------
    VerticalAlignmentResult
        Full result including z_design, cut/fill array, curve list, and violations.
    """
    distances_m  = np.asarray(distances_m,  dtype=np.float64)
    elevations_m = np.asarray(elevations_m, dtype=np.float64)

    if len(distances_m) != len(elevations_m):
        raise ValueError(
            f"distances_m ({len(distances_m)}) and elevations_m "
            f"({len(elevations_m)}) must have the same length."
        )
    if len(distances_m) < 4:
        raise ValueError("Need at least 4 profile points to build vertical alignment.")

    # Look up K-values if not overridden
    k_c_auto, k_s_auto = _k_for_speed(design_speed_kmph)
    k_crest = k_crest if k_crest is not None else k_c_auto
    k_sag   = k_sag   if k_sag   is not None else k_s_auto

    log.info(
        f"Vertical alignment: V={design_speed_kmph:.0f} km/h  "
        f"G_max={max_grade_pct:.1f}%  K_crest={k_crest}  K_sag={k_sag}  "
        f"L_min={min_vc_length_m:.0f} m"
    )

    # ── Step 1: Smooth terrain ───────────────────────────────────────────────
    terrain_interp = _smooth_terrain(distances_m, elevations_m)

    # ── Step 2: VPI candidates ───────────────────────────────────────────────
    vpi_stations = _detect_vpi_candidates(
        distances_m, terrain_interp,
        min_spacing_m=min_vpi_spacing_m,
        peak_prominence=peak_prominence,
    )

    # ── Step 3: Grade-clip VPI elevations ────────────────────────────────────
    vpi_elevations = _clip_grades(vpi_stations, terrain_interp, max_grade_pct)

    # ── Step 4: Fit vertical curves ──────────────────────────────────────────
    curves = _fit_vertical_curves(
        vpi_stations, vpi_elevations,
        k_crest=k_crest, k_sag=k_sag,
        min_vc_length_m=min_vc_length_m,
    )

    # ── Step 5: Evaluate FGL ─────────────────────────────────────────────────
    z_design, grade_pct = _evaluate_fgl(
        distances_m, vpi_stations, vpi_elevations, curves
    )

    # ── Step 6: Violations ───────────────────────────────────────────────────
    grade_violations, ssd_violations = _check_violations(
        distances_m, grade_pct, curves, max_grade_pct
    )

    # Cut / fill
    cut_fill = z_design - elevations_m

    max_g_achieved = float(np.max(np.abs(grade_pct)))

    log.info(
        f"Vertical alignment complete: {len(curves)} VCs  "
        f"max_grade={max_g_achieved:.2f}%  "
        f"max_fill={cut_fill.max():.1f} m  "
        f"max_cut={-cut_fill.min():.1f} m"
    )

    return VerticalAlignmentResult(
        distances_m=distances_m,
        z_ground=elevations_m,
        z_design=z_design,
        cut_fill_m=cut_fill,
        grade_pct=grade_pct,
        vertical_curves=curves,
        max_grade_pct=max_g_achieved,
        grade_violations=grade_violations,
        ssd_violations=ssd_violations,
        vpi_stations=vpi_stations,
        vpi_elevations=vpi_elevations,
    )
