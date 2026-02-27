"""
earthwork.py — Phase 7: Earthwork Volume Estimation
=====================================================
Computes cut and fill volumes along the 3D alignment using simplified
trapezoidal cross-sections, then produces a mass-haul (Brückner) diagram.

Mathematical model
------------------
Cross-section at each station is approximated as a **trapezoid**:

    CUT  (design below terrain, h = |cut_fill_m|):
        Bottom width = formation_width_m
        Top    width = formation_width_m + 2 × h × cut_batter_HV
        Area_cut = h × (formation_width_m + h × cut_batter_HV)

    FILL (design above terrain, h = cut_fill_m):
        Bottom width = toe width  = formation_width_m + 2 × h × fill_batter_HV
        Top width    = subgrade   = formation_width_m
        Area_fill = h × (formation_width_m + h × fill_batter_HV)

Volume between consecutive stations i, i+1 (Average-End-Area method):
    ds  = distances_m[i+1] - distances_m[i]
    V_i = (A_i + A_{i+1}) / 2 × ds

Mass-haul (Brückner) ordinate at station i:
    M_i = Σ_{j < i} (fill_vol_j − cut_vol_j × swell_factor)
    Positive → net fill demand (import required)
    Negative → net surplus (spoil required)

Accuracy: ±20–30% typical for preliminary design (adequate for ADB/WB
pre-feasibility studies at this level of DEM resolution).

Public API
----------
    compute_earthwork(va_result, ...) → EarthworkResult
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("highway_alignment")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class StationEarthwork:
    """Earthwork quantities at a single chainage station."""
    station_m: float
    cut_fill_m: float        # signed: >0 fill, <0 cut
    area_cut_m2: float       # trapezoidal cut cross-section area
    area_fill_m2: float      # trapezoidal fill cross-section area
    vol_cut_m3: float        # cumulative cut volume to this station
    vol_fill_m3: float       # cumulative fill volume to this station
    mass_haul_m3: float      # Brückner ordinate (fill - cut×swell); +ve = deficit


@dataclass
class EarthworkResult:
    """Full output of compute_earthwork()."""
    distances_m: np.ndarray         # chainage (m)
    cut_fill_m: np.ndarray          # signed cut/fill depth (m); input from Phase 6
    area_cut_m2: np.ndarray         # trapezoidal cut area at each station
    area_fill_m2: np.ndarray        # trapezoidal fill area at each station
    seg_cut_vol_m3: np.ndarray      # cut volume per segment (between stations)
    seg_fill_vol_m3: np.ndarray     # fill volume per segment
    cumul_cut_m3: np.ndarray        # cumulative cut volume
    cumul_fill_m3: np.ndarray       # cumulative fill volume
    mass_haul_m3: np.ndarray        # Brückner ordinate (cumulative net)
    total_cut_m3: float             # total cut volume
    total_fill_m3: float            # total fill volume
    net_import_m3: float            # fill − cut×swell; >0 = borrow needed
    balance_stations_m: list[float] # chainages where mass-haul crosses zero
    formation_width_m: float        # carriageway + shoulders (m)
    cut_batter_HV: float            # H:V ratio of cut slope
    fill_batter_HV: float           # H:V ratio of fill slope
    swell_factor: float             # loose/compact volume ratio


# ── Step 1: Trapezoidal cross-section areas ───────────────────────────────────

def _compute_areas(cut_fill_m: np.ndarray,
                   formation_width_m: float,
                   cut_batter_HV: float,
                   fill_batter_HV: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute trapezoidal cross-section areas at each station.

    The formation width (subgrade) is the TOTAL width at finished grade level:
        formation_width_m = carriageway_m + 2 × shoulder_m

    For a CUT section (h = depth of cut, h > 0):
        The excavation widens upward at the cut batter slope (H:V = cut_batter_HV).
        Area = h × (formation_width_m + h × cut_batter_HV)
        This is the area of the trapezoid with:
          - base = formation_width_m (at subgrade level)
          - top  = formation_width_m + 2 × h × cut_batter_HV (at natural ground)
          - height = h
        Simplified (each side only):  Area = h × (B + h × cut_batter_HV)
        where B = formation_width_m and cut_batter_HV covers BOTH sides.

    For a FILL section (h = fill height, h > 0):
        The embankment widens downward at fill batter slope.
        Area = h × (formation_width_m + h × fill_batter_HV)

    Returns
    -------
    area_cut_m2, area_fill_m2 : arrays aligned to cut_fill_m.
        Exactly one of the two is nonzero at each station.
    """
    h_cut  = np.where(cut_fill_m < 0, -cut_fill_m, 0.0)   # depth of cut  (>0)
    h_fill = np.where(cut_fill_m > 0,  cut_fill_m, 0.0)   # height of fill (>0)

    area_cut  = h_cut  * (formation_width_m + h_cut  * cut_batter_HV)
    area_fill = h_fill * (formation_width_m + h_fill * fill_batter_HV)

    return area_cut, area_fill


# ── Step 2: Average-End-Area volume integration ───────────────────────────────

def _compute_volumes(distances_m: np.ndarray,
                     area_cut_m2: np.ndarray,
                     area_fill_m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-segment cut and fill volumes using the Average-End-Area method.

    V_i = (A_i + A_{i+1}) / 2 × Δs_i

    Returns (seg_cut_vol, seg_fill_vol) each of length len(distances_m)-1.
    """
    ds = np.diff(distances_m)
    avg_cut  = (area_cut_m2[:-1]  + area_cut_m2[1:])  / 2.0
    avg_fill = (area_fill_m2[:-1] + area_fill_m2[1:]) / 2.0

    seg_cut  = avg_cut  * ds
    seg_fill = avg_fill * ds

    return seg_cut, seg_fill


# ── Step 3: Mass-haul (Brückner) ordinate ────────────────────────────────────

def _compute_mass_haul(seg_cut_m3: np.ndarray,
                        seg_fill_m3: np.ndarray,
                        swell_factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cumulative volumes and the Brückner mass-haul ordinate.

    Brückner ordinate: M_i = Σ (fill_j − cut_j × swell_factor)

    The swell factor accounts for the fact that excavated material swells
    when loosened (e.g. rock swells ~30–40%, soil ~20–25%).  If M is
    negative the design has a cut surplus (route to spoil); positive means
    a fill deficit (borrow required).

    Returns
    -------
    cumul_cut  : cumulative cut volume (m³) — length = n+1 (station-based)
    cumul_fill : cumulative fill volume (m³)
    mass_haul  : Brückner ordinate (m³)
    """
    n = len(seg_cut_m3)  # number of segments = stations - 1
    cumul_cut  = np.zeros(n + 1)
    cumul_fill = np.zeros(n + 1)
    mass_haul  = np.zeros(n + 1)

    for i in range(n):
        cumul_cut[i + 1]  = cumul_cut[i]  + seg_cut_m3[i]
        cumul_fill[i + 1] = cumul_fill[i] + seg_fill_m3[i]
        # Brückner: positive = fill deficit, negative = cut surplus
        mass_haul[i + 1]  = mass_haul[i] + (
            seg_fill_m3[i] - seg_cut_m3[i] * swell_factor
        )

    return cumul_cut, cumul_fill, mass_haul


# ── Step 4: Balance point detection ──────────────────────────────────────────

def _find_balance_stations(distances_m: np.ndarray,
                            mass_haul_m3: np.ndarray) -> list[float]:
    """
    Find chainages where the mass-haul curve crosses zero (balance points).

    These mark transitions between net-fill and net-cut zones — economically
    significant because material should be hauled within a balance zone rather
    than imported/exported across it.

    Uses linear interpolation between adjacent stations straddling zero.
    """
    balance = []
    for i in range(len(mass_haul_m3) - 1):
        m0, m1 = mass_haul_m3[i], mass_haul_m3[i + 1]
        if m0 * m1 < 0:   # sign change → zero crossing
            # Linear interpolation
            t = -m0 / (m1 - m0)
            s = distances_m[i] + t * (distances_m[i + 1] - distances_m[i])
            balance.append(round(float(s), 1))
    return balance


# ── Step 5: CSV export ────────────────────────────────────────────────────────

def export_earthwork_csv(result: EarthworkResult, output_path: str) -> None:
    """
    Write station-by-station earthwork quantities to CSV.

    Columns:
      chainage_m, cut_fill_m, area_cut_m2, area_fill_m2,
      cumul_cut_m3, cumul_fill_m3, mass_haul_m3
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chainage_m", "cut_fill_m",
            "area_cut_m2", "area_fill_m2",
            "cumul_cut_m3", "cumul_fill_m3", "mass_haul_m3",
        ])
        n = len(result.distances_m)
        for i in range(n):
            writer.writerow([
                round(result.distances_m[i], 1),
                round(result.cut_fill_m[i], 3),
                round(result.area_cut_m2[i], 2),
                round(result.area_fill_m2[i], 2),
                round(result.cumul_cut_m3[i], 1),
                round(result.cumul_fill_m3[i], 1),
                round(result.mass_haul_m3[i], 1),
            ])
    log.info(f"Earthwork CSV exported: {output_path}  ({n} stations)")


# ── Public entry point ────────────────────────────────────────────────────────

def compute_earthwork(
    distances_m: np.ndarray,
    cut_fill_m: np.ndarray,
    formation_width_m: float,
    cut_batter_HV: float = 1.0,
    fill_batter_HV: float = 1.5,
    swell_factor: float = 1.25,
) -> EarthworkResult:
    """
    Compute cut/fill volumes and mass-haul curve for the vertical alignment.

    Parameters
    ----------
    distances_m : 1-D array
        Cumulative chainage along the alignment (m).  Must be monotonically
        increasing.
    cut_fill_m : 1-D array
        Signed cut/fill depth at each station (m).
        Positive  = fill (design ABOVE terrain → embankment).
        Negative  = cut  (design BELOW terrain → excavation).
        Typically sourced from ``VerticalAlignmentResult.cut_fill_m``.
    formation_width_m : float
        Total subgrade formation width (carriageway + both shoulders, m).
        Myanmar DRD typical values:
          - Rural trunk, 2-lane: 11.0 m  (7.0 m carriageway + 2×2.0 m shoulders)
          - Expressway, 4-lane:  24.0 m
    cut_batter_HV : float
        Horizontal : Vertical ratio of cut face.
        Default 1.0 (1H:1V = 45°) — typical for soft rock / stiff clay.
        Use 0.5 for hard rock, 1.5 for loose soil.
    fill_batter_HV : float
        H:V ratio of fill/embankment face.
        Default 1.5 (1.5H:1V = ~34°) — Myanmar DRD standard.
    swell_factor : float
        Loose volume / bank volume ratio for excavated material.
        Default 1.25 (25% swell) — typical mixed soil / decomposed rock.
        Use 1.4–1.5 for granite, 1.1 for dense clay.

    Returns
    -------
    EarthworkResult
        Full result dataclass with all per-station arrays and summary totals.
    """
    distances_m = np.asarray(distances_m, dtype=np.float64)
    cut_fill_m  = np.asarray(cut_fill_m,  dtype=np.float64)

    if len(distances_m) != len(cut_fill_m):
        raise ValueError(
            f"distances_m ({len(distances_m)}) and cut_fill_m "
            f"({len(cut_fill_m)}) must have the same length."
        )
    if len(distances_m) < 2:
        raise ValueError("Need at least 2 stations to compute volumes.")

    log.info(
        f"Earthwork: formation_w={formation_width_m:.1f} m  "
        f"cut_batter={cut_batter_HV:.1f}H:1V  fill_batter={fill_batter_HV:.1f}H:1V  "
        f"swell={swell_factor:.2f}"
    )

    # Step 1 — trapezoidal areas
    area_cut, area_fill = _compute_areas(
        cut_fill_m, formation_width_m, cut_batter_HV, fill_batter_HV
    )

    # Step 2 — segment volumes (Average-End-Area)
    seg_cut, seg_fill = _compute_volumes(distances_m, area_cut, area_fill)

    # Step 3 — cumulative + mass-haul
    cumul_cut, cumul_fill, mass_haul = _compute_mass_haul(seg_cut, seg_fill, swell_factor)

    total_cut  = float(cumul_cut[-1])
    total_fill = float(cumul_fill[-1])
    net_import = float(mass_haul[-1])   # final mass-haul ordinate

    # Step 4 — balance stations
    balance_stations = _find_balance_stations(distances_m, mass_haul)

    log.info(
        f"Earthwork complete:  cut={total_cut/1e6:.3f} Mm³  "
        f"fill={total_fill/1e6:.3f} Mm³  "
        f"net={'import' if net_import > 0 else 'spoil'} "
        f"{abs(net_import)/1e3:.1f} km³  "
        f"balance_points={len(balance_stations)}"
    )

    return EarthworkResult(
        distances_m=distances_m,
        cut_fill_m=cut_fill_m,
        area_cut_m2=area_cut,
        area_fill_m2=area_fill,
        seg_cut_vol_m3=seg_cut,
        seg_fill_vol_m3=seg_fill,
        cumul_cut_m3=cumul_cut,
        cumul_fill_m3=cumul_fill,
        mass_haul_m3=mass_haul,
        total_cut_m3=total_cut,
        total_fill_m3=total_fill,
        net_import_m3=net_import,
        balance_stations_m=balance_stations,
        formation_width_m=formation_width_m,
        cut_batter_HV=cut_batter_HV,
        fill_batter_HV=fill_batter_HV,
        swell_factor=swell_factor,
    )
