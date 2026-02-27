"""
cost_model.py — Phase 9: Parametric Cost Model
===============================================
Aggregates all computed project data into a project-level cost estimate
(USD) with itemized components, contingency, and per-km summary.

Cost components
---------------
1. Earthwork cut          — ew_result.total_cut_m3 × EARTHWORK_CUT_RATE_USD_M3
2. Earthwork import fill  — ew_result.net_import_m3 × EARTHWORK_FILL_RATE_USD_M3
                            (only if net_import_m3 > 0; surplus spoil is free)
3. Pavement               — formation width × length × PAVEMENT_RATE_USD_M2
4. Bridges                — si_result.total_bridge_cost_usd (already in Phase 8)
5. Culverts               — si_result.total_culvert_cost_usd (already in Phase 8)
6. Land acquisition       — corridor area (ha) × LULC-weighted rate
7. Environmental          — ENV_MITIGATION_FACTOR × civil subtotal (items 1–6)
8. Contingency            — CONTINGENCY_FACTOR × civil subtotal
9. Engineering & Admin    — ENGINEERING_FACTOR × civil subtotal

Accuracy: ±25–30% at preliminary design stage (ADB/WB Category B study).

Public API
----------
    compute_cost_model(meta, ew_result, si_result,
                       scenario_profile, lulc_wgs=None) → CostModelResult
    export_cost_csv(result, output_path)
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("highway_alignment")


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class CostModelResult:
    """
    Full output of compute_cost_model().

    All monetary values are in USD.
    Percentage items (env, contingency, engineering) are computed from
    civil_subtotal to keep them independent of each other.
    """
    # ── Civil components ──────────────────────────────────────────────────
    earthwork_cut_usd:      float   # USD for cut excavation
    earthwork_fill_usd:     float   # USD for imported borrow fill (0 if net surplus)
    pavement_usd:           float   # USD for flexible pavement surface
    bridges_usd:            float   # USD from Phase 8 structure inventory
    culverts_usd:           float   # USD from Phase 8 structure inventory
    land_acquisition_usd:   float   # USD for RoW land compensation

    # ── Soft costs ────────────────────────────────────────────────────────
    environmental_usd:      float   # ENV_MITIGATION_FACTOR × civil_subtotal
    civil_subtotal_usd:     float   # sum of all 6 civil components
    contingency_usd:        float   # CONTINGENCY_FACTOR × civil_subtotal
    engineering_usd:        float   # ENGINEERING_FACTOR × civil_subtotal

    # ── Project total ─────────────────────────────────────────────────────
    total_project_cost_usd: float   # civil_subtotal + env + contingency + eng
    cost_per_km_usd:        float   # total / total_length_km

    # ── Input summary ─────────────────────────────────────────────────────
    total_length_km:        float
    formation_width_m:      float
    pavement_area_m2:       float   # carriageway area charged for pavement
    land_acquisition_ha:    float   # total corridor area (ha)
    land_acq_rate_usd_ha:   float   # effective weighted rate (USD/ha)

    # ── Audit trail ───────────────────────────────────────────────────────
    assumptions: dict = field(default_factory=dict)


# ── Helper: LULC-weighted land acquisition rate ───────────────────────────────

def _lulc_weighted_rate(lulc_wgs, land_acq_rates: dict,
                         default_rate: float) -> float:
    """
    Compute area-weighted average land acquisition rate (USD/ha) from the
    LULC GeoDataFrame.

    If lulc_wgs is None / empty, returns the default rate.
    Any LULC category not found in land_acq_rates uses the default rate.
    """
    if lulc_wgs is None:
        return default_rate
    try:
        n = len(lulc_wgs)
    except (TypeError, AttributeError):
        return default_rate
    if n == 0:
        return default_rate

    # Prefer 'landuse' column; fall back to 'natural' or 'leisure'
    tag_col = None
    for col in ("landuse", "natural", "leisure"):
        if col in lulc_wgs.columns:
            tag_col = col
            break

    if tag_col is None:
        return default_rate

    total_area = 0.0
    weighted_sum = 0.0
    for _, row in lulc_wgs.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Use the planar area in the GDF's CRS.
        # If WGS-84 (degrees), approximate with 1 deg² ≈ 1.2×10¹⁰ m².
        # Area is only used as a weight here — absolute value doesn't matter.
        try:
            area = float(geom.area)
        except Exception:
            area = 1.0
        category = str(row.get(tag_col, "")) if hasattr(row, "get") else ""
        rate = land_acq_rates.get(category.lower(), default_rate)
        weighted_sum += rate * area
        total_area += area

    if total_area <= 0.0:
        return default_rate
    return weighted_sum / total_area


# ── Public entry point ────────────────────────────────────────────────────────

def compute_cost_model(
    meta:             dict,
    ew_result,                          # EarthworkResult | None
    si_result,                          # StructureInventory | None
    scenario_profile: str  = "rural_trunk",
    lulc_wgs                = None,     # GeoDataFrame | None (WGS-84)
    *,
    # ── Unit rates (override for sensitivity analysis) ────────────────────
    cut_rate_usd_m3:    float = 8.0,
    fill_rate_usd_m3:   float = 15.0,
    pavement_rate_m2:   float = 120.0,
    corridor_width_m:   float = 60.0,
    land_acq_default:   float = 7_500.0,
    land_acq_rates:     Optional[dict] = None,
    env_factor:         float = 0.03,
    contingency_factor: float = 0.20,
    engineering_factor: float = 0.10,
) -> CostModelResult:
    """
    Build a project-level parametric cost estimate from pipeline outputs.

    Parameters
    ----------
    meta : dict
        Output of compute_metadata() from geometry_utils.  Must contain
        'total_length_km' and should contain 'formation_width_m'.
    ew_result : EarthworkResult | None
        Phase 7 earthwork result.  If None, earthwork costs default to 0.
    si_result : StructureInventory | None
        Phase 8 structure inventory.  If None, structure costs default to 0.
    scenario_profile : str
        One of 'expressway', 'rural_trunk', 'mountain_road'.
        Controls default formation width if not in meta.
    lulc_wgs : GeoDataFrame | None
        OSM land-use polygons in WGS-84.  Used to compute LULC-weighted
        land acquisition rate.  If None, uses land_acq_default.
    cut_rate_usd_m3 : float
        Unit rate for earthwork cut (USD/m³).
    fill_rate_usd_m3 : float
        Unit rate for imported borrow fill (USD/m³).
    pavement_rate_m2 : float
        Flexible pavement unit rate (USD/m²).
    corridor_width_m : float
        Total RoW acquisition width (m).
    land_acq_default : float
        Fallback land acquisition rate when LULC data unavailable (USD/ha).
    land_acq_rates : dict | None
        Per-LULC-category rates (USD/ha).  If None, uses config defaults.
    env_factor : float
        Environmental mitigation as fraction of civil subtotal.
    contingency_factor : float
        Contingency as fraction of civil subtotal.
    engineering_factor : float
        Engineering & administration as fraction of civil subtotal.

    Returns
    -------
    CostModelResult
    """
    from config import FORMATION_WIDTH_M as _FW

    if land_acq_rates is None:
        from config import LAND_ACQ_RATES as _LAR
        land_acq_rates = _LAR

    # ── Input values ──────────────────────────────────────────────────────
    total_length_km = float(meta.get("total_length_km", 0.0))
    # Formation width: meta overrides config default
    formation_width_m = float(
        meta.get("formation_width_m",
                 _FW.get(scenario_profile, 11.0))
    )

    log.info(
        f"CostModel: length={total_length_km:.1f} km  "
        f"formation={formation_width_m:.1f} m  "
        f"scenario={scenario_profile}"
    )

    # ── 1 & 2. Earthwork ──────────────────────────────────────────────────
    if ew_result is not None:
        ew_cut_usd  = ew_result.total_cut_m3  * cut_rate_usd_m3
        net_import  = ew_result.net_import_m3
        ew_fill_usd = max(0.0, net_import) * fill_rate_usd_m3
        log.info(
            f"  Earthwork cut:  {ew_result.total_cut_m3/1e6:.2f} Mm³ "
            f"× USD {cut_rate_usd_m3}/m³ = USD {ew_cut_usd/1e6:.2f} M"
        )
        if net_import > 0:
            log.info(
                f"  Import fill:    {net_import/1e6:.2f} Mm³ "
                f"× USD {fill_rate_usd_m3}/m³ = USD {ew_fill_usd/1e6:.2f} M"
            )
        else:
            log.info(
                f"  Net surplus spoil — no import fill cost "
                f"({abs(net_import)/1e3:.1f} km³ excess)"
            )
    else:
        log.warning("CostModel: ew_result is None — earthwork costs set to 0.")
        ew_cut_usd  = 0.0
        ew_fill_usd = 0.0

    # ── 3. Pavement ───────────────────────────────────────────────────────
    pavement_area_m2 = total_length_km * 1000.0 * formation_width_m
    pavement_usd     = pavement_area_m2 * pavement_rate_m2
    log.info(
        f"  Pavement:       {pavement_area_m2/1e6:.3f} km² "
        f"× USD {pavement_rate_m2}/m² = USD {pavement_usd/1e6:.2f} M"
    )

    # ── 4 & 5. Structures ─────────────────────────────────────────────────
    if si_result is not None:
        bridges_usd  = float(si_result.total_bridge_cost_usd)
        culverts_usd = float(si_result.total_culvert_cost_usd)
        log.info(
            f"  Bridges:        USD {bridges_usd/1e6:.2f} M  "
            f"({si_result.bridge_count} structures)"
        )
        log.info(
            f"  Culverts:       USD {culverts_usd/1e3:.0f} K  "
            f"({si_result.culvert_count} structures)"
        )
    else:
        log.warning("CostModel: si_result is None — structure costs set to 0.")
        bridges_usd  = 0.0
        culverts_usd = 0.0

    # ── 6. Land acquisition ───────────────────────────────────────────────
    land_ha = total_length_km * 1000.0 * corridor_width_m / 10_000.0
    eff_rate = _lulc_weighted_rate(lulc_wgs, land_acq_rates, land_acq_default)
    land_usd = land_ha * eff_rate
    log.info(
        f"  Land acq.:      {land_ha:.1f} ha "
        f"× USD {eff_rate:,.0f}/ha = USD {land_usd/1e6:.2f} M"
    )

    # ── Civil subtotal (items 1–6) ────────────────────────────────────────
    civil_subtotal = (
        ew_cut_usd + ew_fill_usd + pavement_usd +
        bridges_usd + culverts_usd + land_usd
    )

    # ── 7. Environmental mitigation ───────────────────────────────────────
    env_usd = civil_subtotal * env_factor

    # ── 8 & 9. Contingency and Engineering ───────────────────────────────
    contingency_usd = civil_subtotal * contingency_factor
    engineering_usd = civil_subtotal * engineering_factor

    # ── Total ─────────────────────────────────────────────────────────────
    total_usd = civil_subtotal + env_usd + contingency_usd + engineering_usd
    cost_per_km = total_usd / total_length_km if total_length_km > 0 else 0.0

    log.info(
        f"  Civil subtotal: USD {civil_subtotal/1e6:.2f} M  "
        f"(env={env_usd/1e6:.2f} M  "
        f"cont={contingency_usd/1e6:.2f} M  "
        f"eng={engineering_usd/1e6:.2f} M)"
    )
    log.info(
        f"CostModel TOTAL: USD {total_usd/1e6:.2f} M  "
        f"(USD {cost_per_km/1e6:.2f} M/km)"
    )

    assumptions = {
        "cut_rate_usd_m3":    cut_rate_usd_m3,
        "fill_rate_usd_m3":   fill_rate_usd_m3,
        "pavement_rate_m2":   pavement_rate_m2,
        "corridor_width_m":   corridor_width_m,
        "land_acq_rate_usd_ha": round(eff_rate, 0),
        "env_factor_pct":     env_factor * 100,
        "contingency_pct":    contingency_factor * 100,
        "engineering_pct":    engineering_factor * 100,
        "scenario_profile":   scenario_profile,
        "source":             "Myanmar DRD / World Bank ROCKS 2020-2024",
    }

    return CostModelResult(
        earthwork_cut_usd      = round(ew_cut_usd,    2),
        earthwork_fill_usd     = round(ew_fill_usd,   2),
        pavement_usd           = round(pavement_usd,  2),
        bridges_usd            = round(bridges_usd,   2),
        culverts_usd           = round(culverts_usd,  2),
        land_acquisition_usd   = round(land_usd,      2),
        environmental_usd      = round(env_usd,       2),
        civil_subtotal_usd     = round(civil_subtotal, 2),
        contingency_usd        = round(contingency_usd, 2),
        engineering_usd        = round(engineering_usd, 2),
        total_project_cost_usd = round(total_usd,     2),
        cost_per_km_usd        = round(cost_per_km,   2),
        total_length_km        = round(total_length_km, 3),
        formation_width_m      = formation_width_m,
        pavement_area_m2       = round(pavement_area_m2, 1),
        land_acquisition_ha    = round(land_ha,       2),
        land_acq_rate_usd_ha   = round(eff_rate,      0),
        assumptions            = assumptions,
    )


# ── CSV export ────────────────────────────────────────────────────────────────

def export_cost_csv(result: CostModelResult, output_path: str) -> None:
    """
    Write itemized cost estimate to CSV.

    Columns: component, amount_usd, pct_of_total
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    total = result.total_project_cost_usd

    rows = [
        ("Earthwork — Cut Excavation",    result.earthwork_cut_usd),
        ("Earthwork — Import Fill",       result.earthwork_fill_usd),
        ("Pavement (Flexible)",           result.pavement_usd),
        ("Bridges",                       result.bridges_usd),
        ("Culverts",                      result.culverts_usd),
        ("Land Acquisition",              result.land_acquisition_usd),
        ("Environmental Mitigation",      result.environmental_usd),
        ("Contingency (20%)",             result.contingency_usd),
        ("Engineering & Admin (10%)",     result.engineering_usd),
        ("TOTAL PROJECT COST",            result.total_project_cost_usd),
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "amount_usd", "pct_of_total"])
        for name, amt in rows:
            pct = (amt / total * 100.0) if total > 0 else 0.0
            writer.writerow([name, round(amt, 0), round(pct, 1)])
        # Metadata footer
        writer.writerow([])
        writer.writerow(["total_length_km",  result.total_length_km, ""])
        writer.writerow(["cost_per_km_usd",  round(result.cost_per_km_usd, 0), ""])
        writer.writerow(["formation_width_m", result.formation_width_m, ""])
        writer.writerow(["pavement_area_m2",  round(result.pavement_area_m2, 0), ""])
        writer.writerow(["land_ha",           result.land_acquisition_ha, ""])
        writer.writerow(["land_rate_usd_ha",  result.land_acq_rate_usd_ha, ""])
        for k, v in result.assumptions.items():
            writer.writerow([f"assumption.{k}", v, ""])

    log.info(f"Cost estimate CSV exported: {output_path}")
