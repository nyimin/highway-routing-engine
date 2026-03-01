"""
report.py — Phase 10: Automated Feasibility Report
====================================================
Generates a professional HTML feasibility report from all pipeline results
using a Jinja2 template, then optionally exports to PDF via WeasyPrint.

Public API
----------
    generate_report(meta, va_result, ew_result, si_result, cost_result,
                    output_html, output_pdf) → str  (path to HTML file)

WeasyPrint note
---------------
WeasyPrint requires GTK+ / Cairo and Pango libraries.  On Windows these are
available via the Anaconda GTK package or the standalone WeasyPrint wheel.
If WeasyPrint import fails at runtime, the module writes the HTML file only
and emits a clear log.warning() with manual conversion instructions.

Template
--------
    templates/report.html  — Jinja2 template, self-contained CSS, base64 images.
    Located relative to THIS file's directory.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("highway_alignment")

# ── Template resolution ───────────────────────────────────────────────────────

_HERE = Path(__file__).parent
TEMPLATE_DIR  = _HERE / "templates"
TEMPLATE_FILE = "report.html"


# ── Image helper ──────────────────────────────────────────────────────────────

def _img_to_b64(path: str) -> Optional[str]:
    """
    Read an image file and return a base64 string (no prefix).
    Returns None if the file does not exist or cannot be read.
    """
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except (FileNotFoundError, OSError):
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def generate_report(
    meta:         dict,
    va_result,                      # VerticalAlignmentResult | None
    ew_result,                      # EarthworkResult | None
    si_result,                      # StructureInventory | None
    cost_result,                    # CostModelResult | None
    output_html:  str = "output/feasibility_report.html",
    output_pdf:   str = "output/feasibility_report.pdf",
    *,
    waypoints:    Optional[list] = None,
    segment_indices: Optional[list] = None,
    point_a_label: str = "Point A",
    point_b_label: str = "Point B",
    project_title: str = "Myanmar Highway Alignment — Preliminary Feasibility",
) -> str:
    """
    Render the Jinja2 template and write HTML (+ optionally PDF) to disk.

    Parameters
    ----------
    meta : dict
        Pipeline metadata dict from compute_metadata().
    va_result : VerticalAlignmentResult | None
        Phase 6 result.
    ew_result : EarthworkResult | None
        Phase 7 result.
    si_result : StructureInventory | None
        Phase 8 result.
    cost_result : CostModelResult | None
        Phase 9 result.
    output_html : str
        Path for the HTML output file.
    output_pdf : str
        Path for the PDF output file (attempted if WeasyPrint is available).
    point_a_label : str
        Human-readable label for corridor start point.
    point_b_label : str
        Human-readable label for corridor end point.
    project_title : str
        Report title string.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise ImportError(
            "jinja2 is required for Phase 10 report generation. "
            "Install with: pip install jinja2"
        ) from exc

    # ── Build Jinja2 environment ──────────────────────────────────────────
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template(TEMPLATE_FILE)

    # ── Extract scalar values safely ─────────────────────────────────────
    total_length_km  = float(meta.get("total_length_km", 0.0))
    max_slope_pct    = meta.get("max_slope_pct", "N/A")
    min_curve_radius = meta.get("min_curve_radius_achieved_m", "N/A")
    dem_source       = meta.get("dem_source", "Unknown")
    data_confidence  = meta.get("data_confidence", "Unknown")
    data_warnings    = meta.get("data_warnings", [])
    grade_violations = meta.get("sustained_grade_violations", 0)
    scenario_profile = meta.get("scenario_profile", "rural_trunk")

    # ── Vertical alignment variables ──────────────────────────────────────
    va_available = va_result is not None
    vc_count         = len(va_result.vertical_curves) if va_available else 0
    max_grade        = round(va_result.max_grade_pct, 2) if va_available else "N/A"
    ssd_violations   = len(va_result.ssd_violations)  if va_available else 0
    max_fill_m       = round(float(va_result.cut_fill_m.max()), 1) if va_available else "N/A"
    max_cut_m        = round(float(-va_result.cut_fill_m.min()), 1) if va_available else "N/A"

    # ── Earthwork variables ───────────────────────────────────────────────
    ew_available = ew_result is not None
    total_cut_mm3     = ew_result.total_cut_m3  / 1e6 if ew_available else 0.0
    total_fill_mm3    = ew_result.total_fill_m3 / 1e6 if ew_available else 0.0
    net_import_mm3    = ew_result.net_import_m3 / 1e6 if ew_available else 0.0
    balance_points    = len(ew_result.balance_stations_m) if ew_available else 0
    formation_width_m = ew_result.formation_width_m if ew_available else "N/A"
    swell_factor      = ew_result.swell_factor      if ew_available else "N/A"

    # ── Structure variables ───────────────────────────────────────────────
    si_available = si_result is not None
    bridge_count          = si_result.bridge_count   if si_available else 0
    total_bridge_len_m    = si_result.total_bridge_length_m  if si_available else 0.0
    total_bridge_usd      = si_result.total_bridge_cost_usd if si_available else 0.0
    structures            = si_result.structures if si_available else []

    # ── Cost variables ────────────────────────────────────────────────────
    cost_available = cost_result is not None
    total_cost_m   = cost_result.total_project_cost_usd / 1e6 if cost_available else 0.0
    cost_per_km_m  = cost_result.cost_per_km_usd / 1e6 if cost_available else 0.0

    # ── Embed images as base64 ────────────────────────────────────────────
    profile_img_b64  = _img_to_b64("output/vertical_profile.png")
    earthwork_img_b64 = _img_to_b64("output/earthwork_masshual.png")

    # ── Segment Breakdowns ────────────────────────────────────────────────
    import numpy as np
    segment_breakdowns = []
    if waypoints is not None and segment_indices is not None and len(waypoints) > 2:
        num_segments = len(waypoints) - 1
        for i in range(num_segments):
            seg_len_km = 0.0
            if va_result is not None:
                idx_in_seg = np.where(np.array(segment_indices) == i)[0]
                if len(idx_in_seg) > 1:
                    start_m = va_result.distances_m[idx_in_seg[0]]
                    end_m   = va_result.distances_m[idx_in_seg[-1]]
                    seg_len_km = (end_m - start_m) / 1000.0
            
            seg_cut_m3 = 0.0
            seg_fill_m3 = 0.0
            if ew_result is not None:
                idx_in_seg = np.where(np.array(segment_indices[:-1]) == i)[0] 
                if len(idx_in_seg) > 0:
                    seg_cut_m3 = np.sum(ew_result.seg_cut_vol_m3[idx_in_seg])
                    seg_fill_m3 = np.sum(ew_result.seg_fill_vol_m3[idx_in_seg])
            
            seg_bridge_count = 0
            seg_bridge_cost = 0.0
            if si_available:
                seg_bridges = [s for s in structures if s.structure_type == "bridge" and getattr(s, "segment_index", 0) == i]
                seg_bridge_count = len(seg_bridges)
                seg_bridge_cost = sum(b.estimated_cost_usd for b in seg_bridges)
            
            seg_cost = 0.0
            if cost_result is not None and cost_result.total_length_km > 0.0:
                fraction = seg_len_km / cost_result.total_length_km
                cost_exc_ew_bridges = cost_result.total_project_cost_usd - (
                    cost_result.earthwork_cut_usd + cost_result.earthwork_fill_usd + cost_result.bridges_usd
                )
                
                seg_ew_cut_cost = cost_result.earthwork_cut_usd * (seg_cut_m3 / ew_result.total_cut_m3) if ew_result and ew_result.total_cut_m3 > 0 else 0.0
                seg_ew_fill_cost = cost_result.earthwork_fill_usd * (seg_fill_m3 / ew_result.total_fill_m3) if ew_result and ew_result.total_fill_m3 > 0 else 0.0
                
                seg_ew_cost = seg_ew_cut_cost + seg_ew_fill_cost
                seg_cost = (cost_exc_ew_bridges * fraction) + seg_ew_cost + seg_bridge_cost

            segment_breakdowns.append({
                "index": i + 1,
                "name": f"Leg {i+1}",
                "length_km": round(seg_len_km, 1),
                "cut_Mm3": round(seg_cut_m3 / 1e6, 2),
                "fill_Mm3": round(seg_fill_m3 / 1e6, 2),
                "bridges": seg_bridge_count,
                "cost_M": round(seg_cost / 1e6, 1),
            })

    # ── Render ────────────────────────────────────────────────────────────
    html_str = template.render(
        project_title     = project_title,
        report_date       = datetime.now().strftime("%d %B %Y"),
        point_a_label     = point_a_label,
        point_b_label     = point_b_label,
        scenario_profile  = scenario_profile,
        design_speed_kmph = meta.get("design_speed_kmph", "N/A"),
        total_length_km   = total_length_km,
        max_slope_pct     = max_slope_pct,
        min_curve_radius_m = min_curve_radius,
        grade_violations  = grade_violations,
        dem_source        = dem_source,
        data_confidence   = data_confidence,
        data_warnings     = data_warnings,
        map_path          = None,   # Folium HTML cannot be embedded as img
        # VA
        va_available      = va_available,
        vc_count          = vc_count,
        max_grade         = max_grade,
        ssd_violations    = ssd_violations,
        max_fill_m        = max_fill_m,
        max_cut_m         = max_cut_m,
        profile_img_b64   = profile_img_b64,
        # Earthwork
        ew_available      = ew_available,
        total_cut_mm3     = total_cut_mm3,
        total_fill_mm3    = total_fill_mm3,
        net_import_mm3    = net_import_mm3,
        balance_points    = balance_points,
        formation_width_m = formation_width_m,
        swell_factor      = swell_factor,
        earthwork_img_b64 = earthwork_img_b64,
        # Structures
        si_available      = si_available,
        bridge_count      = bridge_count,
        total_bridge_len_m = total_bridge_len_m,
        total_bridge_cost_usd = total_bridge_usd,
        structures        = structures,
        # Cost
        cost_available    = cost_available,
        cost              = cost_result,
        total_cost_m      = total_cost_m,
        cost_per_km_m     = cost_per_km_m,
        # Multi-Waypoint
        segment_breakdowns = segment_breakdowns,
    )

    # ── Write HTML ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_html) if os.path.dirname(output_html) else ".", exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_str)
    log.info(f"Feasibility report HTML written: {output_html}  ({len(html_str):,} bytes)")

    # ── Write PDF (optional) ──────────────────────────────────────────────
    _write_pdf(html_str, output_pdf)

    return os.path.abspath(output_html)


def _write_pdf(html_str: str, output_pdf: str) -> None:
    """
    Attempt to render HTML → PDF using WeasyPrint.
    On failure (GTK not available, etc.) logs a clear warning with alternatives.
    """
    try:
        import weasyprint  # noqa: F401 — import check
        from weasyprint import HTML as _WHTML
        os.makedirs(os.path.dirname(output_pdf) if os.path.dirname(output_pdf) else ".", exist_ok=True)
        _WHTML(string=html_str).write_pdf(output_pdf)
        log.info(f"Feasibility report PDF written: {output_pdf}")
    except ImportError:
        log.warning(
            "WeasyPrint is not installed — PDF not generated. "
            "Install with: pip install weasyprint  "
            "(requires GTK+/Cairo on Windows; see https://doc.courtbouillon.org/weasyprint/)"
        )
    except Exception as exc:
        log.warning(
            f"WeasyPrint PDF generation failed: {exc}  "
            f"— HTML report is still available at the path above. "
            f"Convert manually: weasyprint feasibility_report.html feasibility_report.pdf"
        )
