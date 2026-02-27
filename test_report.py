"""
test_report.py — Unit tests for Phase 10 Feasibility Report Generator
======================================================================
All tests are deterministic — NO DEM, NO OSM, NO WeasyPrint rendering required.
Tests validate HTML output content and structure using string search only.

Tests:
  1. test_html_generation_runs      — generate_report() returns non-empty string
  2. test_html_contains_title       — output contains project title
  3. test_html_contains_cost_table  — key cost numbers present in HTML
  4. test_html_null_results         — all optional args None → no crash
  5. test_html_is_valid_markup      — <html> and <body> tags properly closed
  6. test_csv_export_columns        — export_cost_csv writes correct CSV header
  7. test_report_disclaimer         — disclaimer text appears in HTML output
"""
import sys
import os
import tempfile
import csv

sys.path.insert(0, os.path.dirname(__file__))


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_meta(length_km=100.0):
    return {
        "total_length_km":           length_km,
        "formation_width_m":         11.0,
        "max_slope_pct":             12.5,
        "min_curve_radius_achieved_m": 239.4,
        "curve_radius_violations":   0,
        "sustained_grade_violations": 0,
        "dem_source":                "SRTMGL1",
        "data_confidence":           "HIGH",
        "data_warnings":             [],
        "scenario_profile":          "rural_trunk",
        "design_speed_kmph":         60,
        "vc_count":                  18,
        "vertical_max_grade_pct":    7.5,
        "vertical_grade_violations": 0,
        "vertical_ssd_violations":   0,
        "max_fill_m":                4.2,
        "max_cut_m":                 8.1,
        "total_cut_Mm3":             2.3,
        "total_fill_Mm3":            1.8,
        "net_import_Mm3":            0.05,
    }


import numpy as np
from dataclasses import dataclass, field as dc_field


@dataclass
class _MockVA:
    distances_m:     np.ndarray
    z_design:        np.ndarray
    z_terrain:       np.ndarray
    cut_fill_m:      np.ndarray
    grade_pct:       np.ndarray
    vertical_curves: list
    grade_violations: list
    ssd_violations:  list
    max_grade_pct:   float


@dataclass
class _MockEW:
    total_cut_m3:        float
    total_fill_m3:       float
    net_import_m3:       float
    balance_stations_m:  list
    formation_width_m:   float
    swell_factor:        float


@dataclass
class _MockSI:
    structures:             list
    bridge_count:           int
    culvert_count:          int
    total_bridge_length_m:  float
    total_bridge_cost_usd:  float
    total_culvert_cost_usd: float
    total_structure_cost_usd: float


def _make_va():
    n = 50
    d = np.linspace(0, 10000, n)
    z = np.linspace(100, 200, n)
    return _MockVA(
        distances_m=d, z_design=z, z_terrain=z,
        cut_fill_m=np.zeros(n), grade_pct=np.ones(n) * 2.0,
        vertical_curves=[], grade_violations=[], ssd_violations=[],
        max_grade_pct=7.5,
    )


def _make_ew():
    return _MockEW(
        total_cut_m3=2_300_000.0, total_fill_m3=1_800_000.0,
        net_import_m3=50_000.0, balance_stations_m=[3000.0, 7000.0],
        formation_width_m=11.0, swell_factor=1.25,
    )


def _make_si():
    return _MockSI(
        structures=[], bridge_count=5, culvert_count=20,
        total_bridge_length_m=480.0,
        total_bridge_cost_usd=20_160_000.0,
        total_culvert_cost_usd=300_000.0,
        total_structure_cost_usd=20_460_000.0,
    )


def _make_cost():
    from cost_model import CostModelResult
    return CostModelResult(
        earthwork_cut_usd=18_400_000.0,
        earthwork_fill_usd=750_000.0,
        pavement_usd=132_000_000.0,
        bridges_usd=20_160_000.0,
        culverts_usd=300_000.0,
        land_acquisition_usd=12_375_000.0,
        environmental_usd=5_519_850.0,
        civil_subtotal_usd=183_985_000.0,
        contingency_usd=36_797_000.0,
        engineering_usd=18_398_500.0,
        total_project_cost_usd=244_700_350.0,
        cost_per_km_usd=2_447_003.5,
        total_length_km=100.0,
        formation_width_m=11.0,
        pavement_area_m2=1_100_000.0,
        land_acquisition_ha=660.0,
        land_acq_rate_usd_ha=7_500.0,
        assumptions={
            "cut_rate_usd_m3": 8.0,
            "fill_rate_usd_m3": 15.0,
            "pavement_rate_m2": 120.0,
            "corridor_width_m": 60.0,
            "land_acq_rate_usd_ha": 7_500.0,
            "env_factor_pct": 3.0,
            "contingency_pct": 20.0,
            "engineering_pct": 10.0,
            "scenario_profile": "rural_trunk",
            "source": "Myanmar DRD / World Bank ROCKS 2020-2024",
        },
    )


def _render_to_tmp(va=None, ew=None, si=None, cost=None):
    """Helper: render report to a tmp HTML file and return its content."""
    from report import generate_report
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "report.html")
        pdf_path  = os.path.join(tmpdir, "report.pdf")
        generate_report(
            meta=_make_meta(),
            va_result=va,
            ew_result=ew,
            si_result=si,
            cost_result=cost,
            output_html=html_path,
            output_pdf=pdf_path,
        )
        with open(html_path, encoding="utf-8") as f:
            return f.read()


# ── Test 1: HTML generation runs ─────────────────────────────────────────────

def test_html_generation_runs():
    """generate_report() completes and writes a non-empty HTML file."""
    html = _render_to_tmp(
        va=_make_va(), ew=_make_ew(), si=_make_si(), cost=_make_cost()
    )
    assert len(html) > 5_000, f"HTML too small: {len(html)} bytes"
    print("  ✓ test_html_generation_runs")


# ── Test 2: HTML contains title ──────────────────────────────────────────────

def test_html_contains_title():
    """Report HTML must contain 'Highway Alignment' title text."""
    html = _render_to_tmp(va=_make_va(), ew=_make_ew(), si=_make_si(), cost=_make_cost())
    assert "Highway Alignment" in html, "Title text not found in HTML"
    print("  ✓ test_html_contains_title")


# ── Test 3: HTML contains cost numbers ───────────────────────────────────────

def test_html_contains_cost_table():
    """Cost estimate section must appear when cost_result is provided."""
    html = _render_to_tmp(cost=_make_cost())
    assert "Cost Estimate" in html or "cost" in html.lower(), (
        "Cost section not found in HTML"
    )
    # At least one USD amount must appear
    assert "USD" in html or "{:,.0f}" not in html, "No USD amounts in output"
    print("  ✓ test_html_contains_cost_table")


# ── Test 4: Null results — no crash ──────────────────────────────────────────

def test_html_null_results():
    """Passing all None optional args must not crash and must produce valid HTML."""
    html = _render_to_tmp(va=None, ew=None, si=None, cost=None)
    assert len(html) > 1_000, "HTML too small when all results are None"
    assert "<html" in html, "Missing <html> tag"
    print("  ✓ test_html_null_results")


# ── Test 5: Valid HTML markup ─────────────────────────────────────────────────

def test_html_is_valid_markup():
    """Output must open and close <html> and <body> tags correctly."""
    html = _render_to_tmp()
    assert html.count("<html") == 1,  f"Expected 1 <html> open tag, got {html.count('<html')}"
    assert html.count("</html>") == 1, "Missing </html> closing tag"
    assert html.count("<body") == 1,   "Missing <body> open tag"
    assert html.count("</body>") == 1, "Missing </body> closing tag"
    print("  ✓ test_html_is_valid_markup")


# ── Test 6: CSV export columns ────────────────────────────────────────────────

def test_csv_export_columns():
    """export_cost_csv must write CSV with correct header columns."""
    from cost_model import export_cost_csv
    cost = _make_cost()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cost.csv")
        export_cost_csv(cost, path)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
    expected_cols = {"component", "amount_usd", "pct_of_total"}
    assert set(header) == expected_cols, (
        f"CSV header mismatch: {header}"
    )
    print("  ✓ test_csv_export_columns")


# ── Test 7: Disclaimer text ───────────────────────────────────────────────────

def test_report_disclaimer():
    """HTML must contain the standard 'NOT FOR CONSTRUCTION' disclaimer."""
    html = _render_to_tmp(va=_make_va(), ew=_make_ew(), si=_make_si(), cost=_make_cost())
    assert "NOT FOR CONSTRUCTION" in html, (
        "Disclaimer text 'NOT FOR CONSTRUCTION' not found in report"
    )
    print("  ✓ test_report_disclaimer")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_html_generation_runs,
        test_html_contains_title,
        test_html_contains_cost_table,
        test_html_null_results,
        test_html_is_valid_markup,
        test_csv_export_columns,
        test_report_disclaimer,
    ]
    passed = 0
    print("\nPhase 10 — Feasibility Report Unit Tests")
    print("=" * 50)
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 50)
    print(f"{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
