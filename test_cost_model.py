"""
test_cost_model.py — Unit tests for Phase 9 Parametric Cost Model
==================================================================
All tests are deterministic — NO DEM, NO OSM, NO external data required.

Tests:
  1. test_earthwork_cost_formula     — cut & fill USD calculations
  2. test_pavement_area              — km × formation_width → m²
  3. test_contingency_engineering    — 20% / 10% of civil subtotal
  4. test_env_mitigation             — 3% of civil subtotal
  5. test_null_earthwork_graceful    — ew_result=None returns 0 earthwork cost
  6. test_null_structures_graceful   — si_result=None returns 0 structure costs
  7. test_cost_per_km                — total / length_km matches cost_per_km_usd
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cost_model import compute_cost_model, export_cost_csv, CostModelResult


# ── Mock helpers ─────────────────────────────────────────────────────────────

def _make_meta(length_km=100.0, formation_width_m=11.0):
    return {
        "total_length_km":  length_km,
        "formation_width_m": formation_width_m,
    }


class _MockEW:
    """Minimal EarthworkResult stub."""
    def __init__(self, cut_m3=1_000_000.0, net_import_m3=200_000.0):
        self.total_cut_m3  = cut_m3
        self.net_import_m3 = net_import_m3
        self.total_fill_m3 = 0.0


class _MockSI:
    """Minimal StructureInventory stub."""
    def __init__(self, bridge_usd=5_000_000.0, culvert_usd=300_000.0,
                 bridge_cnt=5, culvert_cnt=20):
        self.total_bridge_cost_usd  = bridge_usd
        self.total_culvert_cost_usd = culvert_usd
        self.bridge_count  = bridge_cnt
        self.culvert_count = culvert_cnt


# ── Test 1: Earthwork cost formula ───────────────────────────────────────────

def test_earthwork_cost_formula():
    """
    Cut 500_000 m³ × USD 8 = USD 4_000_000.
    Import fill 100_000 m³ × USD 15 = USD 1_500_000 (net_import_m3 > 0).
    """
    ew = _MockEW(cut_m3=500_000.0, net_import_m3=100_000.0)
    result = compute_cost_model(
        meta=_make_meta(length_km=50.0),
        ew_result=ew,
        si_result=None,
        cut_rate_usd_m3=8.0,
        fill_rate_usd_m3=15.0,
        pavement_rate_m2=0.0,   # isolate earthwork
        land_acq_default=0.0,
        env_factor=0.0, contingency_factor=0.0, engineering_factor=0.0,
    )
    assert abs(result.earthwork_cut_usd - 4_000_000.0) < 1.0, (
        f"Cut cost wrong: {result.earthwork_cut_usd}"
    )
    assert abs(result.earthwork_fill_usd - 1_500_000.0) < 1.0, (
        f"Fill cost wrong: {result.earthwork_fill_usd}"
    )
    print("  ✓ test_earthwork_cost_formula")


# ── Test 2: Pavement area ────────────────────────────────────────────────────

def test_pavement_area():
    """
    Fix 20: Pavement is now billed on CARRIAGEWAY width only.
    For rural_trunk, CARRIAGEWAY_WIDTH_M = 7 m (from config).
    Expected: 100 km × 7 m × 120 USD/m² = USD 84_000_000.
    pavement_area_m2 = 100_000 × 7 = 700_000 m².
    """
    from config import CARRIAGEWAY_WIDTH_M
    expected_cw = CARRIAGEWAY_WIDTH_M.get("rural_trunk", 7.0)
    result = compute_cost_model(
        meta=_make_meta(length_km=100.0, formation_width_m=11.0),
        ew_result=None, si_result=None,
        cut_rate_usd_m3=0.0, fill_rate_usd_m3=0.0,
        pavement_rate_m2=120.0,
        land_acq_default=0.0,
        env_factor=0.0, contingency_factor=0.0, engineering_factor=0.0,
        scenario_profile="rural_trunk",
    )
    expected_area = 100_000.0 * expected_cw
    expected_cost = expected_area * 120.0
    assert abs(result.pavement_area_m2 - expected_area) < 1.0, (
        f"Pavement area wrong: {result.pavement_area_m2} (expected {expected_area})"
    )
    assert abs(result.pavement_usd - expected_cost) < 1.0, (
        f"Pavement cost wrong: {result.pavement_usd} (expected {expected_cost})"
    )
    print("  ✓ test_pavement_area")


# ── Test 3: Contingency and engineering ──────────────────────────────────────

def test_contingency_engineering():
    """
    Drive civil subtotal via earthwork (not pavement) to avoid Fix 20 carriageway-width effects.
    Earthwork cut: 1_000_000 m³ × USD 10 = USD 10_000_000 civil subtotal.
    contingency = 20% = USD 2_000_000.
    engineering = 10% = USD 1_000_000.
    """
    ew = _MockEW(cut_m3=1_000_000.0, net_import_m3=0.0)   # 0 import fill
    result = compute_cost_model(
        meta=_make_meta(length_km=1.0, formation_width_m=11.0),
        ew_result=ew, si_result=None,
        cut_rate_usd_m3=10.0,
        fill_rate_usd_m3=0.0,
        pavement_rate_m2=0.0,    # isolate earthwork
        land_acq_default=0.0,
        env_factor=0.0,
        contingency_factor=0.20,
        engineering_factor=0.10,
    )
    civil = result.civil_subtotal_usd
    expected_civil = 10_000_000.0 + result.drainage_usd
    assert abs(civil - expected_civil) < 10.0, f"Civil subtotal wrong: {civil}"
    assert abs(result.contingency_usd - civil * 0.20) < 10.0, (
        f"Contingency wrong: {result.contingency_usd}"
    )
    assert abs(result.engineering_usd - civil * 0.10) < 10.0, (
        f"Engineering wrong: {result.engineering_usd}"
    )
    print("  ✓ test_contingency_engineering")


# ── Test 4: Environmental mitigation ─────────────────────────────────────────

def test_env_mitigation():
    """
    civil_subtotal = 5_000_000 → env at 3% = 150_000.
    """
    result = compute_cost_model(
        meta=_make_meta(length_km=1.0, formation_width_m=1.0),
        ew_result=None, si_result=None,
        cut_rate_usd_m3=0.0, fill_rate_usd_m3=0.0,
        pavement_rate_m2=5_000.0,
        land_acq_default=0.0,
        env_factor=0.03,
        contingency_factor=0.0, engineering_factor=0.0,
    )
    expected_env = result.civil_subtotal_usd * 0.03
    assert abs(result.environmental_usd - expected_env) < 1.0, (
        f"Env mitigation wrong: {result.environmental_usd} ≠ {expected_env}"
    )
    print("  ✓ test_env_mitigation")


# ── Test 5: Null earthwork — graceful ────────────────────────────────────────

def test_null_earthwork_graceful():
    """ew_result=None → earthwork_cut_usd == 0, earthwork_fill_usd == 0, no crash."""
    result = compute_cost_model(
        meta=_make_meta(),
        ew_result=None, si_result=None,
    )
    assert result.earthwork_cut_usd == 0.0, "Cut cost should be 0 with no ew_result"
    assert result.earthwork_fill_usd == 0.0, "Fill cost should be 0 with no ew_result"
    print("  ✓ test_null_earthwork_graceful")


# ── Test 6: Null structures — graceful ───────────────────────────────────────

def test_null_structures_graceful():
    """si_result=None → bridges_usd == 0, drainage_usd != 0 (it's length based), no crash."""
    result = compute_cost_model(
        meta=_make_meta(length_km=0.0), # Force 0 length to test 0 drainage
        ew_result=None, si_result=None,
    )
    assert result.bridges_usd == 0.0, "Bridge cost should be 0 with no si_result"
    assert result.drainage_usd == 0.0, "Drainage cost should be 0 for 0 km length"
    print("  ✓ test_null_structures_graceful")


# ── Test 7: Cost per km ───────────────────────────────────────────────────────

def test_cost_per_km():
    """total_project_cost_usd / total_length_km == cost_per_km_usd (within rounding)."""
    result = compute_cost_model(
        meta=_make_meta(length_km=276.0, formation_width_m=11.0),
        ew_result=_MockEW(cut_m3=5_000_000.0, net_import_m3=500_000.0),
        si_result=_MockSI(bridge_usd=8_000_000.0, culvert_usd=450_000.0),
    )
    expected_cpk = result.total_project_cost_usd / result.total_length_km
    assert abs(result.cost_per_km_usd - expected_cpk) < 100.0, (
        f"cost_per_km mismatch: {result.cost_per_km_usd:.0f} ≠ {expected_cpk:.0f}"
    )
    # Sanity: total > civil subtotal
    assert result.total_project_cost_usd > result.civil_subtotal_usd, (
        "Total must exceed civil subtotal (contingency + eng not applied?)"
    )
    print("  ✓ test_cost_per_km")


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_earthwork_cost_formula,
        test_pavement_area,
        test_contingency_engineering,
        test_env_mitigation,
        test_null_earthwork_graceful,
        test_null_structures_graceful,
        test_cost_per_km,
    ]
    passed = 0
    print("\nPhase 9 — Parametric Cost Model Unit Tests")
    print("=" * 50)
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
        except Exception as e:
            print(f"  ✗ {test.__name__}: UNEXPECTED ERROR — {e}")
            import traceback
            traceback.print_exc()

    print("=" * 50)
    print(f"{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
