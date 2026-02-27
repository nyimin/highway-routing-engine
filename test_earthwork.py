"""
test_earthwork.py — Unit tests for Phase 7 earthwork volume estimation
======================================================================
All tests are deterministic — NO DEM, NO OSM, NO external data.

Tests:
  1. test_cut_area_formula     — trapezoidal cut area with known h, batter
  2. test_fill_area_formula    — trapezoidal fill area with known h, batter
  3. test_zero_cut_fill        — flat design produces zero areas and volumes
  4. test_volume_integration   — Average-End-Area volume over simple profile
  5. test_mass_haul_sign       — net-import vs net-spoil sign convention
  6. test_balance_station      — balance point detected at correct chainage
  7. test_swell_factor         — swell > 1 reduces effective cut in mass-haul
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from earthwork import compute_earthwork, _compute_areas, _compute_volumes


# ── Test 1: Cut area formula ──────────────────────────────────────────────────

def test_cut_area_formula():
    """
    h=10 m cut, formation_width=11 m, cut_batter=1.0 H:V.
    Expected: A = h × (B + h × batter) = 10 × (11 + 10×1.0) = 10 × 21 = 210 m²
    """
    d = np.array([0.0, 100.0])
    cf = np.array([-10.0, -10.0])   # 10 m cut everywhere

    area_cut, area_fill = _compute_areas(cf, 11.0, cut_batter_HV=1.0, fill_batter_HV=1.5)

    expected = 10.0 * (11.0 + 10.0 * 1.0)   # 210 m²
    assert abs(area_cut[0] - expected) < 0.001, (
        f"Cut area {area_cut[0]:.2f} ≠ expected {expected:.2f}"
    )
    assert abs(area_fill[0]) < 1e-9, "Fill area should be zero in cut zone"
    print("  ✓ test_cut_area_formula")


# ── Test 2: Fill area formula ─────────────────────────────────────────────────

def test_fill_area_formula():
    """
    h=5 m fill, formation_width=11 m, fill_batter=1.5 H:V.
    Expected: A = 5 × (11 + 5×1.5) = 5 × 18.5 = 92.5 m²
    """
    cf = np.array([5.0, 5.0])
    area_cut, area_fill = _compute_areas(cf, 11.0, cut_batter_HV=1.0, fill_batter_HV=1.5)

    expected = 5.0 * (11.0 + 5.0 * 1.5)   # 92.5 m²
    assert abs(area_fill[0] - expected) < 0.001, (
        f"Fill area {area_fill[0]:.2f} ≠ expected {expected:.2f}"
    )
    assert abs(area_cut[0]) < 1e-9, "Cut area should be zero in fill zone"
    print("  ✓ test_fill_area_formula")


# ── Test 3: Zero cut/fill ─────────────────────────────────────────────────────

def test_zero_cut_fill():
    """
    Design exactly on terrain → cut_fill = 0 → areas = 0 → volumes = 0.
    """
    d  = np.linspace(0, 5000, 100)
    cf = np.zeros(100)

    result = compute_earthwork(d, cf, formation_width_m=11.0)

    assert result.total_cut_m3  < 1e-6, f"Expected ~0 cut, got {result.total_cut_m3}"
    assert result.total_fill_m3 < 1e-6, f"Expected ~0 fill, got {result.total_fill_m3}"
    assert abs(result.net_import_m3) < 1e-6, "Net import should be ~0 for flat profile"
    print("  ✓ test_zero_cut_fill")


# ── Test 4: Volume integration (AEA) ─────────────────────────────────────────

def test_volume_integration():
    """
    Constant 5 m fill over 1000 m, formation=11 m, fill_batter=1.5.
    Cross-section area: 5 × (11 + 5×1.5) = 92.5 m²
    Expected fill volume: 92.5 × 1000 = 92,500 m³
    """
    d  = np.array([0.0, 1000.0])
    cf = np.array([5.0, 5.0])

    result = compute_earthwork(d, cf, formation_width_m=11.0,
                                fill_batter_HV=1.5, swell_factor=1.0)

    expected_vol = 92.5 * 1000.0
    # AEA with constant cross-section is exact
    assert abs(result.total_fill_m3 - expected_vol) < 1.0, (
        f"Fill volume {result.total_fill_m3:.0f} ≠ expected {expected_vol:.0f} m³"
    )
    print("  ✓ test_volume_integration")


# ── Test 5: Mass-haul sign convention ────────────────────────────────────────

def test_mass_haul_sign():
    """
    Pure fill section → mass_haul should end POSITIVE (fill deficit, need borrow).
    Pure cut section  → mass_haul should end NEGATIVE (surplus, need spoil).
    """
    d = np.linspace(0, 5000, 50)

    # Fill profile
    cf_fill = np.full(50, 3.0)
    r_fill = compute_earthwork(d, cf_fill, formation_width_m=11.0, swell_factor=1.25)
    assert r_fill.mass_haul_m3[-1] > 0, (
        f"Pure fill → mass_haul should be +ve, got {r_fill.mass_haul_m3[-1]:.0f}"
    )

    # Cut profile
    cf_cut = np.full(50, -3.0)
    r_cut = compute_earthwork(d, cf_cut, formation_width_m=11.0, swell_factor=1.25)
    assert r_cut.mass_haul_m3[-1] < 0, (
        f"Pure cut → mass_haul should be -ve, got {r_cut.mass_haul_m3[-1]:.0f}"
    )
    print("  ✓ test_mass_haul_sign")


# ── Test 6: Balance station detection ────────────────────────────────────────

def test_balance_station():
    """
    Directly test _find_balance_stations with a mass-haul array that has
    known sign crossings at 2500 m and 7500 m.
    """
    from earthwork import _find_balance_stations

    # Construct stations and mass-haul with two sign crossings
    d  = np.array([0.0, 2000.0, 2500.0, 3000.0, 7000.0, 7500.0, 8000.0, 10000.0])
    mh = np.array([0.0,  1000.0,    0.1,  -500.0, -500.0,     0.0,  600.0,      0.0])

    # Cross zero between d[2] and d[3] (2500→3000 m span)
    # Cross zero between d[4] and d[5] or d[5] and d[6] (7000→7500→8000 m span)
    balance = _find_balance_stations(d, mh)

    assert len(balance) >= 1, (
        f"Expected balance station(s) in constructed profile, got {len(balance)}"
    )
    # The first crossing should be near 2500 m
    assert any(abs(bs - 2500) < 500 for bs in balance), (
        f"No balance station near 2500 m; found: {balance}"
    )
    print("  ✓ test_balance_station")




# ── Test 7: Swell factor effect ───────────────────────────────────────────────

def test_swell_factor():
    """
    For a mixed cut+fill profile, higher swell_factor should make the net
    import LESS (surplus cut is larger in loose volume, fills more of the fill).
    mass_haul_end(swell=1.0) > mass_haul_end(swell=1.4)
    """
    d  = np.linspace(0, 10000, 100)
    # Alternating cut and fill of equal magnitude
    cf = 3.0 * np.sin(2 * np.pi * d / 5000)

    r1 = compute_earthwork(d, cf.copy(), formation_width_m=11.0, swell_factor=1.0)
    r2 = compute_earthwork(d, cf.copy(), formation_width_m=11.0, swell_factor=1.4)

    # With higher swell: cut volume covers more fill → final haul ordinate is lower
    assert r1.mass_haul_m3[-1] >= r2.mass_haul_m3[-1], (
        f"Higher swell should reduce net import: "
        f"swell=1.0 → {r1.mass_haul_m3[-1]:.0f}, "
        f"swell=1.4 → {r2.mass_haul_m3[-1]:.0f}"
    )
    print("  ✓ test_swell_factor")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_cut_area_formula,
        test_fill_area_formula,
        test_zero_cut_fill,
        test_volume_integration,
        test_mass_haul_sign,
        test_balance_station,
        test_swell_factor,
    ]
    passed = 0
    print("\nPhase 7 — Earthwork Volume Unit Tests")
    print("=" * 45)
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

    print("=" * 45)
    print(f"{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
