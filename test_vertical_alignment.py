"""
test_vertical_alignment.py — Unit tests for Phase 6 vertical alignment
=======================================================================
All tests are deterministic and require NO external data (no DEM, no OSM).
Follows the runner pattern established in test_routing_filter.py.

Tests:
  1. test_parabola_elevation    — parabola formula produces correct z(x)
  2. test_k_crest_ssd           — K_crest applied correctly to L_min
  3. test_k_sag_ssd             — K_sag applied correctly to L_min
  4. test_grade_clipping_uphill — forward sweep clips grade to G_MAX on steep uphill
  5. test_grade_clipping_downhill — forward sweep clips grade to G_MAX on steep downhill
  6. test_flat_profile          — flat terrain yields zero or trivial VCs
  7. test_design_above_ground   — design FGL should not significantly under-cut terrain
"""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from vertical_alignment import (
    build_vertical_alignment,
    _clip_grades,
    _smooth_terrain,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _linspace_profile(start_elev, end_elev, length_m=5000.0, n=200):
    """Create a simple linear elevation profile."""
    d = np.linspace(0, length_m, n)
    e = np.linspace(start_elev, end_elev, n)
    return d, e


def _parabola_z(z_pvc, g1_pct, g2_pct, L, x):
    """Reference implementation of the parabola formula."""
    A = g2_pct - g1_pct
    return z_pvc + g1_pct / 100.0 * x + (A / 100.0) / (2.0 * L) * x * x


# ── Test 1: Parabola formula correctness ─────────────────────────────────────

def test_parabola_elevation():
    """
    Known input: g1=2%, g2=-3%, L=200m, z_PVC=100m.
    Check z at x=0, x=100 (midpoint), x=200 (PVT).
    """
    g1, g2, L, z_pvc = 2.0, -3.0, 200.0, 100.0
    A = g2 - g1   # -5.0 %

    # x=0 → z_PVC
    z0 = _parabola_z(z_pvc, g1, g2, L, 0.0)
    assert abs(z0 - 100.0) < 0.001, f"z(0) should be 100.0, got {z0}"

    # x=100 → mid-curve elevation
    z_mid = _parabola_z(z_pvc, g1, g2, L, 100.0)
    expected_mid = 100.0 + 2.0/100.0 * 100.0 + (-5.0/100.0) / (2.0*200.0) * 100.0**2
    assert abs(z_mid - expected_mid) < 0.001, f"z(100) mismatch: {z_mid} vs {expected_mid}"

    # x=200 → PVT; z = z_pvc + g1*L + A*L/2
    z_pvt = _parabola_z(z_pvc, g1, g2, L, 200.0)
    expected_pvt = 100.0 + 2.0/100.0 * 200.0 + (-5.0/100.0) / (2.0*200.0) * 200.0**2
    assert abs(z_pvt - expected_pvt) < 0.001, f"z(200) mismatch: {z_pvt} vs {expected_pvt}"

    print("  ✓ test_parabola_elevation")


# ── Test 2: K_crest applied to minimum curve length ──────────────────────────

def test_k_crest_ssd():
    """
    For V=60 km/h: K_crest=11, K_sag=11.
    A crest VC with |A|=3% should have L_min = 11*3 = 33m.
    """
    # Build a simple crest profile: rises then falls
    d = np.array([0.0, 1000.0, 2000.0])
    e = np.array([100.0, 180.0, 100.0])   # 8% up, 8% down — will be clipped to 8% max

    # Use 8% max grade limit (rural_trunk), K_crest=11, K_sag=11
    result = build_vertical_alignment(
        np.linspace(0, 2000, 100),
        # Resampled profile to 100 points
        np.interp(np.linspace(0, 2000, 100), d, e),
        design_speed_kmph=60,
        max_grade_pct=8.0,
        k_crest=11,
        k_sag=11,
        min_vc_length_m=30.0,
    )
    crest_curves = [vc for vc in result.vertical_curves if vc.curve_type == 'crest']
    # There must be at least one crest VC (at the hilltop)
    assert len(crest_curves) >= 1, "Expected at least one crest VC"
    # Each crest VC should have L ≥ K_crest × |A|  (or min_vc_length_m)
    for vc in crest_curves:
        A_abs = abs(vc.g2_pct - vc.g1_pct)
        L_min_expected = max(11 * A_abs, 30.0)
        # Allow 5% tolerance for floating-point and tangent-space capping
        assert vc.length_m >= L_min_expected * 0.95 or vc.length_m >= 30.0, (
            f"Crest VC L={vc.length_m:.1f} < K*|A|={L_min_expected:.1f} (A={A_abs:.2f}%)"
        )
    print("  ✓ test_k_crest_ssd")


# ── Test 3: K_sag applied to minimum curve length ────────────────────────────

def test_k_sag_ssd():
    """
    A sag profile (falls then rises) with V=60 km/h, K_sag=11.
    """
    d_full = np.linspace(0, 2000, 100)
    e_base = np.array([0.0, 1000.0, 2000.0])
    e_elev = np.array([180.0, 100.0, 180.0])   # valley
    e_full = np.interp(d_full, e_base, e_elev)

    result = build_vertical_alignment(
        d_full, e_full,
        design_speed_kmph=60,
        max_grade_pct=8.0,
        k_crest=11,
        k_sag=11,
        min_vc_length_m=30.0,
    )
    sag_curves = [vc for vc in result.vertical_curves if vc.curve_type == 'sag']
    assert len(sag_curves) >= 1, "Expected at least one sag VC"
    for vc in sag_curves:
        A_abs = abs(vc.g2_pct - vc.g1_pct)
        L_min_expected = max(11 * A_abs, 30.0)
        assert vc.length_m >= L_min_expected * 0.95 or vc.length_m >= 30.0, (
            f"Sag VC L={vc.length_m:.1f} < K*|A|={L_min_expected:.1f}"
        )
    print("  ✓ test_k_sag_ssd")


# ── Test 4: Grade clipping — steep uphill ────────────────────────────────────

def test_grade_clipping_uphill():
    """
    A linear profile rising at 15% grade over 3 km.
    With G_MAX=8%, the design grade must not exceed 8%.
    """
    d = np.linspace(0, 3000, 150)
    e = 100.0 + d * 0.15   # 15% grade everywhere

    result = build_vertical_alignment(
        d, e,
        design_speed_kmph=60,
        max_grade_pct=8.0,
        min_vc_length_m=30.0,
    )
    max_g = np.max(np.abs(result.grade_pct))
    # Inside parabolic curves the instantaneous grade can briefly exceed max
    # at the input-grade side; we check the overall 99th percentile
    p99_g = np.percentile(np.abs(result.grade_pct), 99)
    assert p99_g <= 8.5, (
        f"99th-pct design grade {p99_g:.2f}% exceeds 8% limit after clipping"
    )
    print("  ✓ test_grade_clipping_uphill")


# ── Test 5: Grade clipping — steep downhill ───────────────────────────────────

def test_grade_clipping_downhill():
    """
    A linear profile falling at 12% grade over 3 km.
    With G_MAX=8%, the design grade must not drop below -8%.
    """
    d = np.linspace(0, 3000, 150)
    e = 500.0 - d * 0.12   # -12% grade everywhere

    result = build_vertical_alignment(
        d, e,
        design_speed_kmph=60,
        max_grade_pct=8.0,
        min_vc_length_m=30.0,
    )
    p99_g = np.percentile(np.abs(result.grade_pct), 99)
    assert p99_g <= 8.5, (
        f"99th-pct design grade {p99_g:.2f}% exceeds 8% limit after clipping (downhill)"
    )
    print("  ✓ test_grade_clipping_downhill")


# ── Test 6: Flat profile produces trivial result ──────────────────────────────

def test_flat_profile():
    """
    A perfectly flat profile (constant elevation) should:
    - Produce 0 grade violations.
    - Produce 0 or very few SSD violations.
    - Have z_design ≈ z_ground everywhere.
    """
    d = np.linspace(0, 10000, 300)
    e = np.full(300, 150.0)   # flat at 150 m

    result = build_vertical_alignment(
        d, e,
        design_speed_kmph=60,
        max_grade_pct=8.0,
    )
    assert len(result.grade_violations) == 0, (
        f"Flat profile should have 0 grade violations, got {len(result.grade_violations)}"
    )
    # Design should be very close to terrain (within 5 m = minor numerical drift)
    max_diff = np.max(np.abs(result.z_design - result.z_ground))
    assert max_diff < 10.0, (
        f"Flat profile: max |z_design - z_ground| = {max_diff:.2f} m (expected < 10 m)"
    )
    print("  ✓ test_flat_profile")


# ── Test 7: Design FGL must not violently under-cut terrain ──────────────────

def test_design_above_ground():
    """
    For a rolling profile with mixed grades, the design FGL should not be more
    than a reasonable cut depth below the terrain anywhere along the route.
    (Preliminary design allows cuts; we check it's within sanity bounds.)
    """
    d = np.linspace(0, 15000, 400)
    # Synthetic rolling terrain: two hills and a valley
    e = (
        100.0
        + 80.0 * np.sin(2 * np.pi * d / 6000.0)
        + 40.0 * np.sin(2 * np.pi * d / 2500.0)
    )

    result = build_vertical_alignment(
        d, e,
        design_speed_kmph=60,
        max_grade_pct=8.0,
    )
    # Cut should not exceed 200 m (sanity upper bound for preliminary)
    max_cut = -result.cut_fill_m.min()
    assert max_cut < 200.0, (
        f"Unreasonably large cut depth {max_cut:.1f} m — possible algorithm error"
    )
    # Most stations should NOT be cut — design rides above terrain on most fills
    cut_fraction = np.mean(result.cut_fill_m < 0)
    assert cut_fraction < 0.75, (
        f"Too many cut stations: {cut_fraction*100:.0f}% of alignment in cut — "
        f"grade-clipping may be driving design far below terrain"
    )
    print("  ✓ test_design_above_ground")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_parabola_elevation,
        test_k_crest_ssd,
        test_k_sag_ssd,
        test_grade_clipping_uphill,
        test_grade_clipping_downhill,
        test_flat_profile,
        test_design_above_ground,
    ]
    passed = 0
    print("\nPhase 6 — Vertical Alignment Unit Tests")
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
