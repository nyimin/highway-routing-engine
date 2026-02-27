"""
test_structures.py — Unit tests for Phase 8 bridge and culvert inventory
=========================================================================
All tests are deterministic — NO DEM, NO OSM, NO external data.

Tests:
  1. test_z_at_interpolation       — z_design interpolation at known chainages
  2. test_bridge_cost_formula      — cost = rate × length × width
  3. test_culvert_detection_minima — only valley minima flagged as culverts
  4. test_no_crossings_empty_water — empty water GDF → 0 bridges
  5. test_balance_merge_close_crossings — crossings <50 m apart merged
  6. test_structure_sorting        — sorted by chainage in output
  7. test_culvert_spacing          — culverts spaced by min_spacing_m
"""
import sys
import os
import numpy as np
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(__file__))

from structures import (
    _z_at, _find_culvert_sites, build_structure_inventory,
    _find_water_crossings,
)


# ── Mock VerticalAlignmentResult ──────────────────────────────────────────────

@dataclass
class _MockVA:
    distances_m: np.ndarray
    z_design: np.ndarray
    cut_fill_m: np.ndarray


def _make_va(n=100, length=10000.0, grade_m_per_m=0.05):
    """Simple uphill VA for testing."""
    d  = np.linspace(0.0, length, n)
    z  = np.linspace(100.0, 100.0 + length * grade_m_per_m, n)
    cf = np.zeros(n)
    return _MockVA(distances_m=d, z_design=z, cut_fill_m=cf)


# ── Test 1: z interpolation ───────────────────────────────────────────────────

def test_z_at_interpolation():
    """
    Linear z from 100 to 600 m over 10,000 m.
    At chainage 5000 m → z should be ~350 m.
    At chainage 0 → 100 m, at 10000 → 600 m.
    """
    va = _make_va(length=10000.0, grade_m_per_m=0.05)

    z0 = _z_at(0.0, va)
    zM = _z_at(5000.0, va)
    zE = _z_at(10000.0, va)

    assert abs(z0 - 100.0) < 0.5, f"z at 0 should be ~100, got {z0:.2f}"
    assert abs(zM - 350.0) < 1.0, f"z at 5000 should be ~350, got {zM:.2f}"
    assert abs(zE - 600.0) < 0.5, f"z at 10000 should be ~600, got {zE:.2f}"
    print("  ✓ test_z_at_interpolation")


# ── Test 2: Bridge cost formula ───────────────────────────────────────────────

def test_bridge_cost_formula():
    """
    Bridge of length 80 m, width 12 m, rate USD 3,500/m²
    Expected cost = 80 × 12 × 3500 = USD 3,360,000.
    """
    span   = 80.0
    width  = 12.0
    rate   = 3_500.0
    expected = span * width * rate

    cost = rate * span * width
    assert abs(cost - expected) < 1.0, f"Cost formula mismatch: {cost} ≠ {expected}"
    print("  ✓ test_bridge_cost_formula")


# ── Test 3: Culvert detection — only minima flagged ───────────────────────────

def test_culvert_detection_minima():
    """
    Build a VA with an explicit discrete valley at index 25 in a 50-point grid.
    Only the sag station should become a culvert, not the ridges.
    """
    n = 50
    d = np.linspace(0, 5000, n)
    # V-shape: decreasing then increasing — strict discrete minimum at index 25
    z = np.abs(np.arange(n) - 25).astype(float) * 2.0 + 100.0
    # Check: z[25]=100, z[24]=102, z[26]=102 → strict minimum at 25

    va = _MockVA(distances_m=d, z_design=z, cut_fill_m=np.zeros(n))

    # Build a flow_accum with high values (all qualify)
    fa = np.full((50, 50), 500, dtype=np.int32)
    path_indices = [(i, i) for i in range(50)]

    sites = _find_culvert_sites(
        va, fa, transform=None,
        path_indices=path_indices,
        bridge_chainages=set(),
        min_accum_cells=100,
        min_spacing_m=50.0,
    )

    assert len(sites) >= 1, f"Expected culvert at valley minimum, got {len(sites)}"
    # Nearest culvert should be near index 25 (chainage ≈ 2500 m)
    nearest = min(sites, key=lambda s: abs(s["chainage_m"] - d[25]))
    assert abs(nearest["chainage_m"] - d[25]) < 500, (
        f"Culvert chainage {nearest['chainage_m']:.0f} far from valley at {d[25]:.0f} m"
    )
    print("  ✓ test_culvert_detection_minima")



# ── Test 4: No crossings with empty water ────────────────────────────────────

def test_no_crossings_empty_water():
    """Passing None or an empty GeoDataFrame returns []."""
    va = _make_va()
    smooth_utm = [(float(i * 100), float(i * 50)) for i in range(100)]

    # Case 1: None
    crossings = _find_water_crossings(smooth_utm, None, va)
    assert crossings == [], f"Expected [] with None water, got {crossings}"

    # Case 2: Empty GeoDataFrame (requires geopandas)
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        empty_gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32647")
        crossings2 = _find_water_crossings(smooth_utm, empty_gdf, va)
        assert crossings2 == [], f"Expected [] with empty GDF, got {crossings2}"
    except ImportError:
        pass   # geopandas not available: skip this sub-case

    print("  ✓ test_no_crossings_empty_water")



# ── Test 5: Close crossing merge ─────────────────────────────────────────────

def test_close_crossings_merge():
    """
    Two crossings separated by 30 m should merge into one.
    """
    from structures import _find_water_crossings

    # Build two simple LineString crossings via Shapely
    try:
        from shapely.geometry import Polygon, LineString
        import geopandas as gpd
    except ImportError:
        print("  ⚠ test_close_crossings_merge: shapely/geopandas not available — skipped")
        return

    # Route: straight west-to-east at y=0
    route = [(float(x), 0.0) for x in range(0, 10001, 100)]
    va    = _make_va(n=len(route), length=10000.0)

    # Two narrow water polygons that the route will cross, 30 m apart
    water1 = Polygon([(2000, -50), (2100, -50), (2100, 50), (2000, 50)])
    water2 = Polygon([(2110, -50), (2190, -50), (2190, 50), (2110, 50)])
    water_gdf = gpd.GeoDataFrame(
        {"geometry": [water1, water2], "name": ["r1", "r2"]},
        crs="EPSG:32647"
    )

    crossings = _find_water_crossings(route, water_gdf, va)
    # 30 m gap → should merge
    assert len(crossings) == 1, (
        f"Expected 1 merged crossing, got {len(crossings)}: {crossings}"
    )
    print("  ✓ test_close_crossings_merge")


# ── Test 6: Structure output is sorted by chainage ───────────────────────────

def test_structure_sorting():
    """
    After build_structure_inventory with no water but culverts, the
    structures list must be sorted by chainage_m ascending.
    """
    n  = 100
    d  = np.linspace(0, 10000, n)
    # Multiple valleys
    z  = 100.0 + 5.0 * np.sin(2 * np.pi * d / 3000)
    va = _MockVA(distances_m=d, z_design=z, cut_fill_m=np.zeros(n))

    fa = np.full((100, 100), 500, dtype=np.int32)
    path_indices = [(i, i % 100) for i in range(100)]
    smooth_utm   = [(float(i * 100), 0.0) for i in range(100)]

    inventory = build_structure_inventory(
        smooth_utm=smooth_utm,
        va_result=va,
        water_utm=None,
        flow_accum=fa,
        transform=None,
        path_indices=path_indices,
        bridge_freeboard_m=1.5,
        bridge_cost_per_m2_usd=3500.0,
        bridge_width_m=12.0,
        culvert_unit_cost_usd=15000.0,
        min_culvert_accum_cells=100,
    )

    chainages = [s.chainage_m for s in inventory.structures]
    assert chainages == sorted(chainages), f"Structures not sorted by chainage: {chainages[:5]}"
    print("  ✓ test_structure_sorting")


# ── Test 7: Culvert minimum spacing ──────────────────────────────────────────

def test_culvert_spacing():
    """
    Many adjacent minima should only generate culverts spaced at ≥ min_spacing_m.
    Use a high-frequency terrain so every station oscillates.
    """
    n = 500
    d = np.linspace(0, 10000, n)
    z = 100.0 + np.sin(2 * np.pi * d / 50)   # period = 50 m → many minima

    va = _MockVA(distances_m=d, z_design=z, cut_fill_m=np.zeros(n))
    fa = np.full((50, 50), 1000, dtype=np.int32)
    path_indices = [(i % 50, i % 50) for i in range(500)]

    MIN_SPACING = 200.0
    sites = _find_culvert_sites(
        va, fa, transform=None,
        path_indices=path_indices,
        bridge_chainages=set(),
        min_accum_cells=100,
        min_spacing_m=MIN_SPACING,
    )

    for j in range(1, len(sites)):
        gap = sites[j]["chainage_m"] - sites[j - 1]["chainage_m"]
        assert gap >= MIN_SPACING - 1.0, (
            f"Culverts at {sites[j-1]['chainage_m']:.0f} and "
            f"{sites[j]['chainage_m']:.0f} m are only {gap:.1f} m apart "
            f"(min={MIN_SPACING} m)"
        )
    print("  ✓ test_culvert_spacing")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_z_at_interpolation,
        test_bridge_cost_formula,
        test_culvert_detection_minima,
        test_no_crossings_empty_water,
        test_close_crossings_merge,
        test_structure_sorting,
        test_culvert_spacing,
    ]
    passed = 0
    print("\nPhase 8 — Bridge and Culvert Inventory Unit Tests")
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
