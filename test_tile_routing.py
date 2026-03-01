"""
test_tile_routing.py — Phase 15 unit tests for tile routing
==============================================================
Tests the TilePartitioner and stitch_tile_paths functions.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(__file__))

from tile_routing import TilePartitioner, TileBBox, TileResult, stitch_tile_paths


# ── Myanmar corridor config for testing ──────────────────────────────────────
# Current A→B: ~201 km
POINT_A = (94.2175, 17.5483)
POINT_B = (96.8304, 17.1174)


# ── 1. Partitioner: short corridor → single tile ────────────────────────────

def test_partitioner_single_tile():
    """Short corridor (< tile length) should produce exactly 1 tile."""
    # Use points ~50 km apart
    pt_a = (94.2175, 17.5483)
    pt_b = (94.7, 17.5)  # ~50 km
    tiles = TilePartitioner(pt_a, pt_b, tile_length_km=100, overlap_km=10).partition()
    assert len(tiles) == 1, f"Expected 1 tile, got {len(tiles)}"
    assert tiles[0].is_first and tiles[0].is_last
    assert tiles[0].tile_index == 0


# ── 2. Partitioner: long corridor → multiple tiles ──────────────────────────

def test_partitioner_multi_tile():
    """201 km corridor with 100 km tiles → should produce 3 tiles."""
    tiles = TilePartitioner(
        POINT_A, POINT_B,
        tile_length_km=100, overlap_km=10
    ).partition()
    # 201 km corridor, tiles at 100 km with 10 km overlap → step = 90 km
    # Tile 0: 0-100, Tile 1: 90-190, Tile 2: 180-201
    assert len(tiles) >= 2, f"Expected at least 2 tiles, got {len(tiles)}"
    assert len(tiles) <= 4, f"Expected at most 4 tiles, got {len(tiles)}"

    # First and last flags
    assert tiles[0].is_first
    assert tiles[-1].is_last
    for t in tiles[1:-1]:
        assert not t.is_first and not t.is_last


# ── 3. Tile bboxes contain entry/exit points ─────────────────────────────────

def test_tile_bboxes_contain_corridor():
    """Each tile's bbox should contain its entry and exit points."""
    tiles = TilePartitioner(
        POINT_A, POINT_B,
        tile_length_km=100, overlap_km=10
    ).partition()

    for tile in tiles:
        w, s, e, n = tile.bbox_wgs84
        entry_lon, entry_lat = tile.entry_lonlat
        exit_lon, exit_lat = tile.exit_lonlat

        assert w <= entry_lon <= e, \
            f"Tile {tile.tile_index} entry lon {entry_lon} outside [{w}, {e}]"
        assert s <= entry_lat <= n, \
            f"Tile {tile.tile_index} entry lat {entry_lat} outside [{s}, {n}]"
        assert w <= exit_lon <= e, \
            f"Tile {tile.tile_index} exit lon {exit_lon} outside [{w}, {e}]"
        assert s <= exit_lat <= n, \
            f"Tile {tile.tile_index} exit lat {exit_lat} outside [{s}, {n}]"


# ── 4. First tile starts at A, last tile ends at B ──────────────────────────

def test_tile_endpoints():
    """First tile entry should be A, last tile exit should be B."""
    tiles = TilePartitioner(
        POINT_A, POINT_B,
        tile_length_km=100, overlap_km=10
    ).partition()
    assert tiles[0].entry_lonlat == POINT_A, \
        f"First tile entry should be A: {tiles[0].entry_lonlat} vs {POINT_A}"
    assert tiles[-1].exit_lonlat == POINT_B, \
        f"Last tile exit should be B: {tiles[-1].exit_lonlat} vs {POINT_B}"


# ── 5. Tile indices are sequential ──────────────────────────────────────────

def test_tile_indices_sequential():
    """Tile indices should be 0, 1, 2, ..., N-1."""
    tiles = TilePartitioner(
        POINT_A, POINT_B,
        tile_length_km=100, overlap_km=10
    ).partition()
    for i, tile in enumerate(tiles):
        assert tile.tile_index == i, f"Expected index {i}, got {tile.tile_index}"


# ── 6. Stitcher: single tile → identity ─────────────────────────────────────

def test_stitch_single_tile():
    """Stitching 1 tile result returns its path unchanged."""
    path = [(100.0 + i, 200.0 + i) for i in range(100)]
    results = [
        TileResult(tile_index=0, path_utm=path, entry_utm=path[0], exit_utm=path[-1])
    ]
    stitched = stitch_tile_paths(results)
    assert len(stitched) == len(path)
    assert stitched[0] == path[0]
    assert stitched[-1] == path[-1]


# ── 7. Stitcher: two overlapping paths ──────────────────────────────────────

def test_stitch_two_tiles():
    """Stitch two overlapping path segments — result should be continuous."""
    # Tile 0: straight line from (0,0) to (100000, 0)
    path_a = [(float(i * 100), 0.0) for i in range(1001)]  # 0 to 100km

    # Tile 1: straight line from (90000, 0) to (200000, 0) — 10km overlap
    path_b = [(90000.0 + float(i * 100), 0.0) for i in range(1101)]  # 90km to 200km

    results = [
        TileResult(tile_index=0, path_utm=path_a,
                   entry_utm=(0.0, 0.0), exit_utm=(100000.0, 0.0)),
        TileResult(tile_index=1, path_utm=path_b,
                   entry_utm=(90000.0, 0.0), exit_utm=(200000.0, 0.0)),
    ]
    stitched = stitch_tile_paths(results)

    # Stitched path should start near (0,0) and end near (200000,0)
    assert stitched[0] == (0.0, 0.0), f"Start: {stitched[0]}"
    assert stitched[-1] == (200000.0, 0.0), f"End: {stitched[-1]}"

    # No large gaps
    for i in range(1, len(stitched)):
        dx = stitched[i][0] - stitched[i-1][0]
        dy = stitched[i][1] - stitched[i-1][1]
        gap = math.hypot(dx, dy)
        assert gap < 500, f"Gap at index {i}: {gap:.1f} m"


# ── 8. Stitcher: empty path handling ─────────────────────────────────────────

def test_stitch_empty():
    """Empty input should return empty output."""
    assert stitch_tile_paths([]) == []


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_partitioner_single_tile,
        test_partitioner_multi_tile,
        test_tile_bboxes_contain_corridor,
        test_tile_endpoints,
        test_tile_indices_sequential,
        test_stitch_single_tile,
        test_stitch_two_tiles,
        test_stitch_empty,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
        except Exception as e:
            print(f"  ✗ {test.__name__}: UNEXPECTED {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
