"""
test_routing_filter.py — Unit tests for Phase 5.3 routing post-processing
==========================================================================
Tests the _filter_sharp_reversals function that removes stutter-step
artifacts (near-180° reversals) from raw pathfinding output.
"""
import math
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from routing import _filter_sharp_reversals


def test_no_reversals_preserved():
    """A straight path should pass through unchanged."""
    path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    result = _filter_sharp_reversals(path)
    assert result == path, f"Expected {path}, got {result}"


def test_single_reversal_removed():
    """A stutter (point 2 backtracks to near point 0) should be removed."""
    # Forward, then backwards, then forward again — classic stutter
    path = [(0, 0), (1, 0), (0, 0), (1, 1)]
    result = _filter_sharp_reversals(path, max_turn_deg=160)
    # The reversal at (0,0) should be removed
    assert len(result) < len(path), f"Expected shorter path, got {result}"
    assert result[0] == (0, 0), "Start must be preserved"
    assert result[-1] == (1, 1), "End must be preserved"


def test_short_path_unchanged():
    """Paths with < 3 points can't have reversals."""
    path = [(0, 0), (1, 1)]
    result = _filter_sharp_reversals(path)
    assert result == path


def test_gentle_curve_preserved():
    """A gentle 90-degree turn should NOT be removed."""
    path = [(0, 0), (1, 0), (1, 1), (1, 2)]
    result = _filter_sharp_reversals(path, max_turn_deg=160)
    # 90° turn at (1,0)→(1,1) should be kept
    assert len(result) == len(path), f"Gentle turn should be preserved, got {result}"


def test_multiple_stutters():
    """Multiple reversals in sequence should all be filtered."""
    path = [(0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (2, 2)]
    result = _filter_sharp_reversals(path, max_turn_deg=160)
    assert result[0] == (0, 0)
    assert result[-1] == (2, 2)
    assert len(result) < len(path), f"Expected shorter path, got {result}"


def test_empty_and_single():
    """Edge cases."""
    assert _filter_sharp_reversals([]) == []
    assert _filter_sharp_reversals([(5, 5)]) == [(5, 5)]


if __name__ == "__main__":
    tests = [
        test_no_reversals_preserved,
        test_single_reversal_removed,
        test_short_path_unchanged,
        test_gentle_curve_preserved,
        test_multiple_stutters,
        test_empty_and_single,
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
            print(f"  ✗ {test.__name__}: UNEXPECTED {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
