import numpy as np
import pytest
from routing import (
    _extract_directional_waypoints,
    _map_waypoints_to_high_res,
    _route_segment_worker
)

def test_extract_directional_waypoints():
    """Test waypoint extraction logic using angle threshold and distance fallback."""
    # A path that goes straight right, then turns 90-degrees down
    path = [
        (0, 0), (0, 1), (0, 2), (0, 3), # straight right
        (1, 3), (2, 3), (3, 3)          # sharp turn down
    ]
    
    # 45 degree threshold should catch the 90 degree turn at (0, 3)
    # Plus start and end points
    waypoints = _extract_directional_waypoints(path, angle_thresh_deg=45)
    
    assert len(waypoints) == 3
    assert waypoints[0] == (0, 0)
    assert waypoints[1] == (0, 3)
    assert waypoints[2] == (3, 3)
    
    # Test distance fallback: max_dist_px=2
    # Even on a straight line, it should add waypoints every 2 units
    straight_path = [(0, i) for i in range(10)]
    wps_dist = _extract_directional_waypoints(straight_path, angle_thresh_deg=45, max_dist_px=2.5)
    # Start (0, 0), (0, 3) [dist=3], (0, 6) [dist=3], (0, 9) [dist=3, end]
    # Actually distances:
    # (0,0) -> (0,1),(0,2)->dist=2 < 2.5
    # (0,3) -> dist from last (0,0) = 3 >= 2.5, saved!
    # (0,6) -> dist from last (0,3) = 3 >= 2.5, saved!
    # End point is always saved.
    assert (0, 3) in wps_dist
    assert (0, 6) in wps_dist
    assert wps_dist[0] == straight_path[0]
    assert wps_dist[-1] == straight_path[-1]

def test_map_waypoints_to_high_res():
    """Test mapping low resolution waypoints to high resolution block with min cost."""
    # Create a 4x4 high res grid
    # Ratios = 2 (so this maps a 2x2 low res grid)
    high_cost = np.array([
        [10, 10,  1,  5],
        [10, 10,  2, 10],
        [ 5, 10, 10, 10],
        [10,  2, 10, 10]
    ], dtype=np.float32)
    
    waypoints_low = [
        (0, 0), # Top-Left quadrant, min cost is 10 (all 10s) R=(0..2), C=(0..2)
        (0, 1), # Top-Right quadrant, min cost is 1 at (0, 2)
        (1, 0), # Bottom-Left quadrant, min cost is 2 at (3, 1)
    ]
    
    # Top-Left block: 
    # [10, 10]
    # [10, 10]
    # Min is at (r=0, c=0) relative (first one)
    
    mapped = _map_waypoints_to_high_res(waypoints_low, high_cost, ratio=2)
    
    assert mapped[0] == (0, 0) # Top-Left
    assert mapped[1] == (0, 2) # Top-Right, value 1
    assert mapped[2] == (3, 1) # Bottom-Left, value 2

def test_route_segment_worker():
    """Test localized routing between two high-res mapping points."""
    cost_grid = np.ones((20, 20), dtype=np.float32)
    
    # Create a diagonal impenetrable wall
    for i in range(15):
        cost_grid[i, 15-i] = 1e9
        
    # Except for a gap at (7, 8)
    cost_grid[7, 8] = 1.0
    
    wp_from = (2, 2)
    wp_to = (12, 12)
    
    args = (cost_grid, wp_from, wp_to)
    path = _route_segment_worker(args)
    
    # The path should navigate around/through the gap and reach the destination
    assert path[0] == wp_from
    assert path[-1] == wp_to
    
    # Check that it traverses the gap
    # Because geometric routing is on, the exact path might touch the corner,
    # but it shouldn't hit the 1e9 cells
    for r, c in path:
        assert cost_grid[r, c] < 1e9
