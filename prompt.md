# Phase 14.5: Data Fetching & Caching Pipeline Audit

You are tasked with auditing and robustifying the data fetching, caching, and preprocessing pipeline of the `highway-routing-engine` before we proceed to Phase 15 (Tile Routing).

With tile routing, the engine will be aggressively fetching, caching, and stitching hundreds of overlapping tiles across nationwide bounding boxes. Any fragility, memory leaks, silent failures, or projection mismatches in the current data pipeline will compound and crash the entire run.

## Your Objective

Deeply review `data_fetch.py`, `cost_surface.py`, and how `main.py` uses them. Your goal is to ensure data is downloaded properly, cached reliably, validated before use, and free of silent corruptions.

### Areas to Audit and Improve:

1. **OSM Overpass API Resilience (`fetch_osm_layers`)**:
   - Overpass API is rate-limited and often fails on large bounding boxes.
   - We currently drop to empty GeoDataFrames (`except Exception as exc: return gpd.GeoDataFrame(...)`) on failure. This is dangerous for tile routing — if one tile fails, that section of the highway will run blindly through buildings/water.
   - **Task:** Implement proper retry logic with exponential backoff for Overpass. If it permanently fails, it MUST raise an error and halt, not silently proceed with empty data.

2. **ESA WorldCover Fetch (`fetch_worldcover`)**:
   - The Planetary Computer STAC API is used but relies on optional/missing imports (`planetary_computer`).
   - If it fails, it returns `None, None` and falls back silently.
   - Check the `rioxarray` tile mosaicking logic (`merge_arrays`) and coordinate re-projection to UTM. Ensure it's robust and won't leak memory on large tile requests.

3. **Overture Maps Integration (`fetch_overture_buildings`)**:
   - Overture S3 fetching is currently wrapped in an exception block that also returns empty GeoDataFrames on failure.
   - The spatial deduplication (`merge_building_sources`) between Overture and OSM uses `sjoin_nearest`, which might be a performance bottleneck or memory hog on large tiles. Audit its performance.

4. **DEM Fallback Chain (`fetch_dem`)**:
   - Ensure the void-filling logic (`distance_transform_edt` nearest neighbour) at the UTM reprojection stage correctly handles edge cases without corrupting adjacent tiles.

5. **Cache Management**:
   - Every file is currently saved with a `_bbox_key` like `W94.0_S20.0_E95.0_N21.0`.
   - When we move to tile routing, boundaries might have slight floating-point differences.
   - **Task:** Audit how caching keys are generated. Ensure they are geographically stable or propose a tile-index-based caching strategy (e.g., Z/X/Y or Slippy Map tiles) that will work seamlessly with Phase 15.

### Instructions

1. **Read `data_fetch.py` and `main.py`** to understand the current data flow.
2. **Draft a detailed Implementation Plan** addressing these specific vulnerabilities.
3. **Wait for User Approval** before making code changes.
4. **Implement the fixes** to ensure the pipeline is bulletproof against network hiccups, API rate limits, and partial-data silent failures.
5. Once the pipeline audit is complete and validated, you may proceed to plan **Phase 15: Tile Routing**.
