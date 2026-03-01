# Context

I am continuing work on the Highway Alignment Generator project mapped out in `handoff.md`, which is currently at Phase 14.1. The pipeline successfully executes MS-LCP multi-bridge routing from Point A to Point B.

# The Problem

While the bridge detection and Distance Transform (EDT) span-estimation logic in `structures.py` works mathematically, the OSM data fetched in `data_fetch.py` for major Myanmar rivers (like the Ayeyarwady) only provides 1D single-pixel `LineString` flowlines. Because it lacks explicitly mapped polygon riverbanks or width tags, our bridge spans are artificially constrained to ~40m buffers instead of their true >500m geographical widths, causing inaccurate structure cost estimates.

# The Task

**Phase 14.2: Custom Water Dataset Integration**

1. I have a custom dataset for Myanmar rivers and water bodies that includes accurate polygon widths (e.g., shapefile/GeoJSON).
2. I need you to integrate this dataset into the pipeline so that `water_utm` and `water_mask` in `main.py` utilize this accurate geometry instead of relying purely on the default OSM Overpass queries from `data_fetch.py`.
3. Please review `handoff.md` to understand the current architecture, then ask me to upload or provide the path to the custom water dataset.
4. Once provided, modify `data_fetch.py` and `main.py` to ingest, reproject (to `UTM_EPSG`), and merge this custom water data so the distance transform (EDT) inside `structures.py` can correctly calculate realistic bridge spans over major rivers.
