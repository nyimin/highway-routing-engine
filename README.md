# Myanmar Highway Alignment Generator

An automated Python pipeline for generating optimum-cost highway alignments.
It fetches elevation (DEM) and infrastructure data (OpenStreetMap), builds a
multi-layer cost surface, runs two-resolution pathfinding, and exports
engineering-grade GeoJSON outputs with a 10-product visualization suite.

> **Current phase: 17.0** — Constraint-First Bridge Scouting & River Avoidance.

---

## Features

### Data Acquisition

- **DEM fallback chain**: COP30 → SRTMGL1 → SRTMGL3 → synthetic mock.
- **OSM layers**: buildings, water bodies, roads/tracks, land-use/natural via Overpass.
- **D8 stream fallback**: auto-derives a flow-accumulation stream network from the DEM when OSM water data is sparse.

### Cost Surface

- **Slope cost** with 4-zone thresholds (optimal / moderate / max / cliff) and exponential ramp.
- **River hierarchy** — connected-component analysis classifies water into 5 tiers (culvert → Ayeyarwady-scale) with perpendicular crossing enforcement. Integrated custom GeoPackage datasets for high-fidelity true-width river polygons.
- **Constraint-First Bridge Siting** — major rivers (Tier 3 & 4) are completely `IMPASSABLE` by default. Instead, an upfront Bridge Scouting pass identifies optimal crossing axes and dynamically burns "passable corridors" into the cost surface.
- **Dam & Reservoir Avoidance** — explicit geographical buffering around sensitive water infrastructure permanently excluded with `IMPASSABLE` routing costs.
- **Area-based building penalties** — footprint-scaled peak penalty with continuous EDT distance-decay gradient (Phase 5.2).
- **LULC environmental multipliers** — wetland, forest, farmland, conservation areas.
- **Road discount** — existing OSM tracks receive a 0.5× cost multiplier.

### Routing

- **Multi-Scale LCP (MS-LCP)** — progressive resolution cost pyramid, 45° directional waypoint extraction, and parallel segmented routing (based on Tang & Dou, 2023).
- **Rubber-band centering** — distance-transform penalty prevents meandering; bounding-box-diagonal normalization avoids over-penalization in narrow corridors.
- **Constraint-First Bridge Scouting** — the MS-LCP algorithm initiates with a massive downsampled pyramid search (e.g. 120m) to scout for optimal perpendicular bridges across `IMPASSABLE` rivers. These structures are identified using dynamic width minimization and riverbank stability metrics, then injected back into the cost hierarchy.
- **Sub-pixel FMM gradient descent** — eliminates grid stair-step artifacts (Phase 5.3).
- **Sharp-reversal filter** — removes near-180° waypoint stutters (configurable via `TURNING_ANGLE_FILTER_DEG`).
- **Multi-Waypoint Routing (A → B → C)** — sequential leg processing with recursive stitching and segment-aware tracking.
- **Engine auto-selection**: `scikit-fmm` if available, otherwise Dijkstra.

### Geometry & Vertical Alignment (Phase 6)

- **Segment-aware weighted B-spline smoothing** dynamically adjusts curvature radius based on local terrain capabilities with clothoid transition analysis.
- **Curve-radius verification** against design-speed minimums (e.g., 150 m at 60 km/h).
- **Finished Grade Line (FGL)** vertical alignment with parabolic curves.
- **Sustained-grade verification** and AASHTO stopping sight distance checks.

### Volumetrics & Structures (Phases 7–8)

- **Earthwork computation**: Trapezoidal cross-sections, Average-End-Area volumes, and Brückner mass-haul balancing.
- **Structure inventory**: Shapely-based bridge geometric modeling over tier 3+ rivers.
- **Culvert siting**: Identifying optimal drainage pipes at D8 vertical profile minima.

### Cost & Reporting (Phases 9–10)

- **Parametric Cost Model**: Aggregates earthwork, pavement, structures, and land acquisition (LULC-weighted) into a project USD cost with **Segment-Specific Breakdowns**.
- **Automated Feasibility Report**: Jinja2 + WeasyPrint driven HTML/PDF generation presenting the executive summary, engineering profiles, segment subtotals, and cost estimate.

### Performance

- **Memory auto-tuning** — estimates peak memory before routing; auto-doubles `COARSE_FACTOR` if it exceeds `MEMORY_WARN_GB`.
- **Numba JIT** acceleration via `try_jit` decorator (optional install).
- **Stage checkpointing** — fault-tolerant restart from last completed stage.
- **Per-stage timing + `tracemalloc` peak tracking**.

### Visualization Suite

Automatically generates 12 diagnostic products after routing:

| #   | Output                    | Description                                                                             |
| --- | ------------------------- | --------------------------------------------------------------------------------------- |
| 1   | `cost_heatmap.png`        | Log-scaled unified cost surface + route overlay                                         |
| 2   | `building_decay.png`      | Zoomed EDT decay rings around densest building cluster                                  |
| 3   | `layer_decomposition.png` | 2×3 panel of individual cost layers                                                     |
| 4   | `elevation_profile.png`   | Longitudinal profile with grade-violation bands                                         |
| 5   | `slope_histogram.png`     | Slope distribution with 4-zone thresholds                                               |
| 6   | `cost_along_route.png`    | Per-waypoint cost + dominant-layer indicator                                            |
| 7   | `route_3d.png`            | 3D terrain drape (downsampled, vertical exaggeration)                                   |
| 8   | `cross_sections.png`      | 4 perpendicular terrain slices at key chainage                                          |
| 9   | `route_dashboard.png`     | Statistics summary card + terrain pie chart                                             |
| 10  | `route_map.html`          | Interactive Folium/Leaflet map with slope-coded route, toggleable building/water layers |
| 11  | `vertical_profile.png`    | Finished Grade Line (FGL) vs terrain with parabolic VCs                                 |
| 12  | `earthwork_masshual.png`  | Depth profile, cumulative volumes, and Brückner mass-haul diagram                       |

### Export

- `preliminary_route.geojson` — smoothed alignment with full metadata.
- `output/cost_estimate.csv` — itemised cost quantities.
- `output/feasibility_report.html` (and `.pdf`) — 6-page A4 Feasibility Report.
- Optional intermediate GeoTIFFs (`cost_surface.tif`, `building_penalty.tif`).

---

## Project Structure

```
├── main.py                  # Pipeline orchestrator
├── config.py                # All constants: coordinates, slopes, penalties, unit rates
├── data_fetch.py            # DEM download, OSM queries, D8 streams
├── cost_surface.py          # Multi-layer cost raster
├── routing.py               # Coarse-to-fine routing, gradient descent, filters
├── geometry_utils.py        # Coordinate transforms, B-spline, geometry verification
├── vertical_alignment.py    # Grade-clipping FGL, parabolic VC fitting, SSD check
├── earthwork.py             # Trapezoidal cross-sections, AEA volumes, mass-haul
├── structures.py            # Bridge/culvert detection and cost estimation
├── cost_model.py            # Parametric cost aggregation
├── report.py                # Jinja2/WeasyPrint feasibility report generator
├── visualize_route.py       # 12-product visualization suite
├── serve.py                 # No-cache local HTTP server for auto-viewing outputs
├── templates/               # Jinja2 templates (report.html)
├── requirements_alignment.txt
├── .env                     # OPENTOPOGRAPHY_API_KEY (not committed)
├── data/                    # Cached DEMs and GeoPackages
├── cache/                   # OSM query cache
├── output/                  # Visualizations, GeoTIFFs, HTML map
└── test_*.py                # 35 deterministic unit tests
```

---

## Requirements

**Python 3.9+**

### API Key

An [OpenTopography](https://opentopography.org/) API key is required for DEM downloads.
Create a `.env` file in the project root:

```env
OPENTOPOGRAPHY_API_KEY=your_api_key_here
```

### Install Dependencies

```bash
pip install -r requirements_alignment.txt
```

### Optional Add-ons

```bash
pip install scikit-fmm    # Smoother FMM paths (recommended)
pip install numba>=0.57   # 10–50× speedup on cost surface loops
pip install weasyprint jinja2 # Required for PDF/HTML report generation
```

---

## Usage

### 1. Configure

Edit `config.py` to set your sequence of waypoints (`WAYPOINTS`), UTM zone
(`UTM_EPSG`), resolution, slope thresholds, and feature flags:

```python
WAYPOINTS = [
    (94.2175, 17.5483), # Point A (lon, lat)
    (95.5000, 17.3000), # Midpoint 1
    (96.8304, 17.1174), # Point B
]

EXPORT_INTERMEDIATES     = True   # Save cost_surface.tif, building_penalty.tif
GENERATE_VISUALIZATIONS  = True   # Auto-run the 10-product viz suite
FAST_MODE                = False  # True = coarse-only (300 m) for rapid screening
ROUTING_ENGINE           = 'auto' # 'auto' | 'fmm' | 'dijkstra'
```

### 2. Run

```bash
python main.py
```

The pipeline will:

1. Download / cache DEM and OSM data.
2. Build the multi-layer cost surface.
3. Run MS-LCP routed pathfinding and verify geometry.
4. Compute 3D vertical alignment, earthwork volumes, and structure inventory.
5. Compute the parametric cost model and generate the feasibility report.
6. Export visuals, GeoJSON, CSVs, and the final HTML/PDF report.

### 3. Inspect Outputs

- Open `preliminary_route.geojson` in QGIS or [geojson.io](https://geojson.io).
- Open `output/route_map.html` in a browser for the interactive slope-coded map.
- Review the PNG suite in `output/` for engineering diagnostics.

---

## Configuration Reference

| Constant                    | Default                                     | Description                                         |
| --------------------------- | ------------------------------------------- | --------------------------------------------------- |
| `RESOLUTION`                | 30                                          | Grid cell size (metres)                             |
| `PYRAMID_LEVELS`            | 3                                           | Number of downscaling levels for MS-LCP             |
| `DOWNSAMPLE_RATIO`          | 2                                           | Factor to downscale the cost surface per level      |
| `SLOPE_MAX_PCT`             | 12                                          | Maximum design slope (%)                            |
| `SLOPE_CLIFF_PCT`           | 25                                          | Impassable cliff threshold (%)                      |
| `TURNING_ANGLE_FILTER_DEG`  | 160                                         | Filter near-reversal waypoints (°)                  |
| `MEMORY_WARN_GB`            | 4.0                                         | Auto-escalate coarse factor above this              |
| `WATER_PENALTY_TIERS`       | [2000, 8000, 50000, IMPASSABLE, IMPASSABLE] | Per-tier river crossing multipliers                 |
| `ROAD_DISCOUNT`             | 0.5                                         | Existing-road cost multiplier                       |
| `BRIDGE_SCOUT_RESOLUTION_M` | 120.0                                       | Resolution for the pre-routing bridge scouting pass |
| `TILE_ROUTING_THRESHOLD_KM` | 150                                         | Use tile routing if leg is longer than this         |
| `WAYPOINTS`                 | list                                        | List of (lon, lat) waypoints for the route          |

See `config.py` for the full list.

---

## Tests

We maintain a suite of 35 deterministic unit tests across all phases:

```bash
# E.g. test geometric checks, cost model logic, report templating
python test_earthwork.py
python test_vertical_alignment.py
python test_structures.py
python test_cost_model.py
python test_report.py
```

---

## Known Issues

- **`bridge_siting.py` is deprecated:** This script is rendered obsolete by the recent `multi_pass_routing` integration and structure detection logic. It is currently disconnected from the pipeline and remains in the codebase. Do not use it until it undergoes a complete architectural overhaul. (See `ISSUES.md`).

---

## License

Internal project — not licensed for public distribution.
