# Highway Alignment Generator — Handoff Document (Phase 16 Complete)

## Current State

The pipeline runs end-to-end on real Myanmar terrain (SRTM 30 m DEM, 373 K OSM
buildings, 10 K+ river components) and produces a **276 km alignment** with zero
curve-radius or grade violations. Exit code 0, peak memory ~2.9 GB on 8 GB RAM.

**Phases 6–17 have been added**, implementing a full 3D vertical design profile,
earthwork volumetrics, a bridge/culvert structure inventory, a parametric cost model, automated feasibility reports, minimum design lengths constraints, earthwork proxy awareness, continuous multi-bridge routing, the Tang & Dou (2023) MS-LCP progressive routing algorithm, terrain-adaptive geometric smoothing, and constraint-first bridge scouting for hydrology integration.

---

## Phase History

| Phase    | Scope                                                                                                                 |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| 1–2      | DEM fetch (OpenTopography fallback chain), basic slope cost, OSM buildings/water                                      |
| 3        | D8 flow-accumulation stream fallback when OSM water is sparse                                                         |
| 4        | Two-resolution coarse-to-fine routing, corridor band masking, FMM/Dijkstra engine                                     |
| 5.0      | Numba JIT (`try_jit`), vectorised flow-accumulation, `tracemalloc`, auto `COARSE_FACTOR`                              |
| 5.1      | LULC environmental multipliers, road discount, 5-tier river hierarchy with bridge siting                              |
| 5.2      | Area-based building penalties with EDT distance-decay, chunked rasterization                                          |
| 5.3      | Routing polish, 12-product visualization suite, stability audit                                                       |
| 5.4      | Class-based road discounts, expanded LULC table (18 categories), OSM data mitigations                                 |
| **6**    | **Vertical alignment — grade-clipping FGL, parabolic VCs, 3D GeoJSON export**                                         |
| **7**    | **Earthwork volumes — trapezoidal AEA, Brückner mass-haul, CSV export**                                               |
| **8**    | **Bridge & culvert inventory — geometric crossing detection, D8 culvert siting**                                      |
| **9**    | **Parametric Cost Model — aggregate all cost components into USD breakdown (Supports 8-Lane Expressway scale)**       |
| **10**   | **Automated Feasibility Report — Jinja2 HTML → PDF (with WeasyPrint)**                                                |
| **11**   | **Earthwork Proxy — route pathfinding accounts for local terrain relief volumes**                                     |
| **12**   | **Design Validations — enforces minimum geometric lengths for curves and tangents**                                   |
| **13**   | **Multi-Bridge Support — continuous pathfinding sequences through N identified bridges**                              |
| **14**   | **MS-LCP Optimization — Tang & Dou (2023) progressive pyramid & parallel segmented routing**                          |
| **14.1** | **Geometric Refinement — Segment-aware weighted B-splines for terrain-adaptive smoothing**                            |
| **14.2** | **Custom Water Integration — High-fidelity GeoPackage polygon rivers replacing OSM LineStrings**                      |
| **14.3** | **Dam & Reservoir Avoidance — Absolute routing exclusion buffers around critical waterbodies**                        |
| **14.4** | **Routing Logic Corrections — Fixed multiplicative/additive water penalties, resolving coastal detour anomalies**     |
| **15**   | **Tile Routing — Memory-optimized corridor partitioning for long-distance routes (> 150 km)**                         |
| **16**   | **Multi-Waypoint Routing — Sequential leg processing with segment-aware reporting and visualization**                 |
| **17**   | **Constraint-First Bridge Scouting — Downsampled pre-routing to find optimal crossing axes over `IMPASSABLE` rivers** |

---

## File Structure

| File                         | Responsibility                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `main.py`                    | Orchestrator — steps 12b (VA), 12c (earthwork), 12d (structures)                   |
| `config.py`                  | All constants for Phases 1–8 (see Phase 6/7/8 sections at bottom)                  |
| `data_fetch.py`              | DEM download, OSM Overpass queries, D8 stream derivation                           |
| `cost_surface.py`            | Multi-layer cost raster: slope, water, buildings (EDT), LULC, roads                |
| `routing.py`                 | Tang & Dou MS-LCP, Constraint-First Bridge Scouting, FMM gradient descent, filters |
| `geometry_utils.py`          | Coordinate transforms, B-spline smoothing, curve/grade checks, 2D + 3D GeoJSON     |
| `vertical_alignment.py`      | Grade-clipping FGL, parabolic VC fitting, AASHTO K-values, SSD check               |
| `earthwork.py`               | Trapezoidal cross-sections, AEA volumes, Brückner mass-haul, balance points        |
| `structures.py`              | Bridge crossing detection (Shapely), culvert siting (D8), cost estimation          |
| `cost_model.py`              | Parametric cost model aggregating earthwork, structures, land, and project costs   |
| `report.py`                  | Feasibility report generator (Jinja2 + WeasyPrint HTML/PDF export)                 |
| `templates/report.html`      | Jinja2 template file for the feasibility report                                    |
| `visualize_route.py`         | 12-product suite (plots 11 + 12 respectively for VA and earthwork)                 |
| `jit_utils.py`               | `try_jit` Numba wrapper with CPU fallback                                          |
| `test_vertical_alignment.py` | 7 deterministic tests — Phase 6                                                    |
| `test_earthwork.py`          | 7 deterministic tests — Phase 7                                                    |
| `test_structures.py`         | 7 deterministic tests — Phase 8                                                    |
| `test_cost_model.py`         | 7 deterministic tests — Phase 9                                                    |
| `test_report.py`             | 7 deterministic tests — Phase 10                                                   |

---

## Key Pipeline Steps in `main.py`

```
1  bbox              — bounding box from POINT_A / POINT_B
2  DEM fetch         — OpenTopography COP30 → SRTMGL1 → SRTMGL3 → mock fallback
3  OSM layers        — buildings, water, roads, LULC via Overpass API
4  slope + curvature — compute_slope (numpy gradient)
5  rasterise layers  — building EDT penalty, water_mask, roads_mask, lulc_penalty
6  cost surface      — build_cost_surface (slope × water × building × LULC × road)
7  grid endpoints    — A → B → C sequence of (row, col) coordinates
8  routing           — sequential loop over waypoint pairs (TILE vs MONOLITHIC dispatch)
9  stitch paths      — concatenate paths and build `segment_indices` array
10 smooth path       — segment-aware B-spline → smooth_utm (500 pts)
10 verify geometry   — curve_radius, row_setback, sustained_grade, clothoids
11 extract profile   — longitudinal elevation + slope arrays
12a VA input check   — guard: va_result = None if data insufficient
12b vertical align   — build_vertical_alignment() → va_result (VerticalAlignmentResult)
12c earthwork        — compute_earthwork() → ew_result (EarthworkResult)
12d structures       — build_structure_inventory() → si_result (StructureInventory)
13 reproject + meta  — UTM → WGS-84, compute_metadata(), export_geojson()
14 3D GeoJSON        — export_geojson_3d() if EXPORT_3D_GEOJSON=True
12e cost model       — compute_cost_model() → cost_result (CostModelResult)
12f report           — generate_report() → HTML/PDF paths
15 visualize         — generate_all_visuals(viz_data) — 12 products
```

---

## Phase 6 — Vertical Alignment (`vertical_alignment.py`)

### Key Constants (`config.py`)

```python
GRADE_MAX_PCT      = {"expressway": 5.0, "rural_trunk": 8.0, "mountain_road": 10.0}
VC_K_VALUES        = {"crest": 26, "sag": 37}   # AASHTO K for 80 km/h
MIN_VC_LENGTH_M    = 60.0
MIN_VPI_SPACING_M  = 200.0
SCENARIO_PROFILE   = "rural_trunk"   # controls G_MAX from GRADE_MAX_PCT
DESIGN_SPEED_KMPH  = 80
EXPORT_3D_GEOJSON  = True
OUTPUT_FILE_3D     = "preliminary_route_3d.geojson"
```

### Output Dataclass (`VerticalAlignmentResult`)

```python
distances_m      # 1D array — chainage at each station
z_terrain        # raw DEM terrain elevation
z_design         # Finished Grade Line (FGL) elevation
grade_pct        # instantaneous grade (%) at each station
vertical_curves  # list of (VPI_station, g1, g2, L, K) tuples
cut_fill_m       # z_design − z_terrain (+fill, -cut) at each station
grade_violations # list of (start_m, end_m, avg_grade)
ssd_violations   # list of (VPI_station, K_used, K_required)
max_grade_pct    # float — worst grade in design
```

### Visualization

`output/vertical_profile.png` — Viz #11: dual-panel (FGL vs terrain + grade %)

---

## Phase 7 — Earthwork Volumes (`earthwork.py`)

### Key Constants (`config.py`)

```python
FORMATION_WIDTH_M     = {"expressway": 26.0, "rural_trunk": 11.0, "mountain_road": 7.5}
CUT_BATTER_HV         = 1.0    # 1H:1V cut slope
FILL_BATTER_HV        = 1.5    # 1.5H:1V fill slope
SWELL_FACTOR          = 1.25   # bank m³ → loose m³
OUTPUT_EARTHWORK_CSV  = "output/earthwork_volumes.csv"
```

### Algorithm

- **Cross-section area**: `A = h × (B + h × batter_HV)` (trapezoidal)
- **Volume**: Average-End-Area between stations
- **Brückner ordinate**: cumulative `(fill_vol − cut_vol × swell_factor)`
- **Balance stations**: linear interpolation at zero-crossings

### Output Dataclass (`EarthworkResult`)

```python
distances_m          # chainage array
cut_fill_m           # signed depth (+fill, -cut)
cut_area_m2          # cross-section area per station
fill_area_m2
cumul_cut_m3         # cumulative cut volume
cumul_fill_m3
mass_haul_m3         # Brückner ordinate
balance_stations_m   # list of chainage (m) where mass-haul = 0
total_cut_m3
total_fill_m3
net_import_m3        # >0 → need to import fill; <0 → surplus spoil
swell_factor
formation_width_m
```

### Visualization

`output/earthwork_masshual.png` — Viz #12: 3-panel (depth profile + cumul. volumes + Brückner)

---

## Phase 8 — Structure Inventory (`structures.py`)

### Key Constants (`config.py`)

```python
BRIDGE_FREEBOARD_M      = 1.5       # m above design high-water (Myanmar DRD)
BRIDGE_COST_PER_M2_USD  = 3_500.0   # World Bank ROCKS Myanmar (2020–2024)
BRIDGE_DECK_WIDTH_M     = 12.0      # rural_trunk 2-lane + parapets
CULVERT_UNIT_COST_USD   = 15_000.0  # Myanmar DRD box culvert lump sum
MIN_CULVERT_ACCUM_CELLS = 200       # D8 catchment threshold (~0.18 km² at 30 m)
OUTPUT_STRUCTURES_CSV   = "output/structures.csv"
```

### Detection Logic

- **Bridges**: Shapely intersection of `smooth_utm` LineString with `water_utm` polygons → merged spans
- **Culverts**: Local vertical-alignment minima where `flow_accum > MIN_CULVERT_ACCUM_CELLS`
- **Deck elevation**: `z_design(mid_chainage) + BRIDGE_FREEBOARD_M`
- **Bridge cost**: `BRIDGE_COST_PER_M2_USD × span_m × BRIDGE_DECK_WIDTH_M`

### Output Dataclass (`StructureInventory`)

```python
structures               # list[Structure] — one per bridge or culvert
total_bridge_length_m
total_bridge_cost_usd
total_culvert_cost_usd
total_structure_cost_usd
bridge_count
culvert_count
```

### Visualization

Folium map (`output/route_map.html`) — togglable **Structures** layer: red bridge markers + orange culvert dots with chainage/cost popups

---

## Phase 9 — Parametric Cost Model (`cost_model.py`)

### Key Constants (`config.py`)

```python
EARTHWORK_CUT_RATE_USD_M3   = 8.0
EARTHWORK_FILL_RATE_USD_M3  = 15.0
PAVEMENT_RATE_USD_M2        = 120.0
CORRIDOR_WIDTH_M             = 60.0
LAND_ACQ_DEFAULT_USD_PER_HA = 7_500.0
LAND_ACQ_RATES              = { ... } # 18 categories
ENV_MITIGATION_FACTOR       = 0.03
CONTINGENCY_FACTOR          = 0.20
ENGINEERING_FACTOR          = 0.10
OUTPUT_COST_CSV             = "output/cost_estimate.csv"
```

### Algorithm & Outputs (`CostModelResult`)

- Aggregates earthwork, structure, land acquisition (LULC weighted), pavement, and percentage-based contingency/environmental metrics.
- Computes `cost_per_km_usd` and `total_project_cost_usd`.
- **Expressway Support**: Accurately scales geometric widths (36.0m formation) and includes heavy infrastructure allowances (Ground Improvement piling, Grade-Separated Interchanges, Road Furniture) yielding ~$15M-$20M/km realistic estimates for 8-lane expressways.

---

## Phase 10 — Feasibility Report (`report.py`)

### Details

- Jinja2 HTML template with base64 embedded images for zero external dependencies.
- Output includes Executive Summary, Routing logic, Vault profiles, Mass-haul diagrams, and complete cost breakdown tabulars.
- Graceful `weasyprint` fallback: Outputs standalone HTML and warns user if PDF rendering fails due to missing OS GTK+ libraries.

---

## Smoke-Test Baseline (Phase 17.0)

| Metric             | Value                             |
| ------------------ | --------------------------------- |
| Total length       | 603.0 km (Multi-Waypoint, 3 Legs) |
| Min curve radius   | 213.9 m (≥150 m ✓)                |
| Curve violations   | 0                                 |
| Grade violations   | 0                                 |
| Peak memory        | ~4.3 GB                           |
| DEM source         | SRTMGL1, HIGH                     |
| Viz products       | 10                                |
| Unit tests passing | 35/35 (7 per phase 6/7/8/9/10)    |

---

## Key Dependencies

- **Required**: `numpy`, `scipy`, `geopandas`, `rasterio`, `osmnx`, `shapely`, `pyproj`, `matplotlib`, `folium`, `requests`, `python-dotenv`, `scikit-image`
- **Optional**: `scikit-fmm` (smoother paths), `numba` (JIT acceleration)
- **Phase 10 will need**: `jinja2`, `weasyprint` (or `pdfkit`) for HTML → PDF report

---

## Remaining Phases & Next Steps

| Phase | Scope                                  |
| ----- | -------------------------------------- |
| 18    | Streamlit/Gradio interactive dashboard |

### Immediate Next Step: Streamlit Dashboard Optimization

**Goal:** Implement a real-time interactive dashboard to visualize cost surfaces and route metrics dynamically, allowing for rapid A/B testing of different waypoint configurations.

---

## Known Issues

- The standalone script `bridge_siting.py` is fully deprecated and obsolete following the Phase 13 `routing.py` overhaul. It is currently disconnected and serves no function but is kept strictly as an issue log (see `ISSUES.md`) pending future cleanup.
