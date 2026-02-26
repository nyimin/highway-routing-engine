# Highway Alignment GIS Pipeline

This project is an automated Python pipeline for finding the optimal (least-cost) path for a highway. It fetches elevation data (DEM) and infrastructure/water data (via OpenStreetMap APIs) to dynamically generate a cost surface and uses pathfinding algorithms to trace a realistic highway alignment.

## Core Features

- **Automated Data Fetching**: Retrieves standard Digital Elevation Models (DEM) via the OpenTopography API (COP30 → SRTM fallback chain) and vector features (buildings, water bodies) via OSMnx.
- **Dynamic Cost Surface Generation**: Calculates slope from elevation data, applying heavy penalties to steep inclines while also buffering out buildings and defining "steep bank" abutment zones for water crossings. Includes anisotropic sidehill costs.
- **DEM Stream Fallback**: Automatically derives a D8 flow accumulation stream network purely in numpy/scipy when OSM water data is sparse.
- **Two-Pass "Macro-to-Micro" Routing**:
  - **Coarse Pass**: A fast low-res pathfinding pass downsampled by \`COARSE_FACTOR\` to discover the general corridor.
  - **Band Masking**: Carves an 8km safety corridor around the coarse route, stripping out 80%+ of the grid.
  - **Fine Passes**: Two-pass micro-routing within the band. Applies a rubber band penalty to discover the most direct route, identifying optimal bridge control points and preventing meandering.
  - **Routing Engine**: Transparently uses Scikit-FMM if available (for continuous terrain), otherwise Dijkstra.
- **Geometry Checks**: Applies B-spline smoothing to the jagged grid-based Dijkstra path, computing clothoid transitions and strictly verifying minimum curve radii and maximum sustained grades.
- **Stage Checkpointing**: Fault-tolerant execution. Results of each stage are cached in a local checkpoint file allowing pipeline restarts from the last completed stage.
- **Performance & Memory Auto-tuning**: Uses \`numba\` JIT for critical loops. Estimates potential peak memory footprint before routing and auto-escalates \`COARSE_FACTOR\` to heavily reduce memory. Output includes detailed stage timing and peak memory metrics.
- **GeoJSON Export**: Readily exports the final smoothed path into a standard GeoJSON format containing rich metadata for integration into external GIS software (QGIS, ArcGIS).

## Project Structure

The monolithic pipeline has been extracted into a modular package architecture:

- `main.py`: The entry point and central orchestrator script coordinating all other modules, featuring a fault-tolerant CheckpointManager and performance tracking.
- `config.py`: Stores all configuration variables, route coordinates, engineering constants (max slope, minimum radius, setback), and cost penalties.
- `data_fetch.py`: Handles downloading API responses for raster (DEM) and vector (OSM). Derives D8 stream flow accumulations for hydrology fallback.
- `cost_surface.py`: Generates the friction/cost array. Computes slopes, creates exclusion zones around infrastructure, enforces water-crossing rules, and calculates sidehill aspect costs.
- `routing.py`: Contains coarse-to-fine band masking, bridge siting, and two-pass routing using Scikit-FMM or Dijkstra.
- `geometry_utils.py`: Converts coordinates, manages B-spline smoothing, verifies curve radii, calculates clothoid transitions, and exports to GeoJSON.
- `jit_utils.py`: A `try_jit` decorator wrapper that accelerates key bottlenecks with `numba` if it is installed, operating as a zero-overhead pass-through if not.
- `visualize_route.py`: Utility script to generate top-down map images (`route_overview.png`, `route_start.png`, `route_end.png`) out of the generated data.
- `output/`: Directory where visualization PNG files will be saved.
- `data/`: Directory where downloaded DEMs and Geopackages are cached.

## Requirements

You must have an OpenTopography API key to fetch DEM data correctly. Keep this in a `.env` file in the root project directory:

```env
OPENTOPOGRAPHY_API_KEY=your_api_key_here
```

Requires Python 3.9+. Install the dependencies using:

```bash
pip install -r requirements_alignment.txt
pip install matplotlib  # for visualization
```

_Note: You can greatly accelerate the pipeline by installing optional dependencies like `numba` and `scikit-fmm`._

## Usage

1. **Verify coordinates and settings**: Open `config.py` to inspect the origin point (`START_LON`, `START_LAT`), destination point (`END_LON`, `END_LAT`), and various pathfinding configuration variables.
2. **Execute the pipeline**:
   ```bash
   python main.py
   ```
   This will download data (if not cached), generate the cost surface, trace the path, and save it to `preliminary_route.geojson`.
3. **Generate Visualizations**:
   ```bash
   python visualize_route.py
   ```
   This will render plots of the route and save them as PNGs into the `output/` folder.
