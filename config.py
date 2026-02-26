"""
Highway Alignment Configuration — Myanmar Preliminary Feasibility
=================================================================
All parameters are tunable. Switch scenario via SCENARIO_PROFILE below.
"""
import os
import math
from dotenv import load_dotenv

load_dotenv()

# ── Endpoints ─────────────────────────────────────────────────────────────────
POINT_A = (94.59079006576709503, 17.17894837308733358)
POINT_B = (96.457154, 17.486777)
OUTPUT_FILE = "preliminary_route.geojson"

# ── Coordinate System ─────────────────────────────────────────────────────────
UTM_EPSG = 32646          # WGS-84 / UTM Zone 46N — covers central Myanmar

# ── DEM Source Preference ─────────────────────────────────────────────────────
# Tried in order until one succeeds. COP30 gives 30 m resolution (best available
# globally free). SRTMGL1 is 30 m fallback. SRTMGL3 is 90 m last-resort.
DEM_PREFERENCE = ["COP30", "SRTMGL1", "SRTMGL3"]
RESOLUTION = 30           # Target output resolution in metres
DEM_NODATA_SENTINEL = -9999.0   # Internal sentinel for NoData cells

# ── Scenario Profile ──────────────────────────────────────────────────────────
# Options: "expressway" | "rural_trunk" | "mountain_road"
SCENARIO_PROFILE = "rural_trunk"

_PROFILES = {
    # Myanmar expressway standard (Yangon–Naypyidaw style)
    "expressway": dict(
        design_speed_kmph=100,
        slope_optimal_pct=2.0,
        slope_moderate_pct=5.0,
        slope_max_pct=6.0,
        slope_cliff_pct=15.0,
        min_curve_radius=500,
        superelevation_max=0.08,
        rubber_band_macro_weight=0.05,
        rubber_band_micro_weight=0.01,
    ),
    # Rural/regional trunk (most Myanmar inter-city corridors)
    "rural_trunk": dict(
        design_speed_kmph=60,
        slope_optimal_pct=4.0,
        slope_moderate_pct=8.0,
        slope_max_pct=12.0,
        slope_cliff_pct=20.0,
        min_curve_radius=150,
        superelevation_max=0.08,
        rubber_band_macro_weight=0.03,
        rubber_band_micro_weight=0.005,
    ),
    # Mountain road (Chin Hills, eastern Shan, Kachin — switchbacks acceptable)
    "mountain_road": dict(
        design_speed_kmph=40,
        slope_optimal_pct=6.0,
        slope_moderate_pct=10.0,
        slope_max_pct=14.0,
        slope_cliff_pct=22.0,
        min_curve_radius=60,
        superelevation_max=0.10,
        rubber_band_macro_weight=0.02,
        rubber_band_micro_weight=0.003,
    ),
}

_p = _PROFILES[SCENARIO_PROFILE]

DESIGN_SPEED_KMPH      = _p["design_speed_kmph"]
SLOPE_OPTIMAL_PCT      = _p["slope_optimal_pct"]
SLOPE_MODERATE_PCT     = _p["slope_moderate_pct"]
SLOPE_MAX_PCT          = _p["slope_max_pct"]
SLOPE_CLIFF_PCT        = _p["slope_cliff_pct"]
MIN_CURVE_RADIUS       = _p["min_curve_radius"]
SUPERELEVATION_MAX     = _p["superelevation_max"]
RUBBER_BAND_MACRO_W    = _p["rubber_band_macro_weight"]
RUBBER_BAND_MICRO_W    = _p["rubber_band_micro_weight"]

# ── Right-of-Way ───────────────────────────────────────────────────────────────
ROW_BUFFER_M = 61          # Setback from centreline to structures (200 ft standard)

# ── Water Crossing Penalties (per river tier) ──────────────────────────────────
# Tier 0: micro-stream < 10 m  → culvert / ford
# Tier 1: minor stream 10–50 m → small bridge
# Tier 2: medium river 50–200 m → medium bridge
# Tier 3: large river 200–500 m → major bridge (Chindwin-scale)
# Tier 4: navigation river >500 m → Ayeyarwady-scale; find narrowest crossing
WATER_PENALTY_TIERS = [5, 50, 500, 5_000, 50_000]

# Bridge constraints
MIN_BRIDGE_SPACING_M = 10_000   # Minimum distance between two major bridge sites

# ── Legacy flat penalty (retained as fallback if hierarchy fails) ───────────────
WATER_PENALTY = 5_000

# ── Grid / Routing ────────────────────────────────────────────────────────────
IMPASSABLE   = 1e9
BORDER_CELLS = 20

# ── Phase 4: Two-Resolution Routing ───────────────────────────────────────────
# Coarse pass downsamples the cost grid by COARSE_FACTOR before the first
# Dijkstra run. Fine routing then only occurs inside a CORRIDOR_BAND_KM wide
# band around the coarse route.  Reduces memory ~COARSE_FACTOR² and runtime.
COARSE_FACTOR     = 10         # 30 m × 10 = 300 m coarse resolution
CORRIDOR_BAND_KM  = 8.0        # Half-width of fine-routing band (km each side)

# FAST_MODE: 300 m resolution only (no fine pass) — for rapid scenario screening
FAST_MODE = False

# Routing engine: 'dijkstra' (always available) or 'fmm' (requires scikit-fmm)
# 'auto' tries FMM first and falls back to Dijkstra silently.
ROUTING_ENGINE = 'auto'

# ── Phase 3: DEM Stream Fallback ──────────────────────────────────────────────
# If OSM water_feature count < OSM_WATER_FALLBACK_TRIGGER, the pipeline
# automatically derives a stream network from the DEM using D8 flow accumulation.
# STREAM_ACCUM_THRESHOLD_KM2: upstream catchment area required to call a
# cell a 'stream'. Smaller = more channels (but also more noise).
OSM_WATER_FALLBACK_TRIGGER  = 3      # fewer than this many OSM water features
STREAM_ACCUM_THRESHOLD_KM2  = 0.5   # 0.5 km² upstream → classified as stream

# Stage-checkpoint: saved after each major pipeline stage so a crashed run
# can be resumed without re-downloading data.
CHECKPOINT_FILE = "run_checkpoint.json"
FORCE_RESTART   = False   # set True to ignore existing checkpoint and start fresh

# ── Phase 5: Performance ───────────────────────────────────────────────────────
# MEMORY_WARN_GB: if the estimated peak memory for the fine routing grid exceeds
# this threshold, COARSE_FACTOR is automatically doubled to reduce memory use.
MEMORY_WARN_GB = 4.0

# TILE_ROUTING_THRESHOLD_KM: corridors longer than this are flagged in the log
# with a tile-routing recommendation. Actual tile routing is future work.
TILE_ROUTING_THRESHOLD_KM = 300.0

# PERF_TIMING_ENABLED: track and report wall-clock time per pipeline stage.
PERF_TIMING_ENABLED = True

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"

# ── External APIs ─────────────────────────────────────────────────────────────
OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
