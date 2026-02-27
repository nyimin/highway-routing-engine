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
        # Phase 12: 3D-CHA* inspired parameters
        earthwork_proxy_weight=0.25,      # lower for flat expressway corridors
        min_tangent_length_m=300.0,       # AASHTO at 100 km/h
        min_curve_length_m=200.0,         # 3× superelevation run-off at 100 km/h
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
        # Phase 12: 3D-CHA* inspired parameters
        earthwork_proxy_weight=0.30,      # moderate — balance earthwork vs route length
        min_tangent_length_m=200.0,       # Myanmar DRD rural trunk standard
        min_curve_length_m=100.0,         # 3× superelevation run-off at 60 km/h
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
        # Phase 12: 3D-CHA* inspired parameters
        earthwork_proxy_weight=0.40,      # higher — earthwork dominates mountain cost
        min_tangent_length_m=100.0,       # shorter tangents OK at low speed
        min_curve_length_m=60.0,          # matches min_curve_radius arc at 40 km/h
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
EARTHWORK_PROXY_WEIGHT = _p["earthwork_proxy_weight"]
MIN_TANGENT_LENGTH_M   = _p["min_tangent_length_m"]
MIN_CURVE_LENGTH_M     = _p["min_curve_length_m"]

# ── Right-of-Way & Expropriation ──────────────────────────────────────────────
# Phase 5.2: Industrial Standard Building Buffering
# Instead of a flat penalty, expropriation cost scales with building footprint area.
BUILDING_BASE_PENALTY = 2000      # Base penalty for any structure
BUILDING_AREA_MULT = 20           # Additional penalty per square metre of footprint
BUILDING_MAX_PENALTY = 50000      # Cap to prevent numeric overflow on massive complexes
ROW_BUFFER_M = 30                 # Maximum extent of the concentric penalty buffer


# ── Phase 5.4: Class-Based Road Discount ──────────────────────────────────────
#
# Myanmar OSM data-quality note (2026-02):
#   Trunk/primary roads on major corridors (Yangon–Naypyidaw, Mandalay–Tamu)
#   are reasonably mapped. Secondary, tertiary, and track coverage is patchy
#   to absent outside urban centres.  Positional accuracy of OSM centrelines
#   can drift 20–50 m, especially for tracks derived from GPS field surveys.
#
#   Design principle: presence in OSM is reliable (contributors are careful);
#   *absence* means nothing — do not infer a road-free gap from an empty OSM
#   query.  Discounts on cells WHERE a road IS mapped are still valid.
#
# Cost multiplier per OSM highway= tag value (< 1.0 = cheaper than greenfield).
# Lower multiplier → stronger attractor for the router.
ROAD_CLASS_DISCOUNTS = {
    "motorway":     0.15,  # Grade-separated expressway; existing full-width formation
    "trunk":        0.20,  # Major paved national road — minimal upgrade cost
    "primary":      0.25,  # Paved, maintained — strong reuse attractor
    "secondary":    0.40,  # Partial pavement / gravel — moderate reconstruction
    "tertiary":     0.50,  # Gravel / earthwork — significant upgrade needed
    "unclassified": 0.60,  # Unpaved earthwork formation — full upgrade required
    "track":        0.65,  # Logging / farm track — usefulness as alignment is limited
    # "path" intentionally omitted: OSM 'path' = footpath/hiking trail only;
    # no road formation, no ROW relevance, mostly noise in Myanmar rural areas.
    "default":      0.70,  # Unknown highway= value — conservative assumption
}

# Buffer zone around existing road corridors.
# Even when the router leaves the road centreline, land within this strip has
# already been disturbed / is accessible for construction equipment.
# Kept intentionally conservative (0.88×, not 0.80×) because OSM centreline
# positions in Myanmar can drift 20–50 m — aggressively discounting adjacent
# cells risks rewarding GPS error rather than real corridor value.
ROW_CORRIDOR_BUFFER_M        = 50    # Half-width of the ROW acquisition strip (m)
ROW_CORRIDOR_BUFFER_DISCOUNT = 0.88  # 12 % cost reduction for cells inside this strip

# ── Phase 5.4: Expanded LULC Penalty Table ────────────────────────────────────
#
# Myanmar OSM data-quality note:
#   LULC polygon coverage in Myanmar is severely incomplete.  OSM records
#   the features that volunteers have mapped; vast areas of real forest,
#   wetland, and protected land simply have NO OSM polygon and therefore
#   receive NO penalty from this table.
#
#   Consequence: the cost surface will systematically *under-penalise* remote
#   or unmapped terrain.  Two mitigations are applied:
#     1. LULC_UNMAPPED_BASE — a global background multiplier (>1.0) applied
#        to ALL cells that fall outside any LULC polygon.  Reflects that
#        unmapped terrain in Myanmar is far more likely to be secondary forest
#        or scrubland than clear open farmland.
#     2. OSM_LULC_WARN_THRESHOLD — if the Overpass query returns fewer than
#        this many polygons, the pipeline emits a prominent WARNING in the log
#        and in the output metadata, so downstream users know the LULC layer
#        is likely underrepresented.
#
#   The penalty VALUES themselves (when a polygon IS present) remain high;
#   lowering them would make the model less conservative on the features that
#   ARE captured — which is the wrong direction.

LULC_PENALTIES = {
    # ── Legal / ESG barriers ──────────────────────────────────────────────────
    # These carry permitting and EIA costs that can dwarf construction costs.
    "national_park":  8.0,  # Full EIA + legal consent + carbon offsets required
    "protected_area": 8.0,  # boundary=protected_area — equivalent legal barrier
    "nature_reserve": 7.0,  # leisure=nature_reserve — often enforced in Myanmar
    "conservation":   6.5,  # landuse=conservation (legacy OSM tag, retained)

    # ── High construction / piling cost ───────────────────────────────────────
    "mangrove":  6.0,  # Extreme piling depth, tidal access, international ESG scrutiny
    "wetland":   5.0,  # Seasonal saturation — drainage + subgrade stabilisation
    "swamp":     5.0,  # Similar hydrology to wetland
    "mud":       4.5,  # Tidal/alluvial mud — deep soft subgrade, high surcharge cost
    "water":     5.0,  # Overlaps river tier system; used as LULC baseline floor
    "reef":      6.0,  # Coastal construction — rare but extreme cost

    # ── Forest clearing ───────────────────────────────────────────────────────
    "forest":  4.0,  # Primary / tropical forest — clearing + carbon credit offset
    "wood":    2.5,  # Secondary forest / scrub woodland — significant clearing cost
    "scrub":   1.5,  # Shrubland / regrowth — moderate vegetation removal

    # ── Agricultural compensation ─────────────────────────────────────────────
    # Myanmar land compensation rates are regulated but enforcement varies.
    "orchard":   2.2,  # Mature tree crops (durian, mango) — high compensation value
    "vineyard":  2.0,  # Treated equivalent to orchard
    "rubber":    2.0,  # Rubber plantation — established industry, organised landowners
    "farmland":  1.8,  # Mixed / arable farmland — basic acquisition cost
    "meadow":    1.4,  # Low-value grassland / fallow
    "grass":     1.2,  # Open grass — minimal vegetation, lower acquisition
    "bare_rock": 1.3,  # Hard rock outcrop — blasting + disposal cost
}

# Background multiplier applied to cells with NO LULC polygon (unmapped terrain).
# In Myanmar, unmapped cells are far more likely to be secondary forest or scrub
# than cleared open land.  1.15 ≈ intermediate between bare farmland (1.0) and
# scrubland (1.5); conservative enough to nudge routing toward mapped corridors
# without creating artificial barriers.
LULC_UNMAPPED_BASE = 1.15

# Pipeline emits a WARNING if the OSM LULC query returns fewer than this many
# polygons, flagging likely data incompleteness for downstream reviewers.
OSM_LULC_WARN_THRESHOLD = 10

# ── Slope × LULC interaction ──────────────────────────────────────────────────
# Steep terrain amplifies LULC construction cost because:
#   • Haul-road access to the clearing face is itself expensive;
#   • Drainage structures multiply with slope;
#   • Slope stability risk rises with vegetation removal.
# Formula: lulc_mult_effective = lulc_mult * (1 + SLOPE_LULC_INTERACT * t)
# where t = clamp(slope_pct / SLOPE_MODERATE_PCT, 0, 1).
# At t=1 (moderate slope threshold reached), penalty is 50 % amplified.
SLOPE_LULC_INTERACT = 0.5

# ── LULC boundary soft transition (EDT decay) ─────────────────────────────────
# Hard-edged LULC polygons create 1-cell cost discontinuities that can cause
# the router to hug polygon boundaries (cheapest cell just outside the penalty
# zone).  An EDT-based ramp over LULC_EDGE_DECAY_M blends the penalty to 1.0
# smoothly, eliminating boundary-hugging artefacts.
LULC_EDGE_DECAY_M = 150  # Distance over which penalty decays to 1.0 outside polygon

# ── Water Crossing Penalties (per river tier) ──────────────────────────────────
# Tier 0: micro-stream < 10 m  → culvert / ford
# Tier 1: minor stream 10–50 m → small bridge
# Tier 2: medium river 50–200 m → medium bridge
# Tier 3: large river 200–500 m → major bridge (Chindwin-scale)
# Tier 4: navigation river >500 m → Ayeyarwady-scale; find narrowest crossing
WATER_PENALTY_TIERS = [2, 5, 10, 20, 30]

# Bridge constraints
MIN_BRIDGE_SPACING_M = 10_000   # Minimum distance between two major bridge sites

# ── Legacy flat penalty (retained as fallback if hierarchy fails) ───────────────
WATER_PENALTY = 5_000

# ── Grid / Routing ────────────────────────────────────────────────────────────
IMPASSABLE   = 1e9
BORDER_CELLS = 20

# ── Phase 4: Multi-Resolution Routing Pyramid (Tang & Dou 2023) ───────────────
# Progressively downsamples the cost surface to create a multi-scale pyramid.
# Routing occurs segment-by-segment between waypoints identified at coarser levels.
PYRAMID_LEVELS            = 3          # Number of downsampling levels (e.g., 30m -> 60m -> 120m -> 240m)
DOWNSAMPLE_RATIO          = 2          # Scale factor per level
DOWNSAMPLE_METHOD         = "average"  # "average" or "maximum" (average recommended for mixed terrain)
WAYPOINT_ANGLE_THRESH_DEG = 45.0       # Threshold for extracting directional waypoints
PARALLEL_WAYPOINT_THRESH  = 4          # Min waypoints to spin up ProcessPoolExecutor

# Legacy parameters retained for fallback compatibility
COARSE_FACTOR     = 10         # 30 m × 10 = 300 m coarse resolution
CORRIDOR_BAND_KM  = 8.0        # Half-width of fine-routing band (km each side)

# FAST_MODE: Coarse resolution only — for rapid scenario screening
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

# ── Phase 5.3: Routing Polish ────────────────────────────────────────────────
# TURNING_ANGLE_FILTER_DEG: post-routing filter removes waypoints causing turns
# sharper than this (measured as interior angle). 160° = near-reversal stutter.
TURNING_ANGLE_FILTER_DEG = 160.0

# ── Phase 5.3: Diagnostics & Visualization ───────────────────────────────────
# EXPORT_INTERMEDIATES: save cost surface, building penalty, and per-layer arrays
# as GeoTIFFs in output/ for external GIS inspection (adds ~50–100 MB per run).
EXPORT_INTERMEDIATES = True

# GENERATE_VISUALIZATIONS: auto-run the visualization suite after routing completes.
GENERATE_VISUALIZATIONS = True

# ── Phase 6: Vertical Alignment ───────────────────────────────────────────────
#
# K-value table: minimum parabolic curve parameter (m/%) by design speed.
# K_min = L_min / |A|  where A = algebraic grade change in %.
# Source: AASHTO Green Book 7th ed. Table 3-34 / 3-35.
#         Myanmar DRD Road Design Standard adopts AASHTO values.
#
# Tuple layout: (K_crest_SSD, K_sag_SSD)
#   K_crest: controls stopping sight distance over a crest.
#   K_sag:   controls headlight sight distance in a sag.
#
# K_crest_PSD (passing sight distance, ~185 at 60 km/h) is implemented inside
# vertical_alignment.py as an advisory-only check — NOT enforced as a hard
# constraint at the preliminary design stage.
VC_K_VALUES: dict[int, tuple[int, int]] = {
    #  speed  K_crest  K_sag
    40:  (4,    5),
    60:  (11,  11),
    80:  (26,  21),
    100: (52,  37),
    120: (98,  60),
}

# Maximum allowable sustained design grade per scenario (%).
# Distinct from SLOPE_MAX_PCT which governs the 2-D routing cost surface;
# this applies to the 3-D finished grade line.
GRADE_MAX_PCT: dict[str, float] = {
    "expressway":    6.0,   # Myanmar expressway standard
    "rural_trunk":   8.0,   # Heavy truck design criterion
    "mountain_road": 10.0,  # Switchback sections permitted
}

# Absolute minimum vertical curve length regardless of K·|A|.
# AASHTO §3.4.1: 30 m is the absolute floor for any parabolic VC.
MIN_VC_LENGTH_M: float = 30.0

# Minimum distance between consecutive VPI candidates.
# Reduces the number of VCs on long flat sections.  200 m is a reasonable
# default for 60 km/h design (one VPI every ~3 seconds of travel at design speed).
MIN_VPI_SPACING_M: float = 200.0

# Output flags for Phase 6
EXPORT_3D_GEOJSON: bool = True
OUTPUT_FILE_3D: str = "preliminary_route_3d.geojson"

# ── Phase 7: Earthwork Volumes ────────────────────────────────────────────────
#
# FORMATION_WIDTH_M: total subgrade width in metres = carriageway + both shoulders.
# Myanmar DRD typical sections:
#   rural_trunk  (2-lane): 7.0 m carriageway + 2×2.0 m shoulders = 11.0 m
#   expressway   (4-lane): 2×(3.65 m×2 lanes) + 2×3.0 m shoulders = 20.6 m → use 24 m
#   mountain_road (1.5-lane): 6.0 m carriageway + 2×1.0 m shoulders = 8.0 m
FORMATION_WIDTH_M: dict[str, float] = {
    "expressway":    24.0,
    "rural_trunk":   11.0,
    "mountain_road":  8.0,
}

# Cut batter slopes (H:V) by material class.
# Applied uniformly in preliminary design — refine in detailed design with
# geotechnical investigation.
#   0.5  → hard / competent rock (stable near-vertical)
#   1.0  → soft rock / stiff clay / laterite (DEFAULT)
#   1.5  → residual soil / loose colluvium
CUT_BATTER_HV: float = 1.0

# Fill (embankment) batter slope (H:V).
# Myanmar DRD standard 1.5H:1V for clay/laterite fill.
# Use 2.0 for sandy or silty fills.
FILL_BATTER_HV: float = 1.5

# Material swell factor (bank m³ → loose m³).
# Determines how much cut volume 'expands' when excavated.
#   1.10 → dense clay
#   1.25 → mixed soil / decomposed granite (DEFAULT)
#   1.40 → fresh granite / basalt
SWELL_FACTOR: float = 1.25

# Output paths for Phase 7
OUTPUT_EARTHWORK_CSV: str = "output/earthwork_volumes.csv"

# ── Phase 8: Bridge and Culvert Inventory ─────────────────────────────────────
#
# BRIDGE_FREEBOARD_M: minimum clearance (m) between the design high-water level
# and the underside of the bridge deck.  Myanmar DRD specifies 1.5 m for rural
# trunk roads; ADB recommends 2.0 m for major rivers.
BRIDGE_FREEBOARD_M: float = 1.5

# BRIDGE_COST_PER_M2_USD: composite unit rate for bridge superstructure +
# substructure (USD per m² of deck area).
# Source: World Bank ROCKS Myanmar roads sector (2020–2024): USD 3,000–4,500/m².
# Mid-range 3,500 used as preliminary estimate.
BRIDGE_COST_PER_M2_USD: float = 3_500.0

# BRIDGE_DECK_WIDTH_M: assumed total deck width = carriageway + guardwalls.
# rural_trunk (2-lane): 11.0 m carriageway + 0.5 m per side = 12.0 m
BRIDGE_DECK_WIDTH_M: float = 12.0

# CULVERT_UNIT_COST_USD: lump-sum cost per culvert structure (box or pipe).
# Myanmar DRD standard box culvert (1.2 m × 1.2 m × 12 m): ~USD 12,000–18,000.
CULVERT_UNIT_COST_USD: float = 15_000.0

# MIN_CULVERT_ACCUM_CELLS: minimum D8 flow-accumulation cell count to trigger
# a culvert. At RESOLUTION=30 m, each cell ~ 900 m² catchment.
# 200 cells → ~0.18 km² catchment → warrants a culvert.
MIN_CULVERT_ACCUM_CELLS: int = 200

# ── Phase 8b: Bridge Detection Filtering (Myanmar low-quality OSM context) ────
#
# Myanmar OSM water data contains many features that do NOT warrant a bridge:
#   • natural=wetland — seasonally saturated land, not a waterway
#   • LineString waterways (stream/river centrelines) — no real water surface
#   • Dams, weirs, fish farming ponds, docks — not crossable water
#   • Tiny unmapped ponds — below engineering significance
#
# Only polygon features representing real river/canal water surfaces trigger
# bridge detection.  A crossing-angle check prevents false positives from
# the route running parallel to a river inside its polygon.

# Waterway tags that represent real crossable river water surfaces (polygon)
BRIDGE_WORTHY_WATERWAY_TAGS: set = {"river", "riverbank", "canal"}

# Waterway tags that are structures/facilities, NOT crossable water
BRIDGE_EXCLUDE_WATERWAY_TAGS: set = {
    "dam", "weir", "fish_farming_pond", "dock", "ditch", "drain",
}

# natural= tags that are NOT river water (wetlands are saturated land)
BRIDGE_EXCLUDE_NATURAL_TAGS: set = {"wetland"}

# Minimum water polygon area (m²) to warrant a bridge.
# At 30 m resolution, a single cell ≈ 900 m²; 500 m² catches sub-cell ponds.
BRIDGE_MIN_WATER_AREA_M2: float = 500.0

# Minimum crossing angle (degrees) between route and water body longest axis.
# Prevents false bridges from route running parallel inside a river polygon.
# 15° = very acute; below this means "running alongside", not "crossing".
BRIDGE_MIN_CROSSING_ANGLE_DEG: float = 15.0

# Output paths for Phase 8
OUTPUT_STRUCTURES_CSV: str = "output/structures.csv"

# ── Phase 11: Enhanced Data Sources ───────────────────────────────────────────
#
# ESA WorldCover 10m — satellite-derived wall-to-wall land cover.
# Replaces the incomplete OSM landuse/natural polygon layer for LULC penalties.
# Accessed via Microsoft Planetary Computer STAC API (100% free, no API key).
#
# Class value → LULC cost multiplier mapping.
# Values chosen to match existing LULC_PENALTIES semantics.
WORLDCOVER_PENALTIES: dict[int, float] = {
    10:  4.0,   # Tree cover → forest clearing + carbon offset
    20:  1.5,   # Shrubland → scrub removal
    30:  1.2,   # Grassland → minimal impact
    40:  1.8,   # Cropland → agricultural compensation
    50:  3.0,   # Built-up → urban penalty (supplements building layer)
    60:  1.3,   # Bare / sparse vegetation → hard surface
    70:  1.0,   # Snow and ice → N/A for Myanmar
    80:  5.0,   # Permanent water bodies → water crossing penalty
    90:  5.0,   # Herbaceous wetland → drainage + subgrade stabilisation
    95:  6.0,   # Mangroves → extreme piling + ESG scrutiny
    100: 1.5,   # Moss and lichen → similar to shrubland
}

# ── Data source preference flags ─────────────────────────────────────────────
# True = use the enhanced (satellite/ML) source; False = use OSM-only (legacy).
# If the enhanced source fetch fails, OSM is used as automatic fallback.
USE_WORLDCOVER_LULC:    bool = True   # ESA WorldCover 10m for LULC
USE_OVERTURE_BUILDINGS: bool = True   # Overture Maps ML building footprints

# Microsoft Planetary Computer STAC API endpoint (ESA WorldCover)
PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Overture Maps building deduplication radius (metres).
# Overture buildings within this distance of an OSM building are dropped to avoid
# double-counting.  15 m accounts for GPS drift + footprint alignment differences.
OVERTURE_DEDUP_RADIUS_M: float = 15.0

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"

# ── External APIs ─────────────────────────────────────────────────────────────
OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

# ── Phase 9: Parametric Cost Model ────────────────────────────────────────────
#
# Unit rates from Myanmar DRD schedule-of-rates and World Bank ROCKS 2020–2024.
# All values are in USD; preliminary estimates with ±25% accuracy for ADB/WB
# pre-feasibility studies.

# Earthwork unit rates
EARTHWORK_CUT_RATE_USD_M3: float  = 8.0
EARTHWORK_FILL_RATE_USD_M3: float = 15.0

# Pavement unit rate (flexible pavement, 50 mm AC on 200 mm granular base)
# Myanmar DRD rural_trunk spec. USD 100–140/m²; default 120.
PAVEMENT_RATE_USD_M2: float = 120.0

# Land acquisition corridor width (m).
# rural_trunk: 30 m formation + 2×15 m buffer = 60 m total strip.
# ha acquired = length_km × 1000 × CORRIDOR_WIDTH_M / 10_000
CORRIDOR_WIDTH_M: float = 60.0

# Fallback land acquisition rate when LULC data is unavailable (USD/ha).
LAND_ACQ_DEFAULT_USD_PER_HA: float = 7_500.0

# Land acquisition rates by LULC category (USD/ha).
# Sources: Myanmar Ministry of Agriculture 2022, DRD RAP guidelines,
# ADB Resettlement Framework for Myanmar Roads Projects 2018.
LAND_ACQ_RATES: dict = {
    "national_park":  2_000.0, "protected_area": 2_000.0,
    "nature_reserve": 2_500.0, "conservation":   2_500.0,
    "mangrove":  3_000.0, "wetland":  3_000.0,
    "swamp":     3_000.0, "mud":      2_000.0, "water": 1_000.0,
    "forest":    5_000.0, "wood":     6_000.0, "scrub": 4_000.0,
    "orchard":  12_000.0, "vineyard": 12_000.0, "rubber": 10_000.0,
    "farmland":  8_000.0, "meadow":   6_000.0, "grass":  5_000.0,
    "bare_rock": 3_000.0,
    "urban": 15_000.0, "residential": 15_000.0,
    "commercial": 15_000.0, "industrial": 12_000.0,
}

# Percentage cost factors applied to civil subtotal
# Source: ADB standard contingency guidelines for preliminary design.
ENV_MITIGATION_FACTOR: float = 0.03   # 3%  environmental mitigation
CONTINGENCY_FACTOR:    float = 0.20   # 20% preliminary design contingency
ENGINEERING_FACTOR:    float = 0.10   # 10% engineering, supervision, admin

# Output paths for Phases 9–10
OUTPUT_COST_CSV:    str = "output/cost_estimate.csv"
OUTPUT_REPORT_HTML: str = "output/feasibility_report.html"
OUTPUT_REPORT_PDF:  str = "output/feasibility_report.pdf"
