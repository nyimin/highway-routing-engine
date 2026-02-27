"""
visualize_route.py — 12-product diagnostic visualization suite
================================================================
Phase 5.3 — complete overhaul of the original 3-PNG overview script.
Phase 6   — added Viz Product #11: Vertical alignment design profile.
Phase 7   — added Viz Product #12: Earthwork volumes and mass-haul diagram.

Products generated (all saved to output/):
  1. cost_heatmap.png          — log-scaled cost surface + route overlay
  2. building_decay.png        — zoomed building penalty EDT decay rings
  3. layer_decomposition.png   — 2×3 panel of individual cost layers
  4. elevation_profile.png     — longitudinal profile + grade violations
  5. slope_histogram.png       — slope distribution along the route
  6. cost_along_route.png      — per-waypoint cost + dominant layer bands
  7. route_3d.png              — 3D terrain drape with route ribbon
  8. cross_sections.png        — perpendicular terrain slices at key points
  9. route_dashboard.png       — single-image statistics summary card
 10. route_map.html            — interactive Folium/Leaflet map
 11. vertical_profile.png      — design FGL vs ground line + grade panel  [Phase 6]
 12. earthwork_masshual.png    — cut/fill depths + cumulative volumes + Brückner  [Phase 7]

Usage from main.py:
    from visualize_route import generate_all_visuals
    generate_all_visuals(viz_data)   # viz_data is a dict prepared by main.py
"""
import os
import math
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")       # non-interactive backend — safe on headless / CPU-only
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — needed for 3D projection

log = logging.getLogger("highway_alignment")
OUT = "output"


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_output():
    os.makedirs(OUT, exist_ok=True)


def _route_xy(route_utm):
    """Return (xs, ys) numpy arrays from a list of (x, y) tuples."""
    arr = np.array(route_utm)
    return arr[:, 0], arr[:, 1]


def _cumulative_distance(xs, ys):
    """Cumulative 2-D distance along the route (metres)."""
    dx = np.diff(xs)
    dy = np.diff(ys)
    segs = np.sqrt(dx ** 2 + dy ** 2)
    return np.concatenate(([0.0], np.cumsum(segs)))


def _sample_raster_along_route(raster, route_rc):
    """Sample raster values at each (row, col) in route_rc."""
    rows, cols = raster.shape
    vals = []
    for r, c in route_rc:
        r = max(0, min(rows - 1, r))
        c = max(0, min(cols - 1, c))
        vals.append(float(raster[r, c]))
    return np.array(vals)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Cost Surface Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _plot_cost_heatmap(cost, route_rc, transform, impassable=1e9):
    log.info("Viz [1/10]: Cost surface heatmap …")
    fig, ax = plt.subplots(figsize=(14, 12))
    display = cost.copy().astype(np.float64)
    display[display >= impassable] = np.nan

    vmin = max(np.nanmin(display), 0.1)
    vmax = np.nanpercentile(display, 99.5)
    im = ax.imshow(display, cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax),
                   aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Cost (log scale)", shrink=0.8)

    # Route overlay
    path = np.array(route_rc)
    ax.plot(path[:, 1], path[:, 0], color="#ff3333", linewidth=1.2, alpha=0.9, label="Route")
    ax.plot(path[0, 1], path[0, 0], "go", markersize=8, zorder=5, label="Start")
    ax.plot(path[-1, 1], path[-1, 0], "bs", markersize=8, zorder=5, label="End")

    ax.set_title("Unified Cost Surface (log-scaled)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlabel("Column (px)")
    ax.set_ylabel("Row (px)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cost_heatmap.png"), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Building Penalty Decay Map
# ══════════════════════════════════════════════════════════════════════════════

def _plot_building_decay(building_penalty, route_rc, transform, resolution_m):
    log.info("Viz [2/10]: Building penalty decay map …")
    # Find the densest region (peak penalty area)
    if building_penalty.max() < 1:
        log.info("Viz [2/10]: No building penalties — skipping.")
        return

    # Find peak and center the zoom there
    peak_idx = np.unravel_index(building_penalty.argmax(), building_penalty.shape)
    half_win = int(2000 / resolution_m)  # 2 km window radius
    r0 = max(0, peak_idx[0] - half_win)
    r1 = min(building_penalty.shape[0], peak_idx[0] + half_win)
    c0 = max(0, peak_idx[1] - half_win)
    c1 = min(building_penalty.shape[1], peak_idx[1] + half_win)

    crop = building_penalty[r0:r1, c0:c1]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(crop, cmap="YlOrRd", interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, label="Building Penalty", shrink=0.8)

    # Overlay route segments within the crop window
    path = np.array(route_rc)
    in_box = (path[:, 0] >= r0) & (path[:, 0] < r1) & (path[:, 1] >= c0) & (path[:, 1] < c1)
    if in_box.any():
        ax.plot(path[in_box, 1] - c0, path[in_box, 0] - r0,
                color="cyan", linewidth=2, alpha=0.9, label="Route")
        ax.legend()

    ax.set_title("Building Penalty — Distance Decay (Zoomed)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Column offset (px)")
    ax.set_ylabel("Row offset (px)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "building_decay.png"), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Layer Decomposition Panel (2×3)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_layer_decomposition(layers, route_rc, impassable=1e9):
    """layers: dict with keys like 'slope', 'lulc', 'water', 'building', 'road', 'unified'."""
    log.info("Viz [3/10]: Layer decomposition panel …")
    titles = ["Slope Cost", "LULC Multiplier", "Water/River Tiers",
              "Building Penalty", "Road Discount", "Unified Cost"]
    keys = ["slope", "lulc", "water", "building", "road", "unified"]
    cmaps = ["inferno", "YlGn", "Blues", "YlOrRd", "Greens_r", "viridis"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    path = np.array(route_rc)

    for idx, (ax, key, title, cmap) in enumerate(zip(axes.flat, keys, titles, cmaps)):
        data = layers.get(key)
        if data is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=16, transform=ax.transAxes)
            ax.set_title(title)
            continue

        display = data.copy().astype(np.float64)
        display[display >= impassable] = np.nan
        vmin = np.nanmin(display) if np.nanmin(display) > 0 else 0.01
        vmax = np.nanpercentile(display, 99)
        if vmax <= vmin:
            vmax = vmin + 1

        if key in ("unified", "slope", "water"):
            norm = LogNorm(vmin=max(vmin, 0.01), vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(display, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.plot(path[:, 1], path[:, 0], "r-", linewidth=0.8, alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Cost Layer Decomposition", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "layer_decomposition.png"), dpi=180)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Elevation Profile
# ══════════════════════════════════════════════════════════════════════════════

def _plot_elevation_profile(distances_m, elevations_m, grade_violations, max_grade=0.08):
    log.info("Viz [4/10]: Elevation profile …")
    dist_km = distances_m / 1000.0

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(dist_km, elevations_m, alpha=0.3, color="#5c7cfa", label="Terrain")
    ax.plot(dist_km, elevations_m, color="#364fc7", linewidth=1.0)

    # Highlight grade violations
    for start_m, end_m, avg_grade in grade_violations:
        ax.axvspan(start_m / 1000, end_m / 1000, color="red", alpha=0.2)

    # Design grade reference
    if len(distances_m) > 1:
        # Show max allowable elevation rise line from start
        rise_per_km = max_grade * 1000
        ax.axhline(y=0, color="gray", alpha=0)  # dummy
        ax.text(0.98, 0.95, f"Max design grade: {max_grade * 100:.0f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color="red", alpha=0.7)

    ax.set_xlabel("Distance (km)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.set_title("Longitudinal Elevation Profile", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if grade_violations:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="red", alpha=0.2,
                                 label=f"Grade violations ({len(grade_violations)})")]
        ax.legend(handles=legend_elements, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "elevation_profile.png"), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Slope Histogram
# ══════════════════════════════════════════════════════════════════════════════

def _plot_slope_histogram(slope_along_route, thresholds):
    """thresholds: dict with s_opt, s_mod, s_max, s_cliff."""
    log.info("Viz [5/10]: Slope histogram …")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(slope_along_route, bins=80, color="#5c7cfa", edgecolor="white",
            alpha=0.85, density=True)

    colors = ["#2b8a3e", "#e8c700", "#e8590c", "#c92a2a"]
    labels = ["Optimal", "Moderate", "Max", "Cliff"]
    for (key, color, label) in zip(["s_opt", "s_mod", "s_max", "s_cliff"],
                                    colors, labels):
        val = thresholds.get(key, 0)
        if val > 0:
            ax.axvline(val, color=color, linewidth=2, linestyle="--", label=f"{label} ({val}%)")

    ax.set_xlabel("Slope (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Slope Distribution Along Route", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slope_histogram.png"), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Cost-Along-Route Profile
# ══════════════════════════════════════════════════════════════════════════════

def _plot_cost_along_route(distances_m, cost_along, layers_along):
    """
    cost_along: 1D array of unified cost per waypoint.
    layers_along: dict of key -> 1D array (same length), e.g. slope, building, water, lulc.
    """
    log.info("Viz [6/10]: Cost-along-route profile …")
    dist_km = distances_m / 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Top panel: log-scaled unified cost
    ax1.semilogy(dist_km, np.maximum(cost_along, 0.1), color="#364fc7", linewidth=0.8)
    ax1.fill_between(dist_km, 0.1, np.maximum(cost_along, 0.1), alpha=0.2, color="#5c7cfa")
    ax1.set_ylabel("Unified Cost (log)", fontsize=11)
    ax1.set_title("Cost Along Route — Layer Contributions", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: dominant layer indicator
    layer_names = list(layers_along.keys())
    if layer_names:
        # Stack layer values and find the dominant one per waypoint
        stacked = np.array([layers_along[k] for k in layer_names])
        dominant_idx = np.argmax(stacked, axis=0)

        cmap_layers = plt.colormaps.get_cmap("Set2").resampled(len(layer_names))
        for i, name in enumerate(layer_names):
            mask = dominant_idx == i
            if mask.any():
                ax2.fill_between(dist_km, 0, 1, where=mask,
                                 color=cmap_layers(i), alpha=0.7, label=name)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_ylabel("Dominant\nLayer", fontsize=9)
        ax2.legend(loc="upper right", ncol=len(layer_names), fontsize=8)

    ax2.set_xlabel("Distance (km)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cost_along_route.png"), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. 3D Terrain Drape
# ══════════════════════════════════════════════════════════════════════════════

def _plot_3d_terrain(dem, route_rc, resolution_m, downsample=8):
    log.info("Viz [7/10]: 3D terrain drape …")
    # Downsample DEM for memory safety
    dem_ds = dem[::downsample, ::downsample].copy()
    rows_ds, cols_ds = dem_ds.shape

    x = np.arange(cols_ds) * downsample * resolution_m / 1000.0  # km
    y = np.arange(rows_ds) * downsample * resolution_m / 1000.0
    X, Y = np.meshgrid(x, y)

    # Vertical exaggeration
    z_range = dem_ds.max() - dem_ds.min()
    xy_range = max(x.max(), y.max())
    v_exag = max(2.0, xy_range / max(z_range, 1.0) * 0.3)
    Z = dem_ds * v_exag

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.75, linewidth=0,
                    antialiased=True, rstride=1, cstride=1)

    # Route draped on terrain
    path = np.array(route_rc)
    route_x = path[:, 1] * resolution_m / 1000.0
    route_y = path[:, 0] * resolution_m / 1000.0
    route_rows = np.clip(path[:, 0], 0, dem.shape[0] - 1)
    route_cols = np.clip(path[:, 1], 0, dem.shape[1] - 1)
    route_z = dem[route_rows, route_cols] * v_exag + z_range * 0.02 * v_exag  # slight offset above surface

    ax.plot(route_x, route_y, route_z, color="red", linewidth=2.5, zorder=10, label="Route")
    ax.scatter([route_x[0]], [route_y[0]], [route_z[0]], color="lime", s=60, zorder=11)
    ax.scatter([route_x[-1]], [route_y[-1]], [route_z[-1]], color="blue", s=60, zorder=11)

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_zlabel(f"Elev ×{v_exag:.1f}")
    ax.set_title("3D Terrain Drape", fontsize=14, fontweight="bold")
    ax.view_init(elev=35, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "route_3d.png"), dpi=180)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Cross-Section Profiles
# ══════════════════════════════════════════════════════════════════════════════

def _plot_cross_sections(dem, route_rc, resolution_m, row_buffer_m=30):
    log.info("Viz [8/10]: Cross-section profiles …")
    path = np.array(route_rc)
    n = len(path)
    if n < 10:
        log.info("Viz [8/10]: Route too short for cross-sections — skipping.")
        return

    # Pick 4 key indices: 10%, 35%, 65%, 90% along the route
    indices = [int(n * f) for f in (0.1, 0.35, 0.65, 0.9)]
    labels = ["Approach (10%)", "Section (35%)", "Section (65%)", "End Approach (90%)"]

    half_width_px = int(500 / resolution_m)  # 500m each side

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, idx, label in zip(axes.flat, indices, labels):
        r, c = path[idx]

        # Determine route direction for perpendicular
        if idx > 0 and idx < n - 1:
            dr = path[idx + 1][0] - path[idx - 1][0]
            dc = path[idx + 1][1] - path[idx - 1][1]
        else:
            dr, dc = 0, 1
        mag = math.sqrt(dr * dr + dc * dc) or 1.0
        # Perpendicular direction
        perp_r, perp_c = -dc / mag, dr / mag

        offsets = np.arange(-half_width_px, half_width_px + 1)
        distances = offsets * resolution_m
        elevations = []
        for off in offsets:
            sr = int(round(r + off * perp_r))
            sc = int(round(c + off * perp_c))
            sr = max(0, min(dem.shape[0] - 1, sr))
            sc = max(0, min(dem.shape[1] - 1, sc))
            elevations.append(dem[sr, sc])

        elevations = np.array(elevations)
        ax.fill_between(distances, elevations, alpha=0.3, color="#8b5a2b")
        ax.plot(distances, elevations, color="#5c3a1e", linewidth=1.5)
        ax.axvline(0, color="red", linewidth=2, linestyle="-", label="Centerline")

        # RoW envelope
        ax.axvspan(-row_buffer_m, row_buffer_m, color="yellow", alpha=0.15, label="RoW")

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Offset (m)")
        ax.set_ylabel("Elevation (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Perpendicular Cross-Section Profiles", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "cross_sections.png"), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Route Statistics Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def _plot_dashboard(meta, slope_along_route, thresholds):
    log.info("Viz [9/10]: Route statistics dashboard …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [3, 2]})

    # Left panel: text stats
    ax_text = axes[0]
    ax_text.axis("off")

    stats_lines = [
        ("Total Length", f"{meta.get('total_length_km', '?')} km"),
        ("Min Curve Radius", f"{meta.get('min_curve_radius_achieved_m', '?')} m"),
        ("Curve Violations", f"{meta.get('curve_radius_violations', '?')}"),
        ("Grade Violations", f"{meta.get('sustained_grade_violations', '?')}"),
        ("Max Slope Along Route", f"{np.max(slope_along_route):.1f}%"),
        ("Median Slope", f"{np.median(slope_along_route):.1f}%"),
        ("DEM Source", f"{meta.get('dem_source', '?')}"),
        ("Data Confidence", f"{meta.get('data_confidence', '?')}"),
        ("Peak Memory", f"{meta.get('peak_memory_mb', '?')} MB"),
    ]

    y_start = 0.92
    ax_text.text(0.05, 0.98, "Route Summary", fontsize=18, fontweight="bold",
                 transform=ax_text.transAxes, va="top")

    for i, (label, value) in enumerate(stats_lines):
        y = y_start - i * 0.09
        ax_text.text(0.08, y, f"{label}:", fontsize=12, fontweight="bold",
                     transform=ax_text.transAxes, va="top", color="#333")
        ax_text.text(0.55, y, value, fontsize=12,
                     transform=ax_text.transAxes, va="top", color="#1a1a2e")

    # Right panel: terrain breakdown pie chart
    ax_pie = axes[1]
    s_opt = thresholds.get("s_opt", 4)
    s_mod = thresholds.get("s_mod", 8)
    s_max = thresholds.get("s_max", 12)

    n_flat = np.sum(slope_along_route <= s_opt)
    n_mod = np.sum((slope_along_route > s_opt) & (slope_along_route <= s_mod))
    n_steep = np.sum((slope_along_route > s_mod) & (slope_along_route <= s_max))
    n_cliff = np.sum(slope_along_route > s_max)
    total = max(len(slope_along_route), 1)

    sizes = [n_flat, n_mod, n_steep, n_cliff]
    labels_pie = [f"Flat ≤{s_opt}%\n({n_flat/total*100:.0f}%)",
                  f"Moderate\n({n_mod/total*100:.0f}%)",
                  f"Steep\n({n_steep/total*100:.0f}%)",
                  f"Near-cliff\n({n_cliff/total*100:.0f}%)"]
    colors_pie = ["#2d9f3e", "#f4d03f", "#e67e22", "#c0392b"]

    # Remove zero slices
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 0]
    if non_zero:
        sizes_nz, labels_nz, colors_nz = zip(*non_zero)
        ax_pie.pie(sizes_nz, labels=labels_nz, colors=colors_nz,
                   autopct=None, startangle=90, textprops={"fontsize": 10})
    ax_pie.set_title("Terrain Breakdown", fontsize=13, fontweight="bold")

    fig.suptitle("Myanmar Highway Alignment — Feasibility Dashboard",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "route_dashboard.png"), dpi=200)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 10. Interactive Folium HTML Map
# ══════════════════════════════════════════════════════════════════════════════

def _plot_folium_map(route_wgs84, slope_along_route, meta, thresholds,
                     buildings_wgs=None, water_wgs=None, si_result=None):
    log.info("Viz [10/10]: Interactive Folium map …")
    try:
        import folium
        from folium.plugins import MeasureControl
    except ImportError:
        log.warning("Folium not installed — skipping interactive map. "
                    "Install with: pip install folium")
        return

    # Centre on route midpoint
    mid = len(route_wgs84) // 2
    center = route_wgs84[mid]
    m = folium.Map(location=[center[1], center[0]], zoom_start=10,
                   tiles=None)

    # Basemap layers
    folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap", name="Terrain", overlay=False
    ).add_to(m)

    # Color-code route by slope
    s_opt = thresholds.get("s_opt", 4)
    s_mod = thresholds.get("s_mod", 8)
    s_max = thresholds.get("s_max", 12)

    def _slope_color(s):
        if s <= s_opt:
            return "#2d9f3e"
        elif s <= s_mod:
            return "#f4d03f"
        elif s <= s_max:
            return "#e67e22"
        return "#c0392b"

    # Draw route as segments
    route_fg = folium.FeatureGroup(name="Route", show=True)
    step = max(1, len(route_wgs84) // 500)  # limit to ~500 segments for performance
    for i in range(0, len(route_wgs84) - step, step):
        j = min(i + step, len(route_wgs84) - 1)
        p1 = route_wgs84[i]
        p2 = route_wgs84[j]
        si = int(min(i, len(slope_along_route) - 1))
        slope_val = slope_along_route[si] if si < len(slope_along_route) else 0
        color = _slope_color(slope_val)

        folium.PolyLine(
            locations=[[p1[1], p1[0]], [p2[1], p2[0]]],
            color=color, weight=4, opacity=0.85,
            popup=f"Slope: {slope_val:.1f}%"
        ).add_to(route_fg)
    route_fg.add_to(m)

    # Start/End markers
    start = route_wgs84[0]
    end = route_wgs84[-1]
    folium.Marker(
        [start[1], start[0]], popup="Start (Point A)",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    folium.Marker(
        [end[1], end[0]], popup="End (Point B)",
        icon=folium.Icon(color="blue", icon="stop")
    ).add_to(m)

    # Building layer (togglable)
    if buildings_wgs is not None and len(buildings_wgs) > 0:
        bldg_fg = folium.FeatureGroup(name="Buildings", show=False)
        # Limit to 5000 buildings nearest the route for performance
        sample_size = min(5000, len(buildings_wgs))
        bldg_sample = buildings_wgs.sample(n=sample_size, random_state=42) if len(buildings_wgs) > sample_size else buildings_wgs
        for _, row in bldg_sample.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                centroid = geom.centroid
                folium.CircleMarker(
                    [centroid.y, centroid.x], radius=2,
                    color="black", fill=True, fill_opacity=0.6
                ).add_to(bldg_fg)
            except Exception:
                continue
        bldg_fg.add_to(m)

    # Water layer (togglable)
    if water_wgs is not None and len(water_wgs) > 0:
        water_fg = folium.FeatureGroup(name="Water Bodies", show=False)
        sample_size = min(2000, len(water_wgs))
        water_sample = water_wgs.sample(n=sample_size, random_state=42) if len(water_wgs) > sample_size else water_wgs
        for _, row in water_sample.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                centroid = geom.centroid
                folium.CircleMarker(
                    [centroid.y, centroid.x], radius=3,
                    color="blue", fill=True, fill_color="blue", fill_opacity=0.4
                ).add_to(water_fg)
            except Exception:
                continue
        water_fg.add_to(m)

    # ── Phase 8: Structures layer (bridges + culverts) ─────────────────────
    if si_result is not None and si_result.structures:
        struct_fg = folium.FeatureGroup(name="Structures (bridges/culverts)", show=True)
        for struct in si_result.structures:
            if struct.lat == 0.0 and struct.lon == 0.0:
                continue   # failed coordinate lookup
            if struct.structure_type == "bridge":
                popup_html = (
                    f"<b>Bridge #{struct.structure_id}</b><br>"
                    f"River: {struct.water_name}<br>"
                    f"Chainage: {struct.chainage_m/1000:.2f} km<br>"
                    f"Span: {struct.length_m:.0f} m<br>"
                    f"Deck elev: {struct.deck_elevation_m:.1f} m<br>"
                    f"Est. cost: USD {struct.estimated_cost_usd/1e6:.2f} M"
                )
                folium.Marker(
                    [struct.lat, struct.lon],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color="red", icon="tint", prefix="fa"),
                ).add_to(struct_fg)
            else:  # culvert
                popup_html = (
                    f"<b>Culvert #{struct.structure_id}</b><br>"
                    f"Chainage: {struct.chainage_m/1000:.2f} km<br>"
                    f"Invert elev: {struct.deck_elevation_m:.1f} m<br>"
                    f"Est. cost: USD {struct.estimated_cost_usd/1e3:.0f} K"
                )
                folium.CircleMarker(
                    [struct.lat, struct.lon], radius=6,
                    color="#e67e22", fill=True, fill_color="#e67e22", fill_opacity=0.85,
                    popup=folium.Popup(popup_html, max_width=220),
                ).add_to(struct_fg)
        struct_fg.add_to(m)

    # Legend HTML
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 12px;">
        <b>Slope Legend</b><br>
        <span style="color:#2d9f3e">■</span> Flat (≤{s_opt}%)<br>
        <span style="color:#f4d03f">■</span> Moderate ({s_opt}–{s_mod}%)<br>
        <span style="color:#e67e22">■</span> Steep ({s_mod}–{s_max}%)<br>
        <span style="color:#c0392b">■</span> Near-cliff (>{s_max}%)
    </div>
    """.format(s_opt=s_opt, s_mod=s_mod, s_max=s_max)
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    try:
        MeasureControl().add_to(m)
    except Exception:
        pass

    m.save(os.path.join(OUT, "route_map.html"))
    log.info(f"Interactive map saved to {OUT}/route_map.html")




# ══════════════════════════════════════════════════════════════════════════════
# 11. Vertical Alignment Design Profile  (Phase 6)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_vertical_alignment(va_result, max_grade_pct=8.0):
    """
    Two-panel vertical alignment design plot.

    Upper panel — Profile view
    ──────────────────────────
    • Grey filled polygon  : terrain ground line (z_ground)
    • Blue solid line      : Finished Grade Line (z_design)
    • Green fill above grey: fill embankment zones (z_design > z_ground)
    • Red fill below grey  : cut excavation zones (z_design < z_ground)
    • Magenta diamonds     : VPI (Vertical Point of Intersection) markers
    • Tick marks at PVC/PVT: start and end of each parabolic curve
    • K-value annotation   : shown above each VC for rapid audit

    Lower panel — Design grade (%)
    ───────────────────────────────
    • Solid teal line      : instantaneous design grade at each station
    • Horizontal red bands : ±max_grade_pct limit lines
    • Orange shading       : grade violation zones
    """
    log.info("Viz [11/11]: Vertical alignment design profile …")
    dist_km = va_result.distances_m / 1000.0

    fig, (ax_prof, ax_grade) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor("#f8f9fa")

    # ── Upper panel: profile view ────────────────────────────────────────
    z_gr  = va_result.z_ground
    z_des = va_result.z_design

    # Ground terrain fill (grey)
    z_floor = min(z_gr.min(), z_des.min()) - 5.0
    ax_prof.fill_between(dist_km, z_floor, z_gr,
                         color="#cfd8dc", alpha=0.9, label="Ground terrain")
    ax_prof.plot(dist_km, z_gr, color="#78909c", linewidth=0.8, alpha=0.7)

    # Fill zones (design above ground → green embankment)
    ax_prof.fill_between(dist_km, z_gr, z_des,
                         where=(z_des >= z_gr),
                         color="#2e7d32", alpha=0.35, label="Fill (embankment)")

    # Cut zones (design below ground → orange excavation)
    ax_prof.fill_between(dist_km, z_gr, z_des,
                         where=(z_des < z_gr),
                         color="#b71c1c", alpha=0.30, label="Cut (excavation)")

    # Design Finished Grade Line
    ax_prof.plot(dist_km, z_des, color="#1565c0", linewidth=2.0,
                 label="Design FGL", zorder=5)

    # VPI markers (grade-break knees)
    vpi_km = va_result.vpi_stations / 1000.0
    ax_prof.scatter(vpi_km, va_result.vpi_elevations,
                    marker="D", s=25, color="#7b1fa2", zorder=6,
                    label=f"VPIs ({len(va_result.vpi_stations)})", alpha=0.7)

    # PVC / PVT tick marks and K-value annotations
    for vc in va_result.vertical_curves:
        pvc_km = vc.pvc_station_m / 1000.0
        pvt_km = vc.pvt_station_m / 1000.0
        z_pvt  = vc.z_pvc + vc.g1_pct / 100.0 * vc.length_m
        # Tick at PVC
        ax_prof.plot(pvc_km, vc.z_pvc, "|", color="#6a0dad",
                     markersize=8, markeredgewidth=1.5, zorder=7)
        # Tick at PVT
        ax_prof.plot(pvt_km, z_pvt, "|", color="#6a0dad",
                     markersize=8, markeredgewidth=1.5, zorder=7)
        # K-value label above midpoint
        mid_km  = (pvc_km + pvt_km) / 2.0
        mid_z   = (vc.z_pvc + z_pvt) / 2.0
        z_range = z_des.max() - z_des.min()
        ax_prof.text(
            mid_km, mid_z + z_range * 0.015,
            f"K={vc.k_value:.0f}",
            fontsize=5.5, ha="center", va="bottom",
            color="#4a148c", alpha=0.75,
        )

    ax_prof.set_ylabel("Elevation (m ASL)", fontsize=11)
    ax_prof.set_title(
        "Phase 6 — Vertical Alignment Design Profile",
        fontsize=14, fontweight="bold"
    )
    ax_prof.legend(loc="upper right", fontsize=9, ncol=2)
    ax_prof.grid(True, alpha=0.25, linestyle=":")
    ax_prof.set_facecolor("#f8f9fa")

    # Stats textbox
    n_vc   = len(va_result.vertical_curves)
    g_max  = va_result.max_grade_pct
    n_gv   = len(va_result.grade_violations)
    n_ssd  = len(va_result.ssd_violations)
    fill_max = va_result.cut_fill_m.max()
    cut_max  = -va_result.cut_fill_m.min()
    stats_txt = (
        f"VCs: {n_vc}   Max grade: {g_max:.2f}%   "
        f"Grade viol: {n_gv}   SSD viol: {n_ssd}\n"
        f"Max fill: +{fill_max:.1f} m   Max cut: −{cut_max:.1f} m"
    )
    ax_prof.text(
        0.01, 0.02, stats_txt,
        transform=ax_prof.transAxes,
        fontsize=8.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#aaa", alpha=0.85)
    )

    # ── Lower panel: design grade ────────────────────────────────────────
    ax_grade.plot(dist_km, va_result.grade_pct,
                  color="#00695c", linewidth=1.2, label="Design grade (%)")
    ax_grade.axhline( max_grade_pct, color="red", linewidth=1.0,
                      linestyle="--", alpha=0.7, label=f"+{max_grade_pct:.0f}% limit")
    ax_grade.axhline(-max_grade_pct, color="red", linewidth=1.0,
                      linestyle="--", alpha=0.7, label=f"−{max_grade_pct:.0f}% limit")
    ax_grade.axhline(0, color="#aaa", linewidth=0.7, linestyle=":")

    # Shade violation zones
    ax_grade.fill_between(
        dist_km, max_grade_pct, va_result.grade_pct,
        where=(va_result.grade_pct > max_grade_pct),
        color="#ff5722", alpha=0.4, label="Grade violation"
    )
    ax_grade.fill_between(
        dist_km, -max_grade_pct, va_result.grade_pct,
        where=(va_result.grade_pct < -max_grade_pct),
        color="#ff5722", alpha=0.4
    )

    ax_grade.set_xlabel("Chainage (km)", fontsize=11)
    ax_grade.set_ylabel("Grade (%)", fontsize=11)
    ax_grade.set_ylim(-max_grade_pct * 1.5, max_grade_pct * 1.5)
    ax_grade.legend(loc="upper right", fontsize=8, ncol=3)
    ax_grade.grid(True, alpha=0.25, linestyle=":")
    ax_grade.set_facecolor("#f8f9fa")

    fig.tight_layout(h_pad=0.5)
    out_path = os.path.join(OUT, "vertical_profile.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Viz [11]: Vertical profile saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. Earthwork Volumes + Mass-Haul Diagram  (Phase 7)
# ══════════════════════════════════════════════════════════════════════════════

def _plot_earthwork(ew_result):
    """
    Three-panel earthwork and Brückner mass-haul diagram.

    Panel 1 (top)    — Cut/fill depth profile
    ──────────────────────────
    • Orange bars (negative)  : cut depth below formation
    • Green  bars (positive)  : fill height above terrain

    Panel 2 (middle) — Cumulative cut & fill volumes
    ──────────────────────────
    • Red line   : cumulative cut  (Mm³)
    • Blue line  : cumulative fill (Mm³)

    Panel 3 (bottom) — Brückner mass-haul ordinate
    ──────────────────────────
    • Teal line          : Brückner ordinate (Mm³)
    • Horizontal grey    : zero ordinate
    • Vertical dashed    : balance station markers
    • Shaded zones       : ’import‘ (above 0) and ’spoil‘ (below 0)
    """
    log.info("Viz [12/12]: Earthwork volumes + mass-haul diagram …")
    d_km = ew_result.distances_m / 1000.0

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(18, 12), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5, 2]}
    )
    fig.patch.set_facecolor("#f8f9fa")

    # ── Panel 1: cut/fill depth ─────────────────────────────────────────
    cf = ew_result.cut_fill_m
    ax1.fill_between(d_km, 0, cf, where=(cf >= 0),
                      color="#2e7d32", alpha=0.70, label="Fill (embankment)")
    ax1.fill_between(d_km, 0, cf, where=(cf < 0),
                      color="#e65100", alpha=0.70, label="Cut (excavation)")
    ax1.plot(d_km, cf, color="#333", linewidth=0.5, alpha=0.5)
    ax1.axhline(0, color="#777", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Cut(−) / Fill(+) (m)", fontsize=10)
    ax1.set_title(
        "Phase 7 — Earthwork Volumes & Mass-Haul (Brückner) Diagram",
        fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.2, linestyle=":")
    ax1.set_facecolor("#f8f9fa")

    # Summary text box
    total_cut  = ew_result.total_cut_m3  / 1e6
    total_fill = ew_result.total_fill_m3 / 1e6
    net_val    = abs(ew_result.net_import_m3) / 1e6
    net_lbl    = "import" if ew_result.net_import_m3 > 0 else "spoil"
    stats_txt  = (
        f"Total cut:  {total_cut:.3f} Mm³\n"
        f"Total fill: {total_fill:.3f} Mm³\n"
        f"Net {net_lbl}: {net_val:.3f} Mm³  "
        f"(swell ×{ew_result.swell_factor:.2f})"
    )
    ax1.text(0.01, 0.97, stats_txt,
             transform=ax1.transAxes, fontsize=8.5,
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#aaa", alpha=0.88))

    # ── Panel 2: cumulative volumes ─────────────────────────────────────
    ax2.plot(d_km, ew_result.cumul_cut_m3  / 1e6,
             color="#c62828", linewidth=1.5, label="Cumul. cut (Mm³)")
    ax2.plot(d_km, ew_result.cumul_fill_m3 / 1e6,
             color="#1565c0", linewidth=1.5, label="Cumul. fill (Mm³)")
    ax2.set_ylabel("Cumulative Vol. (Mm³)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.2, linestyle=":")
    ax2.set_facecolor("#f8f9fa")

    # ── Panel 3: Brückner mass-haul ──────────────────────────────────
    mh = ew_result.mass_haul_m3 / 1e6
    ax3.plot(d_km, mh, color="#00695c", linewidth=1.8, label="Mass-haul ordinate")
    ax3.fill_between(d_km, 0, mh, where=(mh >= 0),
                      color="#1565c0", alpha=0.20, label="Fill deficit (import)")
    ax3.fill_between(d_km, 0, mh, where=(mh < 0),
                      color="#e65100", alpha=0.20, label="Cut surplus (spoil)")
    ax3.axhline(0, color="#555", linewidth=1.0, linestyle="-")

    # Balance station markers
    for bs in ew_result.balance_stations_m:
        ax3.axvline(bs / 1000.0, color="#7b1fa2", linewidth=1.2,
                    linestyle="--", alpha=0.8)
        ax3.text(bs / 1000.0, mh.max() * 0.05,
                 f"{bs/1000:.1f} km",
                 fontsize=7, color="#7b1fa2", ha="center", va="bottom", rotation=90)

    ax3.set_xlabel("Chainage (km)", fontsize=11)
    ax3.set_ylabel("Mass-Haul (Mm³)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8, ncol=3)
    ax3.grid(True, alpha=0.2, linestyle=":")
    ax3.set_facecolor("#f8f9fa")
    if ew_result.balance_stations_m:
        ax3.text(0.01, 0.02,
                 f"{len(ew_result.balance_stations_m)} balance station(s): "
                 + ", ".join(f"{s/1000:.1f} km" for s in ew_result.balance_stations_m[:5]),
                 transform=ax3.transAxes, fontsize=7.5, va="bottom",
                 color="#7b1fa2")

    fig.tight_layout(h_pad=0.5)
    out_path = os.path.join(OUT, "earthwork_masshual.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Viz [12]: Earthwork + mass-haul saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Master entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_visuals(viz_data):
    """
    Main entry point called from main.py.

    viz_data is a dictionary with keys:
        cost            — 2D float64 unified cost array
        building_penalty — 2D float32 building penalty map
        layers          — dict of individual layer arrays (slope, lulc, water, building, road, unified)
        dem             — 2D DEM array
        route_rc        — list of (row, col) tuples
        route_utm       — list of (x, y) tuples in UTM
        route_wgs84     — list of (lon, lat) tuples
        distances_m     — 1D array of cumulative distance
        elevations_m    — 1D array of elevation along route
        grade_violations — list of (start_m, end_m, avg_grade)
        slope_along     — 1D array of slope % along route
        slope_thresholds — dict with s_opt, s_mod, s_max, s_cliff
        meta            — route metadata dict
        transform       — rasterio Affine transform
        resolution_m    — cell size in metres
        buildings_wgs   — GeoDataFrame (WGS84) or None
        water_wgs       — GeoDataFrame (WGS84) or None
    """
    _ensure_output()
    log.info("=" * 50)
    log.info("  Generating 10-product visualization suite")
    log.info("=" * 50)

    try:
        _plot_cost_heatmap(
            viz_data["cost"], viz_data["route_rc"],
            viz_data["transform"]
        )
    except Exception as e:
        log.warning(f"Viz [1] cost heatmap failed: {e}")

    try:
        _plot_building_decay(
            viz_data["building_penalty"], viz_data["route_rc"],
            viz_data["transform"], viz_data["resolution_m"]
        )
    except Exception as e:
        log.warning(f"Viz [2] building decay failed: {e}")

    try:
        _plot_layer_decomposition(
            viz_data["layers"], viz_data["route_rc"]
        )
    except Exception as e:
        log.warning(f"Viz [3] layer decomposition failed: {e}")

    try:
        _plot_elevation_profile(
            viz_data["distances_m"], viz_data["elevations_m"],
            viz_data["grade_violations"]
        )
    except Exception as e:
        log.warning(f"Viz [4] elevation profile failed: {e}")

    try:
        _plot_slope_histogram(
            viz_data["slope_along"], viz_data["slope_thresholds"]
        )
    except Exception as e:
        log.warning(f"Viz [5] slope histogram failed: {e}")

    try:
        cost_along = _sample_raster_along_route(viz_data["cost"], viz_data["route_rc"])
        layers_along = {}
        for key in ("slope", "building", "water", "lulc"):
            layer = viz_data["layers"].get(key)
            if layer is not None:
                layers_along[key] = _sample_raster_along_route(layer, viz_data["route_rc"])
        # Use rc_distances_m (matches route_rc length) if available,
        # otherwise fall back to distances_m (may be different length).
        rc_dist = viz_data.get("rc_distances_m", viz_data["distances_m"])
        _plot_cost_along_route(rc_dist, cost_along, layers_along)
    except Exception as e:
        log.warning(f"Viz [6] cost-along-route failed: {e}")

    try:
        _plot_3d_terrain(
            viz_data["dem"], viz_data["route_rc"], viz_data["resolution_m"]
        )
    except Exception as e:
        log.warning(f"Viz [7] 3D terrain failed: {e}")

    try:
        _plot_cross_sections(
            viz_data["dem"], viz_data["route_rc"], viz_data["resolution_m"]
        )
    except Exception as e:
        log.warning(f"Viz [8] cross-sections failed: {e}")

    try:
        _plot_dashboard(
            viz_data["meta"], viz_data["slope_along"],
            viz_data["slope_thresholds"]
        )
    except Exception as e:
        log.warning(f"Viz [9] dashboard failed: {e}")

    try:
        _plot_folium_map(
            viz_data["route_wgs84"], viz_data["slope_along"],
            viz_data["meta"], viz_data["slope_thresholds"],
            buildings_wgs=viz_data.get("buildings_wgs"),
            water_wgs=viz_data.get("water_wgs"),
            si_result=viz_data.get("si_result"),
        )
    except Exception as e:
        log.warning(f"Viz [10] folium map failed: {e}")

    # Viz [11] — Phase 6 vertical alignment profile
    va_result = viz_data.get("va_result")
    if va_result is not None:
        try:
            from config import GRADE_MAX_PCT, SCENARIO_PROFILE
            g_max = GRADE_MAX_PCT.get(SCENARIO_PROFILE, 8.0)
            _plot_vertical_alignment(va_result, max_grade_pct=g_max)
        except Exception as e:
            log.warning(f"Viz [11] vertical profile failed: {e}")
        product_count = 11
    else:
        product_count = 10

    # Viz [12] — Phase 7 earthwork + mass-haul
    ew_result = viz_data.get("ew_result")
    if ew_result is not None:
        try:
            _plot_earthwork(ew_result)
            product_count = 12
        except Exception as e:
            log.warning(f"Viz [12] earthwork mass-haul failed: {e}")

    log.info(f"Visualization suite complete — {product_count} products saved to output/")
