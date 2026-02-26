import sys
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.plot import show
import glob
import os

def visualize():
    os.makedirs("output", exist_ok=True)
    
    route = gpd.read_file("preliminary_route.geojson")
    route_utm = route.to_crs(epsg=32646)
    coords = list(route_utm.geometry[0].coords)
    
    dem_files = glob.glob("data/dem_utm_*.tif")
    if not dem_files:
        print("DEM not found")
        return
        
    dem_path = dem_files[0]
    
    water_files = glob.glob("data/water_*.gpkg")
    building_files = glob.glob("data/buildings_*.gpkg")
    
    water_gdf = gpd.read_file(water_files[0]).to_crs(epsg=32646) if water_files else None
    bldg_gdf = gpd.read_file(building_files[0]).to_crs(epsg=32646) if building_files else None

    # Overview
    fig, ax = plt.subplots(figsize=(12, 12))
    with rasterio.open(dem_path) as src:
        show(src, ax=ax, cmap='terrain', alpha=0.5)
        
    if water_gdf is not None:
        water_gdf.plot(ax=ax, color='blue', alpha=0.4)
    if bldg_gdf is not None:
        bldg_gdf.plot(ax=ax, color='black', alpha=0.5)
        
    route_utm.plot(ax=ax, color='red', linewidth=2)
    ax.plot(coords[0][0], coords[0][1], 'go', markersize=8, label='Start')
    ax.plot(coords[-1][0], coords[-1][1], 'bo', markersize=8, label='End')
    plt.legend()
    plt.title("Overview Route")
    plt.savefig(r"output\route_overview.png", dpi=300)
    plt.close()

    # Start Area
    fig, ax = plt.subplots(figsize=(10, 10))
    with rasterio.open(dem_path) as src:
        show(src, ax=ax, cmap='terrain', alpha=0.5)
    if water_gdf is not None:
        water_gdf.plot(ax=ax, color='blue', alpha=0.4)
    if bldg_gdf is not None:
        bldg_gdf.plot(ax=ax, color='black', alpha=0.5)
        
    route_utm.plot(ax=ax, color='red', linewidth=3)
    ax.plot(coords[0][0], coords[0][1], 'go', markersize=10)
    ax.set_xlim(coords[0][0] - 8000, coords[0][0] + 8000)
    ax.set_ylim(coords[0][1] - 8000, coords[0][1] + 8000)
    plt.title("Start Area")
    plt.savefig(r"output\route_start.png", dpi=300)
    plt.close()

    # End Area
    fig, ax = plt.subplots(figsize=(10, 10))
    with rasterio.open(dem_path) as src:
        show(src, ax=ax, cmap='terrain', alpha=0.5)
    if water_gdf is not None:
        water_gdf.plot(ax=ax, color='blue', alpha=0.4)
    if bldg_gdf is not None:
        bldg_gdf.plot(ax=ax, color='black', alpha=0.5)
        
    route_utm.plot(ax=ax, color='red', linewidth=3)
    ax.plot(coords[-1][0], coords[-1][1], 'bo', markersize=10)
    ax.set_xlim(coords[-1][0] - 8000, coords[-1][0] + 8000)
    ax.set_ylim(coords[-1][1] - 8000, coords[-1][1] + 8000)
    plt.title("End Area")
    plt.savefig(r"output\route_end.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    visualize()
