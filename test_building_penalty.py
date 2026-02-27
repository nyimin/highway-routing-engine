import time
import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Polygon

from cost_surface import _apply_building_penalties

print("Creating mock buildings...")
# Create a grid of 370,000 mock buildings to test performance
polys = []
for x in range(0, 6000, 10):
    for y in range(0, 6000, 10):
        polys.append(Polygon([(x, y), (x+5, y), (x+5, y+5), (x, y+5)]))
        if len(polys) >= 370000:
            break
    if len(polys) >= 370000:
        break

buildings_gdf = gpd.GeoDataFrame(geometry=polys)
transform = from_origin(0, 10000, 30, 30)
shape = (5000, 5000)

print(f"Testing _apply_building_penalties with {len(buildings_gdf)} buildings and shape {shape}...")
start = time.time()
penalty_map = _apply_building_penalties(buildings_gdf, transform, shape, resolution_m=30)
end = time.time()
print(f"Finished in {end-start:.2f} seconds.")
print(f"Max penalty: {np.max(penalty_map)}")
