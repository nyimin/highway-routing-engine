import sys
import geopandas as gpd
from data_fetch import fetch_osm_layers
import logging

logging.basicConfig(level=logging.INFO)

bbox_wgs84 = (94.2175, 17.1174, 96.8304, 17.5483)
_, water, _, _, _ = fetch_osm_layers(bbox_wgs84)

for idx, row in water.iterrows():
    name = row.get("name", "")
    if name and ("irrawaddy" in str(name).lower() or "ဧရာဝတီ" in str(name)):
        print("Found Irrawaddy:", name, row.geometry.geom_type)
        if "Line" in row.geometry.geom_type:
            print("  Length:", row.geometry.length)
        else:
            print("  Area:", row.geometry.area)
        
    if name and ("ပဲခူး" in str(name) or "bago" in str(name).lower()):
        print("Found Bago:", name, row.geometry.geom_type)

print("Total elements:", len(water))
print("Geometry types:")
print(water.geom_type.value_counts())
