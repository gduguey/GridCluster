#!/usr/bin/env python3
"""
extract_NE_population.py

Extracts the Kontur 400 m hexagon population GeoPackage for the six New England states,
saves the clipped GeoPackage next to this script, and plots the result.

Usage:
    1. Download and decompress:
       • Kontur population GeoPackage (.gpkg)
       • Census Cartographic Boundary ZIP (cb_2022_us_state_500k.zip → folder)
    2. Edit POP_GPKG_PATH and STATE_SHP_FOLDER below.
    3. Run: python extract_NE_population.py
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt

def main():
    # ─────────────────────────────────────────────────────────────────────────────
    # USER SETTINGS: edit these two lines to point at your data files
    POP_GPKG_PATH   = r"kontur_population_20231101.gpkg"
    STATE_SHP_FOLDER = r"cb_2022_us_state_500k"
    # ─────────────────────────────────────────────────────────────────────────────

    # Derived paths
    state_shp = os.path.join(STATE_SHP_FOLDER, "cb_2022_us_state_500k.shp")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # output_gpkg = os.path.join(script_dir, "kontur_population_NE.gpkg")
    output_csv = os.path.join(script_dir, "ne_population.csv")

    # 1) Load the Kontur population hexagons
    print("Extracting New England population data...")
    pop_gdf = gpd.read_file(POP_GPKG_PATH)
    print(f"Loaded {len(pop_gdf):,} rows from: {POP_GPKG_PATH}")

    # 2) Load the U.S. state boundaries shapefile
    states_gdf = gpd.read_file(state_shp)

    # 3) Filter to New England states
    ne_abbr = ["CT", "ME", "MA", "NH", "RI", "VT"]
    ne_states = states_gdf[states_gdf["STUSPS"].isin(ne_abbr)]

    # 4) Reproject states to match population data CRS
    ne_states = ne_states.to_crs(pop_gdf.crs)
    
    # 5) Rough clip population data using bounding box
    ne_bbox = ne_states.total_bounds
    pop_ne_rough = pop_gdf.cx[ne_bbox[0]:ne_bbox[2], ne_bbox[1]:ne_bbox[3]]

    # 6) Create exact mask and clip
    ne_mask = ne_states.geometry.union_all()
    print(f"Clipping to {len(ne_states):,} New England states...")
    pop_ne = gpd.clip(pop_ne_rough, ne_mask)
    print(f"Clipped to {len(pop_ne):,} rows")

    # 7) Save as CSV with lat/lon
    # Reproject to WGS84 (latitude/longitude)
    centroids_projected = pop_ne.geometry.centroid
    centroids_wgs84 = centroids_projected.to_crs("EPSG:4326")

    # Add coordinates to DataFrame
    pop_ne["longitude"] = centroids_wgs84.x
    pop_ne["latitude"] = centroids_wgs84.y
    
    # Save to CSV (only keep needed columns)
    print("Saving to CSV...")
    pop_ne[["population", "longitude", "latitude"]].to_csv(output_csv, index=False)
    print(f"Saved CSV with {len(pop_ne):,} rows to: {output_csv}")


if __name__ == "__main__":
    main()