# New England Population Extraction

This repository contains:

- **extract_NE_population.py**  
  Clips the Kontur 400 m hexagon population dataset to the New England region and plots the result.

- **kontur_population_NE.gpkg**  
  (Generated) GeoPackage containing only New England population hexagons.


## Data Sources

1. **Kontur Population Dataset**  
   • Year: 2023  
   • Resolution: 400 m H3 hexagons  
   • Format: Compressed GeoPackage (`.gpkg.gz`)  
   • Download:  
     https://www.kontur.io/datasets/population-dataset/

2. **Census Cartographic Boundary – U.S. States**  
   • Scale: 1:500 000  
   • Filename: `cb_2022_us_state_500k.zip`  
   • Download:  
     https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip


## Setup & Usage

1. **Download data**

   * Download the Kontur dataset `.gpkg.gz` and unzip it using:

     ```bash
     gunzip kontur_population_2023.gpkg.gz
     ```

     This will produce the file `kontur_population_2023.gpkg`.
   * Download and unzip the Census shapefile:

     ```bash
     unzip cb_2022_us_state_500k.zip -d cb_2022_us_state_500k
     ```

2. **Edit script paths**
   Open `extract_NE_population.py` and set:

   ```python
   POP_GPKG_PATH    = r"path/to/kontur_population_2023.gpkg"
   STATE_SHP_FOLDER = r"path/to/cb_2022_us_state_500k"
   ```


