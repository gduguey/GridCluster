# utils.py

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Any
from functools import lru_cache
from dataclasses import asdict
import hashlib
import json
from numba import njit, prange
from datetime import datetime
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd

from .settings import Config


def clamp(value: int, min_val: int, max_val: int) -> int:
    return max(min_val, min(value, max_val))


@njit(parallel=True)
def haversine_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Numba-optimized Haversine distance matrix calculation"""
    R = 6371.0  # Earth radius in km
    n = points1.shape[0]
    m = points2.shape[0]
    dist_matrix = np.empty((n, m), dtype=np.float64)
    
    for i in prange(n):
        lat1 = np.radians(points1[i, 0])
        lon1 = np.radians(points1[i, 1])
        
        for j in prange(m):
            lat2 = np.radians(points2[j, 0])
            lon2 = np.radians(points2[j, 1])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            dist_matrix[i, j] = R * c
            
    return dist_matrix


def load_data(config: Config) -> tuple[xr.DataArray, xr.DataArray, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess wind, solar, and demand data.

    Returns:
        wind (xr.DataArray): Stacked and cleaned wind capacity factors.
        solar (xr.DataArray): Stacked and cleaned solar capacity factors.
        demand_df (pd.DataFrame): Demand time series (in MW).
        population_df (pd.DataFrame): Population data with latitude and longitude.
    """
    wind = xr.open_dataset(config.path.wind_cf_file)['cf']
    solar = xr.open_dataset(config.path.solar_cf_file)['cf']
    wind = wind.stack(z=("lat", "lon")).dropna('z', how='all')
    solar = solar.stack(z=("lat", "lon")).dropna('z', how='all')

    demand_df = pd.read_csv(config.path.demand_file).iloc[:, 1:] * 1000  # Convert from GW to MW
    # demand_lat_lon_df = pd.read_csv(config.path.demand_lat_lon_file)[['Lat', 'Lon']]
    # demand_lat_lon_df['Lon'] = -demand_lat_lon_df['Lon']

    population_df = pd.read_csv(config.path.population_file)
    population_df = population_df.rename(columns={"longitude": "Lon", "latitude": "Lat"})

    return wind, solar, demand_df, population_df

def get_counties(config: Config) -> gpd.GeoDataFrame:
    """ 
    Load and filter New England counties from shapefile.
    Returns:
        counties (gpd.GeoDataFrame): Filtered counties with geometry and FIPS codes.
    """
    # 2) Read New England county polygons
    COUNTY_SHP =  config.path.county_file
    counties = gpd.read_file(COUNTY_SHP).to_crs(epsg=4326)
    ne_fips = ["09","23","25","33","44","50"]  # CT, ME, MA, NH, RI, VT
    counties = counties[counties["STATEFP"].isin(ne_fips)]
    counties["county_id"] = counties["GEOID"]  # e.g. "25027"
    return counties


class FineNetwork():
    """
    Fine network processor for wind, solar, and demand data.
    This class processes capacity factors and demand data, and returns a dictionary of DataFrames.
    """
    def __init__(self, config: Config):     
        self.config = config
        self.wind, self.solar, self.demand_df, self.population_df = load_data(config)
        self.counties = get_counties(config)

    def _process_capacity_factors(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process capacity factors for wind and solar"""
        wind = self.wind
        solar = self.solar
        wind_points = np.column_stack((wind.lat.values, wind.lon.values))
        wind_values = wind.values.astype(np.float64)
        solar_points = np.column_stack((solar.lat.values, solar.lon.values))
        solar_values = solar.values.astype(np.float64)

        wind_values = np.delete(wind_values, 181, axis=1) # Remove the 181st column for wind data
        wind_points = np.delete(wind_points, 181, axis=0)  # Remove the 181st row for wind data

        wind_df = pd.DataFrame(wind_values, columns=range(wind_values.shape[1]))
        solar_df = pd.DataFrame(solar_values, columns=range(solar_values.shape[1]))

        if not np.array_equal(wind_points, solar_points):
            raise ValueError("Wind and solar points do not match!")
        
        nodes_df = pd.DataFrame(wind_points, columns=["Lat", "Lon"])
            
        return nodes_df, wind_df, solar_df
    
    def _process_demand(self, nodes_df) -> pd.DataFrame:
        """Process demand data and disaggregate it to grid nodes"""
        demand_df = self.demand_df
        population_df = self.population_df
        counties = self.counties

        # 3) Attach each population hexagon to a county
        pop_gdf = gpd.GeoDataFrame(
            population_df,
            geometry=gpd.points_from_xy(population_df.Lon, population_df.Lat),
            crs="EPSG:4326"
        )
        # spatial join on “within”
        pop_gdf = gpd.sjoin(pop_gdf, counties[["county_id","geometry"]],
                            how="left", predicate="within")

        # for any population point that fell outside (NaN county_id), snap to the nearest county
        missing_pt = pop_gdf["county_id"].isna()
        if missing_pt.any():
            # build centroids and kd‑tree of county centroids
            cents = np.vstack([
                counties.geometry.centroid.y,
                counties.geometry.centroid.x
            ]).T
            nbr = NearestNeighbors(n_neighbors=1).fit(cents)
            pts = np.vstack([pop_gdf.loc[missing_pt,"Lat"], pop_gdf.loc[missing_pt,"Lon"]]).T
            _, idx = nbr.kneighbors(pts)
            pop_gdf.loc[missing_pt,"county_id"] = counties.iloc[idx.flatten()]["county_id"].values

        # 4) Snap each population point to its nearest grid node
        grid_coords = np.vstack([nodes_df.Lat, nodes_df.Lon]).T
        nbr_nodes = NearestNeighbors(n_neighbors=1).fit(grid_coords)
        pop_pts = np.vstack([pop_gdf.Lat, pop_gdf.Lon]).T
        _, node_idx = nbr_nodes.kneighbors(pop_pts)
        pop_gdf["node_id"] = node_idx.flatten()

        # 5) Compute population per (county_id, node_id)
        pop_cn = (
            pop_gdf
            .groupby(["county_id","node_id"])["population"]
            .sum()
            .unstack(fill_value=0)   # rows=county_id, cols=node_id
        )

        # 6) Build weight matrix W where each row sums to 1
        W = pop_cn.div(pop_cn.sum(axis=1), axis=0).fillna(0)
        #    — if a county had zero total pop (unlikely), its row becomes all zeros

        # 7) Reorder demand_df to match W’s rows, then disaggregate
        demand_df.columns = demand_df.columns.map(lambda x: str(x).zfill(5)) # Ensure demand_df columns are zero-padded to 5 digits (FIPS codes)
        demand_df = demand_df.loc[:, W.index]    # ensure same county order
        grid_demand = demand_df.values.dot(W.values)
        grid_demand_df = pd.DataFrame(
            grid_demand,
            index=demand_df.index,
            columns=W.columns
        )

        if not np.allclose(grid_demand_df.sum(axis=1),demand_df.sum(axis=1),atol=1e-6):
            raise ValueError("Demand conservation check failed!")
        
        return grid_demand_df

    def build_fine_ntw(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """ 
        Process the data and return a dictionary of DataFrames.
        """
        nodes_df, wind_df, solar_df = self._process_capacity_factors()
        grid_demand_df = self._process_demand(nodes_df)
        return {
            'nodes': nodes_df,
            'time_series':{
                'wind': wind_df,
                'solar': solar_df,
                'demand': grid_demand_df
                }
            }
    
class CoarseNetwork():
    """
    Coarse network processor for wind, solar, and demand data.
    This class aggregates capacity factors and demand data from a fine network.
    """
    def __init__(self, config: Config, fine_network: dict[str, pd.DataFrame | dict[str, pd.DataFrame]]):
        self.config = config
        self.fine_network = fine_network
        
        coarse_nodes_df = pd.read_csv(config.path.coarse_node_file)[['Lat', 'Lon']]
        coarse_nodes_df['Lon'] = -coarse_nodes_df['Lon']
        self.coarse_nodes_df = coarse_nodes_df

        nbr = NearestNeighbors(n_neighbors=1).fit(self.coarse_nodes_df.values)
        _, zones = nbr.kneighbors(self.fine_network['nodes'].values)
        self.fine_to_coarse = zones.flatten() 

    def _aggregate_capacity_factors(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate capacity factors for wind and solar by taking the medoid of each zone."""
        wind = self.fine_network['time_series']['wind']
        solar = self.fine_network['time_series']['solar']

        C = len(self.coarse_nodes_df)
        T = wind.shape[0]

        # placeholders
        wind_coarse  = pd.DataFrame(index=wind.index,  columns=range(C), dtype=float)
        solar_coarse = pd.DataFrame(index=solar.index, columns=range(C), dtype=float)

        for zone in range(C):
            # fine nodes assigned to this zone
            mask = (self.fine_to_coarse == zone)
            idxs = np.where(mask)[0]

            if len(idxs) == 0:
                raise ValueError(f"Zone {zone} has no fine nodes assigned!")

            # slice out the sub‑matrices
            wsub = wind.iloc[:, idxs]
            ssub = solar.iloc[:, idxs]

            # compute pairwise distances and pick medoid
            Dw = pairwise_distances(wsub.T)      # k×k
            medoid_w = idxs[ Dw.sum(axis=0).argmin() ]
            Ds = pairwise_distances(ssub.T)
            medoid_s = idxs[ Ds.sum(axis=0).argmin() ]

            # assign the medoid time series
            wind_coarse.iloc[:, zone]  = wind.iloc[:, medoid_w].values
            solar_coarse.iloc[:, zone] = solar.iloc[:, medoid_s].values

        return wind_coarse, solar_coarse

    def _aggregate_demand(self) -> pd.DataFrame:
        """
        Aggregate demand by summing over fine nodes assigned to each coarse node.
        """
        grid_demand_df = self.fine_network['time_series']['demand']

        C = len(self.coarse_nodes_df)

        demand_coarse = pd.DataFrame(
            index=grid_demand_df.index,
            columns=range(C),
            dtype=float
        )

        for zone in range(C):
            # fine nodes assigned to this zone
            mask = (self.fine_to_coarse == zone)
            idxs = np.where(mask)[0]

            if len(idxs) == 0:
                raise ValueError(f"Zone {zone} has no fine nodes assigned!")
            
            # sum demand of all fine nodes in this zone
            demand_coarse.iloc[:, zone] = grid_demand_df.iloc[:, idxs].sum(axis=1).values

        if not np.allclose(
            demand_coarse.sum(axis=1),
            grid_demand_df.sum(axis=1),
            atol=1e-6
        ):
            raise ValueError("Demand conservation check failed!")

        return demand_coarse

    def build_coarse_ntw(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """
        Build the coarse network by aggregating capacity factors and demand.
        """
        wind_df, solar_df = self._aggregate_capacity_factors()
        demand_df = self._aggregate_demand()
        return {
            'nodes': self.coarse_nodes_df,
            'time_series':{
                'wind': wind_df,
                'solar': solar_df,
                'demand': demand_df
                }
            }
    

class Network:
    def __init__(self,
                 nodes_df: pd.DataFrame,
                 time_series: dict[str, pd.DataFrame],
                 config: Config,
                 start_day: str | None = None,
                 end_day: str | None = None):
        """
        Initialize network with configurable features and date filtering
        
        :param nodes_df: DataFrame with 'Lat' and 'Lon' columns
        :param time_series: Dict of time series DataFrames with datetime index
        :param config: Configuration object
        :param start_day: Start date in 'YYYY-MM-DD' format
        :param end_day: End date in 'YYYY-MM-DD' format
        """
        if len(time_series.keys()) <= 1 and 'intra_correlation' in config.data_preproc.active_features:
            raise ValueError(
                "The 'intra_correlation' feature cannot be active when there is only one time series."
            )
        
        self.config = config
        self.nodes_df = nodes_df
        self.time_series = self._filter_time_series(time_series, start_day, end_day)
        self.features = self._compute_features()

    def _date_to_hour(self, date_str: str, end_of_day: bool = False) -> int:
        """Convert date string to hour index (0-8760) with numba-compatible logic"""
        year = self.config.data_preproc.year
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt.year != year:
                raise ValueError(f"Date year {dt.year} doesn't match config year {year}")
        except ValueError as e:
            try:
                dt = datetime.strptime(f"{year}-{date_str}", "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"Invalid date format or mismatched year: {date_str}. "
                    f"Expected format: 'YYYY-MM-DD' or 'MM-DD' with year {year}."
                ) from e

        day_of_year = dt.timetuple().tm_yday - 1  # 0-based
        if end_of_day:
            return clamp((day_of_year + 1) * 24 - 1, 0, 8760)
        return clamp(day_of_year * 24, 0, 8760)

    def _filter_time_series(self, 
                          ts: dict[str, pd.DataFrame],
                          start: str | None,
                          end: str | None) -> dict[str, pd.DataFrame]:
        """Filter time series using hour indices"""
        start_hour = self._date_to_hour(start) if start else 0
        end_hour = self._date_to_hour(end, True) + 1 if end else 8760
        
        # Ensure valid range
        start_hour = clamp(start_hour, 0, 8760)
        end_hour = clamp(end_hour, start_hour, 8760)
        
        return {k: v.iloc[start_hour:end_hour] for k, v in ts.items()}

    def _compute_features(self) -> dict[int, dict[str, Any]]:
        """Compute user requested features for each node of the network"""
        features = {}
        active_features = self.config.data_preproc.active_features
        
        # Convert to numpy arrays for numba
        ts_arrays = {
            name: df.to_numpy().T  # Transpose for node-first access
            for name, df in self.time_series.items()
        }

        for node_idx in prange(len(self.nodes_df)):
            node_data = self.nodes_df.iloc[node_idx]
            features[node_idx] = {}
            
            if 'position' in active_features:
                features[node_idx]['position'] = (node_data['Lat'], node_data['Lon'])
            
            ts_data = {
                name: arr[node_idx] 
                for name, arr in ts_arrays.items()
            }
            
            if 'time_series' in active_features:
                features[node_idx]['time_series'] = ts_data
                
            if 'duration_curves' in active_features:
                features[node_idx]['duration_curves'] = {k: np.sort(values)[::-1] for k, values in ts_data.items()}
                
            if 'ramp_duration_curves' in active_features:
                features[node_idx]['ramp_duration_curves'] = {k: np.sort(np.abs(np.diff(values)))[::-1] for k, values in ts_data.items()}
                
            if 'intra_correlation' in active_features:
                pairs = self._get_all_pairs(list(ts_data.keys()))
                features[node_idx]['intra_correlation'] = self.calculate_correlations(ts_data, pairs)

        return features

    def _get_all_pairs(self, keys: list[str]) -> list[tuple[str, str]]:
        """Generate all unique pairs of time series"""
        return [(k1, k2) for i, k1 in enumerate(keys) for k2 in keys[i+1:]]
    
    def calculate_correlations(self, ts_data: dict[str, np.ndarray], pairs: tuple[tuple[str, str]]) -> dict[tuple[str, str], float]:
        """Correlation calculation for specified pairs"""
        corr_results = {}
        for i in range(len(pairs)):
            key1, key2 = pairs[i]
            ts1 = ts_data[key1]
            ts2 = ts_data[key2]
            if np.std(ts1) == 0 or np.std(ts2) == 0:
                print(f"Warning: Zero variance in time series for keys {key1} or {key2}. Skipping correlation.")
                corr_results[(key1, key2)] = np.nan
                continue
            corr = np.corrcoef(ts1, ts2)[0, 1]
            corr_results[(key1, key2)] = corr
        return corr_results

    def help(self):
        """
        Print descriptions of the features computed in the Network class.
        """
        feature_descriptions = {
            'position': "Tuple (Latitude, Longitude) of each node.",
            'time_series': "Raw time series data for each node.",
            'duration_curves': "Sorted time series values in descending order to create duration curves.",
            'ramp_duration_curves': "Sorted absolute differences of time series values to create ramp duration curves.",
            'intra_correlation': "Correlation coefficients between pairs of time series at each node."
        }
        print("Available Features:")
        for feature, description in feature_descriptions.items():
            print(f"- {feature}: {description}")
        
        print("Selected Features:")
        for feature in self.config.data_preproc.active_features:
            print(f"- {feature}")



class Results():
    """Results processor for joint aggregation results."""
    def __init__(self, config: Config, data: dict, spatial_agg_results: dict, temporal_agg_results: dict, auto_save: bool = True):
        self.config = config
        self.data = data
        self.spatial_agg_results = spatial_agg_results
        self.temporal_agg_results = temporal_agg_results
        self.base_path = Path(config.path.joint_aggregation_results)

        self.results = self._process_results()
        if auto_save:
            self._save_results(self.results)

    def _process_results(self) -> dict[str, dict]:
        """Process results into two batches: spatiotemporal and temporal-only."""
        spatiotemporal = self._get_spatiotemporal_results()
        temporal_only = self._get_temporal_only_results()
        original = {}
        original["nodes"], original["branches"] = temporal_only["nodes"], temporal_only["branches"]
        original["time_series"] = self.data["time_series"]
        
        return {
            'spatiotemporal': spatiotemporal,
            'temporal_only': temporal_only,
            'original': original
        }
    
    def _get_temporal_only_results(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """
        Process results into Dataframes ready for CEP optimization.
        """
        nodes_df = self.data['nodes'].copy()
        nodes_df = nodes_df.reset_index(drop=True)
        nodes_df['bus_id'] = nodes_df.index
        clusters = self.spatial_agg_results["clusters"]
        node_to_rep = {node_id: rep_id for rep_id, members in clusters.items() for node_id in members}
        nodes_df['representative_id'] = nodes_df.index.map(node_to_rep)

        branches_df = pd.DataFrame([
            {
                "branch_id": idx,
                "from_bus_id": from_bus_id,
                "to_bus_id": to_bus_id,
            }
            for idx, (from_bus_id, to_bus_id) in enumerate(
                [(nodes_df.iloc[i]["bus_id"], nodes_df.iloc[j]["bus_id"])
                for i in range(len(nodes_df)) for j in range(len(nodes_df))
                if i != j]
            )
        ])

        agg_ts = {}
        for ts_name, ts_df in self.data['time_series'].items():
            agg_ts[ts_name] = self._filter_temporally_time_series(ts_df)

        return {
            'nodes': nodes_df,
            'branches': branches_df,
            'time_series': agg_ts
            }

    
    def _get_spatiotemporal_results(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """
        Process results into Dataframes ready for CEP optimization.
        """
        nodes_df = self.data['nodes'].copy()
        representative_bus_ids = self.spatial_agg_results["representatives"]
        agg_nodes_df = nodes_df.loc[representative_bus_ids].copy()
        agg_nodes_df["bus_id"] = representative_bus_ids
        agg_nodes_df = agg_nodes_df.reset_index(drop=True)

        agg_branches_df = pd.DataFrame([
            {
                "branch_id": idx,
                "from_bus_id": from_bus_id,
                "to_bus_id": to_bus_id,
            }
            for idx, (from_bus_id, to_bus_id) in enumerate(
                [(agg_nodes_df.iloc[i]["bus_id"], agg_nodes_df.iloc[j]["bus_id"])
                for i in range(len(agg_nodes_df)) for j in range(len(agg_nodes_df))
                if i != j]
            )
        ])

        agg_ts = {}
        for ts_name, ts_df in self.data['time_series'].items():
            if ts_name == 'demand':
                agg_ts[ts_name] = self._aggregate_spatiotemp_demand_time_series(ts_df)
            else:
                agg_ts[ts_name] = self._filter_spatiotemp_representative_time_series(ts_df)


        return {
            'nodes': agg_nodes_df,
            'branches': agg_branches_df,
            'time_series': agg_ts
            }
    
    def _aggregate_spatiotemp_demand_time_series(self, ts: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate demand time series by summing all nodes in each cluster and then
        filter to representative days.
        """
        clusters = self.spatial_agg_results["clusters"]
        column_type = type(ts.columns[0])
        converted_clusters = {
            column_type(rep): [column_type(node) for node in nodes]
            for rep, nodes in clusters.items()
        }
        agg_demand = pd.DataFrame()
        for rep, nodes in converted_clusters.items():
            agg_demand[rep] = ts[nodes].sum(axis=1)

        return self._filter_temporally_time_series(agg_demand)

    def _filter_spatiotemp_representative_time_series(self, ts: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the time series DataFrame to include only the representative nodes
        and the rows corresponding to the representative days.
        """
        column_type = type(ts.columns[0])
        representatives = [column_type(rep) for rep in self.spatial_agg_results["representatives"]]
        representative_ts = ts[representatives]
        
        return self._filter_temporally_time_series(representative_ts)
    
    def _filter_temporally_time_series(self, ts: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the time series DataFrame to include only the rows corresponding 
        to the representative days.
        """
        slices = [
            ts.iloc[day * 24 : (day + 1) * 24]
            for day in self.temporal_agg_results['representatives']
        ]
        
        return pd.concat(slices, axis=0)
    
    def _save_results(self, results: dict[str, dict]):
        """
        Save results to CSV files ready for a CEP optimization problem.
        """
        config_dict = {
            "data_preproc" : asdict(self.config.data_preproc),
            "model_hyper": self.config.model_hyper.__dict__
        }
        version_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]            
        version_path = self.base_path / f'v{version_hash}'
        version_path.mkdir(parents=True, exist_ok=True)
        for results_name, dict in results.items():
            results_path = version_path / results_name
            results_path.mkdir(parents=True, exist_ok=True)
            for name, df in dict.items():
                if isinstance(df, pd.DataFrame):
                    df.to_csv(results_path / f"{name}.csv", index=False)
                else:
                    for sub_name, sub_df in df.items():
                        sub_df.to_csv(results_path / f"{name}_{sub_name}.csv", index=False)

        metadata = {
            'created': datetime.now().isoformat(),
            'config_dict': config_dict
        }
        with open(version_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        print(f"Results saved to {version_path}")






# Deprecated ---------------------------------------------------------------------

@njit(parallel=True)
def knn_interpolation(node_coords, old_points, values, k):
    """
    Perform k-NN interpolation using prange for parallel processing.
    """
    n_nodes = node_coords.shape[0]
    interpolated = np.zeros((n_nodes, values.shape[0]), dtype=np.float64)

    distances = haversine_matrix(node_coords, old_points)

    for i in prange(n_nodes):
        nearest_indices = np.argpartition(distances[i], k)[:k]
        nearest_distances = distances[i][nearest_indices]
        nearest_values = values[:, nearest_indices]

        weights = 1 / nearest_distances
        weights /= weights.sum()

        interpolated[i] = np.dot(nearest_values, weights)

    return interpolated.T


# @njit(parallel=True)
# def exponential_decay_interpolation(node_coords, old_points, values, alpha):
#     """
#     Perform exponential decay interpolation using prange for parallel processing.
#     """
#     n_nodes = node_coords.shape[0]
#     interpolated = np.zeros((n_nodes, values.shape[0]), dtype=np.float64)

#     distances = haversine_matrix(node_coords, old_points)

#     for i in prange(n_nodes):
#         weights = np.exp(-alpha * distances[i])
#         weights /= weights.sum()

#         interpolated[i] = np.dot(values, weights)

#     return interpolated.T


@njit(parallel=True)
def kernel_interpolation(node_coords, old_points, values, alpha):
    """
    Perform mass-conserving kernel interpolation using prange for parallel processing.
    Ensures total interpolated mass equals total original mass at each time step.
    
    Parameters:
    - node_coords: array of shape (n_nodes, 2)
    - old_points: array of shape (n_old, 2)
    - values: array of shape (n_times, n_old)
    - alpha: float, decay parameter
    
    Returns:
    - interpolated: array of shape (n_times, n_nodes)
    """
    n_nodes = node_coords.shape[0]
    n_old = old_points.shape[0]
    n_times = values.shape[0]

    distances = haversine_matrix(node_coords, old_points)

    # Compute kernel weights K[i,j] = exp(-alpha * d_ij)
    K = np.empty((n_nodes, n_old), dtype=np.float64)
    for i in prange(n_nodes):
        for j in range(n_old):
            K[i, j] = np.exp(-alpha * distances[i, j])

    # Column-normalize to preserve mass: sum_i w_ij = 1
    col_sums = np.zeros(n_old, dtype=np.float64)
    for j in range(n_old):
        s = 0.0
        for i in range(n_nodes):
            s += K[i, j]
        col_sums[j] = s if s > 0.0 else 1.0

    # Interpolate: \tilde{x}_i(t) = sum_j (K[i,j]/col_sums[j]) * values[t,j]
    interpolated = np.zeros((n_nodes, n_times), dtype=np.float64)
    for i in prange(n_nodes):
        for t in range(n_times):
            s = 0.0
            for j in range(n_old):
                s += (K[i, j] / col_sums[j]) * values[t, j]
            interpolated[i, t] = s

    return interpolated.T


class HighResDataProcessor():
    """High resolution processor with configurable interpolation methods"""
    
    class StaticProcessor:
        """Handles static data that never changes across configurations"""
        @staticmethod
        @lru_cache(maxsize=None)
        def load_topology(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            """Load nodes, branches, and demand coordinates"""  
            nodes = pd.read_csv(config.path.node_file)
            branches = pd.read_csv(config.path.branch_file)
            demand_locs = pd.read_csv(config.path.demand_lat_lon_file)

            branches = HighResDataProcessor.StaticProcessor._change_units(nodes, branches)
            branches = branches[['branch_id','from_bus_id','to_bus_id','rateA','b']].rename(columns={'rateA': 'max_flow'})
            
            nodes = nodes[['bus_id', 'Lat', 'Lon']]
            nodes['Lon'] = -nodes['Lon']  # Coordinate adjustment for New-England
            deduplicated_nodes = nodes.groupby(['Lat', 'Lon']).agg(list).reset_index()

            demand_locs['Lon'] = -demand_locs['Lon']  # Coordinate adjustment for New-England
            
            return deduplicated_nodes, branches, demand_locs[['Lat', 'Lon']]
        
        @staticmethod
        def _change_units(bus: pd.DataFrame, branch: pd.DataFrame) -> pd.DataFrame:
            """Convert units of a DataFrame column"""
            S_base = 100e6  # in VA

            # Map base voltages
            baseKV_map = bus.set_index("bus_id")["baseKV"]
            branch["V_from_kV"] = branch["from_bus_id"].map(baseKV_map)
            branch["V_to_kV"] = branch["to_bus_id"].map(baseKV_map)

            # Separate transformers and lines
            is_transformer = branch["V_from_kV"] != branch["V_to_kV"]
            transformers = branch[is_transformer].copy()
            lines = branch[~is_transformer].copy()

            # Convert b for transmission lines only
            V_base_volts = lines["V_from_kV"] * 1e3
            lines["b"] = lines["b"] * (S_base / (V_base_volts ** 2))

            # Recombine and clean
            converted = pd.concat([lines, transformers], axis=0).sort_index()
            converted.drop(columns=["V_from_kV", "V_to_kV"], inplace=True)

            print(f"Converted susceptance of {len(lines)} transmission lines. Skipped {len(transformers)} transformers.")
            return converted

        @staticmethod
        @lru_cache(maxsize=None)
        def load_raw_capacity_factors(config: Config) -> tuple[xr.DataArray, xr.DataArray]:
            """Load raw capacity factors"""
            wind = xr.open_dataset(config.path.wind_cf_file)['cf']
            solar = xr.open_dataset(config.path.solar_cf_file)['cf']
            
            return wind.stack(z=("lat", "lon")).dropna('z', how='all'), solar.stack(z=("lat", "lon")).dropna('z', how='all')
        
        @staticmethod
        @lru_cache(maxsize=None)
        def load_raw_demand(config: Config) -> pd.DataFrame:
            """Load raw demand data"""
            return pd.read_csv(config.path.demand_file).iloc[:, 1:] * 1000 # Convert from GW to MW

    def __init__(self, config: Config):
        
        self.config = config

        self.nodes_df, self.branches_df, self.demand_points_df = HighResDataProcessor.StaticProcessor.load_topology(self.config)
        self.raw_wind, self.raw_solar = HighResDataProcessor.StaticProcessor.load_raw_capacity_factors(self.config)
        self.raw_demand_df = HighResDataProcessor.StaticProcessor.load_raw_demand(self.config)

        self.node_coords = self.nodes_df[['Lat', 'Lon']].values

    def _process_capacity_factors(self, cf_type: str) -> pd.DataFrame:
        """Process capacity factors with k-NN linear interpolation""" 
        cf = getattr(self, f"raw_{cf_type}")
        cf_points = np.column_stack((cf.lat.values, cf.lon.values))
        cf_values = cf.values.astype(np.float64) 
        k = self.config.data_preproc.cf_k_neighbors
        interpolated_cf = knn_interpolation(self.node_coords, cf_points, cf_values, k)
            
        return pd.DataFrame(interpolated_cf, columns=range(interpolated_cf.shape[1]))

    def _process_demand(self) -> pd.DataFrame:
        """Exponential decay demand interpolation"""

        demand_df = self.raw_demand_df
        alpha = self.config.data_preproc.demand_decay_alpha
        interpolated_demand = kernel_interpolation(self.node_coords, self.demand_points_df.values, demand_df.values, alpha)
        
        return pd.DataFrame(interpolated_demand, columns=range(interpolated_demand.shape[1]))

    def interpolate(self) -> dict[str, pd.DataFrame]:
        """Process all data types"""
        return {
            'wind': self._process_capacity_factors('wind'),
            'solar': self._process_capacity_factors('solar'),
            'demand': self._process_demand()
        }
    
    def process(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """Process and return all data types"""
        return {
            'nodes': self.nodes_df,
            'branches': self.branches_df,
            'time_series': self.interpolate()
        }
    

# class LowResDataProcessor():
#     """Low resolution data processor with configurable interpolation methods"""

#     def __init__(self, config: Config): 
#         self.config = config
#         self.nodes_df, self.branches_df, self.wind_df, self.solar_df, self.demand_df = self.load_data(self.config)

#     @staticmethod
#     @lru_cache(maxsize=None)
#     def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#         """Load nodes and branches coordinates, and wind, solar and demand time series"""
#         nodes = pd.read_csv(config.path.node_file)
#         nodes = nodes[['node_num', 'Lat', 'Lon']].rename(columns={'node_num': 'bus_id'})
#         branches = pd.read_csv(config.path.branch_file)
#         branches = branches[['line_num','from_node','to_node','maxFlow','susceptance']].rename(columns={'line_num': 'branch_id', 'from_node': 'from_bus_id', 'to_node': 'to_bus_id', 'maxFlow': 'max_flow', 'susceptance': 'b'})
#         wind = pd.read_csv(config.path.wind_cf_file).iloc[:, 1:]
#         solar = pd.read_csv(config.path.solar_cf_file).iloc[:, 1:]
#         demand = pd.read_csv(config.path.demand_file).iloc[:, 1:] * 1000 # Convert from GW to MW
#         wind.columns = range(wind.shape[1])
#         solar.columns = range(solar.shape[1])
#         demand.columns = range(demand.shape[1])
        
#         return nodes, branches, wind, solar, demand
    
#     def process(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
#         """Process and return all data types"""
#         return {
#             'nodes': self.nodes_df,
#             'branches': self.branches_df,
#             'time_series': {
#                 'wind': self.wind_df,
#                 'solar': self.solar_df,
#                 'demand': self.demand_df
#             }
#         }
    
class HighResDataProcessorBis():
    """High resolution processor with configurable interpolation methods"""
    
    class StaticProcessor:
        """Handles static data that never changes across configurations"""
        @staticmethod
        @lru_cache(maxsize=None)
        def load_topology(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            """Load nodes, branches, and demand coordinates"""  
            nodes = pd.read_csv(config.path.node_file)
            branches = pd.read_csv(config.path.branch_file)
            demand_locs = pd.read_csv(config.path.demand_lat_lon_file)

            branches = HighResDataProcessor.StaticProcessor._change_units(nodes, branches)
            branches = branches[['branch_id','from_bus_id','to_bus_id','rateA','b']].rename(columns={'rateA': 'max_flow'})
            
            nodes = nodes[['bus_id', 'Lat', 'Lon']]
            nodes['Lon'] = -nodes['Lon']  # Coordinate adjustment for New-England
            deduplicated_nodes = nodes.groupby(['Lat', 'Lon']).agg(list).reset_index()

            demand_locs['Lon'] = -demand_locs['Lon']  # Coordinate adjustment for New-England
            
            return deduplicated_nodes, branches, demand_locs[['Lat', 'Lon']]
        
        @staticmethod
        def _change_units(bus: pd.DataFrame, branch: pd.DataFrame) -> pd.DataFrame:
            """Convert units of a DataFrame column"""
            S_base = 100e6  # in VA

            # Map base voltages
            baseKV_map = bus.set_index("bus_id")["baseKV"]
            branch["V_from_kV"] = branch["from_bus_id"].map(baseKV_map)
            branch["V_to_kV"] = branch["to_bus_id"].map(baseKV_map)

            # Separate transformers and lines
            is_transformer = branch["V_from_kV"] != branch["V_to_kV"]
            transformers = branch[is_transformer].copy()
            lines = branch[~is_transformer].copy()

            # Convert b for transmission lines only
            V_base_volts = lines["V_from_kV"] * 1e3
            lines["b"] = lines["b"] * (S_base / (V_base_volts ** 2))

            # Recombine and clean
            converted = pd.concat([lines, transformers], axis=0).sort_index()
            converted.drop(columns=["V_from_kV", "V_to_kV"], inplace=True)

            print(f"Converted susceptance of {len(lines)} transmission lines. Skipped {len(transformers)} transformers.")
            return converted

        @staticmethod
        @lru_cache(maxsize=None)
        def load_raw_capacity_factors(config: Config) -> tuple[xr.DataArray, xr.DataArray]:
            """Load raw capacity factors"""
            wind = xr.open_dataset(config.path.wind_cf_file)['cf']
            solar = xr.open_dataset(config.path.solar_cf_file)['cf']
            
            return wind.stack(z=("lat", "lon")).dropna('z', how='all'), solar.stack(z=("lat", "lon")).dropna('z', how='all')
        
        @staticmethod
        @lru_cache(maxsize=None)
        def load_raw_demand(config: Config) -> pd.DataFrame:
            """Load raw demand data"""
            return pd.read_csv(config.path.demand_file).iloc[:, 1:] * 1000 # Convert from GW to MW

    def __init__(self, config: Config):
        
        self.config = config
        self.nodes_df, self.branches_df, self.demand_points_df = HighResDataProcessor.StaticProcessor.load_topology(self.config)
        self.raw_wind, self.raw_solar = HighResDataProcessor.StaticProcessor.load_raw_capacity_factors(self.config)
        self.raw_demand_df = HighResDataProcessor.StaticProcessor.load_raw_demand(self.config)

        self.node_coords = self.nodes_df[['Lat', 'Lon']].values

    def _process_capacity_factors(self, cf_type: str) -> pd.DataFrame:
        """Process capacity factors with k-NN linear interpolation""" 
        cf = getattr(self, f"raw_{cf_type}")
        cf_points = np.column_stack((cf.lat.values, cf.lon.values))
        cf_values = cf.values.astype(np.float64) 
        if cf_type == 'wind':
            cf_values = np.delete(cf_values, 181, axis=1)
            
        return pd.DataFrame(cf_points, columns=["Lat", "Lon"]), pd.DataFrame(cf_values, columns=range(cf_values.shape[1]))

    def _process_demand(self, nodes_df) -> pd.DataFrame:
        """Exponential decay demand interpolation"""

        demand_df = self.raw_demand_df
        alpha = self.config.data_preproc.demand_decay_alpha
        interpolated_demand = kernel_interpolation(nodes_df.values, self.demand_points_df.values, demand_df.values, alpha)
        
        return pd.DataFrame(interpolated_demand, columns=range(interpolated_demand.shape[1]))

    def process(self) -> dict[str, pd.DataFrame]:
        """Process all data types"""
        _, wind_df = self._process_capacity_factors('wind')
        # wind_df.drop(columns=181, inplace=True)
        nodes_df, solar_df = self._process_capacity_factors('solar')
        nodes_df.reset_index(inplace=True)
        nodes_df.rename(columns={"index": "bus_id"}, inplace=True)  
        return {
            'nodes': nodes_df,
            'branches': None,
            'time_series':{
                'wind': wind_df,
                'solar': solar_df,
                'demand': self._process_demand(nodes_df[['Lat', 'Lon']])
                }
            }
    

class LowResDataProcessor():
    """Low resolution data processor with configurable interpolation methods"""

    def __init__(self, config: Config): 
        self.config = config
        self.nodes_df, self.branches_df, self.wind_df, self.solar_df, self.demand_df = self.load_data(self.config)

    @staticmethod
    @lru_cache(maxsize=None)
    def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load nodes and branches coordinates, and wind, solar and demand time series"""
        nodes = pd.read_csv(config.path.node_file)
        nodes = nodes[['node_num', 'Lat', 'Lon']].rename(columns={'node_num': 'bus_id'})
        branches = pd.read_csv(config.path.branch_file)
        branches = branches[['line_num','from_node','to_node','maxFlow','susceptance']].rename(columns={'line_num': 'branch_id', 'from_node': 'from_bus_id', 'to_node': 'to_bus_id', 'maxFlow': 'max_flow', 'susceptance': 'b'})
        wind = pd.read_csv(config.path.wind_cf_file).iloc[:, 1:]
        solar = pd.read_csv(config.path.solar_cf_file).iloc[:, 1:]
        demand = pd.read_csv(config.path.demand_file).iloc[:, 1:] * 1000 # Convert from GW to MW
        wind.columns = range(wind.shape[1])
        solar.columns = range(solar.shape[1])
        demand.columns = range(demand.shape[1])
        
        return nodes, branches, wind, solar, demand
    
    def process(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """Process and return all data types"""
        return {
            'nodes': self.nodes_df,
            'branches': self.branches_df,
            'time_series': {
                'wind': self.wind_df,
                'solar': self.solar_df,
                'demand': self.demand_df
            }
        }


class DataProcessor:
    """
    A unified data processor that automatically selects the correct underlying processor
    based on config.granularity ("low" or "high"). It then either loads processed
    data (if it exists) or processes raw data and saves it.
    
    Usage:
      cfg = Config(**kwargs)
      processor = DataProcessor(cfg)
      data = processor.process()
    """
    def __init__(self, config: Config):
        self.config = config
        self.config_dict = asdict(self.config.data_preproc)
        self.config_dict.pop('active_features', None)
        self._processed_data: dict[str, pd.DataFrame | dict[str, pd.DataFrame]] | None = None
        if self.config.data_preproc.granularity == "high":
            self._processor = HighResDataProcessor(config)
        elif self.config.data_preproc.granularity == "low":
            self._processor = LowResDataProcessor(config)
        elif self.config.data_preproc.granularity == "high_bis":
            self._processor = HighResDataProcessorBis(config)

    def save_data(self, data: dict[str, pd.DataFrame | dict[str, pd.DataFrame]], version_suffix: str = ""):
        """
        Save processed data with versioning and static data separation. 
        Input is a dictionary with keys 'nodes', 'branches', and 'time_series'.
        The 'time_series' value is a dictionary with keys for each time series type.
        """

        version_hash = hashlib.md5(json.dumps(self.config_dict, sort_keys=True).encode()).hexdigest()[:8]
        
        base_path = Path(self.config.path.processed)
        
        # Save static data once
        base_path.mkdir(parents=True, exist_ok=True)
        if not (base_path / 'nodes.parquet').exists():
            data['nodes'].to_parquet(base_path / 'nodes.parquet')
        if not (base_path / 'branches.parquet').exists():
            data['branches'].to_parquet(base_path / 'branches.parquet')
            
        # Save versioned data
        version_path = base_path / f'v{version_hash}{version_suffix}'
        version_path.mkdir(exist_ok=True)
        
        for name, df in data['time_series'].items():
            df.to_parquet(
                version_path / f'{name}.parquet',
                index=False,
                compression='snappy'
            )

        metadata = {
            'created': datetime.now().isoformat(),
            'config_dict': self.config_dict
        }
        with open(version_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
        
        print(f"Processed data saved to {version_path}")
    
    def load_processed_data(self, version_hash: str | None = None) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """
        Load processed data. If version_hash is not provided, compute it from the config.
        """
        base_path = Path(self.config.path.processed)

        if version_hash is None:
            version_hash = hashlib.md5(json.dumps(self.config_dict, sort_keys=True).encode()).hexdigest()[:8]

        data = {}
        data['nodes'] = pd.read_parquet(base_path / 'nodes.parquet')
        data['branches'] = pd.read_parquet(base_path / 'branches.parquet')

        version_path = base_path / f"v{version_hash}"
        data['time_series'] = {
            'wind': pd.read_parquet(version_path / "wind.parquet"),
            'solar': pd.read_parquet(version_path / "solar.parquet"),
            'demand': pd.read_parquet(version_path / "demand.parquet")
        }

        print(f"Cached processed data loaded from {version_path}")
        return data
    
    @property
    def processed_data(self) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
        """Lazy-loaded processed data"""
        if self._processed_data is None:
            # try:
            #     self._processed_data = self.load_processed_data()
            # except FileNotFoundError:
            self._processed_data = self._processor.process()
            # self.save_data(self._processed_data)
        return self._processed_data



# class Results():
#     def __init__(self, config: Config, data: dict, spatial_agg_results: dict, temporal_agg_results: dict, auto_save: bool = True):
#         self.config = config
#         self.data = data
#         self.spatial_agg_results = spatial_agg_results
#         self.temporal_agg_results = temporal_agg_results
#         self.granularity = config.data_preproc.granularity
#         self.base_path = Path(config.path.joint_aggregation_results)

#         self.cluster_mappings = self._precompute_cluster_mappings()
#         self.results = self._process_results(self.cluster_mappings)
#         if auto_save:
#             self._save_results(self.results)

#     def _precompute_cluster_mappings(self):
#         """Precompute cluster mappings for bus IDs to representatives"""
#         bus_to_rep = {}
#         nodes_df = self.data['nodes']

#         for rep, members in self.spatial_agg_results["clusters"].items():
#             if self.granularity == "high":
#                 for member_idx in members:
#                     bus_ids = nodes_df.iloc[member_idx]["bus_id"]
#                     for bus_id in bus_ids:
#                         bus_to_rep[bus_id] = rep
#             elif self.granularity == "low":
#                 for bus_id in members:
#                     bus_to_rep[bus_id] = rep
#             elif self.granularity == "high_bis":
#                 for bus_id in members:
#                     bus_to_rep[bus_id] = rep

#         return bus_to_rep
    
#     def _process_results(self, cluster_mappings: dict[int, int]) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame]]:
#         """
#         Process results into Dataframes ready for CEP optimization.
#         """
#         nodes_df = self.data['nodes']
#         representative_bus_ids = self.spatial_agg_results["representatives"]
#         agg_nodes_df = nodes_df.loc[representative_bus_ids].copy()
#         agg_nodes_df["bus_id"] = representative_bus_ids
#         agg_nodes_df = agg_nodes_df.reset_index(drop=True)

#         if self.granularity == "high_bis":
#             default_max_flow = 1000
#             agg_branches_df = pd.DataFrame([
#                 {
#                     "branch_id": idx,
#                     "from_bus_id": from_bus_id,
#                     "to_bus_id": to_bus_id,
#                     "b": 0,  # Set b to 0
#                     "max_flow": default_max_flow
#                 }
#                 for idx, (from_bus_id, to_bus_id) in enumerate(
#                     [(agg_nodes_df.iloc[i]["bus_id"], agg_nodes_df.iloc[j]["bus_id"])
#                     for i in range(len(agg_nodes_df)) for j in range(len(agg_nodes_df))
#                     if i != j]
#                 )
#             ])
            
#         else:
#             branches_df = self.data['branches']
#             agg_branches = {}
            
#             for row in branches_df.itertuples(index=False):
#                 from_bus_id = row.from_bus_id
#                 to_bus_id = row.to_bus_id
#                 from_rep = cluster_mappings[from_bus_id]
#                 to_rep = cluster_mappings[to_bus_id]

#                 if from_rep != to_rep:
#                     rep_pair = tuple((from_rep, to_rep))
#                     branch_data = agg_branches.setdefault(rep_pair, {"b": 0, "max_flow": 0})
#                     branch_data["b"] += row.b
#                     branch_data["max_flow"] += row.max_flow

#             agg_branches_df = pd.DataFrame([
#                 {
#                     "branch_id": idx, 
#                     "from_bus_id": from_bus_id, 
#                     "to_bus_id": to_bus_id, 
#                     "b": values["b"], 
#                     "max_flow": values["max_flow"]
#                 }
#                 for idx, ((from_bus_id, to_bus_id), values) in enumerate(sorted(agg_branches.items()))
#             ])

#         agg_ts = {
#             ts_name: self._filter_representative_time_series(ts_df)
#             for ts_name, ts_df in self.data['time_series'].items()
#         }

#         return {
#             'nodes': agg_nodes_df,
#             'branches': agg_branches_df,
#             'time_series': agg_ts
#             }

#     def _filter_representative_time_series(self, ts: pd.DataFrame) -> pd.DataFrame:
#         """
#         Filter the time series DataFrame to include only the representative nodes
#         and the rows corresponding to the representative days.
#         """
#         column_type = type(ts.columns[0])
#         representatives = [column_type(rep) for rep in self.spatial_agg_results["representatives"]]
#         representative_ts = ts[representatives]

#         slices = [
#             representative_ts.iloc[day * 24 : (day + 1) * 24]
#             for day in self.temporal_agg_results['representatives']
#         ]
        
#         return pd.concat(slices, axis=0)
    
#     def _save_results(self, results: dict[str, pd.DataFrame | dict[str, pd.DataFrame]]):
#         """
#         Save results to CSV files ready for a CEP optimization problem.
#         """
#         config_dict = {
#             "data_preproc" : asdict(self.config.data_preproc),
#             "model_hyper": self.config.model_hyper.__dict__
#         }
#         version_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]            
#         version_path = self.base_path / f'v{version_hash}'
#         version_path.mkdir(parents=True, exist_ok=True)
#         for name, df in results.items():
#             if isinstance(df, pd.DataFrame):
#                 df.to_csv(version_path / f"{name}.csv", index=False)
#             else:
#                 for sub_name, sub_df in df.items():
#                     sub_df.to_csv(version_path / f"{name}_{sub_name}.csv", index=False)

#         metadata = {
#             'created': datetime.now().isoformat(),
#             'config_dict': config_dict
#         }
#         with open(version_path / 'metadata.json', 'w') as f:
#             json.dump(metadata, f)
        
#         print(f"Results saved to {version_path}")

