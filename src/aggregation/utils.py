# utils.py — data I/O, fine/coarse network building, features, results export

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Any
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
    
    def _process_demand(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
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
        :param time_series: Dict of hourly DataFrames indexed 0..H-1 (H=8760 or 8784)
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
        """Convert date string to hour index (0-8760)"""
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
        eps = 1e-9
        
        # Convert to numpy arrays for numba
        ts_arrays = {
            name: df.to_numpy().T  # Transpose for node-first access
            for name, df in self.time_series.items()
        }

        for node_idx in range(len(self.nodes_df)):
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
                dc = {}
                for k, values in ts_data.items():
                    sorted_vals = np.sort(values)[::-1]
                    mn, mx = sorted_vals.min(), sorted_vals.max()
                    dc[k] = (sorted_vals - mn) / (mx - mn + eps)
                features[node_idx]['duration_curves'] = dc
                # features[node_idx]['duration_curves'] = {k: np.sort(values)[::-1] for k, values in ts_data.items()}
                
            if 'ramp_duration_curves' in active_features:
                rdc = {}
                for k, values in ts_data.items():
                    ramps = np.sort(np.abs(np.diff(values)))[::-1]
                    mn, mx = ramps.min(), ramps.max()
                    rdc[k] = (ramps - mn) / (mx - mn + eps)
                features[node_idx]['ramp_duration_curves'] = rdc
                # features[node_idx]['ramp_duration_curves'] = {k: np.sort(np.abs(np.diff(values)))[::-1] for k, values in ts_data.items()}
                
            if 'intra_correlation' in active_features:
                pairs = self._get_all_pairs(list(ts_data.keys()))
                features[node_idx]['intra_correlation'] = self.calculate_correlations(ts_data, pairs)

        return features

    def _get_all_pairs(self, keys: list[str]) -> list[tuple[str, str]]:
        """Generate all unique pairs of time series"""
        return [(k1, k2) for i, k1 in enumerate(keys) for k2 in keys[i+1:]]
    
    def calculate_correlations(self, ts_data: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
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
            'duration_curves': "Sorted normalized time series values in descending order to create duration curves.",
            'ramp_duration_curves': "Sorted absolute differences of normalized time series values to create ramp duration curves.",
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
            'original': original,
            'clusters': {
                'spatial': self.spatial_agg_results["clusters"],
                'temporal': self.temporal_agg_results["clusters"]
            }
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
                "from_bus_id": nodes_df.iloc[i]["bus_id"],
                "to_bus_id": nodes_df.iloc[j]["bus_id"],
            }
            for idx, (i, j) in enumerate(
                [(i, j)
                for i in range(len(nodes_df)) for j in range(i + 1, len(nodes_df))]
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
                "from_bus_id": agg_nodes_df.iloc[i]["bus_id"],
                "to_bus_id": agg_nodes_df.iloc[j]["bus_id"],
            }
            for idx, (i, j) in enumerate(
                [(i, j)
                for i in range(len(agg_nodes_df)) for j in range(i + 1, len(agg_nodes_df))]
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
            "model_hyper": {
                    **self.config.model_hyper.__dict__,
                    "weights": self.config.model_hyper.weights
                }
        }
        version_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]            
        version_path = self.base_path / f'v{version_hash}'
        version_path.mkdir(parents=True, exist_ok=True)
        for results_name, dict in results.items():
            if results_name == "clusters":
                # Handle clusters separately in JSON format
                cluster_path = version_path / "clustering"
                cluster_path.mkdir(parents=True, exist_ok=True)
                for cluster_type, cluster_dict in dict.items():
                    cluster_file = cluster_path / f"{cluster_type}_clusters.json"
                    with open(cluster_file, "w") as f:
                        json.dump(cluster_dict, f, indent=2)
                continue
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