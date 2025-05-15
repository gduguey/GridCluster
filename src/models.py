# models.py

from pathlib import Path
from datetime import datetime
from functools import lru_cache
from dataclasses import dataclass, field, asdict
from typing import Dict
import hashlib
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from numba import njit, prange
from sklearn_extra.cluster import KMedoids

from .utils import haversine_matrix
from .settings import Config


@dataclass(frozen=True)
class DistanceMetrics:
    """Dynamic container based on active features"""
    features: Dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.features[key]

    def __contains__(self, key: str) -> bool:
        return key in self.features


class SpatialAggregation:
    """Main class for spatial aggregation with config-driven metrics"""

    class DistanceCalculator:
        """Handles distance metric calculations using config-defined features"""
        
        @classmethod
        def compute_metrics(cls, nodes: dict, config: Config) -> DistanceMetrics:
            """Compute metrics using preprocessed arrays for Numba compatibility"""
            
            print(f"Computing distance metrics for {len(nodes)} nodes. This might take a while...")
            t_start_total = datetime.now()
            active_features = config.data_preproc.active_features
            metrics = {}

            if 'position' in active_features:
                t_start = datetime.now()
                print("Computing position distance...")
                pos = np.array([n['position'] for n in nodes.values()])
                metrics['position'] = haversine_matrix(pos, pos)
                print(f"position distance computed in {datetime.now() - t_start}.")

            for feature in set(active_features) - {'position', 'intra_correlation'}:
                t_start = datetime.now()
                print(f"Computing {feature} distance...")
                arr = np.array([np.concatenate(list(n[feature].values())) for n in nodes.values()])
                metrics[feature] = cls._feature_distance(arr)
                print(f"{feature} distance computed in {datetime.now() - t_start}.")

            if 'intra_correlation' in active_features:
                t_start = datetime.now()
                print("Computing intra_correlation distance...")
                arr = np.array([list(n['intra_correlation'].values()) for n in nodes.values()])
                metrics['intra_correlation'] = cls._feature_distance(arr)
                print(f"intra_correlation distance computed in {datetime.now() - t_start}.")

            if config.model_hyper.inter_correlation:
                t_start = datetime.now()
                print("Computing inter_correlation distance...")
                keys = list(next(iter(nodes.values()))['time_series'].keys())
                ts1 = np.array([[n['time_series'][k] for k in keys] for n in nodes.values()])
                metrics['inter_correlation'] = cls._inter_correlation(ts1)
                print(f"inter_correlation distance computed in {datetime.now() - t_start}.")

            print("All distance metrics computed.")
            dist = DistanceMetrics(features=cls._normalize_metrics(metrics))
            print(f"Total computation time: {datetime.now() - t_start_total}.")
            return dist

        @staticmethod
        @njit(parallel=True)
        def _feature_distance(arr: np.ndarray) -> np.ndarray:
            """Generic Euclidean distance calculation"""
            dists = np.empty((arr.shape[0], arr.shape[0]))
            for i in prange(arr.shape[0]):
                for j in prange(i, arr.shape[0]):
                    dists[i,j] = np.sqrt(np.sum((arr[i] - arr[j])**2))
                    dists[j,i] = dists[i,j]
            return dists

        @staticmethod
        @njit(parallel=True)
        def _inter_correlation(ts: np.ndarray) -> np.ndarray:
            """Optimized correlation distance calculation"""
            n_nodes, n_keys, _ = ts.shape
            dist_matrix = np.zeros((n_nodes, n_nodes))

            for i in prange(n_nodes):
                for j in prange(i + 1, n_nodes):
                    total = 0.0
                    count = 0
                    for k1 in range(n_keys):
                        for k2 in range(n_keys):
                            s1 = ts[i, k1]
                            s2 = ts[j, k2]
                            s1_normalized = s1 / np.max(np.abs(s1))
                            s2_normalized = s2 / np.max(np.abs(s2))
                            if np.std(s1_normalized) > 1e-9 and np.std(s2_normalized) > 1e-9:
                                corr = np.corrcoef(s1_normalized, s2_normalized)[0, 1]
                                dist = np.exp(-corr) # Exponential decay for correlation
                                # dist = (1 - c)**2 # Square the distance
                                # dist = 0.5 * (1 - c)  # Normalize to [0, 1]
                                total += dist
                            else:
                                dist = np.exp(1.0)
                                total += dist
                                print(f"Warning: Zero std deviation for time serie {k1} at node {i} or {k2} at node {j}. Setting distance to {dist}.")
                            count += 1
                    dist_matrix[i, j] = total / count
                    dist_matrix[j, i] = dist_matrix[i, j]
            return dist_matrix

        @classmethod
        def _normalize_metrics(cls, metrics: dict) -> dict:
            """Normalize each metric to [0,1] range"""
            t_start = datetime.now()
            print("Starting normalization...")
            normalized = {}
            for name, values in metrics.items():
                vmin, vmax = values.min(), values.max()
                if abs(vmax - vmin) < 1e-9:
                    normalized[name] = np.zeros_like(values)
                    print(f"Warning: Metric '{name}' has no variation (min = max = {vmin}). Normalization is set to zeros.")
                else:
                    normalized[name] = (values - vmin) / (vmax - vmin)
            print(f"Normalization completed in {datetime.now() - t_start}.")
            return normalized
        
    
    class IOManager:
        """Handles metric storage with config-based versioning"""
        
        def __init__(self, config: Config):
            self.config = config
            self.base_path = Path(config.path.distance_metrics)
            self.config_data_preproc_dict = asdict(config.data_preproc)
            
        @lru_cache(maxsize=None)
        def get_metrics_path(self) -> Path:
            """Versioned path based on config hash"""
            hashable = {
                'config_data_dict': self.config_data_preproc_dict, 
                'inter_correlation': self.config.model_hyper.inter_correlation
                }
            config_hash = hashlib.md5(json.dumps(hashable, sort_keys=True).encode()).hexdigest()[:8]
            
            return self.base_path / f"v{config_hash}"
        
        def save_metrics(self, metrics: DistanceMetrics):
            """Save only active features"""
            print("Saving metrics...")
            save_path = self.get_metrics_path()
            save_path.mkdir(parents=True, exist_ok=True)
            
            np.savez(save_path / 'metrics.npz', **metrics.features)
            
            metadata = {
                'created': datetime.now().isoformat(),
                'inter_correlation': self.config.model_hyper.inter_correlation,
                'config_data_preproc_dict': self.config_data_preproc_dict
            }
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f)

            print(f"Metrics saved to {save_path}.")
        
        def load_metrics(self, file_name: str | None = None) -> DistanceMetrics:
            """Load metrics with feature validation"""
            if file_name is None:
                load_path = self.get_metrics_path()
            else:
                load_path = self.base_path / file_name
                if not load_path.exists():
                    raise FileNotFoundError(f"Metrics file '{file_name}' not found.")
                
            data = np.load(load_path / 'metrics.npz')
            
            with open(load_path / 'metadata.json') as f:
                metadata = json.load(f)
                
            if (metadata['config_data_preproc_dict'] != self.config_data_preproc_dict or
                metadata['inter_correlation'] != self.config.model_hyper.inter_correlation):
                print("Cached metrics don't match current config")
            
            print(f"Cached metrics loaded from {load_path}.")
                
            return DistanceMetrics(features=dict(data.items()))

    
    class Optimizer:
        """Handles optimization-based aggregation using Gurobi"""
        
        def __init__(self, config: Config):
            self.config = config
            self.model = gp.Model("node_aggregation")
            
        def solve(self, distance_matrix: np.ndarray) -> dict:
            """Solve the optimization problem"""
            num_nodes = distance_matrix.shape[0]
            n_repr = self.config.model_hyper.n_representative_nodes
            t_build_start = datetime.now()
            # Create optimization variables
            assign = self.model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="assign")
            reprs = self.model.addVars(num_nodes, vtype=GRB.BINARY, name="repre")
            
            # Set objective function
            obj = gp.quicksum(assign[i,j] * distance_matrix[i,j] for i in range(num_nodes) for j in range(num_nodes))
            self.model.setObjective(obj, GRB.MINIMIZE)
            
            # Add constraints
            self.model.addConstrs(
                (assign.sum(i,'*') == 1 for i in range(num_nodes)), 
                "assignment"
            )
            self.model.addConstrs(
                (assign[i,j] <= reprs[j] for i in range(num_nodes) for j in range(num_nodes)),
                "repr_assignment"
            )
            self.model.addConstr(reprs.sum() == n_repr, "num_reprs")
            
            t_build = datetime.now() - t_build_start

            t_solve_start = datetime.now()
            self.model.optimize()
            t_solve = datetime.now() - t_solve_start
            return self._process_solution(assign, reprs, num_nodes, t_build, t_solve)
        
        def _process_solution(self, assign, reprs, num_nodes: int, t_build, t_solve) -> dict:
            """Extract solution from Gurobi variables"""
            assignment = np.zeros((num_nodes, num_nodes))
            representatives = np.zeros(num_nodes)
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if assign[i,j].X > 0.5:
                        assignment[i,j] = 1
                        
            for j in range(num_nodes):
                if reprs[j].X > 0.5:
                    representatives[j] = 1

            representatives = np.where(representatives)[0].tolist()
            clusters = {j: np.where(assignment[:, j])[0].tolist() for j in representatives}
                    
            return {
                'clusters': clusters,
                'representatives': representatives,
                'metadata': {
                    'assignment_matrix': assignment,
                    'objective_value': self.model.objVal,
                    'build_time': t_build,
                    'solve_time': t_solve,
                    }
            }
    

    class Clusterer:
        """Improved k-medoids clustering with proper initialization handling"""
        
        def __init__(self, config: Config):
            self.config = config
            self.rng = np.random.default_rng(config.model_hyper.kmed_seed) if config.model_hyper.kmed_seed else np.random.default_rng()
            
        def cluster(self, distance_matrix: np.ndarray, k: int) -> dict:
            """
            Perform k-medoids clustering with:
            - Medoids guaranteed to be cluster members
            - No overlapping clusters
            - Multiple initializations
            """

            # Validate distance matrix
            if not self._is_valid_distance_matrix(distance_matrix):
                raise ValueError("Invalid distance matrix")

            best_inertia = float('inf')
            best_result = None

            for _ in range(self.config.model_hyper.kmed_n_init):
                random_state = self.rng.integers(0, 1e9)

                kmed = KMedoids(
                    n_clusters=k,
                    metric='precomputed',
                    method='pam',  # Partitioning Around Medoids (better than 'alternate')
                    init='heuristic',
                    max_iter=300,
                    random_state=random_state
                )
                kmed.fit(distance_matrix)

                if kmed.inertia_ < best_inertia:
                    best_inertia = kmed.inertia_
                    best_result = {
                        'clusters': self._build_clusters(kmed.labels_, kmed.medoid_indices_.tolist()),
                        'representatives': sorted(kmed.medoid_indices_.tolist()),
                        'metadata': {
                            'inertia': kmed.inertia_,
                            'n_iter': kmed.n_iter_,
                            'labels': kmed.labels_
                            }
                    }

            return best_result

        def _build_clusters(self, labels: np.ndarray, medoids: list) -> dict:
            """Ensure proper cluster structure"""
            clusters = {m: [] for m in sorted(medoids)}
            
            for idx, label in enumerate(labels):
                medoid = medoids[label]
                if idx == medoid:
                    clusters[medoid].insert(0, medoid)  # Medoid first in list
                else:
                    clusters[medoid].append(idx)
                    
            return clusters

        def _is_valid_distance_matrix(self, matrix: np.ndarray) -> bool:
            """Validate distance matrix properties"""
            if not matrix.shape[0] == matrix.shape[1]:
                print("Distance matrix isn't square.")
            if not (matrix.diagonal() < 1e-9).all():
                print("Distance matrix has non-zero diagonal.")
            if not (matrix >= 0).all():
                print("Distance matrix isn't non-negative.")
            if not np.allclose(matrix, matrix.T):
                print("Distance matrix isn't symmetric.")            
            return (
                matrix.shape[0] == matrix.shape[1] and  # Square matrix
                (matrix.diagonal() < 1e-9).all() and       # Zero diagonal
                (matrix >= 0).all() and                  # Non-negative
                np.allclose(matrix, matrix.T)             # Symmetric
            )
    
    class Evaluator:
        """Handles evaluation metrics calculation"""
        
        @staticmethod
        @njit
        def reeav(original: np.ndarray, aggregated: np.ndarray) -> float:
            """Relative Energy Error Average"""
            energy_orig = np.sum(original)
            energy_agg = np.sum(aggregated)
            return abs(energy_orig - energy_agg) / energy_orig
        
        @staticmethod
        @njit
        def nrmseav(original: np.ndarray, aggregated: np.ndarray) -> float:
            """Normalized Root Mean Square Error Average"""
            rmse = np.sqrt(np.mean((original - aggregated)**2))
            return rmse / (np.max(original) - np.min(original) + 1e-9)
        
        # @staticmethod
        # @njit #######################################
        # def ceav(original: np.ndarray, aggregated: np.ndarray) -> float:
        #     """Correlation Error Average."""

        #     orig_corr = original_features["correlation"][key]
        #     agg_corr = aggregated_features["correlation"][key]
        #     error = abs(orig_corr - agg_corr)
        #     return error
        
        @classmethod
        def evaluate(cls, original: dict, aggregated: dict) -> dict:
            """Calculate evaluation metrics for a node pair"""
            metrics = {}
            for key in original['duration_curves']:
                metrics['REEav'] += cls.reeav(
                    original['duration_curves'][key], 
                    aggregated['duration_curves'][key]
                )
                metrics['NRMSEav'] += cls.nrmseav(
                    original['duration_curves'][key],
                    aggregated['duration_curves'][key]
                )
                metrics['NRMSEavRDC'] += cls.nrmseav(
                    original['ramp_duration_curves'][key],
                    aggregated['ramp_duration_curves'][key]
                )
            count = len(original['duration_curves'])
            metrics['REEav'] /= count
            metrics['NRMSEav'] /= count
            metrics['NRMSEavRDC'] /= count
            return metrics
    
    
    def __init__(self, node_features: dict, config: Config):
        self.config = config
        self.node_features = node_features
        self._distance_metrics: DistanceMetrics | None = None
        self._io = self.IOManager(config)
        
    @property
    def distance_metrics(self) -> DistanceMetrics:
        """Lazy-loaded distance metrics"""
        if self._distance_metrics is None:
            # try:
            #     self._distance_metrics = self._io.load_metrics()
            # except FileNotFoundError:
            self._distance_metrics = self.DistanceCalculator.compute_metrics(self.node_features, self.config)
            self._io.save_metrics(self._distance_metrics)
        return self._distance_metrics
    
    def aggregate(self, method: str = 'kmedoids') -> dict:
        """Main aggregation method"""
        weights = self.config.model_hyper.weights
        total_matrix = self._combine_metrics(self.distance_metrics, weights)
        
        if method == 'optimization':
            result = self.Optimizer(self.config).solve(total_matrix)
        elif method == 'kmedoids':
            result = self.Clusterer(self.config).cluster(total_matrix, self.config.model_hyper.n_representative_nodes)
        else:
            raise ValueError(f"Invalid aggregation method: {method}. Choose 'optimization' or 'kmedoids'.")
        return result
    
    @staticmethod
    def _combine_metrics(distance_metrics: DistanceMetrics, weights: dict) -> np.ndarray:
        """Combine distance metrics using config-defined weights"""
        total = np.zeros_like(next(iter(distance_metrics.features.values())))
        weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        for feature in weights.keys():
            total += distance_metrics[feature] * weights[feature]
            
        return total / sum(weights.values())

    def evaluate_aggregation(self, aggregation: dict, metric_type: str = 'literature') -> dict:
        """Evaluate quality of aggregation"""
        all_metrics = []
        if metric_type == 'literature':
            for repr_id, members in aggregation.items():
                for member in members:
                    orig = self.node_features[member]
                    agg = self.node_features[repr_id]
                    all_metrics.append(self.Evaluator.evaluate(orig, agg))
            return {
                metric: np.mean([m[metric] for m in all_metrics])
                for metric in all_metrics[0].keys()
            }
        elif metric_type == 'custom':
            if "inertia" in aggregation['metadata']:
                eval = aggregation['metadata']['inertia']
            if "objective_value" in aggregation['metadata']:
                eval = aggregation['metadata']['objective_value']
            return {"eval": eval}
        
    def update_config(self, new_config: Config):
        """Update configuration and reset cached properties"""
        if new_config != self.config:
            self.config = new_config
            self._distance_metrics = None
            self._io = self.IOManager(new_config)



class TemporalAggregation:
    """Handles temporal aggregation post spatial aggregation"""
    def __init__(self, config: Config, node_features: dict, spatial_agg_results: dict):
        self.config = config
        self.nodes_features = node_features
        self.spatial_agg_results = spatial_agg_results

    def aggregate(self):
        """
        Aggregate the time series data of the nodes based on the spatial aggregation results.
        """
        sampled_nodes_time_serie = self._sampling(self.spatial_agg_results['clusters'], self.nodes_features)
        day_array = self._create_day_array(sampled_nodes_time_serie)
        distance_matrix = self._distance(day_array)
        return SpatialAggregation.Clusterer(self.config).cluster(distance_matrix, self.config.model_hyper.k_representative_days)
        
    @staticmethod
    def _sampling(aggregation: dict, nodes_features: dict) -> dict:
        """ Randomly sample nodes from each cluster """
        sampled_nodes_ts = {}
        for repr_id, members in aggregation.items():
            sampled_node = np.random.choice(members)
            sampled_nodes_ts[repr_id] = nodes_features[sampled_node]['time_series']
            for type, ts in sampled_nodes_ts[repr_id].items():
                mean_sample = ts.mean()
                means = [nodes_features[k]['time_series'][type].mean() for k in members]
                mean_target = np.mean(means)
                scale = (mean_target / mean_sample)
                ts_adj = ts * scale
                sampled_nodes_ts[repr_id][type] = ts_adj
        return sampled_nodes_ts

    @staticmethod
    def _create_day_array(nodes_time_serie_dict: dict) -> np.ndarray:
        """ 
        Create a 2D array that has:
            - one row for each day
            - one column for each (hour of the day, sampled node, ts type) combination 
        """
        nodes_ts_list = []
        total_hours = list(list(nodes_time_serie_dict.values())[0].values())[0].shape[0]
        number_of_days = total_hours // 24

        for _ , ts_dict in nodes_time_serie_dict.items():
            ts = np.array(list(ts_dict.values()))
            reshaped_ts = ts[:, :number_of_days * 24].reshape(ts.shape[0], number_of_days, 24)
            transposed_ts = reshaped_ts.transpose(1, 0, 2)
            flattened_days = transposed_ts.reshape(number_of_days, -1)
            nodes_ts_list.append(flattened_days)

        nodes_ts_array = np.array(nodes_ts_list)
        reshaped_data = nodes_ts_array.transpose(1, 0, 2)
        final_data = reshaped_data.reshape(number_of_days, -1)

        return final_data
    
    @staticmethod
    # @njit(parallel=True)
    def _distance(arr: np.ndarray) -> np.ndarray:
        """Generic Euclidean distance calculation"""
        dists = np.empty((arr.shape[0], arr.shape[0]))
        for i in range(arr.shape[0]):
            for j in range(i, arr.shape[0]):
                dists[i,j] = np.sqrt(np.sum((arr[i] - arr[j])**2))
                dists[j,i] = dists[i,j]
        return dists