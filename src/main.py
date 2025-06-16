import hashlib
import json
from dataclasses import asdict

from .settings import Config
from .utils import FineNetwork, CoarseNetwork, Network, Results
from .models import SpatialAggregation, TemporalAggregation


class StaticPreprocessor:
    """Handles network construction and feature preparation that doesn't change with hyperparameters"""
    def __init__(self, granularity: str, year: int = 2013, active_features: list = ['position', 'time_series', 'duration_curves', 'ramp_duration_curves', 'intra_correlation'], inter_correlation: bool = True):
        self.config = Config(
            year=year,
            granularity=granularity,
            active_features=active_features,
        )
        self.config.model_hyper.inter_correlation = inter_correlation

        self.network_data = None
        self.fine_data = None
        self.ntw = None

    def preprocess(self):
        """Run all static preprocessing steps"""
        # Build network based on granularity
        config = self.config
        if self.config.data_preproc.granularity == "coarse":
            fine_builder = FineNetwork(config)
            self.fine_data = fine_builder.build_fine_ntw()
            coarse_builder = CoarseNetwork(config, self.fine_data)
            self.network_data = coarse_builder.build_coarse_ntw()
        elif self.config.data_preproc.granularity == "fine":
            fine_builder = FineNetwork(config)
            self.fine_data = fine_builder.build_fine_ntw()
            self.network_data = self.fine_data
        else:
            raise ValueError("Unsupported granularity. Use 'fine' or 'coarse'.")
        
        self.ntw = Network(
            nodes_df=self.network_data["nodes"],
            time_series=self.network_data["time_series"],
            config=config
        )

        return self

class DynamicProcessor:
    """Handles parameter-dependent operations that can vary during grid search"""
    def __init__(self, preprocessor: StaticPreprocessor):
        self.preprocessor = preprocessor
        self.base_config = preprocessor.config
        self.ntw = preprocessor.ntw

    def run_with_hyperparameters(self, 
                               weights: dict,
                               n_representative_nodes: int,
                               k_representative_days: int) -> tuple[dict, str, dict, dict]:
        """Execute parameter-dependent pipeline steps"""
        ntw = self.ntw
        if ntw is None:
            raise ValueError("Network data not initialized. Run static preprocessing first.")
        
        # Update config with current hyperparameters
        current_config = self.base_config
        current_config.model_hyper.weights = weights
        current_config.model_hyper.n_representative_nodes = n_representative_nodes
        current_config.model_hyper.k_representative_days = k_representative_days

        # Spatial aggregation
        spatial_agg = SpatialAggregation(ntw.features, current_config)
        spatial_results = spatial_agg.aggregate()

        # Temporal aggregation
        temporal_agg = TemporalAggregation(current_config, ntw.features, spatial_results)
        temporal_results = temporal_agg.aggregate()
        day_weights = {rep_day: len(days) for rep_day, days in temporal_results["clusters"].items()}

        # Process and save results
        results = Results(current_config, self.preprocessor.network_data, spatial_results, temporal_results)
        res_meta = {
            "weights": current_config.model_hyper.weights,
            "n_representative_nodes": current_config.model_hyper.n_representative_nodes,
            "k_representative_days": current_config.model_hyper.k_representative_days,
            "granularity": current_config.data_preproc.granularity,
            "year": current_config.data_preproc.year,
            "spatial_clusters": spatial_results,
            "temporal_clusters": temporal_results
        }
        
        return results.results, self._get_result_hash(current_config), day_weights, res_meta

    def _get_result_hash(self, config: Config) -> str:
        """Generate unique hash for current configuration"""
        config_dict = {
            "data_preproc" : asdict(config.data_preproc),
            "model_hyper": config.model_hyper.__dict__
        }
        version_hash = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()[:8]   
        return version_hash