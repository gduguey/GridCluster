# settings.py — configuration objects (preprocessing, paths, hyperparameters)
# ---------------------------------------------------------------------
# Configuration Management Module
# This module provides a configuration management system for a data processing pipeline.
# It includes immutable settings for data preprocessing, mutable hyperparameters for model training,
# and derived file paths for data access. The configuration is designed to be user-friendly and robust,
# ensuring that all settings are validated upon assignment.
# ---------------------------------------------------------------------
# Example usage:
# from settings import Config

# config = Config(
#     year=2013,
#     granularity='coarse',
#     active_features=['position', 'time_series', 'duration_curves']
# )

# # Display configuration help.
# config.help()

# # The following assignment should raise an error automatically.
# try:
#     config.model_hyper.n_representative_nodes = "hello"  # Invalid type.
# except Exception as e:
#     print("Error updating n_representative_nodes:", e)

# # Check the auto-generated weights.
# print("Auto-generated weights:", config.model_hyper.weights)
# print("n_representative_nodes:", config.model_hyper.n_representative_nodes)

# # If you try to update the weights with an invalid key set, it will raise an error immediately:
#     try:
#         config.model_hyper.weights = {
#             'position': 0.5,
#             'time_series': 0.3,
#             'duration_curves': 0.2
#             # Missing 'inter_correlation' because inter_correlation is True by default.
#         }
#     except Exception as e:
#         print("Error updating weights:", e)
# ---------------------------------------------------------------------

from typing import List, Dict, Set
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel, PrivateAttr, validator, root_validator

# ---------------------------------------------------------------------
# Helper Function
# ---------------------------------------------------------------------
def auto_generate_weights(active_features: List[str], inter_correlation: bool) -> Dict[str, float]:
    """Auto-generate equal weights for all active features and inter_correlation (if enabled)."""
    keys = set(active_features)
    if inter_correlation:
        keys.add("inter_correlation")
    num_keys = len(keys)
    if num_keys == 0:
        return {}
    weight = 1.0 / num_keys
    return {key: weight for key in keys}

# ---------------------------------------------------------------------
# Data Preproc (Immutable): PreprocessingConfig
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessingConfig:
    """Immutable data preprocessing settings (data preproc)."""
    year: int
    granularity: str
    active_features: List[str]        

    def __post_init__(self):
        if self.year < 2007 or self.year > 2013:
            raise ValueError("Year must be between 2007 and 2013")
        if not isinstance(self.active_features, list):
            raise TypeError("active_features must be a list of strings")
        if len(self.active_features) == 0:
            raise ValueError("active_features cannot be empty")
        allowed_granularities = {"coarse", "fine"}
        if self.granularity not in allowed_granularities:
            raise ValueError(f"granularity must be one of {allowed_granularities}")
        allowed_features = ['position', 'time_series', 'duration_curves', 'ramp_duration_curves', 'intra_correlation']
        if not all(feature in allowed_features for feature in self.active_features):
            raise ValueError(f"Invalid features in active_features. Allowed features: {allowed_features}")
        # Order features for consistent behavior.
        ordered_features = sorted(self.active_features, key=lambda x: allowed_features.index(x))
        object.__setattr__(self, 'active_features', ordered_features)

# ---------------------------------------------------------------------
# Path Configuration (Immutable)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PathConfig:
    """Immutable path configuration based on the data preproc settings."""
    cfg: PreprocessingConfig
    root: Path = Path(__file__).resolve().parents[2]
    # Paths for results
    results: Path = field(init=False)
    distance_metrics: Path = field(init=False)
    joint_aggregation_results: Path = field(init=False)
    # Paths for data files
    data: Path = field(init=False)
    raw: Path = field(init=False)
    processed: Path = field(init=False)
    population_file: Path = field(init=False)
    county_file: Path = field(init=False)
    coarse_node_file: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "results", self.root / "results")
        object.__setattr__(self, "distance_metrics", self.results / "distance_metrics")
        object.__setattr__(self, "joint_aggregation_results", self.results / "joint_aggregation_results")
        object.__setattr__(self, "data", self.root / "DATA")
        object.__setattr__(self, "population_file", self.data / "raw" / "ne_population" / "ne_population.csv")
        object.__setattr__(self, "county_file", self.data / "raw" / "ne_population" / "cb_2021_us_county_500k" / "cb_2021_us_county_500k.shp")
        object.__setattr__(self, "coarse_node_file", self.data / "raw" / "17_zones" / "17_Nodes.csv")
        object.__setattr__(self, "raw", self.data / "raw" / "385_buses")
        object.__setattr__(self, "processed", self.data / "processed" / "385_buses")

    @property
    def demand_file(self) -> Path:
        """Path for the demand file corresponding to the given year"""
        return self.data / "raw" / "67_counties" / "demand_hist" / f"county_demand_local_hourly_{self.cfg.year}.csv"

    @property
    def wind_cf_file(self) -> Path:
        """Path for the wind capacity factor file for the specified year."""
        return self.raw / "CapacityFactors_ISONE" / "Wind" / f"cf_Wind_0.22m_{self.cfg.year}.nc"

    @property
    def solar_cf_file(self) -> Path:
        """Path for the solar capacity factor file for the specified year."""
        return self.raw / "CapacityFactors_ISONE" / "Solar" / f"cf_Solar_0.22m_{self.cfg.year}.nc"

# ---------------------------------------------------------------------
# Model Hyper (Mutable): HyperParameters using Pydantic
# ---------------------------------------------------------------------
class HyperParameters(BaseModel):
    inter_correlation: bool = True
    n_representative_nodes: int = 5
    k_representative_days: int = 10
    kmed_seed: int = 42
    kmed_n_init: int = 10
    active_features: List[str] = []  # Injected from immutable config
    
    # Private state
    _weights_override: Dict[str, float] | None = PrivateAttr(None)

    class Config:
        validate_assignment = True

    # --------------------------------------
    # Validators
    # --------------------------------------
    @root_validator(skip_on_failure=True)
    def check_inter_correlation_and_features(cls, values):
        ic = values.get("inter_correlation")
        af = values.get("active_features", [])
        if ic and "time_series" not in af:
            raise ValueError("'time_series' must be in active_features when inter_correlation is True")
        return values
    
    @validator("n_representative_nodes")
    def validate_n_representative_nodes(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("n_representative_nodes must be a positive integer")
        return v

    @validator("k_representative_days")
    def validate_k_representative_days(cls, v):
        if not isinstance(v, int) or not (1 <= v <= 365):
            raise ValueError("k_representative_days must be between 1 and 365")
        return v

    @validator("kmed_seed")
    def validate_kmed_seed(cls, v):
        if not isinstance(v, int) or v < 0:
            raise ValueError("kmed_seed must be non-negative (0 means no seed)")
        return v

    @validator("kmed_n_init")
    def validate_kmed_n_init(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("kmed_n_init must be a positive integer")
        return v

    # --------------------------------------
    # Core Weights Logic
    # --------------------------------------
    def __setattr__(self, name, value):
        """Clear weight override when dependencies change."""
        super().__setattr__(name, value)
        if name in ("inter_correlation", "active_features"):
            self._weights_override = None

    @property
    def weights(self) -> Dict[str, float]:
        """Get weights - auto-generate if no valid override exists."""
        if self._weights_override is not None:
            return self._weights_override.copy()
        return auto_generate_weights(self.active_features, self.inter_correlation)

    @weights.setter
    def weights(self, value: Dict[str, float]):
        """Validate and store manual weights override."""
        required_keys = self._get_allowed_weight_keys()
        if set(value.keys()) != required_keys:
            raise ValueError(f"Invalid weight keys. Expected: {required_keys}")
        
        self._weights_override = value.copy()

    def _get_allowed_weight_keys(self) -> Set[str]:
        """Dynamic allowed keys based on current state."""
        keys = set(self.active_features)
        if self.inter_correlation:
            keys.add("inter_correlation")
        return keys
    
# ---------------------------------------------------------------------
# Overall Config
# ---------------------------------------------------------------------
class Config:
    """
    Top-level configuration that ties together:
      - Immutable Data Preproc settings (data_preproc)
      - Mutable Model Hyperparameters (model_hyper)
      - Derived file paths (path)
    
    The active_features (from data preproc) determine which node-level features are computed,
    and they also set the allowed keys for the weights dictionary used for spatial aggregation.
    
    The inter_correlation flag indicates whether an extra (network-wide) weight is needed.
    """
    def __init__(self, **data_kwargs):
        # Build immutable data preproc settings.
        self._data_preproc = PreprocessingConfig(**data_kwargs)
        self._path = PathConfig(cfg=self._data_preproc)
        # Build mutable model hyperparameters.
        # Inject active_features from data preproc.
        self.model_hyper = HyperParameters(active_features=self._data_preproc.active_features)
        # Validate overall consistency.
        self._validate_active_features_and_weights()

    @property
    def data_preproc(self) -> PreprocessingConfig:
        return self._data_preproc

    @property
    def path(self) -> PathConfig:
        return self._path

    def _validate_active_features_and_weights(self):
        required = set(self._data_preproc.active_features)
        if self.model_hyper.inter_correlation:
            required.add("inter_correlation")
        if set(self.model_hyper.weights.keys()) != required:
            raise ValueError(
                f"Inconsistent configuration: weights have keys {set(self.model_hyper.weights.keys())}, "
                f"but expected {required} based on active_features and inter_correlation."
            )

    def validate(self):
        self._validate_active_features_and_weights()

    def help(self) -> None:
        print(
            "Configuration Overview:\n"
            "\n"
            "1. Data Preproc (Immutable):\n"
            "   Contains data processing parameters. Key attributes:\n"
            "   - year (int): The data year (2007-2013).\n"
            "   - granularity (str): Data granularity; allowed: 'coarse', 'fine'.\n"
            "   - active_features (list): Features to include; allowed: 'position', 'time_series', 'duration_curves',\n"
            "      'ramp_duration_curves', 'intra_correlation'.\n"
            "2. Model Hyper (Mutable):\n"
            "   Contains model configuration parameters. Key attributes:\n"
            "   - n_representative_nodes (int): Number of representative nodes.\n"
            "   - k_representative_days (int): Number of representative days (1–365).\n"
            "   - inter_correlation (bool): Whether inter-node correlation is included.\n"
            "   - kmed_seed (int): Seed for KMedoids (0 means no seed).\n"
            "   - kmed_n_init (int): Number of KMedoids initializations.\n"
            "   - weights (dict): Weights for various features. Can be manually set or auto-generated based on \n"
            "      data_preproc.active_features and model_hyper.inter_correlation\n"
            "\n"
            "3. Path (Immutable):\n"
            "   Contains dynamically generated paths based on data_preproc.year, including:\n"
            "   - demand_file, wind_cf_file, solar_cf_file, and others.\n"
        )
