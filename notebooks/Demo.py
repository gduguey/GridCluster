# Demo script for the spatial and temporal aggregation of a network of time series.
from src.settings import Config
from src.utils import DataProcessor, Network, Results
from src.models import SpatialAggregation, TemporalAggregation

config = Config(
    year=2013,
    cf_k_neighbors=1,
    demand_decay_alpha=0.4,
    granularity="low",
    active_features=['position', 'time_series', 'duration_curves', 'ramp_duration_curves', 'intra_correlation']
)

processor = DataProcessor(config)
data = processor.processed_data
ntw = Network(data["nodes"], data["time_series"], config)
spatial = SpatialAggregation(ntw.features, config)
distances = spatial.distance_metrics
assignment_dict = spatial.aggregate()
temp = TemporalAggregation(config, ntw.features, assignment_dict)
rep_days = temp.aggregate()
results = Results(config, data, assignment_dict, rep_days)


