#!/usr/bin/env python
from pathlib import Path
import sys

# Setup path
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
EXP_DIR = THIS_FILE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregation.pipeline import StaticPreprocessor, DynamicProcessor
from src.gtep.pipeline import run_experiment, consume

# Define fixed weights
fixed_weights = {
    "position":             0.45,
    "time_series":          0.025,
    "duration_curves":      0.025,
    "ramp_duration_curves": 0.025,
    "intra_correlation":    0.025,
    "inter_correlation":    0.45,
}

# Hyperparameter configuration for the test
test_hp = {
    "id": "test_fine_10_5",
    "desc": "Test run fine network N=10, K=5",
    "weights": fixed_weights,
    "n_representative_nodes": 10,
    "k_representative_days": 5,
}

# Preprocessing
static_prep = StaticPreprocessor(granularity="fine").preprocess()
processor = DynamicProcessor(static_prep)

# Run
df = consume(run_experiment([test_hp], processor, EXP_DIR, threads=8), PROJECT_ROOT)
print(df)
df.to_csv(EXP_DIR / "manifest_test.csv", index=False)
