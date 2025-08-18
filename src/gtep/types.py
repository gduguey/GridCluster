# types.py — typed dataclasses for parameters, results, and input data

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

# =========================================================
# ===============  TYPES & CONSTANTS  =====================
# =========================================================

@dataclass
class Params:
    Initial_SOC: float
    eta_discharge: float
    eta_charge: float
    PV_resource: float        # MW
    Wind_resource: float      # MW
    CCGT_ramp: float          # fraction of nameplate per hour
    CCGT_max_cap: float       # MW per unit
    c_ccgt: float             # $/MW
    c_pv: float               # $/MW
    c_wind: float             # $/MW
    c_storage: float          # $/MW (power)
    c_tran: float             # $/MW/mile
    d_ccgt: float             # $/MWh
    d_shed: float             # $/MWh
    rate_duration: int        # hours
    MW_per_mile: float
    TranMax: Dict[int, float] # MW per line
    Length: Dict[int, float]  # miles per line

@dataclass
class SolveStats:
    build_time: float
    run_time: float
    objective: float

@dataclass
class InvestmentResults:
    PV: Dict[int, float]
    Wind: Dict[int, float]
    CCGT: Dict[int, float]
    Storage: Dict[int, float]
    Tran: Dict[int, float]

@dataclass
class OperationResults:
    y_ccgt: Dict[Tuple[int, int], float]
    y_curtail: Dict[Tuple[int, int], float]
    y_shed: Dict[Tuple[int, int], float]
    y_charge: Dict[Tuple[int, int], float]
    y_discharge: Dict[Tuple[int, int], float]
    y_soc: Dict[Tuple[int, int], float]
    y_flow: Dict[Tuple[int, int], float]

@dataclass
class GTEPResults:
    stats: SolveStats
    inv: InvestmentResults
    op: OperationResults

@dataclass
class RepPeriod:
    Load: pd.DataFrame
    PV: pd.DataFrame
    Wind: pd.DataFrame

    @property
    def T(self) -> int:
        return len(self.Load)
    
@dataclass
class InputData:
    buses: pd.DataFrame
    branches: pd.DataFrame
    rep_period: RepPeriod
    params: Params

@dataclass
class RunArtifacts:
    results: GTEPResults | None
    data: InputData
    agg_hash: str
    gtep_hash: str
    save_path: Path | None
    log_path: Optional[Path] = None
    clusters: dict | None = None