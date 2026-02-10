from .tabular.tree_based import run_tabular_analysis
from .timeseries.lstm_based import run_timeseries_analysis

ANALYSIS_ROUTER = {
    "tabular": run_tabular_analysis,
    "timeseries": run_timeseries_analysis
}


