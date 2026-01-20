# analysis/__init__.py
from .timeseries import run_timeseries_analysis
# from .tabular import run_tabular_analysis 

# Map the "analysis" key from JSON to the sub-package runner
ANALYSIS_ROUTER = {
    "timeseries": run_timeseries_analysis,
    # "tabular": run_tabular_analysis
}