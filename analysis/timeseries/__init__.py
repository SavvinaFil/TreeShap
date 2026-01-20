# analysis/timeseries/__init__.py
from .lstm_pytorch import LSTMExplainer
from .arima_stats import ARIMAExplainer

# Map "model_type" to the specific class
MODEL_MAP = {
    "lstm": LSTMExplainer,
    "arima": ARIMAExplainer
}

def run_timeseries_analysis(config):
    """Orchestrates the specific timeseries model based on config."""
    model_type = config.get("model_type")
    explainer_class = MODEL_MAP.get(model_type)
    
    if not explainer_class:
        raise ValueError(f"Model type '{model_type}' not found in timeseries analysis.")
    
    # Standard Workflow
    explainer = explainer_class(config)
    explainer.load_model()
    explainer.explain()
    
    if config.get("generate_plots"):
        explainer.plot_results()