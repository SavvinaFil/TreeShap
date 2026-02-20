# analysis/tabular/__init__.py
from .tree_based import TREE_MODEL_MAP

# Manager Map: Combines all tabular subtypes
TABULAR_MAP = {
    **TREE_MODEL_MAP,
    # "linear_regression": LinearExplainer, (Future addition)
}

def run_tabular_analysis(config):
    """Orchestrates any tabular model (Tree, Linear, etc.)"""
    model_type = config.get("model_type")
    explainer_class = TABULAR_MAP.get(model_type)
    
    if not explainer_class:
        raise ValueError(f"Model {model_type} not supported in Tabular analysis.")

    # 1. Instantiate (Standard API)
    explainer = explainer_class(config)
    
    # 2. Execute standard workflow
    explainer.load_model()
    explainer.explain()
    
    # 3. Output handling
    if config.get("save_excel"):
        explainer.save_results_to_excel()
        
    if config.get("generate_notebook"):
        explainer.plot_results()