from .rf_explainer import RFExplainer
#from .xgb_explainer import XGBExplainer

# Registry for tree-based models
TREE_MODEL_MAP = {
    "random_forest": RFExplainer,
    #"xgboost": XGBExplainer
}