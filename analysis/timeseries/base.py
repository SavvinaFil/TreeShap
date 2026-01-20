from abc import ABC, abstractmethod
import pandas as pd
import os

class TimeseriesExplainerBase(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.data = None
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
    def get_path(self, key):
        """Helper to resolve paths relative to the project root."""
        relative_path = self.config.get(key)
        if relative_path is None:
            return None
        return os.path.join(self.base_path, relative_path)

    @abstractmethod
    def load_model(self):
        """Each model type loads differently (joblib vs torch.load)."""
        pass

    @abstractmethod
    def explain(self):
        """The core logic that generates the explanation/forecast."""
        pass

    def save_results(self, results: pd.DataFrame):
        """Shared logic: all explainers save results the same way."""
        output_path = self.config.get("output_dir", "outputs")
        results.to_csv(f"{output_path}/explanation_results.csv")
        print(f"Results saved to {output_path}")

    @abstractmethod
    def plot_results(self):
        """Force each explainer to provide a visualization."""
        pass