from abc import ABC, abstractmethod
import os

class ExplainerBase(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None
        self.shap_values = None
        self.raw_data = None
        self.feature_names = config.get("feature_names", [])
        self.output_dir = config.get("output_dir", "output/")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_path(self, key):
        return self.config.get(key)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def explain(self):
        pass

    @abstractmethod
    def plot_results(self):
        pass