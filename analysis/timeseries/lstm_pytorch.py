import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from .base import TimeseriesExplainerBase

class LSTMForecaster(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMExplainer(TimeseriesExplainerBase):
    def load_model(self):
        """Reconstructs the model from state_dict for better stability."""
        # 1. Get parameters from config
        input_dim = self.config.get("input_dim", 12)
        hidden_dim = self.config.get("hidden_size", 16)
        
        # 2. Reconstruct the 'Skeleton' (Architecture)
        # This works because the class LSTMForecaster is defined in this same file
        self.model = LSTMForecaster(input_dim=input_dim, hidden_dim=hidden_dim)
        
        # 3. Load the 'Muscles' (Weights)
        model_full_path = self.get_path("model_path")
        
        # Load the weights (state_dict)
        state_dict = torch.load(model_full_path, map_location=torch.device('cpu'))
        
        # Apply weights to the skeleton
        self.model.load_state_dict(state_dict)
        
        # 4. Set to evaluation mode
        self.model.eval()
        torch.set_grad_enabled(True) 
        print("Model state_dict loaded successfully.")

    def explain(self):
        """Agnostic explanation: Uses pre-processed tensors."""
        # Load tensors directly (agnostic to how they were scaled/merged)
        bg_path = self.get_path("background_data_path")
        test_path = self.get_path("test_data_path")
        
        background = torch.load(bg_path)
        test_data = torch.load(test_path)
        
        # Determine the subset to explain to save time
        explain_len = min(len(test_data), 50)
        test_subset = test_data[:explain_len]

        # Initialize Explainer
        if self.config["explainer_type"] == "gradient":
            explainer = shap.GradientExplainer(self.model, background)
            # This returns a SHAP Explanation object
            self.shap_values = explainer(test_subset)
        
        elif self.config["explainer_type"] == "deep":
            explainer = shap.DeepExplainer(self.model, background)
            # Returns a list of arrays (one for each output)
            self.shap_values = explainer.shap_values(test_subset, check_additivity=False)

        self.raw_data = test_subset.numpy()

    def plot_results(self):
        """Simplified SHAP summary plot for Time-Series."""
        import os
        torch.set_grad_enabled(True)

        # 1. Resolve Absolute Path and Ensure Directory Exists
        output_dir = self.get_path("output_dir")
        os.makedirs(output_dir, exist_ok=True)

        # 2. Flatten feature names for the Y-axis
        look_back = self.config["look_back"]
        features = self.config["feature_names"]
        # Generates: [Feat1_t-6, Feat2_t-6... Feat1_t-1, Feat2_t-1]
        flat_names = [f"{feat}_t-{look_back - i}" for i in range(look_back) for feat in features]

        # 3. Flatten SHAP values and Data from 3D to 2D
        # (Samples, Time, Features) -> (Samples, Time * Features)
        # Note: Using .values for GradientExplainer, or index [0] for DeepExplainer
        val_to_plot = self.shap_values.values if hasattr(self.shap_values, 'values') else self.shap_values[0]
        
        shap_flat = val_to_plot.reshape(val_to_plot.shape[0], -1)
        data_flat = self.raw_data.reshape(self.raw_data.shape[0], -1)

        # 4. Standard SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_flat, 
            data_flat, 
            feature_names=flat_names, 
            show=False
        )
        
        save_path = os.path.join(output_dir, "timeseries_shap_summary.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        print("Analysis complete.")