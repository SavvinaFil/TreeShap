import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from .base import TimeseriesExplainerBase
from datetime import datetime
from output.results import (
    compute_shap_values,
    show_shap_values,
    plot_shap_values,
    save_results_to_excel
)
from output.generate_notebook import generate_analysis_notebook

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
        self.input_dim = self.config.get("input_dim", 12)
        self.hidden_dim = self.config.get("hidden_size", 16)
        self.model_type = self.config.get("model_type", "lstm")
        self.output_dim = 1
        self.output_labels = self.config.get("output_labels", 0)
        
        # 2. Reconstruct the 'Skeleton' (Architecture)
        # This works because the class LSTMForecaster is defined in this same file
        self.model = LSTMForecaster(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        
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
        
        # Create a new figure to avoid overlapping with previous plots
        plt.figure(figsize=(12, 8))

        # 4. Generate the Summary Plot
        # We use the flattened names and data you prepared
        shap.summary_plot(
            shap_flat, 
            data_flat, 
            feature_names=flat_names, 
            show=False  # Crucial: allows us to save before the window closes
        )

        # 5. Save to the Output Directory
        save_path = os.path.join(output_dir, "timeseries_shap_summary.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # 6. Clean up memory
        plt.close()

        print(f"SHAP plot successfully saved to: {save_path}")

        # Common folder with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_type = "regression"
        plots_output_dir = os.path.join(output_dir, f"{timestamp}_{task_type}_plots")
        #os.makedirs(plots_output_dir, exist_ok=True)
        
        model_info = {
            'model_type': type(self.model_type).__name__,
            'n_features': self.input_dim,
            'n_classes': self.output_dim,
            'n_samples': 12,
            'feature_names': flat_names,
            'classes': "PV",
            'output_labels': self.output_labels
        }

        # Add regression-specific info
        is_classification = False
        if not is_classification:
            model_info['prediction_range'] = f"[{0:.2f}, {5:.2f}]"
            model_info['n_bins'] = 1

        # try:
        #     notebook_path = generate_analysis_notebook(
        #         plots_output_dir,
        #         model_info=model_info
        #     )
        #     print(f"Notebook generated: {notebook_path}")

        # except Exception as e:
        #     print(f"\nError generating notebook: {e}")
        #     print("All plots are still available in the output directory.")
        #     import traceback
        #     traceback.print_exc()
        
        # print("Analysis complete.")