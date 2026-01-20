import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .base import TimeseriesExplainerBase

class ARIMAExplainer(TimeseriesExplainerBase):
    def load_model(self):
        """Loads the pmdarima/statsmodels object."""
        # ARIMA models are usually saved as .pkl or .joblib
        self.model = joblib.load(self.config["model_path"])
        print("ARIMA Model loaded successfully.")

    def explain(self):
        """
        Extracts coefficients and statistical significance.
        For ARIMA, 'explanation' is often found in the summary table.
        """
        # Extract coefficients for Exogenous variables
        # Statsmodels stores these in the model results
        summary = self.model.summary()
        
        # We can extract the coefficient table as a DataFrame
        results_as_html = summary.tables[1].as_html()
        self.stats_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
        
        # Logic to separate AR/MA terms from your Exogenous features (ghi, etc.)
        self.exog_importance = self.stats_df.loc[
            self.stats_df.index.isin(self.config["feature_names"])
        ]

    def plot_results(self):
        """Visualizes feature importance based on coefficient weight."""
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # 1. Plot Exogenous Coefficients
        plt.figure(figsize=(10, 6))
        # Use absolute values for 'importance', color by positive/negative impact
        colors = ['red' if x < 0 else 'green' for x in self.exog_importance['coef']]
        self.exog_importance['coef'].plot(kind='barh', color=colors)
        
        plt.axvline(0, color='black', linewidth=0.8)
        plt.title("ARIMA Exogenous Feature Impact (Coefficients)")
        plt.xlabel("Coefficient Value (Direction of Impact)")
        
        save_path = os.path.join(self.config["output_dir"], "arima_coefficients.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # 2. Diagnostic Plots (Standard for ARIMA)
        # This shows if the model is 'healthy' (residuals are white noise)
        self.model.plot_diagnostics(figsize=(12, 8))
        diag_path = os.path.join(self.config["output_dir"], "arima_diagnostics.png")
        plt.savefig(diag_path)
        plt.close()
        
        print(f"ARIMA Explanations saved to {self.config['output_dir']}")