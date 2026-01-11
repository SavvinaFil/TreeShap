import shap
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def compute_shap_values(model, X_df):
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    return explainer, shap_values, X_df

def show_shap_values(shap_array, feature_names, preds=None):
    shap_array = np.array(shap_array)
    if shap_array.ndim == 3:
        if preds is None:
            shap_array = shap_array[:, :, 1]
        else:
            shap_array = shap_array[np.arange(len(preds)), :, preds]
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    print("\nSHAP values per feature:\n")
    for i, row in enumerate(shap_array):
        if preds is not None:
            print(f"Row {i + 1} (predicted class: {preds[i]}):")
        else:
            print(f"Row {i + 1}:")
        for j, feat in enumerate(feature_names):
            print(f"  {feat}: {float(row[j]):+.4f}")
        print("-" * 40)

def save_results_to_excel(X_df, shap_array, feature_names, output_dir):
    shap_array = np.array(shap_array)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    shap_df = pd.DataFrame([shap_array[i] for i in range(len(shap_array))], columns=[f"SHAP_{f}" for f in feature_names])
    output_df = pd.concat([X_df.reset_index(drop=True), shap_df], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "shap_results.xlsx")
    output_df.to_excel(output_path, index=False)
    print(f"SHAP results saved to: {output_path}")

def plot_shap_values(shap_values, X_df, feature_names, output_dir, selected_plots=None):
    os.makedirs(output_dir, exist_ok=True)
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    if selected_plots is None:
        selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'scatter']

    # Beeswarm
    if 'beeswarm' in selected_plots:
        plt.figure()
        shap.summary_plot(shap_array, X_df, feature_names=feature_names, show=False)
        plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"), bbox_inches="tight")
        plt.close()

    # Bar
    if 'bar' in selected_plots:
        plt.figure()
        shap.summary_plot(shap_array, X_df, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig(os.path.join(output_dir, "shap_bar.png"), bbox_inches="tight")
        plt.close()

    # Violin
    if 'violin' in selected_plots:
        plt.figure()
        shap.summary_plot(shap_array, X_df, feature_names=feature_names, plot_type="violin", show=False)
        plt.savefig(os.path.join(output_dir, "shap_violin.png"), bbox_inches="tight")
        plt.close()

    # Dependence (one plot per each feature)
    if 'dependence' in selected_plots:
        for feat in feature_names:
            plt.figure()
            shap.dependence_plot(feat, shap_array, X_df, show=False)
            plt.savefig(os.path.join(output_dir, f"dependence_{feat}.png"), bbox_inches="tight")
            plt.close()


    print(f"SHAP plots saved to folder: {output_dir}")

