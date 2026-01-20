import json
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from analysis.tabular.tree_based.tree_input import load_tree, load_dataset
from output.results import (
    compute_shap_values,
    show_shap_values,
    plot_shap_values,
    save_results_to_excel
)
from output.generate_notebook import generate_analysis_notebook

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        content = f.read().strip()
        if not content:
            raise ValueError("config.json is empty")
        return json.loads(content)


def main():
    print("Trustworthy AI: Decision Tree Explainability\n")

    # Read config.json
    config = load_config()

    # Load model and dataset from config
    model_path = config["model_path"]
    dataset_path = config["dataset_path"]
    output_dir = config.get("output_dir", "outputs")
    generate_plots = config.get("generate_plots", True)
    save_excel = config.get("save_excel", True)
    dataset_scope = config.get("dataset_scope", "whole")
    generate_notebook = config.get("generate_notebook", True)
    auto_open_notebook = config.get("auto_open_notebook", True)

    print(f"Model path   : {model_path}")
    print(f"Dataset path : {dataset_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Plots        : {generate_plots}")
    print(f"Save Excel   : {save_excel}")
    print(f"Dataset scope: {dataset_scope}")
    print(f"Generate Notebook: {generate_notebook}")

    model = load_tree(model_path)

    # Take feature names from the model
    try:
        feature_names = list(model.feature_names_in_)
    except AttributeError:
        feature_names = None

    X_df = load_dataset(choice=2, feature_names=feature_names, path_override=dataset_path)

    if feature_names is None:
        feature_names = list(X_df.columns)

    # Dataset start and stop from config
    if dataset_scope == "subset":
        start = config.get("subset_start", 0)
        end = config.get("subset_end", len(X_df))
        X_sample = X_df.iloc[start:end]
        print(f"Using dataset subset [{start}:{end}]")
    else:
        X_sample = X_df
        print("Using full dataset")

    # Compute Shap values
    explainer, shap_values, X_df_aligned = compute_shap_values(model, X_sample)

    # Calculate the predictions
    try:
        preds = model.predict(X_df_aligned)
        unique_classes, class_counts = np.unique(preds, return_counts=True)

        print("\nModel predictions:")
        for cls, cnt in zip(unique_classes, class_counts):
            percentage = (cnt / len(preds)) * 100
            print(f"  - Class {cls}: {cnt} samples ({percentage:.1f}%)")
    except Exception as e:
        print(f"Error when calculating predictions: {e}")
        return

    # Show Shap values
    show_shap_values(shap_values, feature_names, preds)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save results to Excel
    if save_excel:
        shap_array = np.array(shap_values)
        save_results_to_excel(X_df_aligned, shap_array, feature_names, preds, output_dir)
    else:
        print("Excel output disabled (config).")

    # Create plots
    plots_output_dir = None
    if generate_plots:
        selected_plots = [
            'beeswarm',
            'bar',
            'violin',
            'dependence',
            'decision_map',
            'interactive_decision_map',
            'heatmap',
            'interactive_heatmap',
            'waterfall'
        ]

        # Common folder with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plots_output_dir = os.path.join(output_dir, f"{timestamp}_selected_plots")
        os.makedirs(plots_output_dir, exist_ok=True)

        plot_shap_values(
            shap_values,
            X_df_aligned,
            feature_names,
            preds,
            plots_output_dir,
            selected_plots=selected_plots,
            explainer=explainer
        )
    else:
        print("Plot generation disabled (config).")

    # Generate Jupyter Notebook with analysis
    if generate_notebook and plots_output_dir is not None:
        # Prepare model information for the notebook
        model_info = {
            'model_type': type(model).__name__,
            'n_features': len(feature_names),
            'n_classes': len(unique_classes),
            'n_samples': len(X_df_aligned),
            'feature_names': feature_names,
            'classes': unique_classes.tolist()
        }

        try:
            notebook_path = generate_analysis_notebook(
                plots_output_dir,
                model_info=model_info
            )

        except Exception as e:
            print(f"\nΕrror generating notebook: {e}")
            print("All plots are still available in the output directory.")
            import traceback
            traceback.print_exc()
    elif not generate_plots:
        print("\nΝotebook generation skipped (plots were not generated)")
    else:
        print("\nNotebook generation disabled (config).")


if __name__ == "__main__":
    main()
