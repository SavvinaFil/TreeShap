import json
import numpy as np
import os
from datetime import datetime
from tree_input import load_tree, load_dataset
from results import (
    compute_shap_values,
    show_shap_values,
    plot_shap_values,
    save_results_to_excel
)

CONFIG_PATH = "config.json"


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def ask_yes_no(question):
    while True:
        answer = input(f"{question} (yes/no): ").strip().lower()
        if answer in ["yes", "y"]:
            return True
        if answer in ["no", "n"]:
            return False
        print("Please answer yes or no.")


def ask_int(question, min_val=0):
    while True:
        try:
            value = int(input(question))
            if value >= min_val:
                return value
        except ValueError:
            pass
        print(f"Please enter an integer ≥ {min_val}")


def main():
    print("Trustworthy AI: Decision Tree Explainability\n")

    config = load_config()
    config_changed = False
    first_run = not bool(config)

    # reset paths & preferences
    if not first_run:
        if "model_path" in config or "dataset_path" in config:
            if ask_yes_no("Do you want to reset the dataset/model paths (pkl/csv)?"):
                if ask_yes_no("Reset model_path (.pkl)?"):
                    config.pop("model_path", None)
                if ask_yes_no("Reset dataset_path (.csv)?"):
                    config.pop("dataset_path", None)
                config_changed = True

        if ask_yes_no("Do you want to reset other preferences?"):
            model_path = config.get("model_path")
            dataset_path = config.get("dataset_path")
            output_dir = config.get("output_dir")
            config = {}
            if model_path:
                config["model_path"] = model_path
            if dataset_path:
                config["dataset_path"] = dataset_path
            if output_dir:
                config["output_dir"] = output_dir
            config_changed = True

    # ask for paths if they are missing
    if "model_path" not in config:
        config["model_path"] = input("Enter path to the decision tree model (.pkl): ").strip()
        config_changed = True

    if "dataset_path" not in config:
        config["dataset_path"] = input("Enter path to the dataset (.csv): ").strip()
        config_changed = True

    if "output_dir" not in config:
        config["output_dir"] = input("Enter path for output folder (if not just press enter): ").strip() or "outputs"
        config_changed = True

    # ask basic preferences if they are missing
    if "generate_plots" not in config:
        config["generate_plots"] = ask_yes_no("Do you want to generate SHAP plots?")
        config_changed = True

    if "save_excel" not in config:
        config["save_excel"] = ask_yes_no("Do you want to save results to Excel?")
        config_changed = True

    if "dataset_scope" not in config:
        print("\nChoose dataset scope:")
        print("1 → Whole dataset")
        print("2 → Subset of rows")
        choice = ask_int("Your choice (1 or 2): ", min_val=1)
        if choice == 1:
            config["dataset_scope"] = "whole"
        else:
            config["dataset_scope"] = "subset"
            config["subset_start"] = ask_int("Start row index: ", 0)
            config["subset_end"] = ask_int("End row index (exclusive): ", 1)
        config_changed = True

    if config_changed or first_run:
        save_config(config)
        print("\nPreferences saved to config.json\n")

    # load model and dataset
    model = load_tree(config["model_path"])

    # προσπαθούμε να πάρουμε feature names από το μοντέλο (generic)
    try:
        feature_names = list(model.feature_names_in_)
    except AttributeError:
        feature_names = None

    X_df = load_dataset(
        choice=2,
        feature_names=feature_names,
        path_override=config["dataset_path"]
    )

    # αν δεν είχαμε feature_names από το μοντέλο, τα παίρνουμε τώρα από το dataframe
    if feature_names is None:
        feature_names = list(X_df.columns)

    # πάρουμε και τις κλάσεις του μοντέλου (generic, μπορεί να είναι None για regression)
    class_labels = getattr(model, "classes_", None)

    # apply dataset start and stop
    if config["dataset_scope"] == "subset":
        X_sample = X_df.iloc[config["subset_start"]:config["subset_end"]]
        print(f"Using dataset subset [{config['subset_start']}:{config['subset_end']}]")
    else:
        X_sample = X_df
        print("Using full dataset")

    # compute SHAP values
    explainer, shap_values, X_df_aligned = compute_shap_values(model, X_sample)

    # calculate preds
    try:
        preds = model.predict(X_df_aligned)
        unique_classes, class_counts = np.unique(preds, return_counts=True)

        print(f"\nPredictions of the model:")
        for cls, cnt in zip(unique_classes, class_counts):
            percentage = (cnt / len(preds)) * 100
            print(f"  - Class {cls}: {cnt} samples ({percentage:.1f}%)")
    except Exception as e:
        print(f"Mistake when you calculate the predictions: {e}")
        return

    # show shap values
    show_shap_values(shap_values, feature_names, preds)

    # create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # save predictions to Excel
    if config["save_excel"]:
        shap_array = np.array(shap_values)
        save_results_to_excel(X_df_aligned, shap_array, feature_names, preds, config["output_dir"])
    else:
        print("Excel output disabled (saved preference).")

    # interactive plots
    if config["generate_plots"]:
        print("\nSelect SHAP plots to generate:")
        plots_options = ['beeswarm', 'bar', 'violin', 'dependence', 'heatmap', 'interactive_heatmap', 'all']
        for i, p in enumerate(plots_options, 1):
            print(f"{i} → {p}")
        choice = input("Enter numbers separated by comma (e.g., 1,3) or for all 7: ").strip().lower()
        selected_plots = []

        # Check if user typed 7 (for "all")
        if choice == "7":
            selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'heatmap', 'interactive_heatmap']
        else:
            for c in choice.split(','):
                c = c.strip()
                if not c:
                    continue
                try:
                    idx = int(c) - 1
                    if 0 <= idx < len(plots_options) - 1:
                        selected_plots.append(plots_options[idx])
                    elif idx == len(plots_options) - 1:
                        selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'heatmap', 'interactive_heatmap']
                        break
                except ValueError:
                    continue

        if not selected_plots:
            selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'heatmap', 'interactive_heatmap']

        # create common folder with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plots_output_dir = os.path.join(config["output_dir"], f"{timestamp}_selected_plots")
        os.makedirs(plots_output_dir, exist_ok=True)

        plot_shap_values(
            shap_values,
            X_df_aligned,
            feature_names,
            preds,
            plots_output_dir,
            selected_plots=selected_plots,
            class_labels=class_labels
        )


if __name__ == "__main__":
    main()

