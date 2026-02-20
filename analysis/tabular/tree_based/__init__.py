# This analysis is for decision trees and random forests - Supports both classification and regression
import numpy as np
import os
import shap
from datetime import datetime
from .tree_input import load_tree, load_dataset
from output.results import (
    compute_shap_values,
    show_shap_values,
    plot_shap_values,
    save_results_to_excel
)
from output.generate_notebook import generate_analysis_notebook


def is_classifier(model):
    # Determine if the model is classifier or regressor
    if hasattr(model, 'estimators_'):
        # Check the base estimator
        base_model = model.estimators_[0] if model.estimators_ else model
        return is_classifier(base_model)

    model_class_name = type(model).__name__.lower()

    # Check for common classifier names
    if 'classifier' in model_class_name:
        return True
    elif 'regressor' in model_class_name:
        return False

    # Check for predict_proba method (classifiers have this)
    if hasattr(model, 'predict_proba'):
        return True

    # Default to classifier for backward compatibility
    return True


def convert_regression_to_classes(predictions, n_bins=5):
    # Convert continuous regression predictions into discrete classes

    # Create bins using quantiles for balanced distribution
    bin_edges = np.quantile(predictions, np.linspace(0, 1, n_bins + 1))

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    # Assign predictions to bins
    class_predictions = np.digitize(predictions, bin_edges[1:-1])

    # Create labels
    class_labels = {}
    for i in range(actual_n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        class_labels[str(i)] = f"Range [{lower:.2f}, {upper:.2f}]"

    return class_predictions, bin_edges, class_labels


def validate_model(model, expected_package, expected_model_type):
    # Validate that the loaded model matches the expected package and type
    model_class_name = type(model).__name__
    model_module = type(model).__module__

    # Check package
    if expected_package == "sklearn":
        if not model_module.startswith("sklearn"):
            print(f"WARNING: Expected sklearn model, but got {model_module}")
    elif expected_package == "xgboost":
        if not model_module.startswith("xgboost"):
            print(f"WARNING: Expected xgboost model, but got {model_module}")

    # Handle MultiOutput wrappers
    if model_class_name in ["MultiOutputRegressor", "MultiOutputClassifier"]:
        base_estimator = model.estimators_[0] if hasattr(model, 'estimators_') and model.estimators_ else None
        if base_estimator:
            base_name = type(base_estimator).__name__
            print(f"Model validated: {model_class_name} wrapping {base_name} from {model_module}")
        else:
            print(f"Model validated: {model_class_name} from {model_module}")
        return

    # Check model type
    model_type_mapping = {
        "decision_tree": ["DecisionTreeClassifier", "DecisionTreeRegressor"],
        "random_forest": ["RandomForestClassifier", "RandomForestRegressor",
                          "MultiOutputClassifier", "MultiOutputRegressor"],
        "gradient_boosting": ["GradientBoostingClassifier", "GradientBoostingRegressor"],
        "xgboost": ["XGBClassifier", "XGBRegressor"]
    }

    if expected_model_type in model_type_mapping:
        expected_classes = model_type_mapping[expected_model_type]
        if model_class_name not in expected_classes:
            print(f"WARNING: Expected {expected_model_type}, but got {model_class_name}")

    print(f"Model validated: {model_class_name} from {model_module}")


def run_tabular_analysis(config):
    # Extract config parameters
    model_path = config["model_path"]
    dataset_path = config["dataset_path"]
    output_dir = config.get("output_dir")
    generate_plots = config.get("generate_plots")
    save_excel = config.get("save_excel")
    dataset_scope = config.get("dataset_scope")
    generate_notebook = config.get("generate_notebook")

    # Get feature names and output labels from config
    feature_names = config.get("feature_names", None)
    output_labels = config.get("output_labels", {})

    # Get package and model_type for validation
    expected_package = config.get("package")
    expected_model_type = config.get("model_type")

    # Load model
    model = load_tree(model_path)

    # Validate model
    validate_model(model, expected_package, expected_model_type)

    # Detect if model is classifier or regressor
    is_classification = is_classifier(model)

    if not is_classification:
        print(f"Model type: Regression")
    else:
        print(f"Model type: Classification")

    # Load dataset with feature names from config
    X_df = load_dataset(feature_names=feature_names, path_override=dataset_path)

    if feature_names is None:
        feature_names = list(X_df.columns)

    # Dataset start and stop from config
    if dataset_scope == "subset":
        start = config.get("subset_start", 0)
        end = config.get("subset_end", len(X_df))
        X_sample = X_df.iloc[start:end]
    else:
        X_sample = X_df

    # Compute SHAP values
    explainer, shap_values, X_df_aligned = compute_shap_values(model, X_sample)

    # Check if we have multiple outputs
    # MultiOutput wrappers have estimators_ where each is a separate model
    # RandomForest/GradientBoosting also have estimators_ but they are trees, not separate models
    model_class_name = type(model).__name__
    is_multi_output = model_class_name in ['MultiOutputRegressor', 'MultiOutputClassifier']
    num_outputs = len(model.estimators_) if is_multi_output else 1

    # For multi-output models, create explainers list for each output
    if is_multi_output:
        all_explainers = []
        for estimator in model.estimators_:
            exp = shap.TreeExplainer(estimator)
            all_explainers.append(exp)

        # For multi-output, we compute SHAP values per output later
        # So explainer and shap_values might be None here
        if explainer is None:
            # Use first explainer as a placeholder
            explainer = all_explainers[0]
            shap_values = explainer.shap_values(X_df_aligned)
    else:
        all_explainers = None

    # Calculate the predictions
    try:
        preds = model.predict(X_df_aligned)

        # Handle MultiOutput predictions
        if preds.ndim == 2 and preds.shape[1] > 1:
            all_outputs = preds
            preds_for_main = preds[:, 0]
        else:
            all_outputs = None
            preds_for_main = preds

        if is_classification:
            # Handle multi-output classification
            if is_multi_output:
                for output_idx in range(num_outputs):
                    output_name = output_labels.get(f"{output_idx}_name", f"Output {output_idx}")
                    output_preds = all_outputs[:, output_idx]
                    unique_classes, class_counts = np.unique(output_preds, return_counts=True)

                    for cls, cnt in zip(unique_classes, class_counts):
                        percentage = (cnt / len(output_preds)) * 100
                        # For multi-output classification, labels are stored as output_labels[output_idx][class]
                        if str(output_idx) in output_labels and isinstance(output_labels[str(output_idx)], dict):
                            class_label = output_labels[str(output_idx)].get(str(int(cls)), f"Class {cls}")
                        else:
                            class_label = f"Class {cls}"
            else:
                # Single-output classification
                unique_classes, class_counts = np.unique(preds_for_main, return_counts=True)
                for cls, cnt in zip(unique_classes, class_counts):
                    percentage = (cnt / len(preds_for_main)) * 100
                    class_label = output_labels.get(str(int(cls)), f"Class {cls}")

        else:
            # Regression path
            if is_multi_output:
                # Multi-output regression - no binning needed
                for i in range(num_outputs):
                    output_name = output_labels.get(str(i), f"Output {i}")
                    output_preds = all_outputs[:, i]

                # For multi-output regression, we don't bin
                original_preds = preds_for_main
                bin_labels_for_display = None
                preds = preds_for_main  # Keep as continuous
                unique_classes = None
                class_counts = None
            else:
                # Binary/Single-output regression - apply binning for visualization
                n_bins = config.get("regression_bins", 5)
                preds_binned, bin_edges, auto_labels = convert_regression_to_classes(preds_for_main, n_bins=n_bins)

                bin_labels_for_display = auto_labels

                # Update predictions to binned version for plotting
                original_preds = preds_for_main.copy()
                preds = preds_binned
                unique_classes, class_counts = np.unique(preds, return_counts=True)

    except Exception as e:
        print(f"Error when calculating predictions: {e}")
        import traceback
        traceback.print_exc()
        return

    # Show SHAP values console
    if not is_classification:
        if is_multi_output:
            # For multi-output regression, show SHAP values for first output only
            first_explainer = all_explainers[0]
            first_shap = first_explainer.shap_values(X_df_aligned)
            # Handle 3D SHAP arrays
            if isinstance(first_shap, list) and len(first_shap) > 0:
                first_shap = first_shap[0]
            if isinstance(first_shap, np.ndarray):
                if first_shap.ndim > 2:
                    first_shap = first_shap[:, :, 0] if first_shap.shape[2] > 0 else first_shap[:, :, 0]
            print(f"\nShowing SHAP values for first output only (Output 0: {output_labels.get('0', 'Output 0')})")
            show_shap_values(first_shap, feature_names, preds_for_main, output_labels)
        else:
            # For single-output regression, use bin labels
            show_shap_values(shap_values, feature_names, preds, bin_labels_for_display)
    else:
        # For classification (both single and multi-output)
        show_shap_values(shap_values, feature_names, preds_for_main, output_labels,
                         is_multi_output=is_multi_output, all_outputs=all_outputs)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save results to Excel
    if save_excel:
        # For multi-output, we need to get SHAP values from the first output for the Excel
        if is_multi_output and not is_classification:
            # Get SHAP from first output
            first_shap = all_explainers[0].shap_values(X_df_aligned)
            if isinstance(first_shap, list) and len(first_shap) > 0:
                first_shap = first_shap[0]
            if isinstance(first_shap, np.ndarray) and first_shap.ndim > 2:
                first_shap = first_shap[:, :, 0]
            shap_array = np.array(first_shap)

            # Multi-output regression
            save_results_to_excel(
                X_df_aligned, shap_array, feature_names, preds_for_main, output_dir,
                output_labels, original_predictions=all_outputs, is_multi_output=True,
                is_classification=False
            )
        elif not is_classification and not is_multi_output:
            # Single-output regression with binning
            shap_array = np.array(shap_values)
            save_results_to_excel(
                X_df_aligned, shap_array, feature_names, preds, output_dir,
                bin_labels_for_display, original_predictions=original_preds,
                is_classification=False
            )
        else:
            # Classification (both single and multi-output)
            shap_array = np.array(shap_values)
            save_results_to_excel(
                X_df_aligned, shap_array, feature_names, preds_for_main, output_dir,
                output_labels, is_multi_output=is_multi_output, all_outputs=all_outputs,
                is_classification=True
            )
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
        task_type = "classification" if is_classification else "regression"
        if is_multi_output:
            task_type = f"multioutput_{task_type}"
        plots_output_dir = os.path.join(output_dir, f"{timestamp}_{task_type}_plots")
        os.makedirs(plots_output_dir, exist_ok=True)

        plot_shap_values(
            shap_values,
            X_df_aligned,
            feature_names,
            preds if not (is_multi_output and not is_classification) else preds_for_main,
            plots_output_dir,
            selected_plots=selected_plots,
            explainer=explainer,
            output_labels=output_labels,
            is_multi_output=is_multi_output,
            all_outputs=all_outputs,
            model=model,
            all_explainers=all_explainers,
            is_classification=is_classification
        )
    else:
        print("Plot generation disabled (config).")

    # Generate Jupyter Notebook with analysis
    if generate_notebook and plots_output_dir is not None:
        # Prepare model information for the notebook
        if is_classification:
            if is_multi_output:
                # Multi-output classification
                model_info = {
                    'model_type': type(model).__name__,
                    'task_type': 'Multi-output Classification',
                    'n_features': len(feature_names),
                    'n_outputs': num_outputs,
                    'n_samples': len(X_df_aligned),
                    'feature_names': feature_names,
                    'output_labels': output_labels
                }
            else:
                # Single-output classification
                unique_classes = np.unique(preds_for_main)
                model_info = {
                    'model_type': type(model).__name__,
                    'task_type': 'Classification',
                    'n_features': len(feature_names),
                    'n_classes': len(unique_classes),
                    'n_samples': len(X_df_aligned),
                    'feature_names': feature_names,
                    'classes': unique_classes.tolist(),
                    'output_labels': output_labels
                }
        else:
            if is_multi_output:
                model_info = {
                    'model_type': type(model).__name__,
                    'task_type': 'Multi-output Regression',
                    'n_features': len(feature_names),
                    'n_outputs': num_outputs,
                    'n_samples': len(X_df_aligned),
                    'feature_names': feature_names,
                    'output_labels': output_labels
                }
            else:
                unique_classes = np.unique(preds)
                model_info = {
                    'model_type': type(model).__name__,
                    'task_type': 'Regression',
                    'n_features': len(feature_names),
                    'n_classes': len(unique_classes),
                    'n_samples': len(X_df_aligned),
                    'feature_names': feature_names,
                    'classes': unique_classes.tolist(),
                    'output_labels': bin_labels_for_display,
                    'prediction_range': f"[{original_preds.min():.2f}, {original_preds.max():.2f}]",
                    'n_bins': len(unique_classes)
                }

        try:
            notebook_path = generate_analysis_notebook(
                plots_output_dir,
                model_info=model_info
            )
            print(f"Notebook generated: {notebook_path}")

        except Exception as e:
            print(f"\nError generating notebook: {e}")
            import traceback
            traceback.print_exc()
    elif not generate_plots:
        print("\nNotebook generation skipped (plots were not generated)")
    else:
        print("\nNotebook generation disabled (config).")