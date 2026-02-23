import shap
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import warnings

matplotlib.rcParams['figure.max_open_warning'] = 0
# Suppress tight_layout warnings
warnings.filterwarnings('ignore', message='.*Tight layout not applied.*')


def get_class_mapping(predictions, output_labels=None, is_multi_output=False, output_idx=None):
    """ For multi-output classification, output_labels should be structured as:
    output_labels = {
        "0": {"0": "Label for output 0, class 0", "1": "Label for output 0, class 1"},
        "1": {"0": "Label for output 1, class 0", "1": "Label for output 1, class 1"}
    }"""
    unique_preds = np.unique(predictions)

    if output_labels is None:
        return {pred: f"Class {pred}" for pred in unique_preds}

    # For multi-output classification
    if is_multi_output and output_idx is not None:
        output_key = str(output_idx)
        if output_key in output_labels and isinstance(output_labels[output_key], dict):
            return {
                pred: output_labels[output_key].get(str(int(pred)), f"Class {pred}")
                for pred in unique_preds
            }
        else:
            return {pred: f"Output {output_idx} - Class {pred}" for pred in unique_preds}

    # For single-output classification
    if isinstance(unique_preds[0], (int, np.integer)):
        return {
            pred: output_labels.get(str(int(pred)), f"Class {pred}")
            for pred in unique_preds
        }
    else:
        return {
            pred: output_labels.get(str(pred), str(pred))
            for pred in unique_preds
        }


def compute_shap_values(model, X_df):
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    model_class_name = type(model).__name__

    # For MultiOutput wrappers, we don't create a TreeExplainer for the wrapper
    # Instead, we'll create explainers for each individual estimator later
    # Here we just return the wrapper itself
    if model_class_name in ['MultiOutputRegressor', 'MultiOutputClassifier']:
        # Return None for explainer and shap_values - will be computed per output
        return None, None, X_df

    # For regular models (single-output), create explainer normally
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    return explainer, shap_values, X_df


def show_shap_values(shap_array, feature_names, preds=None, output_labels=None,
                     is_multi_output=False, all_outputs=None):
    shap_array = np.array(shap_array)

    # For multi-output, show only first output in console (to avoid clutter)
    preds_to_show = None
    if is_multi_output and all_outputs is not None:
        preds_to_show = all_outputs[:, 0]
        output_idx = 0
        class_mapping = get_class_mapping(preds_to_show, output_labels, is_multi_output, output_idx)
        print(f"\nShowing SHAP values for first output only (Output 0)")
    else:
        preds_to_show = preds
        class_mapping = get_class_mapping(preds, output_labels) if preds is not None else None

    if shap_array.ndim == 3:
        if preds_to_show is None:
            raise ValueError("Predictions required")

        if isinstance(preds_to_show[0], (int, np.integer)):
            shap_array = shap_array[np.arange(len(preds_to_show)), :, preds_to_show]
        else:
            unique_classes = sorted(set(preds_to_show))
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            pred_indices = np.array([class_to_idx[p] for p in preds_to_show])
            shap_array = shap_array[np.arange(len(preds_to_show)), :, pred_indices]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    print("\nSHAP values per feature:\n")
    max_samples = min(100, len(shap_array))

    preds_display = preds_to_show

    for i, row in enumerate(shap_array[:max_samples]):
        if preds_display is not None and class_mapping:
            label = class_mapping[preds_display[i]]
            print(f"Sample {i} (predicted: {label}):")
        else:
            print(f"Sample {i}:")

        for j, feat in enumerate(feature_names):
            print(f"  {feat}: {float(row[j]):+.4f}")
        print("-" * 40)


def save_results_to_excel(X_df, shap_array, feature_names, preds, output_dir,
                          output_labels=None, original_predictions=None,
                          is_multi_output=False, all_outputs=None, is_classification=True):
    from datetime import datetime

    shap_array = np.array(shap_array)

    # For multi-output classification
    if is_multi_output and is_classification and all_outputs is not None:
        # Create columns for each output
        pred_dfs = []
        for output_idx in range(all_outputs.shape[1]):
            output_preds = all_outputs[:, output_idx]
            class_mapping = get_class_mapping(output_preds, output_labels,
                                              is_multi_output=True, output_idx=output_idx)
            pred_labels = [class_mapping[p] for p in output_preds]

            output_name = output_labels.get(f"{output_idx}_name",
                                            f"Output_{output_idx}") if output_labels else f"Output_{output_idx}"
            pred_dfs.append(pd.DataFrame({
                f"{output_name}_Class": output_preds,
                f"{output_name}_Label": pred_labels
            }))

        pred_df = pd.concat(pred_dfs, axis=1)

    elif is_multi_output and not is_classification and original_predictions is not None:
        # Multi-output regression
        pred_dfs = []
        for output_idx in range(original_predictions.shape[1]):
            output_name = output_labels.get(str(output_idx),
                                            f"Output_{output_idx}") if output_labels else f"Output_{output_idx}"
            pred_dfs.append(pd.DataFrame({
                f"{output_name}_Value": original_predictions[:, output_idx]
            }))
        pred_df = pd.concat(pred_dfs, axis=1)

    else:
        # Single-output (classification or regression)
        class_mapping = get_class_mapping(preds, output_labels)
        pred_labels = [class_mapping[p] for p in preds]

        if original_predictions is not None:
            pred_df = pd.DataFrame({
                "Predicted_Value": original_predictions,
                "Predicted_Bin": preds,
                "Bin_Label": pred_labels
            })
        else:
            pred_df = pd.DataFrame({
                "Predicted_Class": preds,
                "Predicted_Label": pred_labels
            })

    # Handle SHAP values
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions required")

        # Handle 2D preds by flattening
        preds_1d = preds.flatten() if isinstance(preds, np.ndarray) and preds.ndim == 2 else preds

        if isinstance(preds_1d[0], (int, np.integer)):
            # Only apply if preds is 1D
            if isinstance(preds, np.ndarray) and preds.ndim == 1:
                shap_array = shap_array[np.arange(len(preds)), :, preds]
        else:
            # Use np.unique instead of sorted(set()) for arrays
            unique_classes = np.unique(preds_1d).tolist()
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            pred_indices = np.array([class_to_idx[p] for p in preds_1d])

            # Only apply if preds is 1D
            if isinstance(preds, np.ndarray) and preds.ndim == 1:
                shap_array = shap_array[np.arange(len(preds)), :, pred_indices]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    shap_df = pd.DataFrame(shap_array, columns=[f"SHAP_{f}" for f in feature_names])

    output_df = pd.concat([X_df.reset_index(drop=True), pred_df, shap_df], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"shap_results_{timestamp}.xlsx")

    try:
        output_df.to_excel(output_path, index=False)
        print(f"SHAP results saved to: {output_path}")
    except Exception as e:
        csv_path = os.path.join(output_dir, f"shap_results_{timestamp}.csv")
        output_df.to_csv(csv_path, index=False)
        print(f"Results saved as CSV: {csv_path}")


def get_class_label(class_val, output_labels, is_multi_output=False, output_idx=None):
    if is_multi_output and output_idx is not None:
        output_key = str(output_idx)
        if output_key in output_labels and isinstance(output_labels[output_key], dict):
            if isinstance(class_val, (int, np.integer)):
                return output_labels[output_key].get(str(int(class_val)), f"Class {class_val}")
            else:
                return output_labels[output_key].get(str(class_val), str(class_val))
        return f"Output {output_idx} - Class {class_val}"

    if isinstance(class_val, (int, np.integer)):
        if output_labels:
            return output_labels.get(str(int(class_val)), f"Class {class_val}")
        return f"Class {class_val}"
    else:
        if output_labels:
            return output_labels.get(str(class_val), str(class_val))
        return str(class_val)


def get_safe_filename(class_val, output_labels, is_multi_output=False, output_idx=None):
    label = get_class_label(class_val, output_labels, is_multi_output, output_idx)
    safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_label = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe_label)
    return safe_label


def get_expected_value_for_class(explainer, shap_values, class_val, unique_classes):
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        if len(explainer.expected_value) > 1:
            if isinstance(class_val, (int, np.integer)):
                class_idx = int(class_val)
                if class_idx < len(explainer.expected_value):
                    return explainer.expected_value[class_idx]
            else:
                sorted_classes = sorted(unique_classes)
                class_idx = sorted_classes.index(class_val)
                if class_idx < len(explainer.expected_value):
                    return explainer.expected_value[class_idx]
        return explainer.expected_value[0] if len(explainer.expected_value) > 0 else 0
    return explainer.expected_value


def extract_shap_for_class(shap_values, class_val, unique_classes, n_samples, n_features):
    shap_values_array = np.array(shap_values)

    if shap_values_array.ndim == 3:
        if isinstance(class_val, (int, np.integer)):
            class_idx = int(class_val)
        else:
            sorted_classes = sorted(unique_classes)
            class_idx = sorted_classes.index(class_val)

        if shap_values_array.shape[0] == n_samples:
            if class_idx < shap_values_array.shape[2]:
                return shap_values_array[:, :, class_idx]
        elif shap_values_array.shape[2] == n_samples:
            if class_idx < shap_values_array.shape[0]:
                return shap_values_array[class_idx, :, :].T
        elif shap_values_array.shape[1] == n_samples:
            if class_idx < shap_values_array.shape[0]:
                return shap_values_array[class_idx, :, :]
    elif shap_values_array.ndim == 2:
        return shap_values_array

    return None


def create_waterfall_plots(shap_values, shap_array, X_df, feature_names, preds,
                           unique_classes, explainer, output_dir, class_mapping, output_labels,
                           is_multi_output=False, output_idx=None):
    waterfall_dir = os.path.join(output_dir, "waterfall_plots")
    os.makedirs(waterfall_dir, exist_ok=True)

    shap_values_raw = np.array(shap_values)

    if shap_values_raw.ndim == 3:
        if len(unique_classes) == 2:
            class_1_val = sorted(unique_classes)[1]
            shap_for_waterfall = extract_shap_for_class(
                shap_values, class_1_val, unique_classes, len(X_df), len(feature_names)
            )
            expected_value = get_expected_value_for_class(explainer, shap_values, class_1_val, unique_classes)
        else:
            shap_for_waterfall = shap_array
            expected_value = get_expected_value_for_class(explainer, shap_values, unique_classes[0], unique_classes)
    else:
        shap_for_waterfall = shap_array
        expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0

    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
    expected_value = float(expected_value)

    for class_val in unique_classes:
        class_mask = preds == class_val
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        class_label = class_mapping[class_val]
        safe_filename = get_safe_filename(class_val, output_labels, is_multi_output, output_idx)

        n_samples_to_plot = min(5, len(class_indices))
        class_shap_magnitudes = np.abs(shap_for_waterfall[class_indices]).sum(axis=1)
        sorted_idx = np.argsort(class_shap_magnitudes)

        if n_samples_to_plot >= 3:
            sample_positions = [0, len(sorted_idx) // 2, -1]
            if n_samples_to_plot >= 5:
                sample_positions = [0, len(sorted_idx) // 4, len(sorted_idx) // 2, 3 * len(sorted_idx) // 4, -1]
        else:
            sample_positions = list(range(n_samples_to_plot))

        selected_indices = [class_indices[sorted_idx[pos]] for pos in sample_positions[:n_samples_to_plot]]

        for sample_idx in selected_indices:
            try:
                sample_shap = shap_for_waterfall[sample_idx]
                sample_features = X_df.iloc[sample_idx].values

                sample_shap = np.array(sample_shap).flatten()
                sample_features = np.array(sample_features).flatten()
                sample_features_display = np.round(sample_features, 1)

                explanation = shap.Explanation(
                    values=sample_shap,
                    base_values=float(expected_value),
                    data=sample_features_display,
                    feature_names=feature_names
                )

                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(explanation, show=False)
                title = f"SHAP Waterfall Plot - Sample {sample_idx} ({class_label})"
                if is_multi_output and output_idx is not None:
                    title = f"SHAP Waterfall - Sample {sample_idx} - Output {output_idx} ({class_label})"
                plt.title(title, fontsize=12, fontweight='bold')
                try:
                    plt.tight_layout()
                except:
                    pass

                filename = f"waterfall_sample_{sample_idx}_{safe_filename}.png"
                if is_multi_output and output_idx is not None:
                    filename = f"waterfall_output{output_idx}_sample_{sample_idx}_{safe_filename}.png"
                plt.savefig(os.path.join(waterfall_dir, filename), bbox_inches="tight", dpi=150)
                plt.close()

            except Exception as e:
                print(f"Error creating waterfall for sample {sample_idx}: {e}")
                continue

    for class_val in unique_classes:
        class_mask = preds == class_val

        if class_mask.sum() == 0:
            continue

        class_label = class_mapping[class_val]
        safe_filename = get_safe_filename(class_val, output_labels, is_multi_output, output_idx)

        try:
            mean_shap_values = shap_for_waterfall[class_mask].mean(axis=0)
            mean_feature_values = X_df[class_mask].mean().values

            mean_shap_values = np.array(mean_shap_values).flatten()
            mean_feature_values = np.array(mean_feature_values).flatten()
            mean_feature_values_display = np.round(mean_feature_values, 1)

            explanation = shap.Explanation(
                values=mean_shap_values,
                base_values=float(expected_value),
                data=mean_feature_values_display,
                feature_names=feature_names
            )

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, show=False)
            title = f"SHAP Waterfall Plot - Mean Values for {class_label}\n({class_mask.sum()} samples)"
            if is_multi_output and output_idx is not None:
                title = f"SHAP Waterfall - Mean Values - Output {output_idx} ({class_label})\n({class_mask.sum()} samples)"
            plt.title(title, fontsize=12, fontweight='bold')
            try:
                plt.tight_layout()
            except:
                pass

            filename = f"waterfall_mean_{safe_filename}.png"
            if is_multi_output and output_idx is not None:
                filename = f"waterfall_output{output_idx}_mean_{safe_filename}.png"
            plt.savefig(os.path.join(waterfall_dir, filename), bbox_inches="tight", dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error creating aggregate waterfall for {class_label}: {e}")
            continue


def plot_shap_values(shap_values, X_df, feature_names, preds, output_dir,
                     selected_plots=None, explainer=None, output_labels=None,
                     is_multi_output=False, all_outputs=None, model=None,
                     all_explainers=None, is_classification=True):
    os.makedirs(output_dir, exist_ok=True)
    shap_array = np.array(shap_values)

    # Get class mapping (for first output if multi-output classification)
    if is_multi_output and is_classification and all_outputs is not None:
        preds_for_mapping = all_outputs[:, 0]
        class_mapping = get_class_mapping(preds_for_mapping, output_labels,
                                          is_multi_output=True, output_idx=0)
    else:
        class_mapping = get_class_mapping(preds, output_labels)

    # Handle multi-class SHAP
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions required")

        # Handle 2D preds (multi-output) by flattening first
        preds_1d = preds.flatten() if isinstance(preds, np.ndarray) and preds.ndim == 2 else preds

        if isinstance(preds_1d[0], (int, np.integer)):
            # For 1D preds, use directly
            if isinstance(preds, np.ndarray) and preds.ndim == 1:
                shap_array = shap_array[np.arange(len(preds)), :, preds]
        else:
            # Use np.unique instead of sorted(set()) for arrays
            unique_classes_temp = np.unique(preds_1d).tolist()
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes_temp)}
            pred_indices = np.array([class_to_idx[p] for p in preds_1d])

            # Only apply if preds is 1D
            if isinstance(preds, np.ndarray) and preds.ndim == 1:
                shap_array = shap_array[np.arange(len(preds)), :, pred_indices]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    if selected_plots is None:
        selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'decision_map',
                          'interactive_decision_map', 'heatmap', 'interactive_heatmap', 'waterfall']

    # For multi-output regression, preds might be 2D: (n_samples, n_outputs)
    # For single-output, preds is 1D: (n_samples,)
    if isinstance(preds, np.ndarray) and preds.ndim == 2:
        # Multi-output: use first output column for unique_classes
        unique_classes = np.unique(preds[:, 0])
    else:
        # Single-output
        unique_classes = np.unique(preds)

    n_classes = len(unique_classes)

    # Multi-Output Classification
    if is_multi_output and is_classification and model is not None and all_outputs is not None and hasattr(model,
                                                                                                           'estimators_'):

        num_outputs = len(model.estimators_)
        output_names = {}
        for i in range(num_outputs):
            output_names[i] = output_labels.get(f"{i}_name", f"Output_{i}") if output_labels else f"Output_{i}"

        # Generate plots for each output
        for idx in range(num_outputs):
            name = output_names[idx]
            output_preds = all_outputs[:, idx]

            try:
                explainer_out = shap.TreeExplainer(model.estimators_[idx])
                shap_out = np.array(explainer_out.shap_values(X_df))

                # Handle different SHAP value structures
                if shap_out.ndim == 3:
                    # Multi-class output, extract for predicted class
                    shap_out = shap_out[np.arange(len(output_preds)), :, output_preds]
                if shap_out.ndim == 1:
                    shap_out = shap_out.reshape(1, -1)

                subdir = os.path.join(output_dir, name.replace(" ", "_"))
                os.makedirs(subdir, exist_ok=True)

                # Get class mapping for this output
                output_class_mapping = get_class_mapping(output_preds, output_labels,
                                                         is_multi_output=True, output_idx=idx)
                unique_output_classes = np.unique(output_preds)

                # Beeswarm plot per output
                if 'beeswarm' in selected_plots:
                    try:
                        unique_preds = np.unique(output_preds)
                        if len(unique_preds) <= 1:
                            print(f"Skipping beeswarm plot for {name}: only one class present")
                        else:
                            plt.figure(figsize=(10, 6))
                            shap.summary_plot(shap_out, X_df, feature_names=feature_names, show=False)
                            plt.title(f"SHAP Beeswarm - {name}", fontsize=12, fontweight='bold')
                            try:
                                plt.tight_layout()
                            except:
                                pass
                            plt.savefig(os.path.join(subdir, "shap_beeswarm.png"), bbox_inches="tight", dpi=150)
                            plt.close()
                    except Exception as e:
                        print(f"Skipping beeswarm plot for {name}: {str(e)[:100]}...")
                        plt.close()

                # Bar plot per output
                if 'bar' in selected_plots:
                    try:
                        plt.figure(figsize=(10, 6))
                        shap.summary_plot(shap_out, X_df, feature_names=feature_names, plot_type="bar", show=False)
                        plt.title(f"SHAP Feature Importance - {name}", fontsize=12, fontweight='bold')
                        try:
                            plt.tight_layout()
                        except:
                            pass
                        plt.savefig(os.path.join(subdir, "shap_bar.png"), bbox_inches="tight", dpi=150)
                        plt.close()
                    except Exception as e:
                        print(f"Error creating bar plot for {name}: {e}")
                        plt.close()

                # Violin plot per output
                if 'violin' in selected_plots:
                    try:
                        # Check if there's enough variance for violin plot
                        unique_preds = np.unique(output_preds)

                        # Skip if only one class or no variance in SHAP values
                        if len(unique_preds) <= 1:
                            print(f"Skipping violin plot for {name}: only one class present")
                        elif np.std(shap_out) < 1e-10:
                            print(f"Skipping violin plot for {name}: SHAP values have no variance")
                        else:
                            plt.figure(figsize=(10, 6))
                            shap.summary_plot(shap_out, X_df, feature_names=feature_names, plot_type="violin",
                                              show=False)
                            plt.title(f"SHAP Violin - {name}", fontsize=12, fontweight='bold')
                            try:
                                plt.tight_layout()
                            except:
                                pass
                            plt.savefig(os.path.join(subdir, "shap_violin.png"), bbox_inches="tight", dpi=150)
                            plt.close()
                    except (np.linalg.LinAlgError, ValueError) as e:
                        print(f"Skipping violin plot for {name}: {str(e)[:100]}...")
                        plt.close()
                    except Exception as e:
                        print(f"Error creating violin plot for {name}: {e}")
                        plt.close()

                # Dependence plots per output
                if 'dependence' in selected_plots:
                    for feat in feature_names:
                        try:
                            plt.figure(figsize=(10, 7))
                            shap.dependence_plot(feat, shap_out, X_df, show=False)
                            plt.title(f"SHAP Dependence: {feat} - {name}", fontsize=12, fontweight='bold')
                            try:
                                plt.tight_layout()
                            except:
                                pass
                            safe_feat = "".join(c if c.isalnum() or c in "_-" else "_" for c in feat.replace(" ", "_"))
                            plt.savefig(os.path.join(subdir, f"dependence_{safe_feat}.png"), bbox_inches="tight",
                                        dpi=150)
                            plt.close()
                        except Exception as e:
                            print(f"Error creating dependence plot for {feat}, {name}: {e}")
                            plt.close()

                # Decision map per output (for binary classification)
                if 'decision_map' in selected_plots and len(unique_output_classes) <= 2:
                    try:
                        # For binary classification, use class 1
                        raw_shap = explainer_out.shap_values(X_df)

                        # Get SHAP values for class 1
                        if isinstance(raw_shap, list) and len(raw_shap) == 2:
                            shap_for_decision = raw_shap[1]  # Class 1 SHAP values
                            expected_value = explainer_out.expected_value[1] if isinstance(explainer_out.expected_value,
                                                                                           list) else explainer_out.expected_value
                        else:
                            shap_for_decision = shap_out
                            expected_value = explainer_out.expected_value

                        plt.figure(figsize=(12, 8))
                        shap.decision_plot(
                            expected_value,
                            shap_for_decision,
                            X_df,
                            feature_names=feature_names,
                            show=False,
                            link='identity'
                        )
                        plt.title(f"SHAP Decision Plot - {name}\n({len(X_df)} samples)",
                                  fontsize=11, fontweight='bold')
                        try:
                            plt.tight_layout()
                        except:
                            pass
                        plt.savefig(os.path.join(subdir, "shap_decision.png"), bbox_inches="tight", dpi=150)
                        plt.close()
                    except Exception as e:
                        print(f"Error creating decision plot for {name}: {e}")
                        plt.close()

                # Heatmap per output
                if 'heatmap' in selected_plots and len(unique_output_classes) > 1:
                    try:
                        shap_df = pd.DataFrame(shap_out, columns=feature_names)

                        mean_abs_shap = shap_df.abs().mean()
                        feature_order = mean_abs_shap.sort_values(ascending=False).index
                        shap_sorted = shap_df[feature_order]

                        # Sort by predictions
                        sorted_indices = np.argsort(output_preds)
                        shap_sorted = shap_sorted.iloc[sorted_indices]
                        preds_sorted = output_preds[sorted_indices]

                        max_abs = np.abs(shap_sorted.values).max()

                        n_samples, n_features = shap_sorted.shape
                        figsize = (max(14, n_samples / 40), max(8, n_features / 2))

                        fig, ax = plt.subplots(figsize=figsize)

                        im = ax.imshow(shap_sorted.T, aspect='auto', cmap='RdBu_r',
                                       vmin=-max_abs, vmax=+max_abs, interpolation='nearest')

                        cbar = plt.colorbar(im, ax=ax)
                        cbar.set_label("SHAP value", fontsize=10)

                        ax.set_yticks(np.arange(n_features))
                        ax.set_yticklabels(shap_sorted.columns)

                        class_0_label = output_class_mapping.get(0, "Class 0")
                        class_1_label = output_class_mapping.get(1, "Class 1")

                        ax.set_xlabel(f"Samples (sorted by class: {class_0_label}, {class_1_label})", fontsize=10)
                        ax.set_ylabel("Features (sorted by mean |SHAP|)", fontsize=10)
                        ax.set_title(f"SHAP Heatmap - {name}", fontsize=12, fontweight='bold')

                        # Add vertical line to separate classes
                        class_0_count = (preds_sorted == 0).sum()
                        if class_0_count > 0 and class_0_count < len(preds_sorted):
                            ax.axvline(x=class_0_count - 0.5, color='yellow', linewidth=2, linestyle='--')

                            # Add class labels
                            ax.text(class_0_count / 2, -0.5,
                                    f'{class_0_label}\n{class_0_count} samples',
                                    ha='center', va='top', fontsize=9, fontweight='bold')

                            class_1_count = (preds_sorted == 1).sum()
                            ax.text(class_0_count + class_1_count / 2, -0.5,
                                    f'{class_1_label}\n{class_1_count} samples',
                                    ha='center', va='top', fontsize=9, fontweight='bold')

                        try:
                            plt.tight_layout()
                        except:
                            pass
                        plt.savefig(os.path.join(subdir, "shap_heatmap.png"), bbox_inches="tight", dpi=150)
                        plt.close()
                    except Exception as e:
                        print(f"Error creating heatmap for {name}: {e}")
                        plt.close()

                # Waterfall plots per output
                if 'waterfall' in selected_plots:
                    create_waterfall_plots(
                        explainer_out.shap_values(X_df), shap_out, X_df, feature_names,
                        output_preds, unique_output_classes, explainer_out, subdir,
                        output_class_mapping, output_labels, is_multi_output=True, output_idx=idx
                    )

            except Exception as e:
                print(f"Error processing output {name}: {e}")
                import traceback
                traceback.print_exc()

        # Unified plots across all outputs
        all_shap = []
        for idx in range(num_outputs):
            try:
                exp = shap.TreeExplainer(model.estimators_[idx])
                s = np.array(exp.shap_values(X_df))
                output_preds = all_outputs[:, idx]

                if s.ndim == 3:
                    s = s[np.arange(len(output_preds)), :, output_preds]
                if s.ndim == 1:
                    s = s.reshape(1, -1)
                all_shap.append(s)
            except:
                pass

        if all_shap and 'bar' in selected_plots:
            importances = [np.abs(s).mean(axis=0) for s in all_shap]
            unified_imp = np.mean(importances, axis=0)
            sorted_idx = np.argsort(unified_imp)

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_names)), unified_imp[sorted_idx], color='steelblue')
            plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
            plt.xlabel("Mean |SHAP| (across outputs)", fontsize=11)
            plt.title(f"SHAP Unified Feature Importance ({num_outputs} outputs)", fontsize=11, fontweight='bold')
            try:
                plt.tight_layout()
            except:
                pass
            plt.savefig(os.path.join(output_dir, "shap_bar_unified.png"), bbox_inches="tight", dpi=150)
            plt.close()

        print(f"All plots saved to: {output_dir}\n")
        return

    # Multi-Output Regression
    if is_multi_output and not is_classification and model is not None and all_outputs is not None and hasattr(model,
                                                                                                               'estimators_'):

        num_outputs = len(model.estimators_)
        output_names = {i: output_labels.get(str(i), f"Output_{i}") for i in range(num_outputs)}

        for idx in range(num_outputs):
            name = output_names[idx]
            try:
                explainer_out = shap.TreeExplainer(model.estimators_[idx])
                shap_out = np.array(explainer_out.shap_values(X_df))
                if shap_out.ndim > 2:
                    shap_out = shap_out[:, :, 0]
                if shap_out.ndim == 1:
                    shap_out = shap_out.reshape(1, -1)

                subdir = os.path.join(output_dir, name.replace(" ", "_"))
                os.makedirs(subdir, exist_ok=True)

                if 'beeswarm' in selected_plots:
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_out, X_df, feature_names=feature_names, show=False)
                    plt.title(f"SHAP Beeswarm - {name}", fontsize=12, fontweight='bold')
                    try:
                        plt.tight_layout()
                    except:
                        pass
                    plt.savefig(os.path.join(subdir, "shap_beeswarm.png"), bbox_inches="tight", dpi=150)
                    plt.close()

                if 'bar' in selected_plots:
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_out, X_df, feature_names=feature_names, plot_type="bar", show=False)
                    plt.title(f"SHAP Feature Importance - {name}", fontsize=12, fontweight='bold')
                    try:
                        plt.tight_layout()
                    except:
                        pass
                    plt.savefig(os.path.join(subdir, "shap_bar.png"), bbox_inches="tight", dpi=150)
                    plt.close()

                if 'violin' in selected_plots:
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_out, X_df, feature_names=feature_names, plot_type="violin", show=False)
                    plt.title(f"SHAP Violin - {name}", fontsize=12, fontweight='bold')
                    try:
                        plt.tight_layout()
                    except:
                        pass
                    plt.savefig(os.path.join(subdir, "shap_violin.png"), bbox_inches="tight", dpi=150)
                    plt.close()

                if 'dependence' in selected_plots:
                    for feat in feature_names:
                        plt.figure(figsize=(10, 7))
                        shap.dependence_plot(feat, shap_out, X_df, show=False)
                        plt.title(f"SHAP Dependence: {feat} - {name}", fontsize=12, fontweight='bold')
                        try:
                            plt.tight_layout()
                        except:
                            pass
                        safe_feat = "".join(c if c.isalnum() or c in "_-" else "_" for c in feat.replace(" ", "_"))
                        plt.savefig(os.path.join(subdir, f"dependence_{safe_feat}.png"), bbox_inches="tight", dpi=150)
                        plt.close()

            except Exception as e:
                print(f"Error: {e}")

        all_shap = []
        for idx in range(num_outputs):
            try:
                exp = shap.TreeExplainer(model.estimators_[idx])
                s = np.array(exp.shap_values(X_df))
                if s.ndim > 2:
                    s = s[:, :, 0]
                if s.ndim == 1:
                    s = s.reshape(1, -1)
                all_shap.append(s)
            except:
                pass

        if all_shap:
            if 'bar' in selected_plots:
                importances = [np.abs(s).mean(axis=0) for s in all_shap]
                unified_imp = np.mean(importances, axis=0)
                sorted_idx = np.argsort(unified_imp)

                plt.figure(figsize=(10, 6))
                plt.barh(range(len(feature_names)), unified_imp[sorted_idx], color='steelblue')
                plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
                plt.xlabel("Mean |SHAP| (across outputs)", fontsize=11)
                plt.title(f"SHAP Unified ({num_outputs} outputs)", fontsize=11, fontweight='bold')
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(output_dir, "shap_bar_unified.png"), bbox_inches="tight", dpi=150)
                plt.close()

            if 'dependence' in selected_plots:
                stacked = np.vstack(all_shap)
                stacked_X = pd.concat([X_df] * num_outputs, ignore_index=True)

                for feat in feature_names:
                    plt.figure(figsize=(10, 7))
                    shap.dependence_plot(feat, stacked, stacked_X, show=False)
                    plt.title(f"SHAP Dependence: {feat} - Unified ({num_outputs} outputs)", fontsize=12,
                              fontweight='bold')
                    try:
                        plt.tight_layout()
                    except:
                        pass
                    safe_feat = "".join(c if c.isalnum() or c in "_-" else "_" for c in feat.replace(" ", "_"))
                    plt.savefig(os.path.join(output_dir, f"dependence_{safe_feat}_unified.png"), bbox_inches="tight",
                                dpi=150)
                    plt.close()

        # Waterfall plots for multi-output regression
        if 'waterfall' in selected_plots and all_explainers is not None:
            waterfall_dir = os.path.join(output_dir, 'waterfall_plots')
            os.makedirs(waterfall_dir, exist_ok=True)

            n_samples_to_plot = min(10, len(X_df))

            for idx in range(num_outputs):
                name = output_names[idx]
                safe_name = name.replace(" ", "_")

                try:
                    explainer_out = all_explainers[idx]
                    shap_out = np.array(explainer_out.shap_values(X_df))
                    if shap_out.ndim > 2:
                        shap_out = shap_out[:, :, 0]
                    if shap_out.ndim == 1:
                        shap_out = shap_out.reshape(1, -1)

                    expected_value = explainer_out.expected_value
                    output_predictions = all_outputs[:, idx]

                    for i in range(n_samples_to_plot):
                        try:
                            sample_shap = np.array(shap_out[i]).flatten()
                            sample_features = np.array(X_df.iloc[i].values).flatten()
                            pred_value = float(output_predictions[i])

                            if isinstance(expected_value, (list, np.ndarray)):
                                base_val = float(expected_value[0]) if len(expected_value) > 0 else 0.0
                            else:
                                base_val = float(expected_value)

                            explanation = shap.Explanation(
                                values=sample_shap,
                                base_values=base_val,
                                data=sample_features,
                                feature_names=feature_names
                            )

                            plt.figure(figsize=(10, 6))
                            shap.waterfall_plot(explanation, show=False)
                            plt.title(f'Waterfall - Sample {i} - {name}\n(Prediction: {pred_value:.4f})',
                                      fontsize=12, fontweight='bold')
                            try:
                                plt.tight_layout()
                            except:
                                pass

                            filename = f"waterfall_sample_{i}_{safe_name}.png"
                            plt.savefig(os.path.join(waterfall_dir, filename), bbox_inches="tight", dpi=150)
                            plt.close()
                        except Exception as e:
                            print(f"Error creating waterfall for sample {i}, {name}: {e}")
                            plt.close()

                    try:
                        mean_shap = np.array(shap_out.mean(axis=0)).flatten()
                        mean_features = np.array(X_df.mean().values).flatten()
                        mean_pred = float(output_predictions.mean())

                        if isinstance(expected_value, (list, np.ndarray)):
                            base_val = float(expected_value[0]) if len(expected_value) > 0 else 0.0
                        else:
                            base_val = float(expected_value)

                        explanation = shap.Explanation(
                            values=mean_shap,
                            base_values=base_val,
                            data=mean_features,
                            feature_names=feature_names
                        )

                        plt.figure(figsize=(10, 6))
                        shap.waterfall_plot(explanation, show=False)
                        plt.title(f'Waterfall - Mean Values - {name}\n(Mean Prediction: {mean_pred:.4f})',
                                  fontsize=12, fontweight='bold')
                        try:
                            plt.tight_layout()
                        except:
                            pass

                        filename = f"waterfall_mean_{safe_name}.png"
                        plt.savefig(os.path.join(waterfall_dir, filename), bbox_inches="tight", dpi=150)
                        plt.close()

                    except Exception as e:
                        print(f"Error creating mean waterfall for {name}: {e}")
                        plt.close()

                except Exception as e:
                    print(f"Error processing waterfall plots for {name}: {e}")
                    continue

        print(f"All plots saved to: {output_dir}\n")
        return

    # Single-output classification/regression
    for class_val in unique_classes:
        mask = preds == class_val
        n_samples_class = mask.sum()

        if n_samples_class == 0:
            continue

        shap_class = shap_array[mask]
        X_class = X_df[mask]
        class_label = class_mapping[class_val]
        safe_filename = get_safe_filename(class_val, output_labels)

        if 'beeswarm' in selected_plots:
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_class, X_class, feature_names=feature_names, show=False)
                plt.title(f"SHAP Beeswarm Plot - {class_label}", fontsize=12, fontweight='bold')
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(output_dir, f"shap_beeswarm_{safe_filename}.png"), bbox_inches="tight",
                            dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating beeswarm plot for {class_label}: {e}")

        if 'bar' in selected_plots:
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="bar", show=False)
                plt.title(f"SHAP Feature Importance - {class_label}", fontsize=12, fontweight='bold')
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(output_dir, f"shap_bar_{safe_filename}.png"), bbox_inches="tight", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating bar plot for {class_label}: {e}")

        if 'violin' in selected_plots:
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="violin", show=False)
                plt.title(f"SHAP Violin Plot - {class_label}", fontsize=12, fontweight='bold')
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(output_dir, f"shap_violin_{safe_filename}.png"), bbox_inches="tight", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating violin plot for {class_label}: {e}")

        if 'dependence' in selected_plots:
            for feat in feature_names:
                try:
                    plt.figure(figsize=(10, 7))
                    shap.dependence_plot(feat, shap_class, X_class, show=False)
                    plt.title(f"SHAP Dependence: {feat} - {class_label}", fontsize=12, fontweight='bold')
                    try:
                        plt.tight_layout()
                    except:
                        pass

                    safe_feat = "".join(c if c.isalnum() or c in "_-" else "_" for c in feat.replace(" ", "_"))
                    plt.savefig(os.path.join(output_dir, f"dependence_{safe_feat}_{safe_filename}.png"),
                                bbox_inches="tight", dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"Error creating dependence plot for {feat}, {class_label}: {e}")

        if 'decision_map' in selected_plots:
            try:
                expected_value = get_expected_value_for_class(explainer, shap_values, class_val, unique_classes)

                plt.figure(figsize=(12, 8))
                shap.decision_plot(
                    expected_value,
                    shap_class,
                    X_class,
                    feature_names=feature_names,
                    show=False,
                    link='identity'
                )
                plt.title(f"SHAP Decision Plot - {class_label}\n({n_samples_class} samples)",
                          fontsize=11, fontweight='bold')
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(output_dir, f"shap_decision_{safe_filename}.png"),
                            bbox_inches="tight", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating decision plot for {class_label}: {e}")

    if 'decision_map' in selected_plots and n_classes == 2:
        try:
            class_1_val = sorted(unique_classes)[1]
            expected_val_class1 = get_expected_value_for_class(explainer, shap_values, class_1_val, unique_classes)
            shap_for_class1 = extract_shap_for_class(shap_values, class_1_val, unique_classes, len(X_df),
                                                     len(feature_names))

            if shap_for_class1 is not None:
                n_total_samples = shap_for_class1.shape[0]
                class1_label = class_mapping[class_1_val]

                plt.figure(figsize=(14, 8))
                shap.decision_plot(
                    expected_val_class1,
                    shap_for_class1,
                    X_df,
                    feature_names=feature_names,
                    show=False,
                    link='identity'
                )

                plt.axvline(x=0.5, color='red', linestyle='--', linewidth=3, alpha=0.8,
                            label='Decision threshold (0.5)')
                plt.legend(loc='best', fontsize=10)
                plt.title(
                    f"SHAP Unified Decision Plot - Probability for {class1_label}\n(All {n_total_samples} samples)",
                    fontsize=12, fontweight='bold')
                plt.xlabel(f"Model output (Probability for {class1_label})", fontsize=11)
                try:
                    plt.tight_layout()
                except:
                    pass
                plt.savefig(os.path.join(output_dir, "shap_decision_unified.png"), bbox_inches="tight", dpi=150)
                plt.close()

        except Exception as e:
            print(f"Error creating unified decision plot: {e}")

    if 'interactive_decision_map' in selected_plots and n_classes == 2:
        try:
            import plotly.graph_objects as go

            class_1_val = sorted(unique_classes)[1]
            expected_val_class1 = get_expected_value_for_class(explainer, shap_values, class_1_val, unique_classes)
            shap_for_class1 = extract_shap_for_class(shap_values, class_1_val, unique_classes, len(X_df),
                                                     len(feature_names))

            if shap_for_class1 is not None:
                n_total_samples = shap_for_class1.shape[0]

                mean_abs_shap = np.abs(shap_for_class1).mean(axis=0)
                feature_order = np.argsort(mean_abs_shap)

                class0_label = class_mapping[sorted(unique_classes)[0]]
                class1_label = class_mapping[sorted(unique_classes)[1]]

                fig = go.Figure()

                for i in range(n_total_samples):
                    cumsum_values = [expected_val_class1]
                    hover_texts = [f"<b>Sample {i}</b><br>Base value: {expected_val_class1:.4f}"]

                    for feat_idx in feature_order:
                        feat_name = feature_names[feat_idx]
                        shap_val = shap_for_class1[i, feat_idx]
                        feat_val = X_df.iloc[i, feat_idx]

                        cumsum_values.append(cumsum_values[-1] + shap_val)
                        hover_texts.append(
                            f"<b>{feat_name}</b><br>Value: {feat_val:.2f}<br>SHAP: {shap_val:+.4f}<br>Cumulative: {cumsum_values[-1]:.4f}"
                        )

                    pred_val = preds[i]
                    color = 'green' if pred_val == class_1_val else 'red'
                    opacity = 0.4 if n_total_samples > 100 else 0.6

                    y_labels = ["Base value"] + [feature_names[j] for j in feature_order]

                    fig.add_trace(go.Scatter(
                        y=y_labels,
                        x=cumsum_values,
                        mode='lines+markers',
                        line=dict(color=color, width=1.5),
                        marker=dict(size=4),
                        opacity=opacity,
                        hovertext=hover_texts,
                        hoverinfo='text',
                        showlegend=False,
                        name=f"Sample {i}"
                    ))

                fig.add_vline(x=0.5, line_dash="dash", line_color="red", line_width=3,
                              annotation_text="Threshold (0.5)", annotation_position="top", annotation_font_size=12)

                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='green', width=3),
                                         name=class1_label, showlegend=True))
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=3),
                                         name=class0_label, showlegend=True))

                fig.update_layout(
                    title=dict(text=f"Interactive SHAP Decision Plot - Probability for {class1_label}<br>"
                                    f"<sub>All {n_total_samples} samples | Hover for details</sub>",
                               font=dict(size=16)),
                    xaxis_title=f"Model output (Probability for {class1_label})",
                    yaxis_title="Features (ordered by mean |SHAP|)",
                    height=800,
                    hovermode='closest',
                    plot_bgcolor='white',
                    xaxis=dict(gridcolor='lightgray', range=[0, 1], showgrid=True),
                    yaxis=dict(gridcolor='lightgray', showgrid=True),
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )

                interactive_path = os.path.join(output_dir, "shap_decision_unified_interactive.html")
                fig.write_html(interactive_path)

        except ImportError:
            print("Plotly not installed, skipping interactive decision plot")
        except Exception as e:
            print(f"Error creating interactive decision plot: {e}")

    if 'heatmap' in selected_plots:
        shap_df = pd.DataFrame(shap_array, columns=feature_names)

        mean_abs_shap = shap_df.abs().mean()
        feature_order = mean_abs_shap.sort_values(ascending=False).index
        shap_sorted = shap_df[feature_order]

        sorted_indices = np.argsort([str(p) for p in preds])
        shap_sorted = shap_sorted.iloc[sorted_indices]
        preds_sorted = preds[sorted_indices]

        max_abs = np.abs(shap_sorted.values).max()

        n_samples, n_features = shap_sorted.shape
        figsize = (max(14, n_samples / 40), max(8, n_features / 2))

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(shap_sorted.T, aspect='auto', cmap='RdBu_r',
                       vmin=-max_abs, vmax=+max_abs, interpolation='nearest')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("SHAP value", fontsize=10)

        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(shap_sorted.columns)

        class_labels_str = ', '.join([class_mapping[c] for c in unique_classes])

        ax.set_xlabel(f"Samples (sorted by class: {class_labels_str})", fontsize=10)
        ax.set_ylabel("Features (sorted by mean |SHAP|)", fontsize=10)
        ax.set_title("SHAP Heatmap - All samples", fontsize=12, fontweight='bold')

        current_pos = 0
        for i, class_val in enumerate(unique_classes[:-1]):
            class_count = (preds_sorted == class_val).sum()
            current_pos += class_count
            ax.axvline(x=current_pos - 0.5, color='yellow', linewidth=2, linestyle='--')

        current_pos = 0
        for class_val in unique_classes:
            class_count = (preds_sorted == class_val).sum()
            class_label = class_mapping[class_val]
            if class_count > 0:
                ax.text(current_pos + class_count / 2, -0.5,
                        f'{class_label}\n{class_count} samples',
                        ha='center', va='top', fontsize=9, fontweight='bold')
                current_pos += class_count

        try:
            plt.tight_layout()
        except:
            pass
        plt.savefig(os.path.join(output_dir, "shap_heatmap_unified.png"), bbox_inches="tight", dpi=150)
        plt.close()

        if 'interactive_heatmap' in selected_plots:
            try:
                import plotly.graph_objects as go

                shap_df = pd.DataFrame(shap_array, columns=feature_names)

                mean_abs_shap = shap_df.abs().mean()
                feature_order = mean_abs_shap.sort_values(ascending=True).index
                shap_sorted = shap_df[feature_order]

                sorted_indices = np.argsort([str(p) for p in preds])
                shap_sorted = shap_sorted.iloc[sorted_indices]
                preds_sorted = preds[sorted_indices]

                max_abs = np.abs(shap_sorted.values).max()

                hover_text = []
                for i in range(len(shap_sorted)):
                    row_text = []
                    pred_label = class_mapping[preds_sorted[i]]
                    for j, feat in enumerate(shap_sorted.columns):
                        row_text.append(
                            f"Sample: {i}<br>Prediction: {pred_label}<br>Feature: {feat}<br>SHAP: {shap_sorted.iloc[i, j]:.4f}"
                        )
                    hover_text.append(row_text)

                fig = go.Figure(data=go.Heatmap(
                    z=shap_sorted.T.values,
                    x=np.arange(len(shap_sorted)),
                    y=shap_sorted.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    zmin=-max_abs,
                    zmax=+max_abs,
                    hovertext=np.array(hover_text).T,
                    hoverinfo='text',
                    colorbar=dict(title="SHAP value")
                ))

                current_pos = 0
                for i, class_val in enumerate(unique_classes[:-1]):
                    class_count = (preds_sorted == class_val).sum()
                    current_pos += class_count
                    fig.add_vline(x=current_pos - 0.5, line_dash="dash", line_color="yellow", line_width=2)

                class_labels_str = ', '.join([class_mapping[c] for c in unique_classes])

                fig.update_layout(
                    title=f"Interactive SHAP Heatmap ({len(shap_sorted)} samples, {n_classes} classes)",
                    xaxis_title=f"Samples (sorted by class: {class_labels_str})",
                    yaxis_title="Features (sorted by mean |SHAP|)",
                    height=600
                )

                interactive_path = os.path.join(output_dir, "shap_interactive_heatmap_unified.html")
                fig.write_html(interactive_path)

            except ImportError:
                print("Plotly not installed, skipping interactive heatmap")
            except Exception as e:
                print(f"Error creating interactive heatmap: {e}")

    if 'bar' in selected_plots:
        try:
            mean_abs_shap_per_class = []

            for class_val in unique_classes:
                mask = preds == class_val
                if mask.sum() > 0:
                    mean_abs_shap_per_class.append(np.abs(shap_array[mask]).mean(axis=0))

            mean_abs_shap_unified = np.mean(mean_abs_shap_per_class, axis=0)
            sorted_idx = np.argsort(mean_abs_shap_unified)

            plt.figure(figsize=(10, 6))
            plt.barh(range(len(feature_names)), mean_abs_shap_unified[sorted_idx], color='steelblue')
            plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
            plt.xlabel("Mean |SHAP value| (averaged across all classes)", fontsize=11)
            plt.title(f"SHAP Feature Importance - Unified (All {n_classes} Classes)\n"
                      "Average of mean absolute contributions from all classes",
                      fontsize=11, fontweight='bold')
            try:
                plt.tight_layout()
            except:
                pass
            plt.savefig(os.path.join(output_dir, "shap_bar_unified.png"), bbox_inches="tight", dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error creating unified bar plot: {e}")

    if 'dependence' in selected_plots:
        for feat in feature_names:
            try:
                plt.figure(figsize=(10, 7))
                shap.dependence_plot(feat, shap_array, X_df, show=False)
                plt.title(f"SHAP Dependence: {feat}", fontsize=12, fontweight='bold')
                try:
                    plt.tight_layout()
                except:
                    pass

                safe_feat = feat.replace(" ", "_").replace("/", "_").replace("\\", "_")
                safe_feat = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe_feat)

                plt.savefig(os.path.join(output_dir, f"dependence_{safe_feat}.png"), bbox_inches="tight", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating unified dependence plot for {feat}: {e}")

    if 'waterfall' in selected_plots:
        create_waterfall_plots(
            shap_values, shap_array, X_df, feature_names, preds,
            unique_classes, explainer, output_dir, class_mapping, output_labels
        )

    print(f"\nAll plots saved to: {output_dir}")