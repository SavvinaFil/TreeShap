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

    # Predictions to extract correct SHAP values
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions are necessary to show shap values correctly")
        # Multi-class: extract SHAP values for predicted class
        shap_array = shap_array[np.arange(len(preds)), :, preds]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    print("\nSHAP values per feature:\n")
    # Limit display to first 100 samples for large datasets
    max_samples_to_show = min(100, len(shap_array))

    for i, row in enumerate(shap_array[:max_samples_to_show]):
        if preds is not None:
            print(f"Sample {i} (predicted class: {preds[i]}):")
        else:
            print(f"Sample {i}:")
        for j, feat in enumerate(feature_names):
            print(f"  {feat}: {float(row[j]):+.4f}")
        print("-" * 40)

    if len(shap_array) > max_samples_to_show:
        print(f"... (showing first {max_samples_to_show} of {len(shap_array)} samples)")
        print("-" * 40)


def save_results_to_excel(X_df, shap_array, feature_names, preds, output_dir):
    from datetime import datetime

    shap_array = np.array(shap_array)

    # Handle multi-class SHAP values
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions are necessary to save shap values correctly")
        shap_array = shap_array[np.arange(len(preds)), :, preds]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    # Create SHAP DataFrame with proper column names
    shap_df = pd.DataFrame(
        shap_array,
        columns=[f"SHAP_{f}" for f in feature_names]
    )

    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        "Predicted_Class": preds
    })

    # Combine all data
    output_df = pd.concat([
        X_df.reset_index(drop=True),
        pred_df,
        shap_df
    ], axis=1)

    os.makedirs(output_dir, exist_ok=True)

    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"shap_results_{timestamp}.xlsx")

    try:
        output_df.to_excel(output_path, index=False)
        print(f"SHAP results saved to: {output_path}")
    except PermissionError:
        output_path_alt = os.path.join(output_dir, f"shap_results_{timestamp}_alt.xlsx")
        output_df.to_excel(output_path_alt, index=False)
        print(f"SHAP results saved to: {output_path_alt}")
    except Exception as e:
        print(f"⚠️ Error saving Excel file: {e}")
        # Try CSV as fallback
        csv_path = os.path.join(output_dir, f"shap_results_{timestamp}.csv")
        output_df.to_csv(csv_path, index=False)
        print(f"Results saved as CSV instead: {csv_path}")


def plot_shap_values(shap_values, X_df, feature_names, preds, output_dir, selected_plots=None, explainer=None):
    os.makedirs(output_dir, exist_ok=True)
    shap_array = np.array(shap_values)

    # Handle multi-class SHAP values
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions are necessary to create shap plots correctly")
        shap_array = shap_array[np.arange(len(preds)), :, preds]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    if selected_plots is None:
        selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'decision_map',
                          'interactive_decision_map', 'heatmap', 'interactive_heatmap', 'waterfall']

    # Find unique classes
    unique_classes = np.unique(preds)
    n_classes = len(unique_classes)
    print(f"\nFound {n_classes} different class(es): {unique_classes}")

    # Separete data per class
    for class_val in unique_classes:
        mask = preds == class_val
        n_samples_class = mask.sum()

        if n_samples_class == 0:
            continue

        # Take samples for this class
        shap_class = shap_array[mask]
        X_class = X_df[mask]

        # Beeswarm
        if 'beeswarm' in selected_plots:
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_class, X_class, feature_names=feature_names, show=False)
                plt.title(f"SHAP Beeswarm Plot - Class {class_val}", fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"shap_beeswarm_class_{class_val}.png"), bbox_inches="tight",
                            dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating beeswarm plot for class {class_val}: {e}")

        # Bar
        if 'bar' in selected_plots:
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="bar", show=False)
                plt.title(f"SHAP Feature Importance - Class {class_val}", fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"shap_bar_class_{class_val}.png"), bbox_inches="tight", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating bar plot for class {class_val}: {e}")

        # Violin
        if 'violin' in selected_plots:
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="violin", show=False)
                plt.title(f"SHAP Violin Plot - Class {class_val}", fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"shap_violin_class_{class_val}.png"), bbox_inches="tight",
                            dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating violin plot for class {class_val}: {e}")

        # Dependence (one plot for each feature)
        if 'dependence' in selected_plots:
            for feat in feature_names:
                try:
                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(feat, shap_class, X_class, show=False)
                    plt.title(f"SHAP Dependence: {feat} - Class {class_val}", fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"dependence_{feat}_class_{class_val}.png"),
                                bbox_inches="tight", dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"Error creating dependence plot for {feat}, class {class_val}: {e}")

        # Decision Plot per class
        if 'decision_map' in selected_plots:
            try:
                # Use expected value per class
                if hasattr(shap_values, 'ndim') and np.array(shap_values).ndim == 3:
                    # Multi-class
                    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > int(
                            class_val):
                        expected_value = explainer.expected_value[int(class_val)]
                    else:
                        expected_value = explainer.expected_value
                elif isinstance(explainer.expected_value, (list, np.ndarray)):
                    # Binary with array expected values
                    if len(explainer.expected_value) > int(class_val):
                        expected_value = explainer.expected_value[int(class_val)]
                    else:
                        expected_value = explainer.expected_value[0]
                else:
                    expected_value = explainer.expected_value

                # Create decision plot with all samples
                plt.figure(figsize=(12, 8))
                shap.decision_plot(
                    expected_value,
                    shap_class,
                    X_class,
                    feature_names=feature_names,
                    show=False,
                    link='identity'
                )
                plt.title(f"SHAP Decision Plot - Class {class_val}\n"
                          f"({n_samples_class} samples)",
                          fontsize=11, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"shap_decision_class_{class_val}.png"),
                            bbox_inches="tight", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Error creating decision plot for class {class_val}: {e}")

    # Unified Decision Plot
    if 'decision_map' in selected_plots and n_classes == 2:

        try:
            # Expected value for class 1
            if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                expected_val_class1 = explainer.expected_value[1]
            else:
                expected_val_class1 = explainer.expected_value

            # Shap values for class 1
            shap_values_array = np.array(shap_values)

            if shap_values_array.ndim == 3:
                if shap_values_array.shape[0] == len(X_df):
                    shap_for_class1 = shap_values_array[:, :, 1]
                elif shap_values_array.shape[2] == len(X_df):
                    shap_for_class1 = shap_values_array[1, :, :].T
                elif shap_values_array.shape[1] == len(X_df):
                    shap_for_class1 = shap_values_array[1, :, :]
                else:
                    shap_for_class1 = None
            elif shap_values_array.ndim == 2:
                shap_for_class1 = shap_values_array
            else:
                shap_for_class1 = None

            if shap_for_class1 is None or shap_for_class1.shape[1] != len(feature_names):
                print("Skip unified decision plot")
            else:
                n_total_samples = shap_for_class1.shape[0]

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

                plt.title(f"SHAP Unified Decision Plot - Probability for Class 1\n"
                          f"(All {n_total_samples} samples)",
                          fontsize=12, fontweight='bold')
                plt.xlabel("Model output (Probability for Class 1)", fontsize=11)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "shap_decision_unified.png"),
                            bbox_inches="tight", dpi=150)
                plt.close()

        except Exception as e:
            print(f"Error creating unified decision plot: {e}")

    # Interactive Unified Decision Plot (works best for binary classification)
    if 'interactive_decision_map' in selected_plots and n_classes == 2:

        try:
            import plotly.graph_objects as go

            if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                expected_val_class1 = explainer.expected_value[1]
            else:
                expected_val_class1 = explainer.expected_value

            shap_values_array = np.array(shap_values)

            if shap_values_array.ndim == 3:
                if shap_values_array.shape[0] == len(X_df):
                    shap_for_class1 = shap_values_array[:, :, 1]
                elif shap_values_array.shape[2] == len(X_df):
                    shap_for_class1 = shap_values_array[1, :, :].T
                elif shap_values_array.shape[1] == len(X_df):
                    shap_for_class1 = shap_values_array[1, :, :]
                else:
                    shap_for_class1 = None
            elif shap_values_array.ndim == 2:
                shap_for_class1 = shap_values_array
            else:
                shap_for_class1 = None

            if shap_for_class1 is None or shap_for_class1.shape[1] != len(feature_names):
                print("Skip interactive decision plot")
            else:
                n_total_samples = shap_for_class1.shape[0]

                final_values = expected_val_class1 + shap_for_class1.sum(axis=1)

                # Sort features per mean absolute shap
                mean_abs_shap = np.abs(shap_for_class1).mean(axis=0)
                feature_order = np.argsort(mean_abs_shap)

                # Create the interactive plot
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
                            f"<b>{feat_name}</b><br>"
                            f"Value: {feat_val:.2f}<br>"
                            f"SHAP: {shap_val:+.4f}<br>"
                            f"Cumulative: {cumsum_values[-1]:.4f}"
                        )

                    color = 'green' if preds[i] == 1 else 'red'
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
                              annotation_text="Threshold (0.5)",
                              annotation_position="top",
                              annotation_font_size=12)

                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='green', width=3),
                    name='Class 1 (Έγκριση)',
                    showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Class 0 (Απόρριψη)',
                    showlegend=True
                ))

                fig.update_layout(
                    title=dict(
                        text=f"Interactive SHAP Decision Plot - Probability for Class 1<br>"
                             f"<sub>All {n_total_samples} samples | Hover για λεπτομέρειες</sub>",
                        font=dict(size=16)
                    ),
                    xaxis_title="Model output (Probability for Class 1)",
                    yaxis_title="Features (ordered by mean |SHAP|)",
                    height=800,
                    hovermode='closest',
                    plot_bgcolor='white',
                    xaxis=dict(
                        gridcolor='lightgray',
                        range=[0, 1],
                        showgrid=True
                    ),
                    yaxis=dict(
                        gridcolor='lightgray',
                        showgrid=True
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )

                # Save interactive plot
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

        sorted_indices = np.argsort(preds)
        shap_sorted = shap_sorted.iloc[sorted_indices]
        preds_sorted = preds[sorted_indices]

        max_abs = np.abs(shap_sorted.values).max()

        n_samples, n_features = shap_sorted.shape
        figsize = (max(14, n_samples / 40), max(8, n_features / 2))

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            shap_sorted.T,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-max_abs,
            vmax=+max_abs,
            interpolation='nearest'
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("SHAP value", fontsize=10)

        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(shap_sorted.columns)

        ax.set_xlabel(f"Samples (sorted by class: {', '.join(map(str, unique_classes))})", fontsize=10)
        ax.set_ylabel("Features (sorted by mean |SHAP|)", fontsize=10)
        ax.set_title("SHAP Heatmap - Όλα τα δείγματα", fontsize=12, fontweight='bold')

        current_pos = 0
        for i, class_val in enumerate(unique_classes[:-1]):
            class_count = (preds_sorted == class_val).sum()
            current_pos += class_count
            ax.axvline(x=current_pos - 0.5, color='yellow', linewidth=2, linestyle='--')

        current_pos = 0
        for class_val in unique_classes:
            class_count = (preds_sorted == class_val).sum()
            if class_count > 0:
                ax.text(current_pos + class_count / 2, -0.5,
                        f'Class {class_val}\n{class_count} samples',
                        ha='center', va='top', fontsize=9, fontweight='bold')
                current_pos += class_count

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "shap_heatmap_unified.png"),
            bbox_inches="tight",
            dpi=150
        )
        plt.close()

    # Interactive Heatmap
    if 'interactive_heatmap' in selected_plots:
        try:
            import plotly.graph_objects as go

            shap_df = pd.DataFrame(shap_array, columns=feature_names)

            mean_abs_shap = shap_df.abs().mean()
            feature_order = mean_abs_shap.sort_values(ascending=True).index
            shap_sorted = shap_df[feature_order]

            sorted_indices = np.argsort(preds)
            shap_sorted = shap_sorted.iloc[sorted_indices]
            preds_sorted = preds[sorted_indices]

            max_abs = np.abs(shap_sorted.values).max()

            hover_text = []
            for i in range(len(shap_sorted)):
                row_text = []
                for j, feat in enumerate(shap_sorted.columns):
                    row_text.append(
                        f"Sample: {i}<br>"
                        f"Prediction: Class {preds_sorted[i]}<br>"
                        f"Feature: {feat}<br>"
                        f"SHAP: {shap_sorted.iloc[i, j]:.4f}"
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

            # Draw vertical lines between classes
            current_pos = 0
            for i, class_val in enumerate(unique_classes[:-1]):
                class_count = (preds_sorted == class_val).sum()
                current_pos += class_count
                fig.add_vline(x=current_pos - 0.5, line_dash="dash", line_color="yellow", line_width=2)

            fig.update_layout(
                title=f"Interactive SHAP Heatmap ({len(shap_sorted)} samples, {n_classes} classes)",
                xaxis_title=f"Samples (sorted by class: {', '.join(map(str, unique_classes))})",
                yaxis_title="Features (sorted by mean |SHAP|)",
                height=600
            )

            interactive_path = os.path.join(output_dir, "shap_interactive_heatmap_unified.html")
            fig.write_html(interactive_path)

        except ImportError:
            print("Plotly not installed, skipping interactive heatmap")
        except Exception as e:
            print(f"Error creating interactive heatmap: {e}")

    # Waterfall plots
    if 'waterfall' in selected_plots:

        # Create waterfall subfolder
        waterfall_dir = os.path.join(output_dir, "waterfall_plots")
        os.makedirs(waterfall_dir, exist_ok=True)

        # Determine how to extract SHAP values for waterfall
        shap_values_raw = np.array(shap_values)

        # Handle different SHAP value formats
        if shap_values_raw.ndim == 3:
            # Binary classification: shape is (2, n_samples, n_features) or (n_samples, n_features, 2)
            if shap_values_raw.shape[0] == 2 and shap_values_raw.shape[1] == len(X_df):
                # Format: (2, n_samples, n_features) - use class 1
                shap_for_waterfall = shap_values_raw[1]
                if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                    expected_value = explainer.expected_value[1]
                else:
                    expected_value = explainer.expected_value
            elif shap_values_raw.shape[2] == 2 and shap_values_raw.shape[0] == len(X_df):
                # Format: (n_samples, n_features, 2) - use class 1
                shap_for_waterfall = shap_values_raw[:, :, 1]
                if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                    expected_value = explainer.expected_value[1]
                else:
                    expected_value = explainer.expected_value
            else:
                # Multi-class or other format - use aligned array
                shap_for_waterfall = shap_array
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    expected_value = explainer.expected_value[0] if len(explainer.expected_value) > 0 else 0
                else:
                    expected_value = explainer.expected_value
        else:
            # 2D array - already aligned
            shap_for_waterfall = shap_array
            expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0

        # Ensure expected_value is a scalar
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
        expected_value = float(expected_value)

        # Strategy 1: Create waterfall plots for representative samples from each class

        for class_val in unique_classes:
            class_mask = preds == class_val
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            # Select up to 5 representative samples per class
            n_samples_to_plot = min(5, len(class_indices))

            # Get SHAP magnitudes for this class
            class_shap_magnitudes = np.abs(shap_for_waterfall[class_indices]).sum(axis=1)

            # Select samples: highest, median, lowest magnitude
            sorted_idx = np.argsort(class_shap_magnitudes)

            if n_samples_to_plot >= 3:
                # High, medium, low impact samples
                sample_positions = [0, len(sorted_idx) // 2, -1]
                if n_samples_to_plot >= 5:
                    sample_positions = [0, len(sorted_idx) // 4, len(sorted_idx) // 2, 3 * len(sorted_idx) // 4, -1]
            else:
                sample_positions = list(range(n_samples_to_plot))

            selected_indices = [class_indices[sorted_idx[pos]] for pos in sample_positions[:n_samples_to_plot]]

            for sample_idx in selected_indices:
                try:
                    # Get SHAP values and feature values for this sample
                    sample_shap = shap_for_waterfall[sample_idx]
                    sample_features = X_df.iloc[sample_idx].values

                    # Ensure they are 1D arrays
                    sample_shap = np.array(sample_shap).flatten()
                    sample_features = np.array(sample_features).flatten()

                    # Round all features to 1 decimal place for cleaner display
                    sample_features_display = np.round(sample_features, 1)

                    # Create explanation object for this sample
                    explanation = shap.Explanation(
                        values=sample_shap,
                        base_values=float(expected_value),
                        data=sample_features_display,
                        feature_names=feature_names
                    )

                    # Create waterfall plot
                    plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(explanation, show=False)
                    plt.title(f"SHAP Waterfall Plot - Sample {sample_idx} (Class {class_val})",
                              fontsize=12, fontweight='bold')
                    plt.tight_layout()

                    # Save plot
                    filename = f"waterfall_sample_{sample_idx}_class_{class_val}.png"
                    plt.savefig(os.path.join(waterfall_dir, filename), bbox_inches="tight", dpi=150)
                    plt.close()

                except Exception as e:
                    print(f"Error creating waterfall for sample {sample_idx}: {e}")
                    continue

        # Strategy 2: Create aggregate waterfall plots per class (mean SHAP values)

        for class_val in unique_classes:
            class_mask = preds == class_val

            if class_mask.sum() == 0:
                continue

            try:
                # Calculate mean SHAP values for this class
                mean_shap_values = shap_for_waterfall[class_mask].mean(axis=0)
                mean_feature_values = X_df[class_mask].mean().values

                # Ensure they are 1D arrays
                mean_shap_values = np.array(mean_shap_values).flatten()
                mean_feature_values = np.array(mean_feature_values).flatten()

                # Round all features to 1 decimal place for cleaner display
                mean_feature_values_display = np.round(mean_feature_values, 1)

                # Create explanation object
                explanation = shap.Explanation(
                    values=mean_shap_values,
                    base_values=float(expected_value),
                    data=mean_feature_values_display,
                    feature_names=feature_names
                )

                # Create waterfall plot
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(explanation, show=False)
                plt.title(f"SHAP Waterfall Plot - Mean Values for Class {class_val}\n"
                          f"({class_mask.sum()} samples)",
                          fontsize=12, fontweight='bold')
                plt.tight_layout()

                # Save plot
                filename = f"waterfall_mean_class_{class_val}.png"
                plt.savefig(os.path.join(waterfall_dir, filename), bbox_inches="tight", dpi=150)
                plt.close()

            except Exception as e:
                print(f"Error creating aggregate waterfall for class {class_val}: {e}")
                continue

    print(f"\nPlots saved to: {output_dir}")