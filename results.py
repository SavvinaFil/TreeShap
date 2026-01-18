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

    # Predictios to take Shap values
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions are necessary to show shap values correct")
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


def save_results_to_excel(X_df, shap_array, feature_names, preds, output_dir):
    from datetime import datetime

    shap_array = np.array(shap_array)

    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions are important to save shap values correct")
        shap_array = shap_array[np.arange(len(preds)), :, preds]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    shap_df = pd.DataFrame(
        [shap_array[i] for i in range(len(shap_array))],
        columns=[f"SHAP_{f}" for f in feature_names]
    )

    pred_df = pd.DataFrame({
        "Predicted_Class": preds
    })

    output_df = pd.concat([
        X_df.reset_index(drop=True),
        pred_df,
        shap_df
    ], axis=1)

    os.makedirs(output_dir, exist_ok=True)

    # Create unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"shap_results_{timestamp}.xlsx")

    try:
        output_df.to_excel(output_path, index=False)
        print(f"SHAP results saved to: {output_path}")
    except PermissionError:
        output_path_alt = os.path.join(output_dir, f"shap_results_{timestamp}_alt.xlsx")
        output_df.to_excel(output_path_alt, index=False)
        print(f"SHAP results saved to: {output_path_alt}")


def plot_shap_values(shap_values, X_df, feature_names, preds, output_dir, selected_plots=None, explainer=None):
    os.makedirs(output_dir, exist_ok=True)
    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Predictions are important to create shap values correct")
        shap_array = shap_array[np.arange(len(preds)), :, preds]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    if selected_plots is None:
        selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'decision',
                          'interactive_decision', 'heatmap', 'interactive_heatmap']

    # Find the classes
    unique_classes = np.unique(preds)
    print(f"\nFound {len(unique_classes)} different classes: {unique_classes}")

    # Separete data per class
    for class_val in unique_classes:
        mask = preds == class_val
        n_samples_class = mask.sum()

        if n_samples_class == 0:
            continue

        # Take samples for a class
        shap_class = shap_array[mask]
        X_class = X_df[mask]


        # Beeswarm
        if 'beeswarm' in selected_plots:
            plt.figure()
            shap.summary_plot(shap_class, X_class, feature_names=feature_names, show=False)
            plt.title(f"SHAP Beeswarm Plot - Class {class_val}")
            plt.savefig(os.path.join(output_dir, f"shap_beeswarm_class_{class_val}.png"), bbox_inches="tight", dpi=150)
            plt.close()

        # Bar
        if 'bar' in selected_plots:
            plt.figure()
            shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - Class {class_val}")
            plt.savefig(os.path.join(output_dir, f"shap_bar_class_{class_val}.png"), bbox_inches="tight", dpi=150)
            plt.close()

        # Violin
        if 'violin' in selected_plots:
            plt.figure()
            shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="violin", show=False)
            plt.title(f"SHAP Violin Plot - Class {class_val}")
            plt.savefig(os.path.join(output_dir, f"shap_violin_class_{class_val}.png"), bbox_inches="tight", dpi=150)
            plt.close()

        # Dependence (one plot for very feature)
        if 'dependence' in selected_plots:
            for feat in feature_names:
                plt.figure()
                shap.dependence_plot(feat, shap_class, X_class, show=False)
                plt.title(f"SHAP Dependence: {feat} - Class {class_val}")
                plt.savefig(os.path.join(output_dir, f"dependence_{feat}_class_{class_val}.png"), bbox_inches="tight",
                            dpi=150)
                plt.close()

        # Decision Plot per class
        if 'decision_map' in selected_plots:
            # Use expected value per class
            if hasattr(shap_values, 'ndim') and np.array(shap_values).ndim == 3:
                # Multi-class
                expected_value = explainer.expected_value[int(class_val)]
            elif isinstance(explainer.expected_value, (list, np.ndarray)):
                # Binary with array expected values
                expected_value = explainer.expected_value[int(class_val)]
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
                      f"(All {n_samples_class} samples)",
                      fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_decision_class_{class_val}.png"),
                        bbox_inches="tight", dpi=150)
            plt.close()

    # Unified Decision Plot
    if 'decision_map' in selected_plots and len(unique_classes) == 2:

        try:
            # Expected value for class 1
            if isinstance(explainer.expected_value, (list, np.ndarray)):
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
            print(f"   ⚠️ Error: {e}")

    # Interactive Unified Decision Plot
    if 'interactive_decision_map' in selected_plots and len(unique_classes) == 2:

        try:
            import plotly.graph_objects as go

            if isinstance(explainer.expected_value, (list, np.ndarray)):
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
            print("Plotly not installed skipping interactive decision plot")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

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
        for i, class_val in enumerate(unique_classes[:-1]):  # Όχι για την τελευταία κλάση
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
        except ImportError:
            print("⚠️ Το plotly δεν είναι εγκατεστημένο. Παράλειψη interactive heatmap.")
        else:
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

            current_pos = 0
            for i, class_val in enumerate(unique_classes[:-1]):
                class_count = (preds_sorted == class_val).sum()
                current_pos += class_count
                fig.add_vline(x=current_pos - 0.5, line_dash="dash", line_color="yellow", line_width=2)

            fig.update_layout(
                title="Interactive SHAP Heatmap - Όλα τα δείγματα",
                xaxis_title=f"Samples (sorted by class: {', '.join(map(str, unique_classes))})",
                yaxis_title="Features (sorted by mean |SHAP|)",
                height=600
            )

            interactive_path = os.path.join(output_dir, "shap_interactive_heatmap_unified.html")
            fig.write_html(interactive_path)

    print(f"\nPlots saved to: {output_dir}")