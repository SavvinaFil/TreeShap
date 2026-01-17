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

    # use predictions to get shap values (multi-class: (n_classes, n_samples, n_features))
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Error")
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
            raise ValueError("We need the predictions to print shap")
        shap_array = shap_array[np.arange(len(preds)), :, preds]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    shap_df = pd.DataFrame(
        [shap_array[i] for i in range(len(shap_array))],
        columns=[f"SHAP_{f}" for f in feature_names]
    )

    # column with the classes
    pred_df = pd.DataFrame({
        "Predicted_Class": preds
    })

    output_df = pd.concat([
        X_df.reset_index(drop=True),
        pred_df,
        shap_df
    ], axis=1)

    os.makedirs(output_dir, exist_ok=True)

    # create filename with time stamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f"shap_results_{timestamp}.xlsx")

    try:
        output_df.to_excel(output_path, index=False)
        print(f"SHAP results saved to: {output_path}")
    except PermissionError:
        output_path_alt = os.path.join(output_dir, f"shap_results_{timestamp}_alt.xlsx")
        output_df.to_excel(output_path_alt, index=False)
        print(f"SHAP results saved to: {output_path_alt}")


def plot_shap_values(
    shap_values,
    X_df,
    feature_names,
    preds,
    output_dir,
    selected_plots=None,
    class_labels=None
):
    os.makedirs(output_dir, exist_ok=True)
    shap_array = np.array(shap_values)

    # multi-class SHAP: (n_classes, n_samples, n_features)
    if shap_array.ndim == 3:
        if preds is None:
            raise ValueError("Χρειάζονται οι προβλέψεις (preds) για να δημιουργηθούν τα plots σωστά!")
        shap_array = shap_array[np.arange(len(preds)), :, preds]

    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(1, -1)

    if selected_plots is None:
        selected_plots = ['beeswarm', 'bar', 'violin', 'dependence', 'heatmap', 'interactive_heatmap']

    unique_classes = np.unique(preds)
    print(f"\nFound {len(unique_classes)} different classes: {unique_classes}")

    def get_label(c):
        if class_labels is None:
            return str(c)
        return str(c)

    # classify data for each class
    for class_val in unique_classes:
        mask = preds == class_val
        n_samples_class = mask.sum()

        if n_samples_class == 0:
            print(f"No samples for class {class_val}")
            continue

        shap_class = shap_array[mask]
        X_class = X_df[mask]
        label = get_label(class_val)

        # Beeswarm
        if 'beeswarm' in selected_plots:
            plt.figure()
            shap.summary_plot(shap_class, X_class, feature_names=feature_names, show=False)
            plt.title(f"SHAP Beeswarm Plot - Class {label}")
            plt.savefig(os.path.join(output_dir, f"shap_beeswarm_class_{class_val}.png"),
                        bbox_inches="tight", dpi=150)
            plt.close()

        # Bar
        if 'bar' in selected_plots:
            plt.figure()
            shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance - Class {label}")
            plt.savefig(os.path.join(output_dir, f"shap_bar_class_{class_val}.png"),
                        bbox_inches="tight", dpi=150)
            plt.close()

        # Violin
        if 'violin' in selected_plots:
            plt.figure()
            shap.summary_plot(shap_class, X_class, feature_names=feature_names, plot_type="violin", show=False)
            plt.title(f"SHAP Violin Plot - Class {label}")
            plt.savefig(os.path.join(output_dir, f"shap_violin_class_{class_val}.png"),
                        bbox_inches="tight", dpi=150)
            plt.close()

        # Dependence (plot for every feature)
        if 'dependence' in selected_plots:
            for feat in feature_names:
                plt.figure()
                shap.dependence_plot(feat, shap_class, X_class, show=False)
                plt.title(f"SHAP Dependence: {feat} - Class {label}")
                plt.savefig(os.path.join(output_dir, f"dependence_{feat}_class_{class_val}.png"),
                        bbox_inches="tight", dpi=150)
                plt.close()

        # Heatmap
        if 'heatmap' in selected_plots:
            shap_df = pd.DataFrame(shap_array, columns=feature_names)

            # sort features by mean |SHAP|
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
                    label = get_label(class_val)
                    ax.text(current_pos + class_count / 2, -0.5,
                            f'Class {label}\n{class_count} samples',
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
                print("You need to download plotly")
            else:
                shap_df = pd.DataFrame(shap_array, columns=feature_names)

                mean_abs_shap = shap_df.abs().mean()
                feature_order = mean_abs_shap.sort_values(ascending=False).index
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
