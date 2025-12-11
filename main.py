import pickle
import numpy as np
import os
import pandas as pd
from tree_input import load_tree, get_features_from_user, load_dataset
from results import compute_shap_values, show_shap_values, plot_shap_values, save_results_to_excel

def main(config):
    print("!!! Trustworthy AI: Decision Tree Explainability !!!")

    # Load the decision tree
    model_path = config.datapath_decision_tree #input("Enter path to your pickled model (.pkl): ").strip()
    model = load_tree(model_path)
    print(f"Loaded model: {type(model).__name__}")

    # Get the feature names
    try:
        feature_names = list(model.feature_names_in_)
    except AttributeError:
        n_features = model.n_features_in_
        print(f"The model has {n_features} features, please provide names manually.")
        feature_names = []
        for i in range(n_features):
            name = input(f"Enter name for feature {i+1}: ")
            feature_names.append(name)

    # Choose input method
    print("\nInput options:")
    print("1) Manual single-sample entry")
    print("2) Load CSV file")
    print("3) Load Excel (.xlsx) file")
    choice = int(input("Choose 1/2/3: ").strip())

    if choice == 1:
        feature_values = get_features_from_user(feature_names)
        X_df = pd.DataFrame([feature_values], columns=feature_names)
    else:
        X_df = load_dataset(choice, feature_names)

    # SHAP
    if choice != 1:
        print("\nDo you want SHAP for the whole dataset or a single row?")
        print("1) Whole dataset")
        print("2) Single row")
        row_choice = int(input("Choose 1 or 2: ").strip())
        if row_choice == 2:
            row_num = int(input(f"Enter row number between 1 and {len(X_df)}: ").strip())
            X_sample = X_df.iloc[[row_num-1]]
        else:
            X_sample = X_df
    else:
        X_sample = X_df

    # Compute SHAP values
    shap_expl, shap_values, X_df_aligned = compute_shap_values(model, X_sample)
    preds = model.predict(X_df_aligned)

    # Print SHAP values
    show_shap_values(shap_values, feature_names, preds)

    # Output directory
    output_dir = input("\nEnter output directory for results (default='outputs'): ").strip()
    if output_dir == "":
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save the SHAP results to Excel
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        reshaped = []
        for i in range(len(preds)):
            cls = int(preds[i])
            reshaped.append(shap_array[i, :, cls])
        shap_array = np.array(reshaped)
    save_results_to_excel(X_df_aligned, shap_array, feature_names, output_dir)

    # Generate some plots
    generate_plots = input("Do you want to generate SHAP plots? (yes/no): ").strip().lower()
    if generate_plots in ["yes", "y"]:
        plot_shap_values(shap_values, X_df_aligned, feature_names, output_dir, preds=preds)


if __name__ == "__main__":
    main()
