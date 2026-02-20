import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os
import numpy as np

def generate_notebook(explainer_inst, shap_values=None, raw_data=None, output_path=None, all_shap_values=None):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    nb = nbf.v4.new_notebook()
    
    # --- 1. Extract Metadata from Config ---
    config = explainer_inst.config
    model_type = config.get("model_type", "Unknown").upper()
    explainer_type = config.get("explainer_type", "SHAP")
    features = config.get("feature_names", [])
    look_back = config.get("look_back", 1)
    output_labels = config.get("output_labels", {})
    is_timeseries = config.get("analysis") == "timeseries" and raw_data.ndim == 3

    # --- 2. Title and Introduction ---
    text_intro = f"""# SHAP Interpretation Report: {model_type}
This notebook provides a post-hoc explanation of the model's predictions using {explainer_type} SHAP.

**Model Architecture:** {model_type}
**Analysis Context:** {config.get("analysis", "General Interpretation")}
**Dataset Scope:** {config.get("dataset_scope", "N/A")}
"""
    cells = [nbf.v4.new_markdown_cell(text_intro)]

    # --- 3. Setup Code (Handles Single or Multi-Target) ---
    # If all_shap_values is provided, we use it. Otherwise, we wrap shap_values in a dict.
    if all_shap_values is None and shap_values is not None:
        # Backward compatibility: treat single output as index 0
        all_shap_data = {0: shap_values.tolist()}
    else:
        all_shap_data = {k: v.tolist() for k, v in all_shap_values.items()}

    code_setup = f"""import shap
import numpy as np
import matplotlib.pyplot as plt

# Data provided by the explainer
all_shap_dict = {all_shap_data}
data_raw = np.array({raw_data.tolist()})
feature_names = {features}
look_back = {look_back}
is_timeseries = {is_timeseries}
output_labels = {output_labels}

def get_flattened_data(shap_data):
    if is_timeseries:
        shap_flat = shap_data.reshape(shap_data.shape[0], -1)
        data_flat = data_raw.reshape(data_raw.shape[0], -1)
        flat_names = [f"{{feat}}_t-{{look_back - 1 - i}}" for i in range(look_back) for feat in feature_names]
    else:
        shap_flat = shap_data
        data_flat = data_raw
        flat_names = feature_names
    return shap_flat, data_flat, flat_names

print(f"Setup complete. Targets to explain: {{list(all_shap_dict.keys())}}")
"""
    cells.append(nbf.v4.new_code_cell(code_setup))

    # --- 4. Loop Through Targets ---
    for target_idx in all_shap_data.keys():
        # Handle different label formats (Dictionary for Tabular, List for TS)
        if isinstance(output_labels, dict):
            label = output_labels.get(str(target_idx), f"Target {target_idx}")
        elif isinstance(output_labels, list) and target_idx < len(output_labels):
            label = output_labels[target_idx]
        else:
            label = f"Target {target_idx}"
        
        # Section Header
        cells.append(nbf.v4.new_markdown_cell(f"--- \n# Analysis for: **{label}**"))

        # Plotting Code for this specific target
        code_plots = f"""
# Process data for target {target_idx}
current_shap_raw = np.array(all_shap_dict[{target_idx}])
s_flat, d_flat, names = get_flattened_data(current_shap_raw)

print(f"Generating plots for: {label}")

# 1. Summary Plot
plt.figure()
shap.summary_plot(s_flat, d_flat, feature_names=names, show=False)
plt.title(f"Global Importance: {label}")
plt.show()

# 2. Bar Plot
plt.figure()
shap.summary_plot(s_flat, d_flat, feature_names=names, plot_type='bar', show=False)
plt.title(f"Mean Impact: {label}")
plt.show()
"""
        cells.append(nbf.v4.new_code_cell(code_plots))

        # 3. Add Temporal Plot only if it's Time Series
        if is_timeseries:
            code_temporal = f"""
importance_over_time = np.abs(current_shap_raw).mean(axis=(0, 2))
plt.figure(figsize=(10, 4))
plt.plot(range(look_back-1, -1, -1), importance_over_time, marker='o', color='#2ecc71')
plt.title(f"Temporal Importance: {label}")
plt.xlabel("Time Lag (t-n)")
plt.ylabel("Impact (Mean |SHAP|)")
plt.grid(True, alpha=0.3)
plt.show()
"""
            cells.append(nbf.v4.new_code_cell(code_temporal))

    # --- 5. Execution and Saving ---
    nb['cells'] = cells
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(output_path)}})
    except Exception as e:
        print(f"Warning: Execution failed. Error: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook generated successfully: {output_path}")