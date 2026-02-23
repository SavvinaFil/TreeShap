# AI Explainability Toolbox

**Model-agnostic interpretability analysis for any AI tool.**

This toolbox provides a standardized framework for explaining AI model predictions using SHAP (SHapley Additive exPlanations). It is designed to be model-agnostic, supporting both traditional machine learning (Scikit-Learn) and deep learning (PyTorch/LSTM) workflows.

---

## 🔍 Overview

This repository provides an automated pipeline to move from a trained model to a professional interpretability report. By utilizing a configuration-driven approach, users can generate audit-ready Excel files and pre-rendered Jupyter Notebooks without writing new code for every model.

---

## 📁 Repository Structure

```text
/
├── analysis/
│   ├── tabular/                # Logic for CSV-based data (RF, XGB, etc.)
│   │   ├── treebased/          # Tree-specific explainers
│   │   └── __init__.py         # Tabular manager and registry
│   └── timeseries/             # Logic for 3D temporal data (LSTM, GRU)
│       └── lstm_explainer.py   # PyTorch-specific SHAP implementation
│
├── output/                     # Generated Reports and Audit logs
│   ├── utils/
│   │   └── report_gen.py       # The core Notebook generation engine
│   └── (files)                 # .xlsx and .ipynb outputs appear here
│
├── source/                     # Input Assets
│   ├── models/                 # Your .pkl or .pt model files
│   └── data/                   # Your .csv or .pt data files
│
├── examples/                   # Pre-configured JSON templates
├── main.py                     # Central entry point
└── README.md
```

## 🚀 Getting Started

### 1. Prepare your Assets
Place your trained model and the dataset you want to explain in the `source/` directory:
* **Tabular:** `.pkl` model and `.csv` data.
* **Time-Series:** `.pt` PyTorch model and `.pt` pre-processed tensors.

### 2. Configure your Analysis
Create a JSON file to define the analysis scope. This file tells the toolbox where your files are and how to interpret the outputs.

<details>
<summary><b>Example: Tabular Multi-Target Regression (config.json)</b></summary>

```json
{
  "analysis": "tabular",
  "model_type": "random_forest",
  "model_path": "source/models/energy_model.pkl",
  "dataset_path": "source/data/energy_data.csv",
  "output_dir": "output/",
  "feature_names": ["Wind_Speed", "Temp", "Humidity"],
  "target_index": [0, 1, 2],
  "output_labels": {
    "0": "Power Generation",
    "1": "Grid Load",
    "2": "Frequency"
  },
  "save_excel": true,
  "generate_notebook": true
}
```
</details>

### 3. Run the Toolbox
Execute the analysis via the command line using the `--config` flag:

* Run Time-Series Analysis
```bash
python main.py --config examples/timeseries/lstm/config.json
```

* Run Tabular Classification
```bash
python main.py --config examples/tabular/classify/config.json
```

* Run Tabular Regression
```bash
python main.py --config examples/tabular/regress/config.json
```

---

## 📊 Outputs

The toolbox generates two primary artifacts in the output/ folder:

1. SHAP Audit (.xlsx): A multi-sheet spreadsheet containing original feature values, model predictions, and SHAP values for every row. Each target index gets its own sheet.
2. Interpretation Report (.ipynb): A fully executed Jupyter Notebook containing Summary Plots, Feature Importance Bar Charts, and Temporal Analysis (for LSTMs).

You can find examples of the jupyter notebooks here:

|   | **Single Output Regression**  | **Binary Classification** | **Multioutput Regression** |
| **Example** | [LSTM report](./output/explanation_lstm_gradient_20260220_143820.ipynb) | [RF Classify Report](./output/multi_report_random_forrest_20260220_143844.ipynb) | [RF Regress Report](./output/multi_report_random_forrest_20260220_143908.ipynb) |

---

## 🛠️ Supported Models
* Tree-Based: RandomForest, XGBoost, DecisionTrees.
* Deep Learning: LSTM, GRU, MLP (PyTorch, some still under construction).
* Multi-Output: Full support for MultiOutputRegressor and MultiOutputClassifier wrappers.