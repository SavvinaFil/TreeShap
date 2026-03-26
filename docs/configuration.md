## 📄 Configuration & Data Flow

To ensure the toolbox operates correctly, your directory structure should mirror the paths defined in your `config.json`. 

### 1. Storage & Paths
* **Model Storage (`model_path`)**: Your trained PyTorch weights (`.pth`) should be stored in `source/models/`. This allows the explainer to reload the architecture and weights before performing the analysis.
* **Data (`background_data_path` & `test_data_path`)**: Storing these as pre-processed subsets (separate from the raw dataset) ensures that the explainer uses the exact same normalization and windowing as your training pipeline.
* **Output Directory (`output_dir`)**: All generated SHAP plots, summary statistics, and optional reports will be saved here.

**Recommended Structure:**
```text
project_root/
├── energy_forecasting_dataset.csv  # Raw Data
├── source/
│   ├── models/
│   │   └── lstm_model.pth          # Trained Model
│   └── data/
│       ├── background_data.csv     # Reference set
│       └── data_to_explain.csv     # Target samples
└── output/                         # Resulting SHAP plots
```
---


## 🧪 SHAP Data Configuration: Background vs. Test Data

The toolbox requires one or two distinct datasets. The **Background Data** defines the "starting point" (the reference), while the **Test Data** defines the "destination" (the specific event you are explaining).

| Dataset | Role | Context |
| :--- | :--- | :--- |
| **Background** (`background_data_path`) | The **Reference Baseline**. Used to "ignore" features and calculate the model's expected value. | Must represent the "physical envelope" (e.g., typical seasonal/diurnal profiles). |
| **Test** (`test_data_path`) | The **Target Samples**. The specific instances (e.g., a sudden solar drop) to be deconstructed. | Identifying which features (e.g., `PV_lag_24`) drove a specific safety violation. |

---

### ⚠️ Explainer Logic & Background Dependencies

The requirement for background data shifts based on your chosen `explainer_type`:

#### 1. Black-Box (KernelSHAP)
* **Status:** **Mandatory**.
* **Mechanism:** Since the model is a "black box," the explainer must physically replace features with samples from the **Background Set** to measure impact. 
* **Risk:** If the background set lacks diversity (e.g., nighttime only), daytime explanations will be physically nonsensical.

#### 2. Model-Specific (GradientSHAP)
* **Status:** **Required** (as a reference distribution).
* **Mechanism:** Used as the "starting point" for path integration. For **Neural Networks** (LSTMs, MLPs), the explainer calculates how the gradient changes as you move from the background average to the test sample.

#### 3. Tree-Based (TreeSHAP)
* **Status:** **Optional/Optimized**.
* **Mechanism:** TreeSHAP can use the internal tree structure (node sample counts) to define the baseline. It is the most robust against small background sets but still benefits from a representative sample to align with physical reality.



---

### Best Practices for Selecting Background Data
1. **Size:** 100–500 samples is usually the "sweet spot" between accuracy and computation time.
2. **Diversity:** Use a **K-Means summarized** version of your training set rather than the first 100 rows to ensure you capture the full range of solar/wind variability.
3. **Consistency:** The background data must have the exact same `normalization` and `look_back` windowing as your `test_data`.

---

## ⚙️ Configuration Parameter Breakdown

| Parameter | Purpose |
| :--- | :--- |
| `analysis` | The class of problem, such as `timeseries` for forecasting with LSTMs. |
| `explainer_type` | Specifies the algorithm (e.g., `gradient` for NNs, `tree` for Random Forest, `kernel` for black-box). |
| `look_back` | The temporal horizon; e.g., `6` means the explainer audits the 6 previous time steps. |
| `input_dim` | The number of features per time step. |
| `dataset_scope` | Usually set to `whole` to ensure scaling context is maintained across the entire project. |
| `generate_notebook` | If `true`, exports a `.ipynb` file for interactive post-run analysis. |

---

## 📊 Feature & Label Setup
The toolbox maps raw tensor dimensions back to human-readable names:
* **`feature_names`**: Maps the input dimensions (e.g., Physics: `ghi`, Temporal: `hour_sin`, Historical: `PV_lag_24`).
* **`output_labels`**: Defines the target variable being explained (e.g., `PV` power production).


### 📂 Model & Data Requirements

Please provide files in the following formats:

| Surrogate Type | Model Format | Data Format | Library Basis |
| :--- | :--- | :--- | :--- |
| **LSTM / Neural Networks** | `.pt` (PyTorch) | `.pt` (Tensors) | `torch`|
| **Tree-Based Models** | `.pkl` (Pickle) | `.csv` (Table) | `scikit-learn`, `XGBoost` |

> **Note:** For LSTM models, ensure the `.pt` data follows the $(Batch, Seq, Feature)$ dimensionality. For Tree-based models, the `.csv` headers must exactly match the feature names used during the initial training phase.


