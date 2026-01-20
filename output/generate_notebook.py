import nbformat as nbf
import os
import glob
import subprocess
import sys
from datetime import datetime


def create_markdown_cell(text):
    return nbf.v4.new_markdown_cell(text)


def create_code_cell(code):
    return nbf.v4.new_code_cell(code)


def create_image_display_code(image_path, title=""):
    rel_path = image_path.replace('\\', '/')  # Convert Windows paths to forward slashes
    code = f"""# {title}
from IPython.display import Image, display
display(Image(filename='{rel_path}'))"""
    return code


def create_image_html(image_path, title=""):
    rel_path = image_path.replace('\\', '/')  # Convert Windows paths to forward slashes
    html = f"""### {title}

<img src="{rel_path}" alt="{title}" style="max-width:100%; height:auto;">"""
    return html


def create_html_embed_code(html_path, height=600):
    rel_path = html_path.replace('\\', '/')
    code = f"""from IPython.display import IFrame
IFrame(src='{rel_path}', width='100%', height={height})"""
    return code


def get_plot_explanation(plot_type):
    #Get explanation text for each plot type
    explanations = {
        'beeswarm': """
## SHAP Beeswarm Plot

The **Beeswarm Plot** shows the distribution of SHAP values for each feature across all samples.

**How to read it:**
- Each dot represents one sample
- Features are sorted by importance (top = most important)
- Position on x-axis shows the SHAP value (impact on prediction)
- Color shows the feature value (red = high, blue = low)
- Width shows density of samples

**Key insights:**
- Features at the top have the biggest impact on predictions
- Spread of dots shows how consistently a feature affects predictions
- Color patterns reveal relationships (e.g., high values → positive impact)
""",

        'bar': """
## SHAP Feature Importance (Bar Plot)

The **Bar Plot** shows the mean absolute SHAP value for each feature, indicating overall feature importance.

**How to read it:**
- Longer bars = more important features
- This is the average impact magnitude across all predictions
- Doesn't show direction (positive or negative)

**Key insights:**
- Quickly identify which features matter most to your model
- Compare relative importance between features
""",

        'violin': """
## SHAP Violin Plot

The **Violin Plot** combines the distribution and density of SHAP values for each feature.

**How to read it:**
- Width shows the density of SHAP values at each level
- Wider sections = more samples with that SHAP value
- Color gradient shows feature value (red = high, blue = low)

**Key insights:**
- See the full distribution of feature impacts
- Identify bimodal or unusual distributions
- Understand variability in feature effects
""",

        'dependence': """
## SHAP Dependence Plot

The **Dependence Plot** shows how a feature's value affects its SHAP value (impact on prediction).

**How to read it:**
- X-axis: feature value
- Y-axis: SHAP value (impact on prediction)
- Color: another feature that interacts with this one
- Each dot is one sample

**Key insights:**
- Reveals non-linear relationships
- Shows feature interactions (via color)
- Identifies thresholds or turning points
""",

        'decision_map': """
## SHAP Decision Plot

The **Decision Plot** shows how features combine to create each prediction.

**How to read it:**
- Each line represents one sample's prediction path
- Starts from base value (left) to final prediction (right)
- Features are ordered by importance
- Lines moving right = positive contribution
- Lines moving left = negative contribution

**Key insights:**
- Visualize the "decision journey" for each prediction
- See which features push predictions in which direction
- Compare decision paths between different samples
- Identify threshold effects (e.g., decision boundary at 0.5)
""",

        'heatmap': """
## SHAP Heatmap

The **Heatmap** shows SHAP values for all features across all samples.

**How to read it:**
- Each column is one sample
- Each row is one feature
- Color intensity shows SHAP value (red = positive, blue = negative)
- Samples are sorted by predicted class
- Features are sorted by importance

**Key insights:**
- See patterns across all predictions at once
- Identify feature clusters that work together
- Compare SHAP patterns between classes
""",

        'waterfall': """
## SHAP Waterfall Plot

The **Waterfall Plot** shows how each feature contributes to moving the prediction from the base value to the final prediction for individual samples.

**How to read it:**
- Starts from E[f(X)] = base value (expected model output)
- Each bar shows a feature's contribution
- Red/positive bars push prediction higher
- Blue/negative bars push prediction lower
- Ends at f(x) = final prediction

**Key insights:**
- **Individual sample explanation**: See exactly why this specific prediction was made
- **Feature attribution**: Which features increased/decreased the prediction
- **Magnitude**: How much each feature contributed
- **Cumulative effect**: How contributions add up to reach the final prediction

**For loan approval example:**
- Base value (~0.48): Average probability across all applicants
- Income > 60k: +0.21 → Strong positive contribution (increases approval chance)
- Age > 50: +0.11 → Moderate positive contribution
- Final prediction: 0.925 → High approval probability
""",

        'interactive': """
## Interactive Visualization

This is an **interactive plot** embedded directly in the notebook. You can:
- Hover over elements to see detailed information
- Zoom in/out using your mouse or trackpad
- Pan across the visualization
- Click on legend items to toggle visibility

**Note:** The plot is fully interactive within this notebook - no need to open external files!
"""
    }
    return explanations.get(plot_type, "")


def generate_notebook(plots_dir, output_path, model_info=None):
    #Generate a Jupyter notebook with all plots and explanations

    # Create new notebook
    nb = nbf.v4.new_notebook()
    cells = []

    # Add title and introduction
    title = f"""# Model Agnostic Explainability Analysis for any AI decision tree
## Decision Tree Model Explainability with SHAP

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This notebook contains a comprehensive explainability analysis of your decision tree model using SHAP (SHapley Additive exPlanations) values.

SHAP values provide a unified measure of feature importance and show how each feature contributes to individual predictions.
"""
    cells.append(create_markdown_cell(title))

    # Add SHAP explanation section
    shap_intro = """
## What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

### Key Principles:

**1. Unified Framework**
- SHAP values unify multiple existing methods (LIME, DeepLIFT, Layer-Wise Relevance Propagation, etc.)
- Provides a single, theoretically sound approach to model interpretation

**2. Local Explanations**
- Explains individual predictions by computing the contribution of each feature
- Shows how each feature value pushes the prediction away from the base value (average model output)

**3. Consistency and Accuracy**
- Based on Shapley values from cooperative game theory
- Satisfies desirable properties: local accuracy, missingness, and consistency

**4. Model-Agnostic Approach**
- Works with any machine learning model
- For tree-based models (like Random Forests, XGBoost), SHAP provides exact solutions efficiently

### How SHAP Values Work:

For a prediction, SHAP answers: **"How much did each feature contribute to moving the prediction from the base value to the final prediction?"**

- **Base value**: The average model output over the training dataset
- **SHAP value**: The contribution of each feature to the prediction
- **Final prediction** = Base value + sum of all SHAP values

### Why SHAP for Decision Trees?

TreeExplainer, the algorithm used in this analysis, provides:
- Fast, exact computation for tree-based models
- No sampling required
- Polynomial time complexity
- Consistent and accurate feature attributions

---
"""
    cells.append(create_markdown_cell(shap_intro))

    # Add model information if provided
    if model_info:
        info_text = f"""
## Model Information

- **Model Type:** {model_info.get('model_type', 'N/A')}
- **Number of Features:** {model_info.get('n_features', 'N/A')}
- **Number of Classes:** {model_info.get('n_classes', 'N/A')}
- **Total Samples Analyzed:** {model_info.get('n_samples', 'N/A')}
"""
        cells.append(create_markdown_cell(info_text))

    # Add table of contents
    toc = """
## Table of Contents

1. [Feature Importance (Bar Plots)](#feature-importance)
2. [Feature Distribution (Beeswarm Plots)](#beeswarm-plots)
3. [Feature Distribution (Violin Plots)](#violin-plots)
4. [Feature Dependence Plots](#dependence-plots)
5. [Decision Plots](#decision-plots)
6. [Interactive Decision Plots](#interactive-decision)
7. [SHAP Heatmaps](#heatmaps)
8. [Interactive Heatmaps](#interactive-heatmaps)
9. [Waterfall Plots (Individual Explanations)](#waterfall-plots)

---
"""
    cells.append(create_markdown_cell(toc))

    # Define plot order and their patterns
    plot_sections = [
        {
            'name': 'Feature Importance',
            'anchor': 'feature-importance',
            'pattern': '**/shap_bar_class_*.png',
            'type': 'bar',
            'description': 'Overall feature importance ranked by mean absolute SHAP value.',
            'is_interactive': False
        },
        {
            'name': 'Beeswarm Plots',
            'anchor': 'beeswarm-plots',
            'pattern': '**/shap_beeswarm_class_*.png',
            'type': 'beeswarm',
            'description': 'Distribution of SHAP values showing feature impacts across all samples.',
            'is_interactive': False
        },
        {
            'name': 'Violin Plots',
            'anchor': 'violin-plots',
            'pattern': '**/shap_violin_class_*.png',
            'type': 'violin',
            'description': 'Density distribution of SHAP values for each feature.',
            'is_interactive': False
        },
        {
            'name': 'Feature Dependence Plots',
            'anchor': 'dependence-plots',
            'pattern': '**/dependence_*.png',
            'type': 'dependence',
            'description': 'Relationship between feature values and their SHAP contributions.',
            'is_interactive': False
        },
        {
            'name': 'Decision Plots',
            'anchor': 'decision-plots',
            'pattern': '**/shap_decision_*.png',
            'type': 'decision_map',
            'description': 'Visualization of prediction paths from base value to final prediction.',
            'is_interactive': False
        },
        {
            'name': 'Interactive Decision Plots',
            'anchor': 'interactive-decision',
            'pattern': '**/shap_decision_*_interactive.html',
            'type': 'interactive',
            'description': 'Interactive visualization of decision paths - fully interactive within this notebook!',
            'is_interactive': True,
            'height': 850
        },
        {
            'name': 'SHAP Heatmaps',
            'anchor': 'heatmaps',
            'pattern': '**/shap_heatmap_*.png',
            'type': 'heatmap',
            'description': 'Heatmap showing SHAP values across all samples and features.',
            'is_interactive': False
        },
        {
            'name': 'Interactive Heatmaps',
            'anchor': 'interactive-heatmaps',
            'pattern': '**/shap_interactive_heatmap_*.html',
            'type': 'interactive',
            'description': 'Interactive heatmap - fully interactive within this notebook!',
            'is_interactive': True,
            'height': 650
        },
        {
            'name': 'Waterfall Plots - Individual Samples',
            'anchor': 'waterfall-plots',
            'pattern': '**/waterfall_plots/waterfall_sample_*.png',
            'type': 'waterfall',
            'description': 'Individual sample explanations showing feature contributions.',
            'is_interactive': False
        },
        {
            'name': 'Waterfall Plots - Class Averages',
            'anchor': 'waterfall-mean',
            'pattern': '**/waterfall_plots/waterfall_mean_*.png',
            'type': 'waterfall',
            'description': 'Average feature contributions per class.',
            'is_interactive': False
        }
    ]

    for section in plot_sections:
        # Find plots matching pattern
        plot_files = glob.glob(os.path.join(plots_dir, section['pattern']), recursive=True)

        if not plot_files:
            continue

        # Sort files for consistent ordering
        plot_files.sort()

        # Add section header
        section_header = f"""
---
<a id='{section['anchor']}'></a>
# {section['name']}

{section['description']}
"""
        cells.append(create_markdown_cell(section_header))

        # Add plot type explanation
        explanation = get_plot_explanation(section['type'])
        if explanation:
            cells.append(create_markdown_cell(explanation))

        # Add each plot
        for plot_file in plot_files:
            # Get relative path from plots_dir
            rel_path = os.path.relpath(plot_file, plots_dir)

            # Extract meaningful title from filename
            filename = os.path.basename(plot_file)
            title = filename.replace('_', ' ').replace('.png', '').replace('.html', '').title()

            if section.get('is_interactive', False) and plot_file.endswith('.html'):
                # For interactive HTML files, embed them directly using IFrame
                cells.append(create_markdown_cell(f"### {title}\n"))

                # Create code cell with IFrame to display the interactive plot
                height = section.get('height', 600)
                embed_code = create_html_embed_code(rel_path, height)
                cells.append(create_code_cell(embed_code))

            else:
                # For image files, use HTML img tag in markdown for immediate display
                image_html = create_image_html(rel_path, title)
                cells.append(create_markdown_cell(image_html))

    # Add conclusion
    conclusion = """
---

## Summary and Next Steps

This report provides a comprehensive view of how your decision tree AI model makes predictions using SHAP values.

### Key Takeaways:
1. **Feature Importance**: Identify which features drive your model's decisions
2. **Individual Predictions**: Understand why specific predictions were made (waterfall plots)
3. **Feature Interactions**: Discover how features work together (dependence plots)
4. **Model Behavior**: Visualize decision paths and patterns across all samples

### Recommendations:
- Focus on the top important features shown in bar plots
- Investigate unexpected patterns in dependence plots
- Use waterfall plots to explain individual predictions to stakeholders
- Compare decision paths between different classes using decision plots
- Interact with the embedded visualizations to explore your data in detail

### For More Information:
- SHAP Documentation: https://shap.readthedocs.io/
- Original SHAP Paper: Lundberg & Lee (2017)

---

**Report Generated by Explainable AI Tool for Decision Trees**
"""
    cells.append(create_markdown_cell(conclusion))

    # Add all cells to notebook
    nb['cells'] = cells

    # Write notebook to file
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def open_notebook(notebook_path):
    abs_notebook_path = os.path.abspath(notebook_path)

    # Try different methods in order of preference
    methods = [
        ('Jupyter Notebook', ['jupyter', 'notebook', abs_notebook_path]),
        ('JupyterLab', ['jupyter', 'lab', abs_notebook_path]),
    ]

    for name, command in methods:
        try:
            # Check if command exists first
            check_result = subprocess.run(
                command[:2] + ['--version'],
                capture_output=True,
                timeout=5,
                text=True
            )
            if check_result.returncode == 0:
                # Command exists, try to open notebook
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
                )

                # Wait a bit for the server to start
                import time
                time.sleep(3)

                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            continue

    try:
        if sys.platform == 'win32':
            os.startfile(abs_notebook_path)
            return True
        elif sys.platform == 'darwin':  # macOS
            subprocess.Popen(['open', abs_notebook_path])
            return True
        else:  # linux
            subprocess.Popen(['xdg-open', abs_notebook_path])
            return True
    except Exception as e:
        return False


def generate_analysis_notebook(plots_output_dir, model_info=None):
    # Create notebook filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    notebook_name = f"SHAP_Analysis_Report_{timestamp}.ipynb"
    notebook_path = os.path.join(plots_output_dir, notebook_name)

    # Generate the notebook
    generate_notebook(plots_output_dir, notebook_path, model_info)

    return notebook_path