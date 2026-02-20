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
    rel_path = image_path.replace('\\', '/')
    code = f"""# {title}
from IPython.display import Image, display
display(Image(filename='{rel_path}'))"""
    return code


def create_image_html(image_path, title=""):
    rel_path = image_path.replace('\\', '/')
    html = f"""### {title}

<img src="{rel_path}" alt="{title}" style="max-width:100%; height:auto;">"""
    return html


def create_html_embed_code(html_path, height=600):
    rel_path = html_path.replace('\\', '/')
    code = f"""from IPython.display import IFrame
IFrame(src='{rel_path}', width='100%', height={height})"""
    return code


def get_plot_explanation(plot_type):
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

**Example interpretation:**
- Base value: Average probability across all samples
- High income: Strong positive contribution (increases prediction)
- High age: Moderate positive contribution
- Final prediction: High probability for the predicted class
""",

        'interactive': """
## Interactive Visualization

This is an **interactive plot** that can be viewed in your web browser. Features include:
- Hover over elements to see detailed information
- Zoom in/out using your mouse or trackpad
- Pan across the visualization
- Click on legend items to toggle visibility

**Note:** These plots are saved as HTML files in the output folder for the best interactive experience!
"""
    }
    return explanations.get(plot_type, "")


def generate_notebook(plots_dir, output_path, model_info=None):
    nb = nbf.v4.new_notebook()
    cells = []

    output_labels = model_info.get('output_labels', {}) if model_info else {}

    title = f"""# Model agnostic explainability analysis of any AI tool. \n\n This is the explainability analysis toolbox for the AI-EFFECT project. It is model agnostic, and can take any AI tool as input and returns a clear explanation of your tool.

## AI Model Explainability Analysis with SHAP

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This notebook contains a comprehensive explainability analysis of your model using SHAP (SHapley Additive exPlanations) values.

SHAP values provide a unified measure of feature importance and show how each feature contributes to individual predictions.
"""
    cells.append(create_markdown_cell(title))

    shap_intro = """
## What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

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

    if model_info:
        classes_display = []
        for cls in model_info.get('classes', []):
            class_label = output_labels.get(str(int(cls)), f"Class {cls}")
            classes_display.append(f"{cls} ({class_label})")

        info_text = f"""
## Model Information

- **Model Type:** {model_info.get('model_type', 'N/A')}
- **Task Type:** {model_info.get('task_type', 'N/A')}
- **Number of Features:** {model_info.get('n_features', 'N/A')}
- **Number of Classes/Outputs:** {model_info.get('n_classes', model_info.get('n_outputs', 'N/A'))}
- **Output Classes:** {', '.join(classes_display) if classes_display else 'N/A'}
- **Total Samples Analyzed:** {model_info.get('n_samples', 'N/A')}
"""
        cells.append(create_markdown_cell(info_text))

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

    # Multiple patterns to match both binary and multi-output structures
    plot_sections = [
        {
            'name': 'Feature Importance - Unified (All Classes/Outputs)',
            'anchor': 'unified-feature-importance',
            'pattern': 'shap_bar_unified.png',
            'type': 'bar',
            'description': 'Overall feature importance averaged across all classes/outputs, ranked by mean absolute SHAP value.',
            'is_interactive': False,
            'recursive': False
        },
        {
            'name': 'Feature Importance',
            'anchor': 'feature-importance',
            'patterns': ['**/shap_bar.png', '**/shap_bar_*.png'],
            'type': 'bar',
            'description': 'Overall feature importance ranked by mean absolute SHAP value.',
            'is_interactive': False,
            'exclude_pattern': 'unified'
        },
        {
            'name': 'Beeswarm Plots',
            'anchor': 'beeswarm-plots',
            'patterns': ['**/shap_beeswarm.png', '**/shap_beeswarm_*.png'],
            'type': 'beeswarm',
            'description': 'Distribution of SHAP values showing feature impacts across all samples.',
            'is_interactive': False
        },
        {
            'name': 'Violin Plots',
            'anchor': 'violin-plots',
            'patterns': ['**/shap_violin.png', '**/shap_violin_*.png'],
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
            'patterns': ['**/shap_decision.png', '**/shap_decision_*.png'],
            'type': 'decision_map',
            'description': 'Visualization of prediction paths from base value to final prediction.',
            'is_interactive': False,
            'exclude_pattern': 'interactive'
        },
        {
            'name': 'Interactive Decision Plots',
            'anchor': 'interactive-decision',
            'pattern': '**/shap_decision_*_interactive.html',
            'type': 'interactive',
            'description': 'Interactive visualization of decision paths.',
            'is_interactive': True,
            'height': 850
        },
        {
            'name': 'SHAP Heatmaps',
            'anchor': 'heatmaps',
            'patterns': ['**/shap_heatmap.png', '**/shap_heatmap_*.png'],
            'type': 'heatmap',
            'description': 'Heatmap showing SHAP values across all samples and features.',
            'is_interactive': False,
            'exclude_pattern': 'interactive'
        },
        {
            'name': 'Interactive Heatmaps',
            'anchor': 'interactive-heatmaps',
            'pattern': '**/shap_interactive_heatmap_*.html',
            'type': 'interactive',
            'description': 'Interactive heatmap visualization.',
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
            'name': 'Waterfall Plots - Class/Output Averages',
            'anchor': 'waterfall-mean',
            'pattern': '**/waterfall_plots/waterfall_mean_*.png',
            'type': 'waterfall',
            'description': 'Average feature contributions per class or output.',
            'is_interactive': False
        }
    ]

    for section in plot_sections:
        # Handle multiple patterns
        plot_files = []

        if 'patterns' in section:
            # Multiple patterns - merge results
            for pattern in section['patterns']:
                files = glob.glob(os.path.join(plots_dir, pattern), recursive=True)
                plot_files.extend(files)
            # Remove duplicates
            plot_files = list(set(plot_files))
        elif 'pattern' in section:
            # Single pattern
            if section.get('recursive', True):
                plot_files = glob.glob(os.path.join(plots_dir, section['pattern']), recursive=True)
            else:
                # Non-recursive - only root level
                plot_files = glob.glob(os.path.join(plots_dir, section['pattern']), recursive=False)

        # Exclude files matching exclude_pattern
        if section.get('exclude_pattern'):
            exclude_keyword = section['exclude_pattern']
            plot_files = [f for f in plot_files if exclude_keyword not in os.path.basename(f)]

        if not plot_files:
            continue

        plot_files.sort()

        section_header = f"""
---
<a id='{section['anchor']}'></a>
# {section['name']}

{section['description']}
"""
        cells.append(create_markdown_cell(section_header))

        explanation = get_plot_explanation(section['type'])
        if explanation:
            cells.append(create_markdown_cell(explanation))

        # Skip interactive plots in notebook, just reference them
        if section.get('is_interactive', False):
            interactive_note = f"""
### Interactive Plots Available

Interactive {section['name'].lower()} are available in the output folder but not embedded in this notebook for optimal performance.

**To view interactive plots:**
1. Navigate to the output folder: `{plots_dir}`
2. Look for files matching pattern: `{section.get('pattern', section.get('patterns', [''])[0]).replace('**/', '')}`
3. Open the `.html` files in your web browser

**Features of interactive plots:**
- Hover over elements for detailed information
- Zoom and pan to explore the data
- Click legend items to toggle visibility
- Fully interactive experience in your browser

**Found {len(plot_files)} interactive plot(s) for this section.**

---

*Interactive plots provide the best experience when opened directly in a web browser.*
"""
            cells.append(create_markdown_cell(interactive_note))
            continue

        # Embed images
        for plot_file in plot_files:
            rel_path = os.path.relpath(plot_file, plots_dir)

            filename = os.path.basename(plot_file)

            # Extract meaningful title
            # For multi-output: "Power_Forecast/shap_bar.png" → "Power Forecast - Feature Importance"
            # For binary: "shap_bar_Loan_Approved.png" → "Loan Approved - Feature Importance"
            parent_dir = os.path.basename(os.path.dirname(plot_file))

            if parent_dir and parent_dir != os.path.basename(plots_dir):
                # Multi-output structure
                output_name = parent_dir.replace('_', ' ')
                plot_type_name = filename.replace('shap_', '').replace('.png', '').replace('_', ' ').title()
                title = f"{output_name} - {plot_type_name}"
            else:
                # Binary structure
                title = filename.replace('_', ' ').replace('.png', '').replace('.html', '').title()

            image_html = create_image_html(rel_path, title)
            cells.append(create_markdown_cell(image_html))

    conclusion = """
---

## Summary and Next Steps

This report provides a comprehensive view of how your AI model makes predictions using SHAP values.

### Key Takeaways:
1. **Feature Importance**: Identify which features drive your model's decisions
2. **Individual Predictions**: Understand why specific predictions were made (waterfall plots)
3. **Feature Interactions**: Discover how features work together (dependence plots)
4. **Model Behavior**: Visualize decision paths and patterns across all samples

### Recommendations:
- Focus on the top important features shown in bar plots
- Investigate unexpected patterns in dependence plots
- Use waterfall plots to explain individual predictions to stakeholders
- Compare decision paths between different classes/outputs using decision plots
- **Open the interactive HTML plots** in the output folder for full exploration capabilities

### For More Information:
- SHAP Documentation: https://shap.readthedocs.io/
- Original SHAP Paper: Lundberg & Lee (2017)

---

**Report Generated by Explainable AI Tool**  
**Project: AI-EFFECT**
"""
    cells.append(create_markdown_cell(conclusion))

    nb['cells'] = cells

    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def generate_analysis_notebook(plots_output_dir, model_info=None):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    notebook_name = f"SHAP_Analysis_Report_{timestamp}.ipynb"
    notebook_path = os.path.join(plots_output_dir, notebook_name)

    generate_notebook(plots_output_dir, notebook_path, model_info)

    return notebook_path