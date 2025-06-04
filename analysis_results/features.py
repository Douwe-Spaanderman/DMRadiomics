import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

def read_performance_data(file_path, statistics="Mann-Whitney"):
    """Reads performance data from a CSV file and returns it as a DataFrame."""
    try:
        data = pd.read_csv(file_path, skiprows=1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return pd.DataFrame()
    
    objects = data["Label"]
    labels = []
    for o in objects:
        if 'hf_' in o:
            labels.append(0)
        elif 'sf_' in o:
            labels.append(1)
        elif 'of_' in o:
            labels.append(2)
        elif 'GLCM_' in o or 'GLCMMS_' in o:
            labels.append(3)
        elif 'GLRLM_' in o:
            labels.append(4)
        elif 'GLSZM_' in o:
            labels.append(5)
        elif 'GLDM_' in o:
            labels.append(6)
        elif 'NGTDM_' in o:
            labels.append(7)
        elif 'Gabor_' in o:
            labels.append(8)
        elif 'semf_' in o:
            labels.append(9)
        elif 'df_' in o:
            labels.append(10)
        elif 'logf_' in o:
            labels.append(11)
        elif 'vf_' in o:
            labels.append(12)
        elif 'LBP_' in o:
            labels.append(13)
        elif 'phasef_' in o:
            labels.append(14)
        else:
            raise KeyError(o)
        
    mapping = {0: 'Histogram',
            1: 'Shape',
            2: 'Orientation',
            3: 'GLCM',
            4: 'GLRLM',
            5: 'GLSZM',
            6: 'GLDM',
            7: 'NGTDM',
            8: 'Gabor',
            9: 'Semantic',
            10: 'DICOM',
            11: 'LoG',
            12: 'Vessel',
            13: 'LBP',
            14: 'Phase'
            }
            
    # Replace several labels
    objects = [o.replace('CalcFeatures_', '') for o in objects]
    objects = [o.replace('featureconverter_', '') for o in objects]
    objects = [o.replace('PREDICT_', '') for o in objects]
    objects = [o.replace('PyRadiomics_', '') for o in objects]
    objects = [o.replace('Pyradiomics_', '') for o in objects]
    objects = [o.replace('predict_', '') for o in objects]
    objects = [o.replace('pyradiomics_', '') for o in objects]
    objects = [o.replace('_predict', '') for o in objects]
    objects = [o.replace('_pyradiomics', '') for o in objects]
    objects = [o.replace('original_', '') for o in objects]
    objects = [o.replace('train_', '') for o in objects]
    objects = [o.replace('test_', '') for o in objects]
    objects = [o.replace('1_0_', '') for o in objects]
    objects = [o.replace('hf_', '') for o in objects]
    objects = [o.replace('sf_', '') for o in objects]
    objects = [o.replace('of_', '') for o in objects]
    objects = [o.replace('GLCM_', '') for o in objects]
    objects = [o.replace('GLCMMS_', '') for o in objects]
    objects = [o.replace('GLRLM_', '') for o in objects]
    objects = [o.replace('GLSZM_', '') for o in objects]
    objects = [o.replace('GLDM_', '') for o in objects]
    objects = [o.replace('NGTDM_', '') for o in objects]
    objects = [o.replace('Gabor_', '') for o in objects]
    objects = [o.replace('semf_', '') for o in objects]
    objects = [o.replace('df_', '') for o in objects]
    objects = [o.replace('logf_', '') for o in objects]
    objects = [o.replace('vf_', '') for o in objects]
    objects = [o.replace('Frangi_', '') for o in objects]
    objects = [o.replace('LBP_', '') for o in objects]
    objects = [o.replace('phasef_', '') for o in objects]
    objects = [o.replace('tf_', '') for o in objects]
    objects = [o.replace('_MRI_0', '') for o in objects]
    objects = [o.replace('MRI_0', '') for o in objects]
    objects = [o.replace('_CT_0', '') for o in objects]
    objects = [o.replace('_MR_0', '') for o in objects]
    objects = [o.replace('CT_0', '') for o in objects]
    objects = [o.replace('MR_0', '') for o in objects]

    data["group"] = labels
    data["name"] = objects
    data = data.sort_values('group')
    data = data.replace({"group": mapping})
    data = data.reset_index(drop=True)
    data = data[["Label", "group", "name", statistics]]
    return data

def plot_significant_features(data, statistics="Mann-Whitney", output_path="feature_plots", p_thresh=0.05, annotate_thresh=9e-5):
    os.makedirs(output_path, exist_ok=True)

    # Marker styles for centers
    center_markers = ['*', '^', '.']
    center_marker_map = {}

    for i, center in enumerate(sorted(data["center"].unique())):
        center_marker_map[center] = center_markers[i % len(center_markers)]

    for (label_name, sequence_name), subset in data.groupby(["label", "sequence"]):
        subset = subset.copy()
        subset["Feature"] = subset["Label"]

        # Identify features (Label) that are significant in at least one center
        feature_significance_map = subset.groupby("Label")[statistics].min() < p_thresh

        # Assign colors to significant features based on their group
        feature_colors = {}

        groups = subset.set_index("Label")["group"].to_dict()
        palette = sns.color_palette("colorblind", len(set(groups.values())))
        group_color_map = {group: palette[i] for i, group in enumerate(set(groups.values()))}

        for label, is_significant in feature_significance_map.items():
            if is_significant:
                feature_colors[label] = group_color_map[groups[label]]
            else:
                feature_colors[label] = "lightgray"

        # Add x-axis indexing for each feature (Label)
        subset = subset.sort_values(by=["group", "Feature"])
        feature_order = subset["Feature"].unique()
        feature_to_x = {feature: i for i, feature in enumerate(feature_order)}
        subset["x"] = subset["Feature"].map(feature_to_x)

        plt.figure(figsize=(28, 10))

        # Plot points per center with different markers
        for center in subset["center"].unique():
            center_data = subset[subset["center"] == center]
            for _, row in center_data.iterrows():
                plt.scatter(
                    row["x"],
                    row[statistics],
                    color=feature_colors[row["Feature"]],
                    marker=center_marker_map[center],
                    edgecolor="black",
                    s=120,
                    label=center  # may duplicate in legend, will fix below
                )

        plt.yscale("log")
        plt.gca().invert_yaxis()

        # Annotate very significant features
        annotations = []
        for _, row in subset[subset[statistics] < annotate_thresh].iterrows():
            annotations.append(plt.text(row["x"], row[statistics], row["name"], ha='center', va='bottom'))
        adjust_text(annotations)
        
        # Add vertical lines and custom xticks for feature groups
        group_boundaries = []
        xtick_positions = []
        xtick_labels = []

        grouped = subset.groupby("group")
        current_idx = 0

        for group_name, group_df in grouped:
            n_features = len(group_df["Feature"].unique())
            mid = current_idx + n_features / 2 - 0.5
            xtick_positions.append(mid)
            xtick_labels.append(group_name)
            current_idx += n_features
            group_boundaries.append(current_idx - 0.5)

        # Draw vertical lines between groups
        for boundary in group_boundaries[:-1]:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)

        # Custom group x-ticks
        plt.xticks(xtick_positions, xtick_labels, rotation=45, ha='right', rotation_mode='anchor')

        plt.axhline(y=p_thresh, color='red', linestyle='--', dashes=(5, 10))
        plt.axhline(y=annotate_thresh, color='red', linestyle='--', dashes=(5, 5))
        plt.ylabel(f"{statistics} P-value", fontsize=18)

        # Legend: unique handles for centers
        legend_elements = []
        for center, marker in center_marker_map.items():
            legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=center,
                                          markerfacecolor='black', markeredgecolor='black', markersize=18))
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=18, frameon=False)

        plt.tight_layout()
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlim(-0.1, len(subset["x"].unique()) - 0.1)
        out_path = os.path.join(output_path, f"{label_name}_{sequence_name}_significance.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

def compare_features_by_center(data: pd.DataFrame, output_excel="significant_features.xlsx", p_threshold=0.05):
    data = data.copy()

    # Group by label, sequence, and feature name
    grouped = data.groupby(["label", "sequence", "name"])

    results = []

    for (label, sequence, name), group in grouped:
        sig_centers = group[group["Mann-Whitney"] < p_threshold]["center"].tolist()
        min_p = group["Mann-Whitney"].min()
        results.append({
            "label": label,
            "sequence": sequence,
            "feature": name,
            "significant_centers": sig_centers,
            "min_p_value": min_p,
            "n_centers_significant": len(sig_centers)
        })

    results_df = pd.DataFrame(results)

    # Save to Excel, one sheet per label+sequence
    with pd.ExcelWriter(output_excel) as writer:
        for (label, sequence), group_df in results_df.groupby(["label", "sequence"]):
            sheet_name = f"{label}_{sequence}"[:31].replace(":", "_").replace("/", "_")
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)

    return results_df

def extract_features(input_root="../data/results", output_root="../data/results"):
    # Loop through results folders to find all unique experiments
    data = []
    for experiment in os.listdir(input_root):
        experiment_path = os.path.join(input_root, experiment)
        if not os.path.isdir(experiment_path):
            continue

        # Split experiment name to get cohort names
        experiment_parts = experiment.split("_")
        center = experiment_parts[-1]
        if experiment_parts[0] == "WORC":
            label = experiment_parts[1]
            sequence = " ".join(experiment_parts[2:-1])
        else:
            label = experiment_parts[0]
            sequence = " ".join(experiment_parts[1:-1])

        print(f"Found experiment: {experiment} with label: {label}, sequence: {sequence} and center: {center}")

        # Load data from pandas DataFrame
        d = read_performance_data(os.path.join(experiment_path, "Evaluation", 'StatisticalTestFeatures_all_0.csv'))
        d["center"] = center
        d["label"] = label
        d["sequence"] = sequence
        data.append(d)

    data = pd.concat(data, ignore_index=True)
    compare_features_by_center(data, output_excel=os.path.join(output_root, "significant_features.xlsx"))
    plot_significant_features(data, output_path=os.path.join(output_root, "feature_plots"))

if __name__ == "__main__":
    extract_features()