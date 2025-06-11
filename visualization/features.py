import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from adjustText import adjust_text
from functools import reduce

def read_feature_data(file_path, statistics="Mann-Whitney"):
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
    # Marker styles for centers
    center_marker_map = {
        "Italy" : "*",
        "Canada" : "^",
        "Netherlands" : ".",
    }
    
    # Find unique centers and check of each center in center_marker_map is present in data
    unique_centers = [meta["metadata"].get("internalcenter", "None") for meta in data.values()]
    if center_marker_map.keys() != set(unique_centers):
        missing_centers = set(center_marker_map.keys()) - set(unique_centers)
        print(f"Warning: Not all experiments have been run, and {missing_centers} is/are missing.")

    feature_data = []

    for experiment, content in data.items():
        features_df = content["features_data"]
        meta = content["metadata"]

        if features_df is not None:
            features_df = features_df.copy()
            features_df["experiment"] = experiment
            features_df["center"] = meta.get("internalcenter")
            feature_data.append(features_df)

    # Combine into a single DataFrame
    feature_data = pd.concat(feature_data, ignore_index=True)
    feature_data["Feature"] = feature_data["Label"]

    # Identify features (Label) that are significant in at least one center
    feature_significance_map = feature_data.groupby("Label")[statistics].min() < p_thresh

    # Assign colors to significant features based on their group
    feature_colors = {}

    groups = feature_data.set_index("Label")["group"].to_dict()
    palette = sns.color_palette("colorblind", len(set(groups.values())))
    group_color_map = {group: palette[i] for i, group in enumerate(set(groups.values()))}

    for label, is_significant in feature_significance_map.items():
        if is_significant:
            feature_colors[label] = group_color_map[groups[label]]
        else:
            feature_colors[label] = "lightgray"

    # Add x-axis indexing for each feature (Label)
    feature_data = feature_data.sort_values(by=["group", "Feature"])
    feature_order = feature_data["Feature"].unique()
    feature_to_x = {feature: i for i, feature in enumerate(feature_order)}
    feature_data["x"] = feature_data["Feature"].map(feature_to_x)

    plt.figure(figsize=(28, 10))
    # Plot points per center with different markers
    for center in unique_centers:
        center_data = feature_data[feature_data["center"] == center]
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
    for _, row in feature_data[feature_data[statistics] < annotate_thresh].iterrows():
        annotations.append(plt.text(row["x"], row[statistics], row["name"], ha='center', va='bottom'))
    adjust_text(annotations)
    
    # Add vertical lines and custom xticks for feature groups
    group_boundaries = []
    xtick_positions = []
    xtick_labels = []

    grouped = feature_data.groupby("group")
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

    # Add Legend header
    legend_elements.append(Patch(label=r'$\bf{Center:}$', facecolor='none', edgecolor='none'))
    for center, marker in center_marker_map.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', label=center,
                                    markerfacecolor='black', markeredgecolor='black', markersize=18))
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=18, frameon=False)

    plt.tight_layout()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xlim(-0.1, len(feature_data["x"].unique()) - 0.1)
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_features_by_center(data: pd.DataFrame, output_path="significant_features.xlsx", statistics="Mann-Whitney", p_threshold=0.05):
    feature_data = []
    unique_centers = []
    for experiment, content in data.items():
        features_df = content["features_data"]
        meta = content["metadata"]

        if features_df is not None:
            features_df = features_df.copy()
            features_df[statistics] = features_df[statistics].astype(float)
            features_df.rename(columns={
                statistics: f"{meta.get('internalcenter', None)}: {statistics}"
                }, inplace=True
            )
            unique_centers.append(meta.get("internalcenter", "None"))
            
            feature_data.append(features_df)

    data = reduce(lambda left, right: pd.merge(left, right, on=["Label", "group", "name"], how='inner'), feature_data)

    # Extract columns with p-values (those renamed to include ': statistics')
    pval_columns = [col for col in data.columns if f": {statistics}" in col]

    # Create a binary matrix where 1 = significant (p < threshold), 0 = not significant
    sig_matrix = data[pval_columns] < p_threshold
    sig_counts = sig_matrix.sum(axis=1)  # Count how many centers each feature is significant in

    # Add this information to the dataframe
    data["significant_in_n_centers"] = sig_counts

    # Remove non significant features
    data = data[data["significant_in_n_centers"] > 0]

    data.to_csv(output_path, index=False)