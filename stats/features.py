
import json
import os
import pandas as pd
import glob
from scipy.stats import mannwhitneyu
from typing import Union
from visualization import plot_significant_features

center_rules = {
    "Canada": ["T"],
    "Italy": ["3", "6", "7", "9"],
    "Netherlands": ["11", "12", "13", "14", "15", "16", "18"],
}

def read_feature_data(file_path:Union[pd.DataFrame, str], statistics="Mann-Whitney"):
    """Reads performance data from a CSV file and returns it as a DataFrame."""
    if isinstance(file_path, str):
        try:
            data = pd.read_csv(file_path, skiprows=1)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return pd.DataFrame()
    elif isinstance(file_path, pd.DataFrame):
        data = file_path.copy()
    else:
        raise ValueError("Input must be a file path (str) or a pandas DataFrame.")
        
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

def calculate_mann_whitney(data, group_col, value_col):
    """
    Calculate the Mann-Whitney U statistic for two groups in the data.

    Parameters:
    - data: DataFrame containing the data.
    - group_col: Column name for the grouping variable.
    - value_col: Column name for the values to compare.

    Returns:
    - u_statistic: The Mann-Whitney U statistic.
    - p_value: The p-value of the test.
    """
    groups = data.groupby(group_col)
    
    group_keys = list(groups.groups.keys())
    group1 = groups.get_group(group_keys[0])[value_col]
    group2 = groups.get_group(group_keys[1])[value_col]

    # Remove NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()

    p_value = mannwhitneyu(group1, group2, alternative="two-sided")[1]
    
    return p_value

def extract_features(input_root, label_file, output_root):
    # Extract label from label file
    labels = pd.read_csv(label_file, sep="\t")
    label = input_root.split("/")[-1]
    labels = labels[["Patient", label]]

    data = []
    for feature_file in glob.glob(os.path.join(input_root, "*.hdf5")):
        patient = feature_file.split("_")[-2]

        # Skip dummy patients
        if "Dummy_" in feature_file:
            continue

        # Determine the center based if patient starts with a specific rule
        center = None
        for center_name, rules in center_rules.items():
            if any(patient.startswith(rule) for rule in rules):
                center = center_name
                break

        if center == None:
            raise ValueError(f"Patient {patient} does not match any center rules: {center_rules}")

        if "MRI_0_" in feature_file:
            sequence = "T2"
        elif "MRI_1_" in feature_file:
            sequence = "T1"
        else:
            raise ValueError(f"Unknown sequence in file: {feature_file}")
        
        feature_data = pd.read_hdf(feature_file)
        data.append({
            "patient": patient,
            "center": center,
            "sequence": sequence,
            **{k: v for k, v in zip(feature_data["feature_labels"], feature_data["feature_values"])},
        })

    data = pd.DataFrame(data)
    data = data.merge(labels, left_on="patient", right_on="Patient", how="left").drop(columns=["Patient"])

    for sequence in data["sequence"].unique():
        plot_data = {}
        for center in data["center"].unique():
            center_data = data[(data["center"] == center) & (data["sequence"] == sequence)]
            if center_data.empty:
                raise ValueError(f"No data found for center {center} and sequence {sequence}")

            # Calculate Mann-Whitney U statistic for each feature
            features = center_data.columns.difference(["patient", "center", "sequence", label])
            results = []
            for feature in features:
                p_value = calculate_mann_whitney(center_data, label, feature)
                results.append({
                    "Label": feature,
                    "Mann-Whitney": p_value
                })

            results = pd.DataFrame(results)
            # This is a bit hacky, but needed to make work with the plotting function
            plot_data[f"{label}_{center}_{sequence}"] = {
                "features_data": read_feature_data(results),
                "metadata": {
                    "internalcenter": center,
                    "sequence": sequence,
                    "label": label
                }
            }
        # Save features by center
        output_file = os.path.join(output_root, "features_importance", f"{label}_{sequence}.png")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plot_significant_features(data=plot_data, output_path=output_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features from hfd5 data and calculate Mann-Whitney.")
    parser.add_argument("--input", type=str, default="data/features/Treatment", help="Input directory containing results")
    parser.add_argument("--label", type=str, default="data/final/labels.txt", help="Label file")
    parser.add_argument("--output", type=str, default="data/analyzed_results/V1", help="Output directory for results")

    args = parser.parse_args()
    extract_features(args.input, args.label, args.output)

