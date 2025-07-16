import os
import json
from visualization import read_roc_data, plot_significant_features, save_features_by_center, plot_ROC
from stats import read_posterior_data, calculate_DeLong, compute_auc_ci, read_feature_data
import pandas as pd
import numpy as np
from typing import Union, Dict

default_experiment_dict = {
    'label': 'PD', 'seq': 'T2FS T2', 'addseq': 'None', 'externalcenter': 'None', 'internalcenter': 'All', 'ComBat': 'False', 'Clinical': 'None'
}
measurements = ["AUC", "F1-score", "Accuracy", "Sensitivity", "Specificity", "NPV", "Precision", "BCA"]

subgroups = {
    'Age': ['<=38', '>38'],
    'Sex': ['Female', 'Male'],
    'Location': ['Abdominal wall ', 'Extremities', 'Chest wall', 'Other'],
    'Magnetic Field Strength': ['<=1.5T', '>1.5T'],
    'Manufacturer': ['GE', 'Siemens', 'Philips', 'Unknown'],
    'Fat saturation available': [False, True],
    'Imputation required': [False, True],
    'Radiomics on segmented scan': [False, True]
}

def format_value(cell: str) -> str:
    """
    Format a cell value from the performance measurements.
    If the cell contains a string with parentheses and commas, it will be formatted to a string with two decimal places.
    If the cell is not in the expected format, it will return the cell as-is.
    Parameters:
    - cell: str, the cell value to format.
    Returns:
    - str, formatted cell value.
    """
    try:
        # Remove parentheses and commas, split by space
        parts = [round(float(x.replace("(", "").replace(",", "").replace(")", "")), 2) for x in cell.split()]
        return f"{parts[0]:.2f} [{parts[1]:.2f}, {parts[2]:.2f}]"
    except Exception:
        return cell  # If the cell can't be parsed, return as-is

def extract_performance_measurements(data: Dict[str, dict], DeLong: Dict[str, float]) -> Dict[str, dict]:
    """
    Extract performance measurements from the data dictionary and format them for output.
    Parameters:
    - data: dict, contains performance data for each experiment.
    - DeLong: dict, contains p-values from the DeLong statistical test.
    Returns:
    - dict, formatted performance measurements for each experiment.
    """
    final = {}
    for experiment, content in data.items():
        performance = content["performance"]
        meta = content["metadata"]

        final[experiment] = {
            **meta,
            **{measurement: performance.get(measurement + " 95%:", None) for measurement in measurements},
            **{"p-value": DeLong.get(experiment, "")}
        }
    return final

def parse_experiment_name(name:str) -> Dict[str, Union[str, bool]]:
    """
    Parse the experiment name to extract metadata.
    Parameters:
    - name: str, the name of the experiment.
    Returns:
    - dict, containing parsed metadata with default values.
    """
    # skip WORC_ prefix if it exists
    if name.startswith("WORC_"):
        name = name[5:]
        
    parsed = dict(
        part.split("_", 1)
        for part in name.split("__")
        if "_" in part  # ensure valid key-value structure
    )
    # Add default values for missing keys
    for key, value in default_experiment_dict.items():
        parsed.setdefault(key, value)

    return parsed

def run_analyze(experiment_root: str, output_root: str) -> None:
    """
    Run the analysis on all WORC experiments for desmoid-type fibromatosis.
    Parameters:
    - experiment_root: str, path to the directory containing WORC experiment folders.
    - output_root: str, path to the directory where results will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_root, exist_ok=True)

    # Load all data from each experiment
    data = {}
    experiments = [f for f in os.listdir(experiment_root) if f.startswith("WORC_")]
    for experiment in experiments:
        experiment_path = os.path.join(experiment_root, experiment)
        if not os.path.isdir(experiment_path):
            continue

        if not os.path.exists(os.path.join(experiment_path, "performance_all_0.json")):
            print(f"Skipping {experiment}: No performance_all_0.json found.")
            continue
    
        # Parse the experiment name
        experiment_dict = parse_experiment_name(experiment)

        # Load performance data from JSON
        with open(os.path.join(experiment_path, "performance_all_0.json"), "r") as f:
            performance = json.load(f)

        ranking = performance.get("Rankings", {})
        performance = performance.get("Statistics", {})

        # Load ROC data from CSV
        roc_data = read_roc_data(os.path.join(experiment_path, "Evaluation", 'ROC_all_0.csv'))

        # Load probability data for statistical test
        ranked_posterior = read_posterior_data(os.path.join(experiment_path, "Evaluation", 'RankedPosteriors_all_0.csv'))

        # Load feature importance data if available
        features_data = read_feature_data(os.path.join(experiment_path, "Evaluation", 'StatisticalTestFeatures_all_0.csv'))

        data[experiment] = {
            "performance": performance,
            "ranking": ranking,
            "roc_data": roc_data,
            "features_data": features_data,
            "ranked_posterior": ranked_posterior,
            "metadata": experiment_dict
        }

    # Load clinical metadata
    metadata = pd.read_csv(os.path.join(experiment_root, "clinical_updated.csv"))

    # Set the groups based on robustness analysis paper
    metadata['Age'] = np.where(metadata['Age'] <= 38, '<=38', '>38')
    metadata['Sex'] = metadata['Sex'].str.capitalize()
    metadata['Magnetic Field Strength'] = metadata['Magnetic Field Strength'].map({
        '1T': '<=1.5T',
        '1.5T': '<=1.5T',
        '3T': '>1.5T'
    })

    # Analyze combinations
    combinations = [
        {
            'label': meta["metadata"].get("label"),
            'seq': meta["metadata"].get("seq"),
            'ComBat': meta["metadata"].get("ComBat"),
            'addseq': meta["metadata"].get("addseq")
        }
        for meta in data.values()
    ]
    unique_combinations = {tuple(sorted(combo.items())) for combo in combinations}
    # Convert back to dictionaries
    unique_combinations = [dict(combo) for combo in unique_combinations]

    performance_data = {}
    for combination in unique_combinations:
        print(f"Processing combination: {combination}")
        combination_name = f"label={combination['label']}__Sequence={combination['seq']}_AdditionalSequence={combination['addseq']}__ComBat={combination['ComBat']}"
        combination_path = os.path.join(output_root, combination_name)
        os.makedirs(combination_path, exist_ok=True)

        # Find experiments matching this combo
        matching_experiments = {
            name: data for name, data in data.items()
            if all(data["metadata"].get(k) == v for k, v in combination.items())
        }
        print(f"Found {len(matching_experiments)} experiments for this combination.")

        # Plot features importance - Only for internal centers and not clinical
        feature_importance = {
            name: data for name, data in matching_experiments.items()
            if data["metadata"].get("externalcenter", "") == "None" and data["metadata"].get("internalcenter", "") != "All" and data["metadata"].get("Clinical", "") == "None"
        }
        if feature_importance:
            if len(feature_importance) > 3:
                print("Warning: More than 3 experiments found for feature importance, expected 3.")

            save_features_by_center(data=feature_importance, output_path=os.path.join(combination_path, "features_by_center.xlsx"))
            plot_significant_features(data=feature_importance, output_path=os.path.join(combination_path, "features_importance.png"))

        # Plot ROC curves
        # For normal experiment
        ROC_data = {
            name: data for name, data in matching_experiments.items()
            if data["metadata"].get("Clinical", "") == "None"
        }
        plot_ROC(data=ROC_data, output_path=os.path.join(combination_path, "ROC.png"))
        # For clinical models
        ROC_data = {
            name: data for name, data in matching_experiments.items()
            if data["metadata"].get("Clinical", "") == "Location"
        }
        plot_ROC(data=ROC_data, output_path=os.path.join(combination_path, "ROC_Imaging_and_Location.png"))
        ROC_data = {
            name: data for name, data in matching_experiments.items()
            if data["metadata"].get("Clinical", "") == "Age_Sex_Location"
        }
        plot_ROC(data=ROC_data, output_path=os.path.join(combination_path, "ROC_Imaging_and_Clinical.png"))
        ROC_data = {
            name: data for name, data in matching_experiments.items()
            if data["metadata"].get("Clinical", "") == "Age_Sex_Location_Only"
        }
        plot_ROC(data=ROC_data, output_path=os.path.join(combination_path, "ROC_Clinical_only.png"))

        # Calculate DeLong statistical test - and test robustness against other variables
        external_data = {
            name: data for name, data in matching_experiments.items()
            if data["metadata"].get("externalcenter", "") != "None"
        }
        unique_centers = set([data["metadata"]["externalcenter"] for data in external_data.values()])
        DeLong_results = {}
        ranked_posteriors = []
        for center in unique_centers:
            baseline = [data for data in external_data.values() if data["metadata"].get("Clinical", "") == "None" and data["metadata"].get("externalcenter", "") == center]
            if len(baseline) < 1 or len(baseline) > 1:
                print("Weird, didn't expect multiple baseline experiments")
            baseline = baseline[0]
            baseline_ranked_posterior = baseline["ranked_posterior"]
                
            # Calculate DeLong statistical test
            DeLong_rocB = {
                name: data for name, data in external_data.items()
                if data["metadata"].get("Clinical", "") != "None" and data["metadata"].get("externalcenter", "") == center
            }
            for key, roc in DeLong_rocB.items():
                DeLong_results[key] = calculate_DeLong(baseline_ranked_posterior, roc["ranked_posterior"])

            # Save ranked posterior data for the baseline
            ranked_posteriors.append(baseline_ranked_posterior)

        # Robustness assessment
        ranked_posteriors = pd.concat(ranked_posteriors, ignore_index=True)
        ranked_posteriors["PatientID"] = ranked_posteriors["PatientID"].astype(str)
        ranked_posteriors = ranked_posteriors.merge(metadata, left_on="PatientID", right_on="Patient", how="left")
        robustness = []
        for subgroup, values in subgroups.items():
            for value in values:
                subset = ranked_posteriors[ranked_posteriors[subgroup] == value]
                if subset['TrueLabel'].nunique() < 2:
                    print(f"Skipping subgroup {subgroup} with value {value}: not enough unique labels.")
                    continue

                auc, lower, upper = compute_auc_ci(
                    subset['TrueLabel'].values, subset['Probability'].values
                )
                robustness.append({
                    'Feature': subgroup,
                    'Level': value,
                    'AUC': auc,
                    '95% CI Lower': lower,
                    '95% CI Upper': upper,
                    'AUC [95% CI]': f"{auc:.2f} [{lower:.2f}, {upper:.2f}]",
                    'N': len(subset)
                })

        robustness = pd.DataFrame(robustness)
        robustness.to_csv(os.path.join(combination_path, "robustness.csv"), index=False)

        # Save performance measurements
        performance_data.update(extract_performance_measurements(matching_experiments, DeLong_results))

    # Convert performance data to DataFrame
    performance_data = pd.DataFrame.from_dict(performance_data, orient='index')
    performance_data = performance_data.applymap(format_value)

    # Sort rows
    performance_data = performance_data.sort_values(['label', 'externalcenter', 'internalcenter', 'seq', 'addseq', 'Clinical'])

    with pd.ExcelWriter(os.path.join(output_root, "performance.xlsx")) as writer:
        # Save the performance data to a new sheet
        performance_data.to_excel(writer, sheet_name="Performance")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Analyze all WORC experiments for desmoid-type fibromatosis")
    parser.add_argument(
        "-e",
        "--experiment_path",
        default="data/results",
        type=str,
        help="Path to the experiment directory"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="data/analyzed_results",
        type=str,
        help="Path to the result directory"
    )

    args = parser.parse_args()

    run_analyze(args.experiment_path, args.output_path)
