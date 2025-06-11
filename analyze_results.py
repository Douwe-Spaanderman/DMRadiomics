import os
import json
from visualization import read_roc_data, read_feature_data, available_imaging, plot_significant_features, save_features_by_center, plot_ROC
from itertools import groupby
from operator import itemgetter
import pandas as pd

default_experiment_dict = {
    'label': 'PD', 'seq': 'T2FS T2', 'addseq': 'None', 'externalcenter': 'None', 'internalcenter': 'All', 'ComBat': 'False'
}
measurements = ["AUC", "F1-score", "Accuracy", "Sensitivity", "Specificity", "NPV", "Precision", "BCA"]

def format_value(cell):
    try:
        # Remove parentheses and commas, split by space
        parts = [round(float(x.replace("(", "").replace(",", "").replace(")", "")), 2) for x in cell.split()]
        return f"{parts[0]:.2f} [{parts[1]:.2f}, {parts[2]:.2f}]"
    except Exception:
        return cell  # If the cell can't be parsed, return as-is

def extract_performance_measurements(data):
    """

    """
    final = {}
    for experiment, content in data.items():
        performance = content["performance"]
        meta = content["metadata"]

        final[experiment] = {
            **meta,
            **{measurement: performance.get(measurement + " 95%:", None) for measurement in measurements}
        }
    return final

def parse_experiment_name(name):
    """
    Parse the experiment name into a dictionary.
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

def run_analyze(experiment_root, output_root, data_root):
    """
    Analyze all WORC experiments for de
    
    return finalsmoid-type fibromatosis.
    
    Parameters:
    - experiment_root: Path to the experiment directory
    - output_root: Path to the result directory
    - data_root: Path to the raw data directory
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

        # Load feature importance data if available
        features_data = read_feature_data(os.path.join(experiment_path, "Evaluation", 'StatisticalTestFeatures_all_0.csv'))

        data[experiment] = {
            "performance": performance,
            "ranking": ranking,
            "roc_data": roc_data,
            "features_data": features_data,
            "metadata": experiment_dict
        }

    # Plot available imaging
    available_imaging(data_root, output_root)

    # Analyze combinations
    combinations = [
        {
            'label': meta["metadata"].get("label"),
            'seq': meta["metadata"].get("seq"),
            'ComBat': meta["metadata"].get("ComBat"),
            'addseq': meta["metadata"].get("addseq"),
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

        # Plot features importance - Only for internal centers
        feature_importance = {
            name: data for name, data in matching_experiments.items()
            if data["metadata"].get("externalcenter", "") == "None" and data["metadata"].get("internalcenter", "") != "All"
        }
        if feature_importance:
            if len(feature_importance) > 3:
                print("Warning: More than 3 experiments found for feature importance, expected 3.")

            save_features_by_center(data=feature_importance, output_path=os.path.join(combination_path, "features_by_center.xlsx"))
            plot_significant_features(data=feature_importance, output_path=os.path.join(combination_path, "features_importance.png"))

        # Plot ROC curves
        plot_ROC(data=matching_experiments, hue="externalcenter", output_path=os.path.join(combination_path, "External_ROC.png"))
        plot_ROC(data=matching_experiments, hue="internalcenter", output_path=os.path.join(combination_path, "Internal_ROC.png"))

        # Save performance measurements
        performance_data.update(extract_performance_measurements(matching_experiments))

    # Convert performance data to DataFrame
    performance_data = pd.DataFrame.from_dict(performance_data, orient='index')
    performance_data = performance_data.applymap(format_value)

    # Sort rows
    performance_data = performance_data.sort_values(['label', 'seq', 'addseq', 'externalcenter', 'internalcenter'])

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
    parser.add_argument(
        "-d",
        "--data_path",
        default="data/final",
        type=str,
        help="Path to the raw data directory"
    )

    args = parser.parse_args()

    run_analyze(args.experiment_path, args.output_path, args.data_path)
