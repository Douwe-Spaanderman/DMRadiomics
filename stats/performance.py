import json
import os
import pandas as pd

measurements = ["AUC", "F1-score", "Accuracy", "Sensitivity", "Specificity", "NPV", "Precision", "BCA"]

def format_value(cell: str) -> str:
    """
    Format a cell value from the performance measurements.
    If the cell contains a string with parentheses and commas, it will be formatted to a string with two decimal places.
    If the cell is not in the expected format, it will return the cell as-is.
    """
    try:
        # Remove parentheses and commas, split by space
        parts = [round(float(x.replace("(", "").replace(",", "").replace(")", "")), 2) for x in cell.split()]
        return f"{parts[0]:.2f} [{parts[1]:.2f}, {parts[2]:.2f}]"
    except Exception:
        return cell  # If the cell can't be parsed, return as-is

def extract_performance_measurements(input_root: str = "../data/results", output_root: str = "../data/results") -> None:
    """
    Extract performance measurements from JSON files in the input directory and save them to an Excel file.
    Parameters:
    - input_root: str, path to the directory containing results folders.
    - output_root: str, path to the directory where the Excel file will be saved.
    """
    # Loop through results folders to find all unique experiments
    data = {}
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

        # Load data from JSON files
        with open(os.path.join(experiment_path, "performance_all_0.json"), "r") as f:
            performance = json.load(f)

        data[experiment] = {
            "center": center,
            "label": label,
            "sequence": sequence,            
            **{measurement: performance["Statistics"].get(measurement + " 95%:", None) for measurement in measurements}
        }

    # Order dictionary based on center
    data = dict(sorted(data.items(), key=lambda item: item[1]["center"]))

    with pd.ExcelWriter(os.path.join(output_root, "performance.xlsx")) as writer:
        for label in set(item["label"] for item in data.values()):
            for sequence in set(item["sequence"] for item in data.values()):        
                print(f"Label: {label}, Sequence: {sequence}")
                # Get all configurations for this label and sequence
                d = {key: value for key, value in data.items() if value["label"] == label and value["sequence"] == sequence}
                d = pd.DataFrame(d).T
                d = d.set_index('center')
                d = d.drop(columns=["label", "sequence"])

                # Format the measurements
                d = d.applymap(format_value)

                # Clean sheet name
                sheet_name = f"{label}_{sequence}"[:31].replace(":", "_").replace("/", "_")

                # Save as new sheet
                d.to_excel(writer, sheet_name=sheet_name)

if __name__ == "__main__":
    extract_performance_measurements()