import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import json
import pandas as pd
import os

# Configuration for line styles and markers
linestyles = ["-", "--", ":", "-."]
markers = ["*", "x", "o", "s"]
colorpalette = "colorblind"

def read_roc_data(file_path):
    data = pd.read_csv(
        file_path,
        converters={
            "FPR": lambda x: x.strip("[]").split(" "), 
            "TPR": lambda x: x.strip("[]").split(" ")
        }
    )

    data["FPR"] = [[float(y) for y in x if y != ""] for x in data["FPR"]] 
    data["TPR"] = [[float(y) for y in x if y != ""] for x in data["TPR"]] 
    data["1-Specificity"] = [(x[0] + x[1]) / 2 for x in data["FPR"]]
    data["Sensitivity"] = [(x[0] + x[1]) / 2 for x in data["TPR"]]

    data["1-Specificity"] = data["1-Specificity"].clip(upper=1)
    data["Sensitivity"] = data.Sensitivity.clip(upper=1)

    return data

def create_legend_info(data, hue):
    legend_info = []

    for d in data.values():
        AUC = [round(float(x.replace("(", "").replace(",", "").replace(")", "")), 2) for x in d["AUC"].split(" ")]
        legend_info.append(f"{d[hue]}: {AUC[0]:.2f}({AUC[1]:.2f}-{AUC[2]:.2f})")

    return legend_info

def create_ROC(data, hue, radiologists=None, ax=None, title=None, figsize=(6, 6)):
    colors = sns.color_palette(colorpalette, len(data))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for d, color, linestyle in zip(data.values(), colors, linestyles):
        ax.plot(d["1-Specificity"], d["Sensitivity"], label=d[hue], color=color, linestyle=linestyle)

    if radiologists is not None:
        for d, color in zip(data.values(), colors):
            for i, radiologist in enumerate(radiologists["radiologist"].unique()):
                d = radiologists[(radiologists[hue] == d[hue]) & (radiologists["radiologist"] == radiologist)]
                ax.scatter([1-d["specificity"]], [d["sensitivity"]], color=color, marker=markers[i], label=f"{d[hue]} {radiologist}")

    # Reference line
    ax.plot([-0.5, 1.5], [-0.5, 1.5], color='k', linestyle='-', linewidth=0.2)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.05)

    ax.set_xlabel('1-Specificity', fontsize=14)
    ax.set_ylabel('Sensitivity', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    l1 = ax.legend(title='AUC(95% CI)', loc='center right', bbox_to_anchor=(1, 0.15), labels=create_legend_info(data, hue), frameon=False, title_fontsize=12, fontsize=11)

    if radiologists is not None:
        handle_markers = []
        handle_labels = []
        for i, radiologist in enumerate(radiologists["radiologist"].unique()):
            handle_markers.append(Line2D([0], [0], label=f'radiologist {radiologist}', marker=markers[i], color='black', linestyle=''))
            handle_labels.append(f'Radiologist {i+1}')

        ax.legend(handles=[handle_markers, handle_labels], title='', loc='center right', bbox_to_anchor=(1, 0.1), scatterpoints=1, frameon=False, title_fontsize=12, fontsize=11)

        # Add custom legend for cohorts
        ax.gca().add_artist(l1)

    if title is not None:
        ax.set_title(title, fontsize=16)

    return ax

def create_all_ROCs(input_root="../data/results", output_root="../data/results"):
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

        roc_data = read_roc_data(os.path.join(experiment_path, "Evaluation", 'ROC_all_0.csv'))

        data[experiment] = {
            "center": center,
            "label": label,
            "sequence": sequence,            
            "AUC": performance["Statistics"]["AUC 95%:"],
            "1-Specificity": roc_data["1-Specificity"].tolist(),
            "Sensitivity": roc_data["Sensitivity"].tolist(),
        }

    # Order dictionary based on center
    data = dict(sorted(data.items(), key=lambda item: item[1]["center"]))

    unique_labels = set(item["label"] for item in data.values())
    unique_sequences = set(item["sequence"] for item in data.values())
    fig, ax = plt.subplots(nrows=len(unique_labels), ncols=len(unique_sequences), figsize=(6*len(unique_sequences), 6*len(unique_labels)))
    for i, label in enumerate(unique_labels):
        for j, sequence in enumerate(unique_sequences): 
            print(f"Label: {label}, Sequence: {sequence}")
            # Get all configurations for this label and sequence
            roc_data = {key: value for key, value in data.items() if value["label"] == label and value["sequence"] == sequence}
            _ = create_ROC(roc_data, hue="center", ax=ax[i,j], title=f"{label} - {sequence}")

    fig.savefig(os.path.join(output_root, f"ROC.png"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    create_all_ROCs()