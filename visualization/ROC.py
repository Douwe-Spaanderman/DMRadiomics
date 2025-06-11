import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import json
import pandas as pd
import os

# Configuration for line styles and markers
centers = ["Italy", "Canada", "Netherlands", "All"]

linestyles = ["-", "--", ":", "-."]
linestyles = {
    center: linestyle for center, linestyle in zip(centers, linestyles)
}

colorpalette = "colorblind"
colors = {
    center: color for center, color in zip(centers, sns.color_palette(colorpalette, len(centers)))
}

markers = ["*", "x", "o", "s"]


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

def create_legend_info(data):
    legend_info = []
    for k, v in data.items():
        AUC = [round(float(x.replace("(", "").replace(",", "").replace(")", "")), 2) for x in v["AUC 95%:"].split(" ")]
        legend_info.append(f"{k}: {AUC[0]:.2f}({AUC[1]:.2f}-{AUC[2]:.2f})")

    return legend_info

def plot_ROC(data, hue, radiologists=None, ax=None, title=None, figsize=(6, 6), output_path=None):
    """
    Plot ROC curves from the given data.

    Note, hue only works atm for "internalcenter" and "externalcenter".
    """
    plot_data = []
    AUC_data = {}
    for experiment, content in data.items():
        ROC_data = content["roc_data"]
        meta = content["metadata"]

        if ROC_data is not None:
            ROC_data = ROC_data.copy()
            ROC_data["experiment"] = experiment
            if meta.get(hue, "None") != "None":
                if hue == "internalcenter" and meta.get("externalcenter", "") != "None":
                    continue
                
                if hue == "externalcenter" and meta.get("internalcenter", "") != "All":
                    continue

                ROC_data[hue] = meta.get(hue, "None")
                plot_data.append(ROC_data)

                # Add AUC data for legend
                AUC_data[meta.get(hue, "None")] = content["performance"]

    if not plot_data:
        return None
    
    plot_data = pd.concat(plot_data, ignore_index=True)
    # Order plot_data based on center
    plot_data[hue] = pd.Categorical(plot_data[hue], categories=centers, ordered=True)
    plot_data = plot_data.sort_values(by=hue)

    # order AUC based on center
    AUC_data = {center: AUC_data[center] for center in centers if center in AUC_data}

    # Find unique hues and assign colors
    unique_hues = plot_data[hue].unique()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for unique in unique_hues:
        # Filter data for the current hue
        filtered_data = plot_data[plot_data[hue] == unique]

        # Order based on index
        filtered_data = filtered_data.sort_index()

        if not filtered_data.empty:
            ax.plot(filtered_data["1-Specificity"], filtered_data["Sensitivity"], label=unique, color=colors.get(unique, "black"), linestyle=linestyles.get(unique, "-"))

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
    l1 = ax.legend(loc='center right', bbox_to_anchor=(1, 0.15), labels=create_legend_info(AUC_data), frameon=False, fontsize=12)
    l1.set_title(r'$\bf{AUC\ (95\%\ CI)}$', prop={'size': 12})

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

    if output_path is not None:
        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    
    else:
        return ax