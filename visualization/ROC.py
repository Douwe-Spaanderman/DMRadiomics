import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
from typing import Dict, Optional, Tuple

# Configuration for line styles and markers
centers = ["Italy", "Canada", "Netherlands", "All"]

linestyles = ["-", "--", ":", "-."]
linestyles = {center: linestyle for center, linestyle in zip(centers, linestyles)}

colorpalette = "colorblind"
colors = {
    center: color
    for center, color in zip(centers, sns.color_palette(colorpalette, len(centers)))
}

markers = ["*", "x", "o", "s"]


def read_roc_data(file_path: str) -> pd.DataFrame:
    """
    Read ROC data from a CSV file and return a DataFrame with relevant columns.
    Parameters:
    - file_path: str, path to the CSV file containing ROC data.
    Returns:
    - pd.DataFrame with columns: FPR, TPR, 1-Specificity, Sensitivity.
    """
    data = pd.read_csv(
        file_path,
        converters={
            "FPR": lambda x: x.strip("[]").split(" "),
            "TPR": lambda x: x.strip("[]").split(" "),
        },
    )

    data["FPR"] = [[float(y) for y in x if y != ""] for x in data["FPR"]]
    data["TPR"] = [[float(y) for y in x if y != ""] for x in data["TPR"]]
    data["1-Specificity"] = [(x[0] + x[1]) / 2 for x in data["FPR"]]
    data["Sensitivity"] = [(x[0] + x[1]) / 2 for x in data["TPR"]]

    data["1-Specificity"] = data["1-Specificity"].clip(upper=1)
    data["Sensitivity"] = data.Sensitivity.clip(upper=1)

    return data


def create_legend_info(data: Dict[str, dict]) -> list:
    """
    Create legend information from the performance data.
    Parameters:
    - data: dict, contains performance data for each center.
    Returns:
    - list of strings for legend information.
    """
    legend_info = []
    for k, v in data.items():
        if k == "All":
            k = "All Centers"
        AUC = [
            round(float(x.replace("(", "").replace(",", "").replace(")", "")), 2)
            for x in v["AUC 95%:"].split(" ")
        ]
        legend_info.append(f"{k}: {AUC[0]:.2f} ({AUC[1]:.2f}-{AUC[2]:.2f})")

    return legend_info


def plot_ROC(
    data: Dict[str, dict],
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int] = (6, 6),
    output_path: Optional[str] = None,
) -> Optional[Axes]:
    """
    Plot ROC curves for external (LOCO) validation and internal cross-validation.
    Parameters:
    - data: dict, contains ROC data for each experiment.
    - ax: matplotlib Axes object, if None a new figure and axes will be created.
    - title: str, title of the plot.
    - figsize: tuple, size of the figure.
    - output_path: str, path to save the plot. If None, the plot will be shown.
    """
    external_plot_data = []
    internal_plot_data = []
    AUC_data = {}
    for experiment, content in data.items():
        ROC_data = content["roc_data"]
        meta = content["metadata"]

        if ROC_data is not None:
            ROC_data = ROC_data.copy()
            ROC_data["experiment"] = experiment
            if (
                meta.get("internalcenter", "None") == "All"
                and meta.get("externalcenter", "None") == "None"
            ):
                ROC_data["center"] = meta.get("internalcenter", "None")
                internal_plot_data.append(ROC_data)

                # Add AUC data for legend
                AUC_data[meta.get("internalcenter", "None")] = content["performance"]
            elif (
                meta.get("internalcenter", "None") == "All"
                and meta.get("externalcenter", "None") != "None"
            ):
                ROC_data["center"] = meta.get("externalcenter", "None")
                external_plot_data.append(ROC_data)

                # Add AUC data for legend
                AUC_data[meta.get("externalcenter", "None")] = content["performance"]

    if not external_plot_data:
        return None

    external_plot_data = pd.concat(external_plot_data, ignore_index=True)

    # Order plot_data based on center
    external_plot_data["center"] = pd.Categorical(
        external_plot_data["center"], categories=centers, ordered=True
    )
    external_plot_data = external_plot_data.sort_values(by="center")

    # order AUC based on center
    AUC_data = {center: AUC_data[center] for center in centers if center in AUC_data}

    # Find unique hues and assign colors
    unique_hues = external_plot_data["center"].unique()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    legend_handles = [
        Line2D([], [], color="white", alpha=0.0, label="External (LOCO) Validation")
    ]
    legend_info = ["External (LOCO) Validation"]
    for unique in unique_hues:
        # Filter data for the current hue
        filtered_data = external_plot_data[external_plot_data["center"] == unique]

        # Order based on index
        filtered_data = filtered_data.sort_index()

        if not filtered_data.empty:
            ax.plot(
                filtered_data["1-Specificity"],
                filtered_data["Sensitivity"],
                label=unique,
                color=colors.get(unique, "black"),
                linestyle=linestyles.get(unique, "-"),
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    label=unique,
                    color=colors.get(unique, "black"),
                    linestyle=linestyles.get(unique, "-"),
                )
            )
            legend_info.extend(
                create_legend_info({k: v for k, v in AUC_data.items() if k == unique})
            )

    # Reference line
    ax.plot([-0.5, 1.5], [-0.5, 1.5], color="k", linestyle="-", linewidth=0.2)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.05)

    ax.set_xlabel("1-Specificity", fontsize=14)
    ax.set_ylabel("Sensitivity", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    if internal_plot_data:
        internal_plot_data = pd.concat(internal_plot_data, ignore_index=True)
        # Spacer for legend
        legend_handles.append(Line2D([0], [0], color="white", alpha=0.0, label=""))
        legend_info.append("")

        ax.plot(
            internal_plot_data["1-Specificity"],
            internal_plot_data["Sensitivity"],
            label="Cross-Validation",
            color="black",
            linestyle="--",
            alpha=0.5,
        )
        legend_handles.append(
            Line2D([], [], color="white", alpha=0.0, label="Internal Cross-Validation")
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                label="Cross-Validation",
                color="black",
                linestyle="--",
                alpha=0.5,
            )
        )

        legend_info.extend(
            ["Internal Cross-Validation"]
            + create_legend_info({k: v for k, v in AUC_data.items() if k == "All"})
        )

    l1 = ax.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1, 0.18),
        labels=legend_info,
        frameon=False,
        fontsize=12,
    )

    # Bold only the section headers manually
    for text in l1.get_texts():
        if text.get_text() in [
            "External (LOCO) Validation",
            "Internal Cross-Validation",
        ]:
            text.set_weight("bold")

    ax.add_artist(l1)

    if title is not None:
        ax.set_title(title, fontsize=16)

    if output_path is not None:
        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    else:
        return ax
