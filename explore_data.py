import pandas as pd
import numpy as np
import os
from visualization import available_imaging
from collections import defaultdict

from collections import Counter
from scipy.stats import chi2_contingency, kruskal

mapping = {
    "Magnetic Field Strength": {"1.0": "1T", "1.5": "1.5T", "3.0": "3T"},
    "Manufacturer": {
        "UMC_Utrecht_MR4": "Unknown",
        "Siemens Healthineers": "Siemens",
        "Philips Healthcare": "Philips",
        "MRIAVANTO": "Siemens",
        "Philips Medical Systems": "Philips",
        "SIEMENS": "Siemens",
        "PHILIPS-HJ24JVI": "Philips",
        "GE MEDICAL SYSTEMS": "GE",
        "MRC30792": "Unknown",
    },
}
order = ["T1-FS", "T2-FS", "T1", "T2", "T1-FS-C", "T1-C"]


def explore_data(output_root: str, data_root: str, overwrite: bool = True) -> None:
    """
    Explore all data for desmoid-type fibromatosis, including clinical and imaging data.
    This function collects information about available MRI sequences, their parameters,
    and the centers where the data was collected. It also performs statistical tests to compare
    the distributions of these parameters across different centers.
    Parameters:
    - output_root: str, path to the directory where results will be saved.
    - data_root: str, path to the directory containing the raw data.
    - overwrite: bool, whether to overwrite existing clinical data with new information.
    """
    # Load clinical data
    clinical = pd.read_excel(os.path.join(data_root, "clinical.xlsx"))
    clinical = clinical[clinical["Excluded"].isnull() & clinical["Inclusion"] == 1]
    valid_patients = set(clinical["Study number"].astype(str))

    # Load scan metadata
    scanmeta = pd.read_csv(os.path.join(data_root, "scanmeta.csv"))

    # Plot available imaging
    available_imaging(data_root, output_root)

    # Prepare dictionary to collect info per center
    centers = scanmeta["Center"].dropna().unique()
    table = {center: defaultdict(list) for center in centers}

    for patient_id in valid_patients:
        scanmeta_patient = scanmeta[scanmeta["patient"].astype(str) == patient_id]
        if scanmeta_patient.empty:
            continue

        # Get segmented orientation(s)
        segmented = scanmeta_patient[scanmeta_patient["Segmented sequence"] == True]
        if segmented.empty or "Imaging Plane" not in segmented.columns:
            continue
        elif segmented["scan"].nunique() > 1:
            # If multiple scans are available, pick one manually
            import ipdb

            ipdb.set_trace()
        else:
            segmented = segmented.iloc[0]

        # Get session variables
        metadata = (
            eval(segmented["metadata"])
            if isinstance(segmented["metadata"], str)
            else segmented["metadata"]
        )
        scan_data = metadata.get("scan_data", {})
        center = segmented.get("Center", "Unknown")
        table[center]["Patient ID"].append(patient_id)
        table[center]["MRI sequences segmented on"].append(
            segmented.get("Sequence Name", "Unknown")
        )

        # Magnetic field strength
        field_strength = metadata.get("MagneticFieldStrength") or scan_data.get(
            "fieldStrength"
        )
        table[center]["Magnetic Field Strength"].append(
            mapping["Magnetic Field Strength"].get(field_strength, "Unknown")
        )

        # Manufacturer
        manufacturer = (
            scan_data.get("scanner/manufacturer")
            or scan_data.get("scanner")
            or scan_data.get("manufacturer")
        )
        if not manufacturer:
            import ipdb

            ipdb.set_trace()
        table[center]["Manufacturer"].append(
            mapping["Manufacturer"].get(manufacturer, "Unknown")
        )

        # Now check additional sequences - filter scans to same orientation
        filtered = scanmeta_patient[
            (scanmeta_patient["Available"] == True)
            & (scanmeta_patient["Exclude"] == False)
            & (scanmeta_patient["Imaging Plane"] == segmented["Imaging Plane"])
        ]

        # Keep only one of the sequences
        filtered = filtered.drop_duplicates(subset=["Sequence Name"], keep="first")
        filtered["Sequence Name"] = pd.Categorical(
            filtered["Sequence Name"], categories=order, ordered=True
        )
        filtered = filtered.sort_values(by="Sequence Name")
        filtered = filtered[filtered["Sequence Name"].notna()]

        T1 = False
        T2 = False
        for _, row in filtered.iterrows():
            center = row["Center"]
            if pd.isna(center) or pd.isna(row["metadata"]):
                continue

            metadata = (
                eval(row["metadata"])
                if isinstance(row["metadata"], str)
                else row["metadata"]
            )
            scan_data = metadata.get("scan_data", {})

            # Available Sequences
            seq_name = row.get("Sequence Name", "Unknown")
            table[center]["Available MRI sequences"].append(seq_name)

            if seq_name in ["T1-FS-C", "T1-C"]:
                continue

            if seq_name.startswith("T1") and T1 == False:
                T1 = True
                seq = "T1"
            elif seq_name.startswith("T2") and T2 == False:
                T2 = True
                seq = "T2"
            else:
                continue

            # Slice Thickness (mm)
            slice_thickness = metadata.get("SliceThickness")
            if slice_thickness:
                table[center][f"{seq}: Slice Thickness (mm)"].append(
                    float(slice_thickness)
                )

            # Repetition Time (ms)
            repetition_time = metadata.get("RepetitionTime") or scan_data.get(
                "parameters/tr"
            )
            if repetition_time:
                table[center][f"{seq}: Repetition Time (ms)"].append(
                    float(repetition_time)
                )

            # Echo Time (ms)
            echo_time = metadata.get("EchoTime") or scan_data.get("parameters/te")
            if echo_time:
                table[center][f"{seq}: Echo Time (ms)"].append(float(echo_time))

    # Define structure of output table with hierarchical rows (store as (section, subrow))
    raw_index_rows = [
        ("Magnetic Field Strength", None),
        ("Magnetic Field Strength", "1T"),
        ("Magnetic Field Strength", "1.5T"),
        ("Magnetic Field Strength", "3T"),
        ("Manufacturer", None),
        ("Manufacturer", "Siemens"),
        ("Manufacturer", "GE"),
        ("Manufacturer", "Philips"),
        ("Manufacturer", "Toshiba"),
        ("Manufacturer", "Hitachi"),
        ("Manufacturer", "Unknown"),
        ("Setting (Unit)", None),
        ("Setting (Unit)", "T1: Slice Thickness (mm)"),
        ("Setting (Unit)", "T1: Repetition Time (ms)"),
        ("Setting (Unit)", "T1: Echo Time (ms)"),
        ("Setting (Unit)", "T2: Slice Thickness (mm)"),
        ("Setting (Unit)", "T2: Repetition Time (ms)"),
        ("Setting (Unit)", "T2: Echo Time (ms)"),
        ("Available MRI sequences", None),
        ("Available MRI sequences", "T1"),
        ("Available MRI sequences", "T1-FS"),
        ("Available MRI sequences", "T1-C"),
        ("Available MRI sequences", "T1-FS-C"),
        ("Available MRI sequences", "T2"),
        ("Available MRI sequences", "T2-FS"),
        ("MRI sequences segmented on", None),
        ("MRI sequences segmented on", "T1"),
        ("MRI sequences segmented on", "T1-FS"),
        ("MRI sequences segmented on", "T1-C"),
        ("MRI sequences segmented on", "T1-FS-C"),
        ("MRI sequences segmented on", "T2"),
        ("MRI sequences segmented on", "T2-FS"),
    ]

    # Create readable index names (use full paths for sub-items)
    index_rows = []
    index_row_mapping = {}  # Map from full index label to (parent, child)

    for parent, child in raw_index_rows:
        if child is None:
            index_rows.append(parent)
            index_row_mapping[parent] = (parent, None)
        else:
            full_label = f"{parent} - {child}"
            index_rows.append(full_label)
            index_row_mapping[full_label] = (parent, child)

    # Create DataFrame
    summary = pd.DataFrame(index=index_rows, columns=centers)

    # Count total patients per center
    patient_counts = {
        center: len(clinical[clinical["Country"] == center]) for center in centers
    }

    for center in centers:
        for row in index_rows:
            parent, child = index_row_mapping[row]
            value = ""

            if child is None:
                # Header row — leave empty
                value = ""
            elif parent == "Setting (Unit)":
                values = table[center].get(child, [])
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    value = f"{mean:.1f} ± {std:.1f}"
            else:
                items = table[center].get(parent, [])
                count = sum(child.lower() == s.lower() for s in items)
                total = patient_counts[center]
                if count > 0:
                    percent = 100 * count / total if total > 0 else 0
                    value = f"{count} ({percent:.1f}%)"

            summary.loc[row, center] = value

    # Rename columns to include N
    summary.columns = [
        f"{center} (N = {patient_counts[center]})" for center in summary.columns
    ]

    # Add p-value column and calculations
    summary["p-value"] = ""

    tests = {
        "Magnetic Field Strength": "chi2",
        "Manufacturer": "chi2",
        "T1: Slice Thickness (mm)": "Kruskal-Wallis",
        "T1: Repetition Time (ms)": "Kruskal-Wallis",
        "T1: Echo Time (ms)": "Kruskal-Wallis",
        "T2: Slice Thickness (mm)": "Kruskal-Wallis",
        "T2: Repetition Time (ms)": "Kruskal-Wallis",
        "T2: Echo Time (ms)": "Kruskal-Wallis",
        "Available MRI sequences": "chi2",
        "MRI sequences segmented on": "chi2",
    }

    for item, test in tests.items():
        values = [table[center].get(item, []) for center in centers]

        if test == "chi2":
            unique_values = sorted(set(v for sublist in values for v in sublist))
            contingency = pd.DataFrame(
                {
                    f"Center {i+1}": [
                        Counter(center_data).get(k, 0) for k in unique_values
                    ]
                    for i, center_data in enumerate(values)
                },
                index=unique_values,
            )
            chi2, p, dof, expected = chi2_contingency(contingency.values)
            summary.loc[item, "p-value"] = f"{p:.3f}"
        elif test == "Kruskal-Wallis":
            stat, p = kruskal(*values)
            summary.loc["Setting (Unit) - " + item, "p-value"] = f"{p:.3f}"
        else:
            continue

    if overwrite:
        # Saving information to clinical needed for downstream analysis
        patient_information = {}
        for _, values in table.items():
            patient_information.update(
                {
                    patient_id: {
                        "MRI sequences segmented on": segmented,
                        "Magnetic Field Strength": field_strength,
                        "Manufacturer": manufacturer,
                    }
                    for patient_id, segmented, field_strength, manufacturer in zip(
                        values["Patient ID"],
                        values["MRI sequences segmented on"],
                        values["Magnetic Field Strength"],
                        values["Manufacturer"],
                    )
                }
            )

        # Save patient information to existing clinical
        tmp = pd.DataFrame.from_dict(patient_information, orient="index")
        tmp.index.name = "Study number"
        tmp = tmp.reset_index()

        clinical["Study number"] = clinical["Study number"].astype(str)
        clinical = clinical.merge(tmp, on="Study number", how="left")
        clinical.to_csv(os.path.join(data_root, "clinical_updated.csv"), index=False)

    # Save summary to csv
    output_file = os.path.join(output_root, "scan_summary.csv")
    os.makedirs(output_root, exist_ok=True)
    summary.to_csv(output_file, index=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Explore all data for desmoid-type fibromatosis"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="data/analyzed_results",
        type=str,
        help="Path to the result directory",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        default="data/final",
        type=str,
        help="Path to the raw data directory",
    )

    args = parser.parse_args()

    explore_data(args.output_path, args.data_path)
