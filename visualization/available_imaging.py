import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.text as mtext

exclude = ['140015', '140025', '3523749', '3615751', 'T012', 'T019', 'T077']
colorpalette = "colorblind"

def read_included_patients(label_file, label_name='PD'):
    """Read included patients from label file."""
    included_patients = []
    labels = {}
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            # Read the header to identify the column index for label_name
            header = f.readline().strip().split('\t')
            if label_name not in header:
                raise ValueError(f"Label name {label_name} not found in file header.")
            label_index = header.index(label_name)

            # Process the remaining lines
            for line in f:
                patient_data = line.strip().split('\t')
                if len(patient_data) <= label_index:
                    continue  # Skip invalid lines
                patient, label = patient_data[0], patient_data[label_index]
                if label and label != 'None':
                    included_patients.append(patient)
                    labels[patient] = label
    else:
        raise FileNotFoundError(f"Label file {label_file} does not exist.")
    
    return included_patients, labels

def get_sequences(imagedatadir, sequences, included_patients):
    """Get images and labels from the datadir."""
    result = {}
    for sequence in sequences:
        # Get all images masks in the datadir
        for patient in glob.glob(os.path.join(imagedatadir, '*')):
            if patient in result:
                continue

            # Get the patient name
            patient_name = os.path.basename(patient)
            if patient_name not in included_patients:
                continue

            if patient_name in exclude:
                continue

            # Get all images in the datadir
            image = os.path.abspath(os.path.join(patient, f'{sequence}.nii.gz'))
            label = os.path.abspath(os.path.join(patient, f'{sequence}-mask.nii.gz'))

            if os.path.exists(image) and os.path.exists(label):
                # Create a dictionary with the patient name as key and the image and label as values
                result[patient_name] = sequence

    return result

def find_center(sequences):
    centers = {}

    rules = {
        "Canada": ["T"],
        "Italy": ["2", "3", "6", "7", "9"],
        "Netherlands": ["11", "12", "13", "14", "15", "16", "18"],
    }

    for patient in sequences.keys():
        if any(patient.startswith(rule) for rule in rules["Canada"]):
            centers[patient] = "Canada"
        elif any(patient.startswith(rule) for rule in rules["Netherlands"]):
            centers[patient] = "Netherlands"
        elif any(patient.startswith(rule) for rule in rules["Italy"]):
            centers[patient] = "Italy"
        else:
            raise ValueError(f"Unknown patient: {patient}")

    return centers

def create_barplot(data, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sequences = sorted(data['sequences'].unique())
    centers = sorted(data['center'].unique())
    labels = sorted(data['label'].unique())

    # Define hatches for each label
    hatch_map = {
        '0': '',
        '1': '//',
    }

    # Define colors for each center
    color_map = sns.color_palette(colorpalette, len(centers))
    center_colors = {center: color_map[i] for i, center in enumerate(centers)}

    bar_width = 0.8 / len(centers)  # space per group, one bar per center

    x = np.arange(len(sequences))

    for seq_idx, seq in enumerate(sequences):
        for c_idx, center in enumerate(centers):
            xpos = x[seq_idx] - (bar_width * len(centers) / 2) + (c_idx + 0.5) * bar_width
            bottom = 0

            # 1. Stack present bars by label
            for label in labels:
                present_subset = data[
                    (data['sequences'] == seq) &
                    (data['center'] == center) &
                    (data['label'] == label) &
                    (data['missing'] == False)
                ]
                if not present_subset.empty:
                    count = present_subset['count'].values[0]
                    ax.bar(
                        xpos,
                        count,
                        width=bar_width,
                        color=center_colors[center],
                        edgecolor='black',
                        hatch=hatch_map[label],
                        bottom=bottom,
                        label='_nolegend_'
                    )
                    bottom += count

            # 2. Overlay missing bars in same label order
            for label in labels:
                missing_subset = data[
                    (data['sequences'] == seq) &
                    (data['center'] == center) &
                    (data['label'] == label) &
                    (data['missing'] == True)
                ]
                if not missing_subset.empty:
                    count = missing_subset['count'].values[0]
                    ax.bar(
                        xpos,
                        count,
                        width=bar_width,
                        color='none',
                        edgecolor='gray',
                        hatch=hatch_map[label],
                        bottom=bottom,
                        label='_nolegend_',
                        alpha=0.5
                    )
                    bottom += count

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(sequences)
    ax.set_ylabel("Number of Patients")
    if title:
        ax.set_title(title)

    # Build legends
    header_style = dict(facecolor='none', edgecolor='none')
    center_header = Patch(label=r'$\bf{Center:}$', **header_style)
    label_header = Patch(label=r'$\bf{Labels:}$', **header_style)
    missing_header = Patch(label=r'$\bf{Sequence:}$', **header_style)
    spacer = Patch(label='', **header_style)

    center_handles = [Patch(facecolor=center_colors[c], edgecolor='black', label=c) for c in centers]
    label_handles = [Patch(facecolor='white', edgecolor='black', hatch=hatch_map[l], label=l) for l in labels]
    available_handle = Patch(facecolor='white', edgecolor='black', label="Available", hatch='', alpha=1.0)
    missing_handle = Patch(facecolor='none', edgecolor='gray', hatch='', label="Missing", alpha=0.5)

    all_handles = (
        [center_header] + center_handles +
        [spacer] +
        [label_header] + label_handles +
        [spacer] +
        [missing_header, available_handle, missing_handle]
    )

    return ax, all_handles

def available_imaging(data_path='../data/final', output_root="../data/results", label_names=['PD', 'Treatment'], sequence_options=['T1', 'T1-FS', 'T1-C', 'T1-FS-C', 'T2', 'T2-FS']):
    label_file = os.path.join(data_path, 'labels.txt')

    fig, ax = plt.subplots(nrows=len(label_names), figsize=(3*len(sequence_options), 6*len(label_names)), squeeze=False)
    for i, label in enumerate(label_names):
        data = []
        for j, sequence in enumerate(sequence_options):
            included_patients, labels = read_included_patients(label_file, label_name=label)

            # Get the images and labels from the datadir for the specified sequences
            sequences = get_sequences(data_path, [sequence], included_patients)
            centers = find_center(labels)

            d = pd.DataFrame({'center': centers, 'label': labels, 'sequences': sequences})

            d['missing'] = d['sequences'].isna()
            d['sequences'] = sequence

            data.append(d)

        data = pd.concat(data, ignore_index=True)
        data = data.groupby(['center', 'label', 'sequences', 'missing']).size().reset_index(name='count')
        _, legend_handles = create_barplot(data, ax=ax[i][0], title=f"Label: {label}")

    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(0.9, 0.5),
        frameon=False
    )
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(os.path.join(output_root, f"available_sequences.png"), dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    available_imaging()