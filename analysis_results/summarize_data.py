import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

exclude = ['140015', '140025', '3523749', '3615751', 'T012', 'T019', 'T077']

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
                print(f"Skipping patient {patient_name} because not in included list")
                continue

            if patient_name in exclude:
                print(f"Skipping patient {patient_name} because in excluded list") 
                continue

            # Get all images in the datadir
            image = os.path.abspath(os.path.join(patient, f'{sequence}.nii.gz'))
            label = os.path.abspath(os.path.join(patient, f'{sequence}-mask.nii.gz'))

            if os.path.exists(image) and os.path.exists(label):
                print(f"Patient: {patient_name}, Image: {image}, Label: {label}")
                
                # Create a dictionary with the patient name as key and the image and label as values
                result[patient_name] = sequence

    return result

def find_center(sequences, external_center):
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

def create_boxplot(data, ax=None):
    """Create a boxplot for the given data."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    sns.boxplot(x='sequences', y='center', hue='label', data=data, ax=ax)
    ax.set_xlabel('Sequences')
    ax.set_ylabel('Label')

    return ax

def main(data_path='../data/final', label_names=['PD', 'Treatment']):
    label_file = os.path.join(data_path, 'labels.txt')
    
    sequences_combinations = [
        ['T1'],
        ['T1-FS-C', 'T1-C', 'T1-FS', 'T1'],
        ['T2'],
        ['T2-FS', 'T2'],
        ['T2-FS', 'T2', 'T1-FS-C', 'T1-FS', 'T1-C', 'T1'],
    ]

    fig, ax = plt.subplots(nrows=len(label_names), ncols=len(sequences_combinations), figsize=(6*len(label_names), 6*len(sequences_combinations)))
    for i, label in enumerate(label_names):
        for j, combination in enumerate(sequences_combinations):
            included_patients, labels = read_included_patients(label_file, label_name=label)

            # Get the images and labels from the datadir for the specified sequences
            sequences = get_sequences(data_path, combination, included_patients)
            centers = find_center(labels, external_center='Canada')

            data = pd.DataFrame({'center': centers, 'label': labels, 'sequences': sequences})
            import ipdb; ipdb.set_trace()
            # Create the boxplot
            create_boxplot(data, ax=ax[i, j])

if __name__ == '__main__':
    main()
