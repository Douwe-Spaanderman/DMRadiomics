import WORC
from WORC import BasicWORC
import os
import pandas as pd
import json
import fastr
import glob

script_path = os.path.dirname(os.path.abspath(__file__))

modus = 'binary_classification'

def get_images_and_labels(imagedatadir, sequences):
    """Get images and labels from the datadir."""
    images, labels = {}, {}
    for sequence in sequences:
        # Get all images masks in the datadir
        for patient in glob.glob(os.path.join(imagedatadir, '*')):
            if patient in images:
                continue

            # Get all images in the datadir
            image = os.path.abspath(os.path.join(patient, f'{sequence}.nii.gz'))
            label = os.path.abspath(os.path.join(patient, f'{sequence}-mask.nii.gz'))

            if os.path.exists(image) and os.path.exists(label):
                # Get the patient name
                patient_name = os.path.basename(patient)
                print(f"Patient: {patient_name}, Image: {image}, Label: {label}")
                
                # Create a dictionary with the patient name as key and the image and label as values
                images[patient_name] = image
                labels[patient_name] = label

    return images, labels

def leave_one_out(images, labels, external_center):
    """Leave one out cross-validation."""
    # Create a leave-one-out cross-validation
    Trimages, Trlabels, Tsimages, Tslabels = {}, {}, {}, {}

    rules = {
        "Canada": ["T"],
        "Italy": ["3", "6", "7", "9"],
        "Netherlands": ["11", "12", "13", "14", "15", "18"],
    }

    for patient in images.keys():
        if external_center == "Canada":
            if any(patient.startswith(rule) for rule in rules["Canada"]):
                Tsimages[patient] = images[patient]
                Tslabels[patient] = labels[patient]
            else:
                Trimages[patient] = images[patient]
                Trlabels[patient] = labels[patient]
        elif external_center == "Netherlands":
            if any(patient.startswith(rule) for rule in rules["Netherlands"]):
                Tsimages[patient] = images[patient]
                Tslabels[patient] = labels[patient]
            else:
                Trimages[patient] = images[patient]
                Trlabels[patient] = labels[patient]
        elif external_center == "Italy":
            if any(patient.startswith(rule) for rule in rules["Italy"]):
                Tsimages[patient] = images[patient]
                Tslabels[patient] = labels[patient]
            else:
                Trimages[patient] = images[patient]
                Trlabels[patient] = labels[patient]
        else:
            raise ValueError(f"Unknown external center: {external_center}")

    print(f"Training images: {len(Trimages)}, Training labels: {len(Trlabels)}")
    print(f"Testing images: {len(Tsimages)}, Testing labels: {len(Tslabels)}")

    return Trimages, Trlabels, Tsimages, Tslabels

def main(data_path, experiment_name, sequences=["T1"], external_center="Canada"):
    """Execute WORC Tutorial experiment."""
    print(f"Running in folder: {script_path}.")
    # ---------------------------------------------------------------------------
    # Input
    # ---------------------------------------------------------------------------
    # The minimal inputs to WORC are:
    #   - Images
    #   - Segmentations
    #   - Labels
    #
    # In BasicWORC, we assume you have a folder "datadir", in which there is a
    # folder for each patient, where in each folder there is a image.nii.gz and a mask.nii.gz:
    #           Datadir
    #               Patient_001
    #                   image.nii.gz
    #                   mask.nii.gz
    #               Patient_002
    #                   image.nii.gz
    #                   mask.nii.gz
    #               ...
    imagedatadir = os.path.join(data_path)

    # File in which the labels (i.e. outcome you want to predict) is stated
    # Again, change this accordingly if you use your own data.
    label_file = os.path.join(data_path, 'labels.txt')
    label_name = ['PD']

    # Determine whether we want to do a coarse quick experiment, or a full lengthy
    # one. Again, change this accordingly if you use your own data.
    coarse = True

    # Instead of the default tempdir, let's but the temporary output in a subfolder
    # in the same folder as this script
    tmpdir = os.path.join(script_path, 'WORC_' + experiment_name + "_" + external_center)
    print(f"Temporary folder: {tmpdir}.")

    # ---------------------------------------------------------------------------
    # The actual experiment: here we will use BasicWORC
    # ---------------------------------------------------------------------------

    # Create a BasicWORC object
    experiment = BasicWORC(experiment_name)

    # Get the images and labels from the datadir for the specified sequences
    images, labels = get_images_and_labels(imagedatadir, sequences)

    # Create a leave-one-out cross-validation
    Trimages, Trlabels, Tsimages, Tslabels = leave_one_out(images, labels, external_center)

    # Add the images and segmentations to the experiment
    experiment.images_train.append(Trimages)
    experiment.segmentations_train.append(Trlabels)
    experiment.images_test.append(Tsimages)
    experiment.segmentations_test.append(Tslabels)

    # Set the experiment parameters
    experiment.labels_from_this_file(label_file, is_training=True)
    experiment.labels_from_this_file(label_file, is_training=False)
    experiment.predict_labels(label_name)
    experiment.set_image_types(['MRI'])

    tmp = WORC.WORC('DMRadiomics')
    config = tmp.defaultconfig()
    config['General']['AssumeSameImageAndMaskMetadata'] = 'True'
    config['General']['tempsave'] = 'True'
    config['Classification']['fastr_plugin'] = 'DRMAAExecution'
    config['Bootstrap']['Use'] = 'True'

    experiment.add_config_overrides(config)

    experiment.binary_classification(coarse=coarse)

    # Set the temporary directory
    experiment.set_tmpdir(tmpdir)

    # evaluation
    experiment.add_evaluation()

    # Run the experiment!
    experiment.execute()

    # ---------------------------------------------------------------------------
    # Analysis of results
    # ---------------------------------------------------------------------------

    # Locate output folder
    outputfolder = fastr.config.mounts['output']
    experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name + "_" + external_center)

    print(f"Your output is stored in {experiment_folder}.")

    # Read the features for the first patient
    # NOTE: we use the glob package for scanning a folder to find specific files
    feature_files = glob.glob(os.path.join(experiment_folder,
                                           'Features',
                                           'features_*.hdf5'))

    if len(feature_files) == 0:
        raise ValueError('No feature files found: your network has failed.')

    feature_files.sort()
    featurefile_p1 = feature_files[0]
    features_p1 = pd.read_hdf(featurefile_p1)

    # Read the overall peformance
    performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
    if not os.path.exists(performance_file):
        raise ValueError(f'No performance file {performance_file} found: your network has failed.')

    with open(performance_file, 'r') as fp:
        performance = json.load(fp)

    # Print the feature values and names
    print("Feature values from first patient:")
    for v, l in zip(features_p1.feature_values, features_p1.feature_labels):
        print(f"\t {l} : {v}.")

    # Print the output performance
    print("\n Performance:")
    stats = performance['Statistics']
    for k, v in stats.items():
        print(f"\t {k} {v}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="WORC experiments for desmoid-type fibromatosis")
    parser.add_argument(
        "-d",
        "--data_path",
        default="data/final",
        type=str,
        help="Path to the data directory"
    )
    parser.add_argument(
        "-n",
        "--experiment_name",
        default="baseline",
        type=str,
        help="Name of the experiment"
    )
    parser.add_argument(
        "-s",
        "--sequences",
        default=["T1"],
        nargs='+',
        choices=['T1', 'T1-FS', 'T1-C', 'T1-FS-C', 'T2', 'T2-FS'],
        help="sequences to include in the experiment"
    )
    parser.add_argument(
        "-e",
        "--external_center",
        default="Canada",
        type=str,
        choices=["Canada", "Italy", "Netherlands"],
        help="External dataset to use"
    )

    args = parser.parse_args()

    main(args.data_path, args.experiment_name, args.sequences, args.external_center)