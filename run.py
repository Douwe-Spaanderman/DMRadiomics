from WORC import BasicWORC
import os
import pandas as pd
import json
import fastr
import glob
from typing import List, Tuple, Dict

script_path = os.path.dirname(os.path.abspath(__file__))

modus = 'binary_classification'
exclude = ['140015', '140025', '3523749', '3615751', 'T012', 'T019', 'T077']

rules = {
    "Canada": ["T"],
    "Italy": ["3", "6", "7", "9"],
    "Netherlands": ["11", "12", "13", "14", "15", "18"],
}

def read_included_patients(label_file: str, label_name: str = 'PD') -> List[str]:
    """
    Read the included patients from the label file.
    The label file is expected to be a tab-separated file with the first column as patient ID
    and the second column as the label.
    If the label_name is not found in the file, it raises a ValueError.
    If the label is 'None' or empty, the patient is not included.
    Parameters:
    - label_file: str, Path to the label file.
    - label_name: str, The name of the label to include patients for.
    Returns:
    - included_patients: list, List of patient IDs that are included based on the label.
    """
    included_patients = []
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
    else:
        raise FileNotFoundError(f"Label file {label_file} does not exist.")
    
    return included_patients

def get_images_and_labels(imagedatadir: str, sequences: List[str], included_patients: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Get images and labels from the datadir for the specified sequences.
    The images are expected to be in the format:
        {patient_name}/{sequence}.nii.gz
    The labels are expected to be in the format:
        {patient_name}/{sequence}-mask.nii.gz
    Parameters:
    - imagedatadir: str, Path to the directory containing patient folders.
    - sequences: list, List of sequences to include in the experiment.
    - included_patients: list, List of patient names to include.
    Returns:
    - images: dict, Dictionary with patient names as keys and image paths as values.
    - labels: dict, Dictionary with patient names as keys and label paths as values.
    """
    images, labels = {}, {}
    for sequence in sequences:
        # Get all images masks in the datadir
        for patient in glob.glob(os.path.join(imagedatadir, '*')):
            if patient in images:
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
                images[patient_name] = image
                labels[patient_name] = label

    return images, labels

def leave_one_out(images: Dict[str, str], labels: Dict[str, str], external_center: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Create a leave-one-out cross-validation for the specified external center.
    Parameters:
    - images: dict, Dictionary with patient names as keys and image paths as values.
    - labels: dict, Dictionary with patient names as keys and label paths as values.
    - external_center: str, The external center to use for cross-validation.
    Returns:
    - Trimages: dict, Training images for the specified external center.
    - Trlabels: dict, Training labels for the specified external center.
    - Tsimages: dict, Testing images for the specified external center.
    - Tslabels: dict, Testing labels for the specified external center.
    """
    # Create a leave-one-out cross-validation
    Trimages, Trlabels, Tsimages, Tslabels = {}, {}, {}, {}

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

def extract_center(images: Dict[str, str], labels: Dict[str, str], center: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Extract one center from images and labels based on the specified center.
    Parameters:
    - images: dict, Dictionary with patient names as keys and image paths as values.
    - labels: dict, Dictionary with patient names as keys and label paths as values.
    - center: str, The center to extract images and labels for.
    Returns:
    - Trimages: dict, Training images for the specified center.
    - Trlabels: dict, Training labels for the specified center.
    """
    # Extract one center from images and labels
    Trimages, Trlabels = {}, {}

    for patient in images.keys():
        if center == "Canada":
            if any(patient.startswith(rule) for rule in rules["Canada"]):
                Trimages[patient] = images[patient]
                Trlabels[patient] = labels[patient]
        elif center == "Netherlands":
            if any(patient.startswith(rule) for rule in rules["Netherlands"]):
                Trimages[patient] = images[patient]
                Trlabels[patient] = labels[patient]
        elif center == "Italy":
            if any(patient.startswith(rule) for rule in rules["Italy"]):
                Trimages[patient] = images[patient]
                Trlabels[patient] = labels[patient]
        else:
            raise ValueError(f"Unknown center: {center}")

    print(f"Training images: {len(Trimages)}, Training labels: {len(Trlabels)}")

    return Trimages, Trlabels

def create_dummy(dictionaries_A: List[Dict[str, str]], dictionaries_B: List[Dict[str, str]]) -> Tuple[Tuple[Dict[str, str]], Tuple[Dict[str, str]]]:
    """
    Create dummy entries for missing keys in two dictionaries.
    This function takes two lists of dictionaries and ensures that both lists have the same keys.
    If a key is missing in one dictionary, it creates a dummy entry in the other dictionary.
    Parameters:
    - dictionaries_A: list, List of dictionaries A.
    - dictionaries_B: list, List of dictionaries B.
    Returns:
    - final_A: tuple, Tuple of dictionaries A with dummy entries added.
    - final_B: tuple, Tuple of dictionaries B with dummy entries added.
    """
    final_A = ()
    final_B = ()
    for A, B in zip(dictionaries_A, dictionaries_B):
        keysA = set(A.keys())
        keysB = set(B.keys())

        # Find missing keys
        missing_in_dictB = keysA - keysB
        missing_in_dictA = keysB - keysA

        # Create dummy entries for missing keys
        for key in missing_in_dictB:
            B[f"{key}_Dummy"] = A[key]
        for key in missing_in_dictA:
            A[f"{key}_Dummy"] = B[key]

        # Add to tuples
        final_A += (A,)
        final_B += (B,)

    return final_A, final_B

def add_clinical_data(data_path: str, tmpdir: str, Trimages: Dict[str, str], Tsimages: Dict[str, str] = None, clinical: List[str] = ["Age", "Sex", "Location"]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Add clinical data to the experiment.
    Parameters:
    - data_path: str, Path to the directory containing clinical data.
    - tmpdir: str, Temporary directory to store the clinical data.
    - Trimages: dict, Training images for the experiment.
    - Tsimages: dict, Testing images for the experiment (optional).
    - clinical: list, List of clinical features to include in the experiment.
    Returns:
    - Trfeatures_files: dict, Dictionary with patient names as keys and feature file paths as values for training images.
    - Tsfeatures_files: dict, Dictionary with patient names as keys and feature file paths as values for testing images (optional).
    """
    clinical_path = os.path.join(data_path, 'clinical_data.csv')
    if not os.path.exists(clinical_path):
        raise FileNotFoundError(f"Clinical data file {clinical_path} does not exist.")

    # Make sure tmpdir exists
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # Read the clinical data
    clinical_data = pd.read_csv(clinical_path)

    # Extract the relevant columns based on the clinical input
    columns_to_extract = ['Patient'] + clinical
    columns_to_extract = [col for col in columns_to_extract if col in clinical_data.columns]
    clinical_data = clinical_data[columns_to_extract]

    # Ensure the 'Patient' column is of type string
    clinical_data['Patient'] = clinical_data['Patient'].astype(str)

    # Create Trsemantics DataFrame at tmpdir
    if "Only" in clinical:
        feature_directory = os.path.join(tmpdir, 'ManualInputFeatures')
        if not os.path.exists(feature_directory):
            os.makedirs(feature_directory)
            
        clinical_data = clinical_data.set_index('Patient')
        clinical_data = clinical_data.fillna(0)
        for column in clinical_data.columns:
            if clinical_data[column].dtype == "float" or clinical_data[column].dtype == "int":
                clinical_data[column] = clinical_data[column].astype(int)
            elif clinical_data[column].dtype == "object":
                clinical_data[column] = pd.Categorical(clinical_data[column])
                clinical_data[column] = clinical_data[column].cat.codes

        # Filter the clinical data for the patients in Trimages and Tsimages
        Trpatients = list(Trimages.keys())
        Trpatients = [patient.replace("_Dummy", "") for patient in Trpatients]
        Trfeatures = clinical_data[clinical_data.index.isin(Trpatients)]

        Trfeatures_files = {}
        for patient, row in Trfeatures.iterrows(): 
            panda_data = pd.Series([
                [float(x) for x in row.values.tolist()],
                row.keys().tolist()],
                index=['feature_values', 'feature_labels'],
                name='Image features'
            )

            file_path = os.path.join(feature_directory, patient + ".hdf5")
            panda_data.to_hdf(file_path, 'image_features')
            Trfeatures_files[patient] = file_path

        if Tsimages:
            Tspatients = list(Tsimages.keys())
            Tspatients = [patient.replace("_Dummy", "") for patient in Tspatients]
            Tsfeatures = clinical_data[clinical_data.index.isin(Tspatients)]

            Tsfeatures_files = {}
            for patient, row in Tsfeatures.iterrows(): 
                panda_data = pd.Series([
                    [float(x) for x in row.values.tolist()],
                    row.keys().tolist()],
                    index=['feature_values', 'feature_labels'],
                    name='Image features'
                )

                file_path = os.path.join(feature_directory, patient + ".hdf5")
                panda_data.to_hdf(file_path, 'image_features')
                Tsfeatures_files[patient] = file_path

            return Trfeatures_files, Tsfeatures_files
        else:
            return Trfeatures_files
    else:
        # Filter the clinical data for the patients in Trimages and Tsimages
        Trpatients = list(Trimages.keys())
        Trsemantics = clinical_data[clinical_data['Patient'].isin(Trpatients)]
        
        Trsemantics_file = os.path.join(tmpdir, 'Trsemantics.csv')
        Trsemantics.to_csv(Trsemantics_file, index=False)

        if Tsimages:
            Tspatients = list(Tsimages.keys())
            Tssemantics = clinical_data[clinical_data['Patient'].isin(Tspatients)]

            # Create Tssemantics DataFrame at tmpdir
            Tssemantics_file = os.path.join(tmpdir, 'Tssemantics.csv')
            Tssemantics.to_csv(Tssemantics_file, index=False)

            return Trsemantics_file, Tssemantics_file
        else:
            return Trsemantics_file

def main(data_path: str, experiment_name: str, sequences: List[str] = ["T1"], external_center: str = "Canada", include_center: str = "All", label_name: List[str] = ["PD"], combat: bool = False, additional_sequences: List[str] = ["None"], clinical: List[str] = ["None"]):
    """
    Main function to run the WORC experiment for desmoid-type fibromatosis.
    Parameters:
    - data_path: str, Path to the data directory.
    - experiment_name: str, Name of the experiment.
    - sequences: list, List of sequences to include in the experiment.
    - external_center: str, External center to use for cross-validation.
    - include_center: str, Center for inclusion to use.
    - label_name: list, List of labels to predict.
    - combat: bool, Whether to use ComBat for data homogenization.
    - additional_sequences: list, List of additional sequences to include in the experiment.
    - clinical: list, List of clinical features to include in the experiment.
    """
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
    # Read the labels from the file
    included_patients = read_included_patients(label_file, label_name=label_name[0])

    # Determine whether we want to do a coarse quick experiment, or a full lengthy
    # one. Again, change this accordingly if you use your own data.
    coarse = False

    # Instead of the default tempdir, let's but the temporary output in a subfolder
    # in the same folder as this script
    tmpfolder = fastr.config.mounts['tmp']
    tmpdir = os.path.join(tmpfolder, 'WORC_' + experiment_name)
    print(f"Temporary folder: {tmpdir}.")

    # ---------------------------------------------------------------------------
    # The actual experiment: here we will use BasicWORC
    # ---------------------------------------------------------------------------

    # Create a BasicWORC object
    experiment = BasicWORC(experiment_name)

    # Get the images and labels from the datadir for the specified sequences
    images, labels = get_images_and_labels(imagedatadir, sequences, included_patients)

    # Create a leave-one-out cross-validation
    if external_center != "None":
        Trimages, Trlabels, Tsimages, Tslabels = leave_one_out(images, labels, external_center)
    else:
        Trimages, Trlabels = images, labels

    if include_center != "All":
        Trimages, Trlabels = extract_center(Trimages, Trlabels, include_center)

    if "None" not in additional_sequences:
        print(f"Adding additional sequence: {additional_sequences}")
        additional_images, additional_labels = get_images_and_labels(imagedatadir, additional_sequences, included_patients)
        if include_center != "All":
            additional_images, additional_labels = extract_center(additional_images, additional_labels, include_center)
        
        if external_center != "None":
            Trimages2, Trlabels2, Tsimages2, Tslabels2 = leave_one_out(additional_images, additional_labels, external_center)
            (Trimages, Trlabels, Tsimages, Tslabels), (Trimages2, Trlabels2, Tsimages2, Tslabels2) = create_dummy([Trimages, Trlabels, Tsimages, Tslabels], [Trimages2, Trlabels2, Tsimages2, Tslabels2])
        else:
            Trimages2, Trlabels2 = additional_images, additional_labels
            (Trimages, Trlabels), (Trimages2, Trlabels2) = create_dummy([Trimages, Trlabels], [Trimages2, Trlabels2])

    # Add the images and segmentations to the experiment
    experiment.images_train.append(Trimages)
    experiment.segmentations_train.append(Trlabels)
    experiment.labels_from_this_file(label_file, is_training=True)

    if external_center != "None":
        experiment.images_test.append(Tsimages)
        experiment.segmentations_test.append(Tslabels)
        experiment.labels_from_this_file(label_file, is_training=False)

    # Adding additional sequences:
    images_types = ["MRI"]
    if "None" not in additional_sequences and "Only" not in clinical:
        images_types.append("MRI")
        experiment.images_train.append(Trimages2)
        experiment.segmentations_train.append(Trlabels2)

        if external_center != "None":
            experiment.images_test.append(Tsimages2)
            experiment.segmentations_test.append(Tslabels2)

    # Set the experiment parameters
    experiment.predict_labels(label_name)
    experiment.set_image_types(images_types)

    overwrite_config = {
        'General': {
            'AssumeSameImageAndMaskMetadata': 'True',
            'tempsave': 'True'
        },
        'Classification': {
            'fastr_plugin': 'ProcessPoolExecution'
        },
        'Bootstrap': {
            'Use': 'False'
        },
        "SelectFeatGroup": {
            'semantic_features': 'False'
        },
        "Featsel": {
            'GroupwiseSearch': 'True'
        }
    }
    if external_center != "None":
        overwrite_config['Bootstrap'].update({'Use': True})

    if combat:
        print("Using ComBat")
        overwrite_config['General'].update({'ComBat': True})
    
    if "None" not in clinical:
        print("Adding clinical data")
        if "Only" in clinical:
            if external_center != "None":
                Trfeatures, Tsfeatures = add_clinical_data(data_path, tmpdir, Trimages, Tsimages, clinical=clinical)
                experiment.features_train.append(Trfeatures)
                experiment.features_test.append(Tsfeatures)
            else:
                Trfeatures = add_clinical_data(data_path, tmpdir, Trimages, clinical=clinical)
                experiment.features_train.append(Trfeatures)

            overwrite_config['SelectFeatGroup'].update({
                'semantic_features': 'True',
                'shape_features': 'False',
                'histogram_features': 'False',
                'orientation_features': 'False',
                'texture_Gabor_features': 'False',
                'texture_GLCM_features': 'False',
                'texture_GLDM_features': 'False',
                'texture_GLCMMS_features': 'False',
                'texture_GLRLM_features': 'False',
                'texture_GLSZM_features': 'False',
                'texture_GLDZM_features': 'False',
                'texture_NGTDM_features': 'False',
                'texture_NGLDM_features': 'False',
                'texture_LBP_features': 'False',
                'dicom_features': 'False',
                'vessel_features': 'False',
                'phase_features': 'False',
                'fractal_features': 'False',
                'location_features': 'False',
                'rgrd_features': 'False',
                'original_features': 'False',
                'wavelet_features': 'False',
                'log_features': 'False',
            })
            overwrite_config['Featsel'].update({'GroupwiseSearch': 'False'})
        else:
            if external_center != "None":
                Trsementics, Tssemantics = add_clinical_data(data_path, tmpdir, Trimages, Tsimages, clinical=clinical)
                experiment.semantics_file_train.append(Trsementics)
                experiment.semantics_file_test.append(Tssemantics)
            else:
                Trsementics = add_clinical_data(data_path, tmpdir, Trimages, clinical=clinical)
                experiment.semantics_file_train.append(Trsementics)

            overwrite_config['SelectFeatGroup'].update({'semantic_features': 'True'})

    print(overwrite_config)
    experiment.add_config_overrides(overwrite_config)

    if label_name[0] == 'Mutation':
        experiment.multiclass_classification(coarse=coarse)
    else:
        experiment.binary_classification(coarse=coarse)

    # Set multicore
    experiment.set_multicore_execution()    

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
    experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

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
        choices=["None", "Canada", "Italy", "Netherlands"],
        help="External dataset to use"
    )
    parser.add_argument(
        "-i",
        "--include_center",
        default="All",
        type=str,
        choices=["All", "Canada", "Italy", "Netherlands"],
        help="Center for inclusion to use"
    )
    parser.add_argument(
        "-l",
        "--label_name",
        default=["PD"],
        nargs='+',
        choices=['PD', 'Treatment', 'Mutation'],
        help="Name of the label to predict"
    )
    parser.add_argument(
        "-co",
        "--combat",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Do you want to use ComBat to homogenous data"
    )
    parser.add_argument(
        "-a",
        "--additional_sequences",
        default=["None"],
        nargs='+',
        choices=['None', 'T1', 'T1-FS', 'T1-C', 'T1-FS-C', 'T2', 'T2-FS'],
        help="Additional sequences to include in the experiment"
    )
    parser.add_argument(
        "-cl",
        "--clinical",
        default=["None"],
        nargs='+',
        choices=['None', 'Age', 'Sex', 'Location', 'Only'],
        help="Do you want to include clinical data in the experiment"
    )

    args = parser.parse_args()

    main(args.data_path, args.experiment_name, args.sequences, args.external_center, args.include_center, args.label_name, args.combat, args.additional_sequences, args.clinical)
