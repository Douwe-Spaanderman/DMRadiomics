import os
import shutil
import subprocess
import pandas as pd
import glob

dicom_root = '../data/DICOMs'
output_root = '../data/output'

def load_metadata(metadata="metadata_final.csv"):
    """Loads the metadata CSV file into a pandas DataFrame."""
    return pd.read_csv(os.path.join(dicom_root, metadata))

def construct_dicom_path(dicom_root, patient_id, session_id, scan_id):
    """Constructs the path to the DICOM directory."""
    return os.path.join(dicom_root, patient_id, session_id, 'DICOM', 'ADDITIONAL_IMAGES', scan_id)

def construct_output_path(output_root, patient_id, sequence_name, sequence_counts):
    """Determines the output path for the NIfTI file, assigning sequential suffixes."""
    patient_output_dir = os.path.join(output_root, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    base_filename = sequence_name.strip().replace(' ', '_')
    count = sequence_counts.get((patient_id, base_filename), 0) + 1
    sequence_counts[(patient_id, base_filename)] = count

    filename = f"{base_filename}-{count}.nii.gz"
    return os.path.join(patient_output_dir, filename), base_filename

def convert_dicom_to_nifti(dicom_dir, temp_output_dir, base_filename):
    """Uses dcm2niix to convert DICOM files to NIfTI format."""
    cmd = [
        'dcm2niix',
        '-z', 'y',  # gzip compression
        '-f', base_filename,
        '-o', temp_output_dir,
        dicom_dir
    ]
    subprocess.run(cmd, check=True)

def process_row(row, dicom_root, output_root, sequence_counts):
    """Processes a single row from the DataFrame."""
    patient_id = str(row['patient'])
    session_id = str(row['session'])
    scan_id = str(row['scan'])
    sequence_name = str(row['Sequence Name'])

    dicom_dir = construct_dicom_path(dicom_root, patient_id, session_id, scan_id)
    output_path, base_filename = construct_output_path(output_root, patient_id, sequence_name, sequence_counts)

    if os.path.exists(output_path):
        print(f"Skipping existing file: {output_path}")
        return

    temp_output_dir = os.path.join(output_root, patient_id, 'temp_conversion')
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        convert_dicom_to_nifti(dicom_dir, temp_output_dir, base_filename)
        converted_file = os.path.join(temp_output_dir, f"{base_filename}.nii.gz")
        if not os.path.exists(converted_file):
            raise FileNotFoundError(f"Converted file not found: {converted_file}")
        shutil.move(converted_file, output_path)
        print(f"Successfully converted: {output_path}")
    except Exception as e:
        print(f"Error processing {dicom_dir}: {e}")
    finally:
        shutil.rmtree(temp_output_dir, ignore_errors=True)

def main():
    df = load_metadata()
    sequence_counts = {}

    for patient_id, patient_df in df.groupby('patient'):
        print(f"\nProcessing patient: {patient_id}")
        for _, row in patient_df.iterrows():
            process_row(row, dicom_root, output_root, sequence_counts)

if __name__ == "__main__":
    main()
