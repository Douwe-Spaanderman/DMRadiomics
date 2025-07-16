import os
import subprocess
import pandas as pd
import json
import nibabel as nib
from nibabel.orientations import (
    aff2axcodes,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)

# Configure paths
INPUT_ROOT = "../data"
OUTPUT_ROOT = "../data/output"


def convert_dicom_to_nifti(dicom_dir, output_file):
    """Convert DICOM to NIfTI using dcm2niix"""
    subprocess.run(
        [
            "dcm2niix",
            "-f",
            "image",
            "-o",
            os.path.dirname(output_file),
            "-m",
            "y",  # Merge 2D slices into 3D volume
            "-z",
            "y",  # Compress to .nii.gz
            dicom_dir,
        ],
        check=True,
    )

    # Rename to ensure exact filename (dcm2niix may add suffixes)
    for f in os.listdir(os.path.dirname(output_file)):
        if f.startswith("image") and f.endswith(".nii.gz"):
            os.rename(os.path.join(os.path.dirname(output_file), f), output_file)


def convert_seg_to_nifti(dicom_seg_dir, output_file):
    """Convert DICOM-SEG to NIfTI mask"""
    seg_files = [f for f in os.listdir(dicom_seg_dir) if f.endswith(".dcm")]
    if not seg_files:
        raise RuntimeError("No DICOM-SEG file found")

    output_dir = os.path.dirname(output_file)

    # Run conversion (outputs to same dir as reference image)
    subprocess.run(
        [
            "segimage2itkimage",
            "--inputDICOM",
            os.path.join(dicom_seg_dir, seg_files[0]),
            "--outputDirectory",
            os.path.dirname(output_file),
            "--outputType",
            "nifti",
            "--prefix",
            "mask",  # Output will be mask.nii.gz
        ],
        check=True,
    )

    # Handle the expected filename like mask-1.nii.gz
    generated_mask = os.path.join(output_dir, "mask-1.nii.gz")
    if not os.path.exists(generated_mask):
        raise RuntimeError("Expected output file 'mask-1.nii.gz' not found")

    os.rename(generated_mask, output_file)


def force_nifti_header(image_file, seg_file, output_file=None):
    # Load the image and segmentation
    img = nib.load(image_file)
    seg = nib.load(seg_file)

    # Get orientation codes, e.g., ('R', 'A', 'S')
    img_ornt = aff2axcodes(img.affine)
    seg_ornt = aff2axcodes(seg.affine)

    if img_ornt != seg_ornt:
        print("Orientation do not match so transforming segmentation")
        # Compute orientation transform from segmentation to image
        seg_ornt_obj = axcodes2ornt(seg_ornt)
        img_ornt_obj = axcodes2ornt(img_ornt)
        transform = ornt_transform(seg_ornt_obj, img_ornt_obj)

        # Apply the transform
        seg_data = seg.get_fdata()
        reoriented_data = apply_orientation(seg_data, transform)

        # Create a new Nifti image with the image's affine
        new_seg = nib.Nifti1Image(reoriented_data, affine=img.affine, header=seg.header)
    else:
        # Orientations already match; just copy affine
        new_seg = nib.Nifti1Image(seg.get_fdata(), affine=img.affine, header=seg.header)

    # Save result
    if output_file is None:
        output_file = seg_file  # overwrite original

    nib.save(new_seg, output_file)


def find_metadata_match(metadata, study_folder):
    """Find metadata entry matching the patient metadata"""
    with open(os.path.join(study_folder, "metadata.json")) as f:
        meta = json.load(f)

    patient_meta = metadata[metadata["patient"] == meta["patient_name"]]

    if patient_meta.empty:
        print("❌ No metadata match found for this study, set manually")
        import ipdb

        ipdb.set_trace()

    patient_meta["Segmented sequence"] = (
        patient_meta["series_instance_uid"] == meta["referenced_series_uid"]
    )
    if patient_meta["Segmented sequence"].sum() == 0:
        print("❌ No segmented sequence found, set manually")
        import ipdb

        ipdb.set_trace()
    elif patient_meta["Segmented sequence"].sum() > 1:
        print("❌ Multiple segmented sequences found, set manually")
        import ipdb

        ipdb.set_trace()

    # Get the matching image name
    image_name = patient_meta.loc[
        patient_meta["Segmented sequence"], "Sequence Name"
    ].values[0]
    patient_meta = patient_meta[
        patient_meta["session"]
        == patient_meta.loc[patient_meta["Segmented sequence"], "session"].values[0]
    ]
    if "Unclassified" in image_name or "T2-C" in image_name or "T2-FS-C" in image_name:
        print("❌ weird sequence segmented on, manually check this")
        import ipdb

        ipdb.set_trace()

    patient_meta = patient_meta.reset_index(drop=True)
    return patient_meta, image_name


def generate_sequence_name(row):
    name = row["Sequence Type"]
    modifiers = []
    if row["Fat Saturated"]:
        modifiers.append("FS")
    if row["Inversion Recovery"]:
        if "FS" not in modifiers:
            modifiers.append("FS")
    if row["Contrast Enhanced"]:
        modifiers.append("C")
    if modifiers:
        name += "-" + "-".join(modifiers)
    return name


def process_patient(patient_dir, metadata):
    """Process all studies for one patient"""
    patient_name = os.path.basename(patient_dir)
    output_dir = os.path.join(OUTPUT_ROOT, patient_name)
    os.makedirs(output_dir, exist_ok=True)

    patients_meta = []
    # Find all studies for this patient
    for study_folder in os.listdir(patient_dir):
        study_path = os.path.join(patient_dir, study_folder)
        if not os.path.isdir(study_path):
            continue

        # Read metadata for this study
        patient_meta, image_name = find_metadata_match(metadata, study_path)

        # Paths to DICOMs
        dicom_image_dir = os.path.join(study_path, "DICOM", "IMAGE")
        dicom_seg_dir = os.path.join(study_path, "DICOM", "SEGMENTATION")

        # Output paths (directly under patient_name)
        image_nifti = os.path.join(output_dir, image_name + ".nii.gz")
        mask_nifti = os.path.join(output_dir, image_name + "-mask.nii.gz")

        # Convert image
        print(f"Converting DICOM images for {patient_name}...")
        convert_dicom_to_nifti(dicom_image_dir, image_nifti)

        # Convert segmentation (requires image as reference)
        print(f"Converting DICOM-SEG for {patient_name}...")
        convert_seg_to_nifti(dicom_seg_dir, mask_nifti)

        # Force the mask to have the same header as the image
        print(f"Aligning headers for {patient_name}...")
        force_nifti_header(image_nifti, mask_nifti)

        print(
            f"✅ Saved NIfTIs for {patient_name} in:\n  {image_nifti}\n  {mask_nifti}"
        )

        patients_meta.append(patient_meta)

    patients_meta = pd.concat(patients_meta)
    return patients_meta


def main():
    """Process all patients"""
    metadata = pd.read_csv(os.path.join(INPUT_ROOT, "metadata_analyzed.csv"))
    metadata["Sequence Name"] = metadata.apply(generate_sequence_name, axis=1)

    patients_meta = []
    for patient_name in os.listdir(os.path.join(INPUT_ROOT, "DICOMs")):
        patient_path = os.path.join(INPUT_ROOT, "DICOMs", patient_name)
        if not os.path.isdir(patient_path):
            continue

        try:
            patient_meta = process_patient(patient_path, metadata)
            patients_meta.append(patient_meta)
        except Exception as e:
            print(f"❌ Failed to process {patient_name}: {e}")

    # Concatenate all metadata for this patient
    print("Combining all patient meta data into dataframe")
    patients_meta = pd.concat(patients_meta)
    patients_meta.to_csv(os.path.join(OUTPUT_ROOT, "metadata.csv"), index=False)


if __name__ == "__main__":
    main()
