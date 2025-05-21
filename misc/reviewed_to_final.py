import os
import json
import shutil
import re
from glob import glob

def strip_numbering(filename):
    """Convert T1-3.nii.gz -> T1.nii.gz, T1-FS-1.nii.gz -> T1-FS.nii.gz"""
    return re.sub(r'-(\d+)(?=\.nii\.gz$)', '', filename)

def postprocess_patient(patient_input_dir, patient_output_dir):
    review_file = os.path.join(patient_input_dir, "registration_review.json")
    if not os.path.exists(review_file):
        print(f"Skipping {patient_input_dir}: No review file found.")
        return

    os.makedirs(patient_output_dir, exist_ok=True)

    with open(review_file, "r") as f:
        review = json.load(f)

    used_output_names = {}

    for image_name, review_info in review.items():
        if not review_info.get("RegistrationOK", False):
            continue

        chosen_mask_type = review_info["ChosenMask"]
        image_path = os.path.join(patient_input_dir, image_name)
        base_name = strip_numbering(image_name)
        mask_name = base_name.replace(".nii.gz", "-mask.nii.gz")

        # Check for duplicates
        if base_name in used_output_names:
            raise RuntimeError(f"Conflict: Multiple reviewed sequences map to {base_name} in {patient_input_dir}")
        used_output_names[base_name] = image_name

        # Copy image
        shutil.copy(image_path, os.path.join(patient_output_dir, base_name))

        # Find corresponding mask
        original_mask_name = image_name.replace(".nii.gz", f"-{chosen_mask_type}-mask.nii.gz")
        original_mask_path = os.path.join(patient_input_dir, original_mask_name)

        if not os.path.exists(original_mask_path):
            raise FileNotFoundError(f"Missing mask file: {original_mask_path}")

        shutil.copy(original_mask_path, os.path.join(patient_output_dir, mask_name))

    # Copy reference T2.nii.gz and T2-mask.nii.gz
    for ref_file in ["T2.nii.gz", "T2-mask.nii.gz"]:
        ref_src = os.path.join(patient_input_dir, ref_file)
        ref_dst = os.path.join(patient_output_dir, ref_file)
        if os.path.exists(ref_src):
            shutil.copy(ref_src, ref_dst)

def postprocess_all_patients(input_root="../data/output", output_root="../data/final"):
    os.makedirs(output_root, exist_ok=True)
    patient_dirs = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

    for patient_id in sorted(patient_dirs):
        input_dir = os.path.join(input_root, patient_id)
        output_dir = os.path.join(output_root, patient_id)
        print(f"Processing {patient_id}")
        postprocess_patient(input_dir, output_dir)

if __name__ == "__main__":
    postprocess_all_patients()
