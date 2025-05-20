import itk
import os
import glob
from collections import defaultdict
from typing import Tuple, Optional

rigid_custom = {
    'MaximumNumberOfIterations': ['500'],
    'NumberOfResolutions': ['4']
}

affine_custom = {
    'MaximumNumberOfIterations': ['1000'],
    'ImageSampler': ['Random']
}

def get_sequence_variations(patient_dir):
    """Organize sequences by their base name and variations."""
    sequences = defaultdict(list)
    
    # Find all NIfTI files (excluding masks and registered masks)
    nii_files = glob.glob(os.path.join(patient_dir, "*.nii.gz"))
    nii_files = [f for f in nii_files if not any(x in f for x in ["-mask", "-registered-mask"])]
    
    for nii_file in nii_files:
        base_name = os.path.basename(nii_file).replace(".nii.gz", "")
        
        # Handle numbered sequences (e.g., T1-FS-C-1, T1-FS-C-2)
        if "-" in base_name and base_name.split("-")[-1].isdigit():
            base_seq = "-".join(base_name.split("-")[:-1])
            seq_number = base_name.split("-")[-1]
            sequences[base_seq].append((int(seq_number), nii_file))
        else:
            sequences[base_name].append((0, nii_file))  # 0 indicates no number
    
    # Sort each sequence group by number
    for seq in sequences:
        sequences[seq].sort()
    
    return sequences

def find_reference_sequence(patient_dir, sequences):
    """Find a sequence that has a corresponding mask to use as reference."""
    for seq_name in sequences:
        # Check for both possible mask naming patterns
        mask_path_v1 = os.path.join(patient_dir, f"{seq_name}-mask.nii.gz")
        mask_path_v2 = os.path.join(patient_dir, f"{seq_name.split('-')[0]}-mask.nii.gz")
        
        if os.path.exists(mask_path_v1):
            return seq_name, mask_path_v1
        elif os.path.exists(mask_path_v2):
            return seq_name.split('-')[0], mask_path_v2
    
    return None, None

def register_segmentation(
    fixed_img_path: str,
    moving_img_path: str,
    moving_mask_path: str,
    output_dir: str,
    basename: str,
    rigid_params: Optional[dict] = None,
    affine_params: Optional[dict] = None
) -> Tuple[bool, bool]:
    """
    Perform both rigid and affine registration using ITK Elastix.
    
    Args:
        fixed_img_path: Path to the fixed reference image
        moving_img_path: Path to the moving image to be registered
        moving_mask_path: Path to the mask associated with the moving image
        output_dir: Directory to save output masks
        basename: Base name for output files (without extension)
        rigid_params: Optional custom parameters for rigid registration
        affine_params: Optional custom parameters for affine registration
        
    Returns:
        Tuple indicating success of rigid and affine registrations (rigid_success, affine_success)
    """
    rigid_success, affine_success = False, False
    
    try:
        # Load images with correct pixel types
        fixed_image = itk.imread(fixed_img_path, itk.F)
        moving_image = itk.imread(moving_img_path, itk.F)
        moving_mask = itk.imread(moving_mask_path, itk.UC)
        
        # Create parameter object
        parameter_object = itk.ParameterObject.New()
        
        # Common parameters for both registration types
        common_mask_params = {
            'FinalBSplineInterpolationOrder': ['0'],  # Nearest neighbor
            'DefaultPixelValue': ['0'],  # Background value
            'ResultImagePixelType': ['unsigned char']  # Maintain mask type
        }
        
        # --- Rigid Registration ---
        rigid_param_map = parameter_object.GetDefaultParameterMap('rigid')
        
        # Update with common mask parameters
        rigid_param_map.update(common_mask_params)
        
        # Apply any custom rigid parameters
        if rigid_params:
            rigid_param_map.update(rigid_params)
            
        # Clear existing maps and add rigid
        parameter_object.AddParameterMap(rigid_param_map)
        
        # Perform rigid registration
        try:
            rigid_result, rigid_transform_params = itk.elastix_registration_method(
                fixed_image, 
                moving_image,
                parameter_object=parameter_object,
                log_to_console=False
            )
            
            # Transform the mask with rigid
            rigid_transformed_mask = itk.transformix_filter(
                moving_mask,
                rigid_transform_params,
                log_to_console=False
            )
            
            # Save rigid-transformed mask
            rigid_output_path = f"{output_dir}/{basename}-rigid-mask.nii.gz"
            itk.imwrite(rigid_transformed_mask, rigid_output_path)
            rigid_success = True
        except Exception as rigid_e:
            print(f"Rigid registration failed: {str(rigid_e)}")
        
        # --- Affine Registration ---
        affine_param_map = parameter_object.GetDefaultParameterMap('affine')
        
        # Update with common mask parameters
        affine_param_map.update(common_mask_params)
        
        # Apply any custom affine parameters
        if affine_params:
            affine_param_map.update(affine_params)
            
        # Clear existing maps and add affine
        parameter_object.AddParameterMap(affine_param_map)
        
        # Perform affine registration
        try:
            affine_result, affine_transform_params = itk.elastix_registration_method(
                fixed_image, 
                moving_image,
                parameter_object=parameter_object,
                log_to_console=False
            )
            
            # Transform the mask with affine
            affine_transformed_mask = itk.transformix_filter(
                moving_mask,
                affine_transform_params,
                log_to_console=False
            )
            
            # Save affine-transformed mask
            affine_output_path = f"{output_dir}/{basename}-affine-mask.nii.gz"
            itk.imwrite(affine_transformed_mask, affine_output_path)
            affine_success = True
        except Exception as affine_e:
            print(f"Affine registration failed: {str(affine_e)}")
        
        return (rigid_success, affine_success)
        
    except Exception as e:
        print(f"Registration failed for {fixed_img_path}: {str(e)}")
        return (False, False)

def process_patient(patient_dir):
    """Process all sequences for a single patient."""
    sequences = get_sequence_variations(patient_dir)
    if not sequences:
        return
    
    # Find reference sequence with mask
    ref_seq_name, ref_mask_path = find_reference_sequence(patient_dir, sequences)
    if not ref_seq_name:
        print(f"No reference mask found in {patient_dir}")
        return
    
    print(f"\nProcessing {os.path.basename(patient_dir)}")
    print(f"Using {ref_seq_name} as reference sequence")
    
    # Get reference image (prefer unnumbered if exists)
    ref_images = sequences.get(ref_seq_name, [])
    if not ref_images:
        print(f"No reference images found for {ref_seq_name}")
        return
    
    # Find the best reference image (prefer unnumbered version)
    ref_img_path = None
    for (num, path) in ref_images:
        if num == 0:  # This is the unnumbered version
            ref_img_path = path
            break
    if ref_img_path is None:  # If no unnumbered version, use first numbered
        ref_img_path = ref_images[0][1]
    
    # Process all sequences
    for seq_name, variations in sequences.items():
        if seq_name == ref_seq_name:
            continue  # Skip reference sequence
            
        for (seq_num, img_path) in variations:
            # Determine output name based on whether sequence is numbered
            base_name = os.path.basename(img_path).replace(".nii.gz", "")
            if seq_num > 0:
                basename = base_name
            else:
                basename = seq_name
                
            print(f"Registering to {os.path.basename(img_path)}...")
            success = register_segmentation(
                fixed_img_path=img_path,
                moving_img_path=ref_img_path,
                moving_mask_path=ref_mask_path,
                output_dir=patient_dir,
                basename=basename,
                rigid_params=rigid_custom,
                affine_params=affine_custom
            )
            
            if success:
                print(f"Created registered mask: {basename}")

def main(output_root="../data/output"):
    """Main processing function."""
    patient_dirs = [d for d in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(d)]
    
    for patient_dir in patient_dirs:
        process_patient(patient_dir)

if __name__ == "__main__":
    main()