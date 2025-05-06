from PIL import Image
import numpy as np
import json
import os

def convert_npz_to_rlds(input_dir, output_file):
    """
    Convert NPZ files to RLDS format for OpenVLA.
    
    Args:
        input_dir: Directory containing NPZ files
        output_file: Path to save the RLDS JSON file
    """
    rlds_dataset = []
    
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".npz"):
            data = np.load(os.path.join(input_dir, file))
            sample = {
                'observation': {
                    'image_primary': np.array(Image.fromarray(data['observation/image_primary'])),
                    'proprio': data['observation/proprio'],
                    'camera_pose': data['observation/camera_pose'],
                },
                'task': {
                    'language_instruction': data['task/language_instruction']
                },
                'action': data['action'],
                'dataset_name': data['dataset_name'],
            }
            rlds_dataset.append(sample)
    
    with open(output_file, "w") as f:
        json.dump(rlds_dataset, f, indent=2)
    
    return rlds_dataset
