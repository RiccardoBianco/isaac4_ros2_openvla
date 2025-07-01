import os
import numpy as np
import argparse
from pprint import pprint
from PIL import Image

EPISODE_PATH = "isaac_ws/src/output/episodes_prev/episode_0000.npy"

def load_first_step(episode_path):
    data = np.load(episode_path, allow_pickle=True)
    first_step = data[0]

    print(f"\nLoaded episode: {episode_path}")
    print(f"Total steps: {len(data)}\n")

    print("First step:")
    for key, value in first_step.items():
        if isinstance(value, np.ndarray):
            if key == 'image' or key == 'wrist_image':
                print(f"{key}: {value.shape} (Image)")
            else:
                print(f"{key}: {value}")
        elif isinstance(value, Image.Image):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    # print(f"Camera pose: {first_step['camera_pose']}")

    
    

if __name__ == "__main__":

    episode_path = os.path.join(EPISODE_PATH)
    load_first_step(episode_path)
