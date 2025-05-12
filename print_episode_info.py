import os
import numpy as np
import argparse
from pprint import pprint
from PIL import Image

EPISODE_PATH = "rlds_dataset_builder/sim_data_custom_v0/data/train/episode_0006.npy"

def load_first_step(episode_path):
    data = np.load(episode_path, allow_pickle=True)
    first_step = data[0]

    print(f"\nLoaded episode: {episode_path}")
    print(f"Total steps: {len(data)}\n")

    print("First step:")
    print(f"Object pose: {first_step['object_pose'][0, :3]}")
    print(f"Goal pose: {first_step['goal_pose'][0, :3]}")
    
    # print(f"Camera pose: {first_step['camera_pose']}")

    
    

if __name__ == "__main__":

    episode_path = os.path.join(EPISODE_PATH)
    load_first_step(episode_path)
