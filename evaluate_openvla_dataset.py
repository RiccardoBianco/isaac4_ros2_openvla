
import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import json_numpy
import yaml
import requests
import glob


# Apply patch for handling numpy arrays in JSON
json_numpy.patch()



DATASET_PATH = "/home/wanghan/Desktop/PLRItalians/isaac4_ros2_openvla/rlds_dataset_builder/sim_data_custom_v0/data/val"  # <-- Update this with the actual path
OPENVLA_UNNORM_KEY = "sim_data_custom_v0"

# Define the URL of the server endpoint
def set_server_url():
    # if user is "wanghan"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"

    if user == "wanghan":
        server_url = "http://0.0.0.0:8000/act"
    else:
        print("Current working directory:", os.getcwd())

        config_path = os.path.abspath("./isaac_ws/src/config.yaml")  # assuming you are in /root/isaac_ws folder
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ip_address = config["ip_address"]
        port = config["port"]
        server_url = f'http://{ip_address}:{port}/act'
        print(f"Server URL: {server_url}")

    return server_url


SERVER_URL = set_server_url()


def send_request(payload):

    # Send POST request to the server
    response = requests.post(SERVER_URL, json=payload)

    # Check the response
    if response.status_code == 200:
        # print("Response from server:", response.json())
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None


def main():

    episode_paths = sorted(glob.glob(os.path.join(DATASET_PATH, "episode_*.npy")))
    
    if not episode_paths:
        print(f"No episodes found in {DATASET_PATH}")
        return
    
    total_steps = 0
    total_correct = 0

    for ep_idx, episode_path in enumerate(episode_paths):
        print(f"\n\nProcessing {episode_path}")
        episode_data = np.load(episode_path, allow_pickle=True)

        ep_total = 0
        ep_correct = 0

        for step_idx, step_data in enumerate(episode_data):
            image_array = step_data["image"]  # Assuming RGB image (H, W, 3)
            instruction = step_data["language_instruction"]
            gt_action = step_data["action"]  # Ground-truth action (7,)

            payload = {
                "image": image_array,  # Send as list via JSON
                "instruction": instruction,
                "unnorm_key": OPENVLA_UNNORM_KEY
            }
            #Send request to the server
            # print("Sending request to OpenVLA...")
            res = send_request(payload)
            if res is None:
                print("Error in sending request to OpenVLA.")
                continue

            # Simple accuracy: all elements within a threshold
            threshold = 0.001 
            is_correct = np.allclose(gt_action[:6], res[:6], atol=threshold)
            gripper_correct = np.isclose(gt_action[6], res[6], atol=0.05)

            if is_correct and gripper_correct:
                print(f"Episode {ep_idx}, Step {step_idx}: CORRECT")
                # print(f"    GT: {gt_action}")
                # print(f"    Pred: {res}")
                ep_correct += 1
            else:
                if not is_correct:
                    print(f"Episode {ep_idx}, Step {step_idx}: INCORRECT POSE")
                    print(f"GT: {gt_action}")
                    print(f"Pred: {res}")
                if not gripper_correct:
                    print(f"Episode {ep_idx}, Step {step_idx}: INCORRECT GRIPPER")
                    print(f"GT: {gt_action[6]}")
                    print(f"Pred: {res[6]}")
            ep_total += 1
        
        ep_accuracy = ep_correct / ep_total if ep_total > 0 else 0.0
        print(f"\n\nEpisode {ep_idx} accuracy: {ep_accuracy:.2%}\n\n")

        total_steps += ep_total
        total_correct += ep_correct
    
    avg_accuracy = total_correct / total_steps if total_steps > 0 else 0.0
    print(f"\n\nAverage accuracy across all episodes: {avg_accuracy:.2%}\n\n")




if __name__ == "__main__":
    main()

