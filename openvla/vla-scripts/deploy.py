"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse

# === Globals ===
SERVER_SIDE_FOLDER = "server_side_images"
DEFAULT_IMAGE_NAME = "received_image"
BASE_MODEL = "openvla/openvla-7b"
FINETUNED_MODEL = "/home/wanghan/Desktop/PLRItalians/isaac4_ros2_openvla/models/openvla-7b+sim_data_custom_v0+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"
OPENVLA_MODEL = FINETUNED_MODEL  # ^ Change this to the desired model path


# === Command Line Arguments ===
parser = argparse.ArgumentParser(description="Get command line arguments")
# Parse the arguments
parser.add_argument("--save", action="store_true", help="Save images to server-side folder")
args = parser.parse_args()

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    # TODO: check because we trained with a different prompt, here it's being concatenated with other
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            #attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            if args.save:
                # Save image in .jpg format
                save_image_with_progressive_filename(image, reset_folder=False)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        print("The server is running.... Waiting for post requests")
        uvicorn.run(self.app, host=host, port=port)

# === Image Saving ===
def save_image_with_progressive_filename(image, image_directory=SERVER_SIDE_FOLDER, base_filename=DEFAULT_IMAGE_NAME, extension=".png"):
    """
    Saves an image with a progressively higher filename.
    Optionally resets the folder (deletes all previous images).

    Parameters:
    - image: The image data (numpy array) to save.
    - image_directory: Directory to save the images (default is "server_side_images").
    - base_filename: Base name for the image files (default is "received_image").
    - extension: File extension (default is ".png").
    """

    # Create the image directory if it doesn't exist
    os.makedirs(image_directory, exist_ok=True)

    # Find the highest number in existing filenames
    existing_files = [f for f in os.listdir(image_directory) if f.startswith(base_filename) and f.endswith(extension)]

    # Find the highest index by extracting the numbers from the filenames
    max_index = 0
    for filename in existing_files:
        try:
            # Extract the number from filenames like 'received_image_1.jpg', 'received_image_2.jpg', etc.
            index = int(filename[len(base_filename)+1:-len(extension)])
            if index > max_index:
                max_index = index
        except ValueError:
            # In case there's a filename that doesn't have the expected format, we skip it
            continue

    # Create the new filename by incrementing the highest index
    new_index = max_index + 1
    image_filename = f"{base_filename}_{new_index}{extension}"

    # Save the image with the new filename
    image_pil = Image.fromarray(image).convert("RGB")
    image_path = os.path.join(image_directory, image_filename)
    image_pil.save(image_path)

    print(f"Image saved as: {image_filename}")
    return image_filename

# === Image Folder Clearing ===
def clear_img_folder():
    if os.path.exists(SERVER_SIDE_FOLDER):
        for file in os.listdir(SERVER_SIDE_FOLDER):
            if file.startswith(DEFAULT_IMAGE_NAME) and file.endswith(".png"):
                os.remove(os.path.join(SERVER_SIDE_FOLDER, file))

@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = OPENVLA_MODEL                      # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    if args.save:
        clear_img_folder()
    server = OpenVLAServer(cfg.openvla_path)

    if cfg.openvla_path == FINETUNED_MODEL:
        print("Using finetuned model")
    elif cfg.openvla_path == BASE_MODEL:
        print("Using base model")

    server.run(cfg.host, port=cfg.port)

if __name__ == "__main__":
    deploy()
