import requests
import json_numpy
import numpy as np
from PIL import Image

# Apply patch for handling numpy arrays in JSON
json_numpy.patch()

# Define the URL of the server endpoint
SERVER_URL = "http://0.0.0.0:8000/act"

def send_request(image_path: str, instruction: str, unnorm_key: str):
    # Open the image
    image = Image.open(image_path)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Prepare the payload with the image (as a numpy array) and instruction
    payload = {
        "image": image_array,  # Sending as numpy array, no conversion to list
        "instruction": instruction,
        "unnorm_key": unnorm_key  # Add the unnorm_key to the payload
    }

    # Send POST request to the server
    response = requests.post(SERVER_URL, json=payload)

    # Check the response
    if response.status_code == 200:
        print("Response from server:", response.json())
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    # Define the image path and instruction
    image_path = "im_0.jpg"
    instruction = "pick up the object"

    # Select an appropriate unnorm_key from the available dataset options
    unnorm_key = "austin_buds_dataset_converted_externally_to_rlds"  # Replace with the dataset you want

    # Send request to the server
    send_request(image_path, instruction, unnorm_key)
