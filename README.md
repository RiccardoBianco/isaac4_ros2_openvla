# Finetuning OpenVLA using IsaacSim

## 1. Clone the Main Repository

Clone the project and navigate into the working directory:

    git clone git@github.com:RiccardoBianco/isaacsim_openvla.git
    cd isaacsim_openvla

---

## 2. Create Conda Environment for RLDS
Make sure Conda is installed from: https://www.anaconda.com/docs/getting-started/miniconda/main
Create the `rlds_env` environment using the `environment_ubuntu.yml` (or `environment_macos.yml` if you're on macOS):

    cd rlds_dataset_builder
    conda env create -f environment_ubuntu.yml
    cd ..

---

## 3. Create Conda Environment for OpenVLA
The openvla conda environment must be installed on the device or cluster where you intend to run the OpenVLA server or perform fine-tuning. It is not required on the device running the Isaac Sim simulation, which only communicates with the OpenVLA server remotely.

    cd .. # move outside isaacsim_openvla

    # Create and activate the environment
    conda create -n openvla python=3.10 -y
    conda activate openvla

    # Install PyTorch (check the best version for your system: https://pytorch.org/get-started/locally/)
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

    # Clone and install the OpenVLA repository
    git clone https://github.com/openvla/openvla.git
    cd openvla
    pip install -e .

    # Install Flash Attention 2 (for training)
    pip install packaging ninja
    ninja --version; echo $?  # Should return exit code "0"
    pip install "flash-attn==2.5.5" --no-build-isolation

    # (Optional) Remove the openvla repo after installation
    cd ..
    rm -rf openvla
    cd isaacsim_openvla

---

## 4. Use Isaac Sim inside the Docker container

Make sure Docker is installed from: https://www.docker.com/get-started/

Start the container running the following commands inside `isaacsim_openvla` folder:

    xhost +local:root                     # Enable GUI for Isaac Sim
    docker compose up -d isaac_sim       # Build and launch the container
    docker exec -it isaac-sim bash       # Open a terminal inside the container
    isaac_ws/setup_env.sh           # Setup the environment correctly

---

## 5. Running Simulations Inside the Container

The following `.sh` scripts are available in the container to start simulations:

### Data collection
Start a data collection in simulation:

    ./isaac_ws/data_collection.sh -x      # x = -s (single cube), -m (multi cube), -r (real objects)

### Evaluation
Start the client for OpenVLA evaluation in isaacsim, sending requests to openvla server and updating the simulation:

    ./isaac_ws/client.sh -x              # x = -s (single cube), -m (multi cube), -r (real objects)

Evaluate a complex task using OpenVLA and T5 module to decompose a complex task into subtasks:

    ./isaac_ws/complex.sh 

**Important:** Parameters that define the scenario (e.g., object types, camera views, etc.) must be set manually at the beginning of the corresponding Python files:
- `data_collection_*.py`
- `evaluate_openvla_*.py`

Each time a data collection is started, a `.json` file with all parameters is saved in the `output/` directory with a timestamp.

---

## 6. Split Training and Validation Data

After collecting simulation data, split it into training and validation sets:

    python3 split_train_val.py 

You can specify the percentage of training and validation dataset using the global variables at the beginning of the python file

---

## 7. Build RLDS Dataset Outside the Docker Container

From a separate terminal on the host system:

    conda activate rlds_env
    cd rlds_dataset_builder/sim_data_custom_v0
    tfds build --overwrite
    conda deactivate

> If any errors occur or if you want to recreate the dataset from scratch, refer to the README inside `rlds_dataset_builder`.

---

## 8. Start Finetuning with OpenVLA

Once the RLDS dataset is created, follow OpenVLA's official instructions for LoRA finetuning.  
In particular, edit the file `openvla/start_finetuning.sh` to include:
- The dataset repository path (`data_root_dir`)
- The output directory for the fine-tuned model (`run_root_dir`)

Then run the script to begin training.

---

## 9. Load and Run Fine-tuned Model

Once training is complete:

1. In the file `openvla/vla-scripts/deploy.py`, update the global parameters to point to the fine-tuned model path (saved in `run_root_dir`).

2. Launch the OpenVLA inference server:

       python3 openvla/vla-scripts/deploy.py

---

## 10. Run Evaluation Simulation with Isaac Sim

Back inside the Docker container, use the following command to launch evaluation and connect to the OpenVLA server:

    ./client.sh -x      # where x = -s, -m, or -r

This script:
- Starts the Isaac Sim simulation
- Captures images from the virtual camera
- Sends images to the OpenVLA inference server
- Receives delta instructions for the robot end-effector
- Applies them in real-time to continue task execution

---

## Summary

- All development and data processing steps releted to OpenVLA finetuning and dataset creation are kept outside Docker.
- Simulation and Isaac Sim execution are isolated inside the Docker container to avoid errors due to different IsaacSim and IsaacLab versions.
- Parameters and file paths must remain consistent between data collection and evaluation to ensure proper behavior.

> For additional details or if any error occurs, please refer to each sub-repository’s README file.


## References

- [OpenVLA](https://github.com/openvla/openvla) — VLA robotic manipulation
- [Isaac Sim](https://developer.nvidia.com/isaac-sim) — NVIDIA simulator for robotics
- [RLDS Dataset Format](https://github.com/google-research/rlds) — Standard format for robot learning datasets
