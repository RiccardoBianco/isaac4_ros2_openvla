#!/bin/bash
#SBATCH -n 6 # Request 4 cores (useful for parallel processing)
#SBATCH --time=04:00:00 # Set a short run-time (30 minutes)
#SBATCH --mem-per-cpu=16000 # Request 1GB memory per CPU
#SBATCH -J finetuning_openvla # Job name
#SBATCH -o finetuning_openvla.out # Output log file
#SBATCH -e finetuning_openvla.err # Error log file
#SBATCH --mail-type=BEGIN,END,FAIL # Notify via email when job ends or fails
#SBATCH --gpus=a100_80gb:1 # Request 1 GPU (A100 80GB)
#SBATCH --gres=gpumem:80g # Request 80GB of GPU memory

module load eth_proxy
module load stack/2024-06   
module load gcc/13.2.0
module load cuda/12.4.1  
module load python/3.11.6_cuda 
module load code-server/4.89.1

# Use conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla

cd ~/isaac4_ros2_openvla/openvla

nvidia-smi

# Install the required packages 
export CUDA_HOME=$CUDATOOLKIT_HOME  # or set manually if undefined
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install flash-attn==2.5.5 --no-build-isolation

# Run the Python script
#python3 -m project_1.Q4.Q4
#accelerate launch --num_processes=2 --num_machines=1 python3 -m project_1.Q4.Q4
./start_finetuning.sh

