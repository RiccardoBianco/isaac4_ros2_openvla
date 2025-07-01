# Create a symbolic link to the Isaac Sim directory inside isaac_lab
# This is needed so that isaaclab can find the Isaac Sim installation
ln -s /isaac-sim /root/isaac_ws/isaac_lab/_isaac_sim

# Upgrade pip using Isaac Sim's Python interpreter
# This ensures that pip inside Isaac Sim's environment is up to date
/root/isaac_ws/isaac_lab/_isaac_sim/kit/python/bin/python3 -m pip install --upgrade pip

# Install isaaclab dependencies and set up the environment
# This will configure Isaac Lab to work with Isaac Sim
/root/isaac_ws/isaac_lab/isaaclab.sh --install

# Install the json_numpy Python package inside Isaac Sim's environment
# This package is required for simulation and data exchange
/root/isaac_ws/isaac_lab/_isaac_sim/python.sh -m pip install json_numpy

