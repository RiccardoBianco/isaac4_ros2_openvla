apt-get update
sudo apt install ros-humble-cv-bridge # TODO check is necessary

apt install pip
pip install flask
pip install requests
pip install json_numpy



rosdep install --from-paths src --ignore-src --rosdistro=humble -y

apt-get install -y ros-humble-moveit ros-humble-moveit-ros-planning-interface ros-humble-tf-transformations
apt install ros-humble-joint-state-broadcaster
apt install ros-humble-position-controllers
sudo apt install ros-humble-ros2-control ros-humble-controller-manager
sudo apt install ros-humble-ros2-controllers
sudo apt install ros-humble-gripper-controllers
