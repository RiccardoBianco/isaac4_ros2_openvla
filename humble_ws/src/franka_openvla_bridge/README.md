# Franka OpenVLA Bridge

This ROS2 package provides a bridge between OpenVLA and Franka robot control. It subscribes to OpenVLA pose messages and publishes joint states for the Franka robot.


## Dependencies

- ROS2 Humble
- Python 3.8+
- NumPy
- SciPy
- MoveIt
- TF Transformations



### Docker Installation

If you're using the provided Docker setup:

1. Connect to the ROS2 Humble container:
```bash
docker exec -it ros2_humble bash
```

2. Install required ROS2 packages:
```bash
apt-get update
apt-get install -y ros-humble-moveit ros-humble-moveit-ros-planning-interface ros-humble-tf-transformations
```

3. Install Python dependencies:
```bash
pip install numpy scipy
```

4. Build the package:
```bash
cd /root/humble_ws
colcon build --packages-select franka_openvla_bridge
source install/setup.bash
```

## Usage

1. Source your workspace:
```bash
source install/setup.bash
```

2. Launch the bridge:
```bash
ros2 launch franka_openvla_bridge franka_openvla_bridge.launch.py
```

## Topics

### Subscribed Topics
- `/cmd_pose` (geometry_msgs/PoseStamped): OpenVLA pose data
- `/openvla/delta_ee` (std_msgs/Float64MultiArray): Delta commands [dx, dy, dz, droll, dpitch, dyaw, dgrip]

### Published Topics
- `/gripper/joint_commands` (sensor_msgs/JointState): Franka robot joint states
- `/franka_openvla_bridge/target_marker` (visualization_msgs/Marker): Visualization of target pose

## Delta Commands

The package supports delta commands in the format:
```
[dx, dy, dz, droll, dpitch, dyaw, dgrip]
```
Where:
- dx, dy, dz: Position deltas in meters
- droll, dpitch, dyaw: Orientation deltas in radians
- dgrip: Gripper opening delta in meters

## MoveIt Integration

This package uses MoveIt for robust inverse kinematics solving. It:
1. Sets the target pose in MoveIt
2. Plans a trajectory to the target
3. Extracts the joint values from the plan
4. Checks joint limits for safety
5. Publishes the joint values to the robot

## TODO
- Add error handling and safety checks
- Add configuration parameters for robot parameters 
