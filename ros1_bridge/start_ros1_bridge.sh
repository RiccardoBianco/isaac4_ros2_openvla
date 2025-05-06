#!/bin/bash

echo "[INFO] Sourcing ROS 1 Noetic..."
source /opt/ros/noetic/setup.bash

echo "[INFO] Starting roscore..."
roscore &
ROSCORE_PID=$!

# Attendi che roscore sia pronto
echo "[INFO] Waiting for roscore to be ready..."
until rosnode list > /dev/null 2>&1; do
  sleep 0.5
done

echo "[INFO] Sourcing ROS 2 Foxy..."
source /opt/ros/foxy/setup.bash

echo "[INFO] Starting ros1_bridge..."

#ros2 run ros1_bridge static_bridge # /ee_pose@geometry_msgs/msg/PoseStamped[geometry_msgs/PoseStamped
#ros2 run ros1_bridge dynamic_bridge --bridge-all-topics 
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics --topics-regex "/(ee_pose|joint_states|camera/color/image_raw)"


# Se chiudi il bridge, chiudi anche roscore
kill $ROSCORE_PID