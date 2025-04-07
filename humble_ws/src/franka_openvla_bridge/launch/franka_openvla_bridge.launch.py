from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='franka_openvla_bridge',
            executable='franka_openvla_bridge',
            name='franka_openvla_bridge',
            output='screen'
        )
    ]) 