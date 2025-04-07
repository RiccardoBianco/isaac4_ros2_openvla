
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
import numpy as np
from scipy.spatial.transform import Rotation
import math
import tf_transformations
from moveit.planning import MoveGroupInterface
from moveit.core.robot_state import RobotState
from moveit.core.robot_model import RobotModel
from moveit.core.planning_scene import PlanningScene
from moveit.core.planning_scene_interface import PlanningSceneInterface

class FrankaOpenVLABridge(Node):
    def __init__(self):
        super().__init__('franka_openvla_bridge')
        
        # Create subscribers for OpenVLA 
        self.openvla_sub = self.create_subscription(
            PoseStamped,
            '/cmd_pose',
            self.openvla_callback,
            10)
            
        # Create publisher for Franka joint states
        self.joint_pub = self.create_publisher(
            JointState,
            '/gripper/joint_commands', 
            10)
            
        # Create publisher for visualization marker
        self.marker_pub = self.create_publisher(
            Marker,
            '/franka_openvla_bridge/target_marker',
            10)
            
        # Initialize MoveIt components
        self.robot_model = RobotModel()
        self.robot_state = RobotState(self.robot_model)
        self.planning_scene = PlanningScene(self.robot_model)
        self.planning_scene_interface = PlanningSceneInterface()
        
        # Initialize MoveGroup for the panda arm
        self.move_group = MoveGroupInterface(
            node=self,
            wait_for_servers=10.0,
            name="panda_arm"
        )
        
        # Franka robot parameters (DH parameters)
        self.dh_params = {
            'd': [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107],  # Link offsets
            'a': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],         # Link lengths
            'alpha': [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, np.pi/2, np.pi/2]  # Link twists
        }
        
        # Joint limits for safety
        self.joint_limits = {
            'lower': [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, -0.0168, -2.8065],
            'upper': [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 3.7525, 2.8065]
        }
        
        # Gripper parameters
        self.gripper_open = 1.0  # Maximum gripper opening
        self.gripper_closed = 0.0  # Minimum gripper opening
        self.current_gripper_value = self.gripper_open
        
        self.get_logger().info('Franka OpenVLA Bridge has been started')

    def publish_marker(self, pose):
        """Publish a marker to visualize the target pose"""
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "franka_openvla_bridge"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.1
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def compute_ik_with_moveit(self, target_pose):
        """
        Compute inverse kinematics using MoveIt
        """
        try:
            # Set the target pose
            self.move_group.set_pose_target(target_pose)
            
            # Plan to the target pose
            plan_result = self.move_group.plan()
            
            if not plan_result.success:
                self.get_logger().warn('IK planning failed')
                return None
                
            # Get the joint values from the plan
            joint_values = plan_result.joint_trajectory.points[-1].positions
            
            # Check joint limits
            for i, angle in enumerate(joint_values):
                if angle < self.joint_limits['lower'][i] or angle > self.joint_limits['upper'][i]:
                    self.get_logger().warn(f'Joint {i+1} angle {angle} outside limits')
                    return None
            
            return joint_values
            
        except Exception as e:
            self.get_logger().error(f'MoveIt IK computation failed: {str(e)}')
            return None

    def apply_delta(self, pose, delta):
        """
        Apply a delta transformation to a pose
        delta: [dx, dy, dz, droll, dpitch, dyaw, dgrip] in end-effector frame
        """
        # Extract position and orientation
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, 
                              pose.orientation.z, pose.orientation.w])
        
        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat(orientation)
        R = rot.as_matrix()
        
        # Extract euler angles
        roll, pitch, yaw = rot.as_euler('xyz')
        
        # Transform position delta from end-effector frame to world frame
        # The delta is in the end-effector's local frame, so we need to rotate it
        position_delta = np.array(delta[:3])
        world_position_delta = R @ position_delta
        
        # Apply position delta in world frame
        new_position = position + world_position_delta
        
        # For orientation, we need to be careful with the order of rotations
        # The delta angles are in the end-effector's local frame
        delta_rot = Rotation.from_euler('xyz', [delta[3], delta[4], delta[5]])
        delta_R = delta_rot.as_matrix()
        
        # Compose the rotations: R_new = R_current * R_delta
        new_R = R @ delta_R
        
        # Convert back to quaternion
        new_rot = Rotation.from_matrix(new_R)
        new_orientation = new_rot.as_quat()
        
        # Create new pose
        new_pose = Pose()
        new_pose.position.x, new_pose.position.y, new_pose.position.z = new_position
        new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z, new_pose.orientation.w = new_orientation
        
        # Update gripper value
        self.current_gripper_value = np.clip(
            self.current_gripper_value + delta[6], 
            self.gripper_closed, 
            self.gripper_open
        )
        
        return new_pose

    def openvla_callback(self, msg):
        # Check if this is a delta command or a full pose
        if hasattr(msg, 'data') and isinstance(msg.data, list) and len(msg.data) == 7:
            # This is a delta command
            delta = msg.data
            self.get_logger().info(f'Received delta command: {delta}')
            
            # Get current pose
            current_pose = self.move_group.get_current_pose().pose
            
            # Apply delta to get target pose
            target_pose = self.apply_delta(current_pose, delta)
        else:
            # This is a full pose command
            target_pose = msg.pose
            self.get_logger().info(f'Received full pose command')
        
        # Publish marker for visualization
        self.publish_marker(target_pose)
        
        # Compute inverse kinematics
        joint_angles = self.compute_ik_with_moveit(target_pose)
        
        if joint_angles is None:
            self.get_logger().warn('Failed to compute IK, skipping message')
            return
        
        # Create joint state message
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set joint names (7 joints + 2 gripper joints)
        joint_msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3',
                         'panda_joint4', 'panda_joint5', 'panda_joint6',
                         'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        
        # Set joint positions (7 robot joints + 2 gripper joints)
        joint_msg.position = list(joint_angles) + [self.current_gripper_value, self.current_gripper_value]
        
        # Set zero velocities and efforts
        joint_msg.velocity = [0.0] * 9
        joint_msg.effort = [0.0] * 9
        
        # Publish joint states
        self.joint_pub.publish(joint_msg)
        self.get_logger().info('Published joint command')

def main(args=None):
    rclpy.init(args=args)
    node = FrankaOpenVLABridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
