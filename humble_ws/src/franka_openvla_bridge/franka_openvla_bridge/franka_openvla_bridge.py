import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation
import numpy as np
from moveit_msgs.srv import GetPositionIK
from moveit2 import MoveIt2
from moveit2.robots import Panda

class FrankaOpenVLABridge(Node):
    def __init__(self):
        super().__init__('franka_openvla_bridge')

        # ROS 2 interfaces
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/cmd_pose',
            self.pose_callback,
            10)

        self.delta_sub = self.create_subscription(
            Float64MultiArray,
            '/openvla/delta_ee',
            self.delta_callback,
            10)

        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_commands',  # controller topic
            10)

        self.marker_pub = self.create_publisher(
            Marker,
            '/franka_openvla_bridge/target_marker',
            10)

        # MoveIt 2 interface (requires move_group to be running)
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[
                'panda_joint1', 'panda_joint2', 'panda_joint3',
                'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
            ],
            base_link_name='panda_link0',
            end_effector_name='panda_link8',
            group_name='panda_arm',
            execute_via_moveit=True
        )

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        self.current_gripper_value = 1.0  # Open gripper by default
        self.get_logger().info('Franka OpenVLA Bridge started.')

    def get_current_pose(self):
        """Use MoveIt2 interface to get current end-effector pose"""
        pose_stamped = self.moveit2.get_current_pose()
        if pose_stamped:
            return pose_stamped.pose
        else:
            self.get_logger().warn("Failed to get current pose from MoveIt2")
            return None

    def pose_callback(self, msg: PoseStamped):
        self.get_logger().info('Received full pose command')
        self.process_target_pose(msg.pose)

    def delta_callback(self, msg: Float64MultiArray):
        if len(msg.data) != 7:
            self.get_logger().warn("Delta command must be 7 elements long")
            return

        self.get_logger().info(f'Received delta command: {msg.data}')
        current_pose = self.get_current_pose()
        if current_pose is None:
            self.get_logger().warn('Failed to get current pose')
            return

        target_pose = self.apply_delta(current_pose, msg.data)
        self.process_target_pose(target_pose)

    def process_target_pose(self, pose: Pose):
        self.publish_marker(pose)
        joint_angles = self.compute_ik(pose)

        if joint_angles is None:
            self.get_logger().warn('IK solution failed.')
            return

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
            'panda_finger_joint1', 'panda_finger_joint2'
        ]
        joint_msg.position = list(joint_angles) + [self.current_gripper_value] * 2
        joint_msg.velocity = [0.0] * 9
        joint_msg.effort = [0.0] * 9

        self.joint_pub.publish(joint_msg)
        self.get_logger().info('Published joint command')

    def publish_marker(self, pose: Pose):
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

    def apply_delta(self, pose: Pose, delta):
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w])
        R = Rotation.from_quat(orientation).as_matrix()
        world_delta = R @ np.array(delta[:3])
        new_position = position + world_delta

        delta_rot = Rotation.from_euler('xyz', delta[3:6])
        new_orientation = (Rotation.from_matrix(R @ delta_rot.as_matrix())).as_quat()

        new_pose = Pose()
        new_pose.position.x, new_pose.position.y, new_pose.position.z = new_position
        new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z, new_pose.orientation.w = new_orientation

        self.current_gripper_value = np.clip(self.current_gripper_value + delta[6], 0.0, 1.0)
        return new_pose

    def compute_ik(self, pose: Pose):
        request = GetPositionIK.Request()
        request.ik_request.group_name = "panda_arm"
        request.ik_request.pose_stamped.header.frame_id = "panda_link0"
        request.ik_request.pose_stamped.pose = pose
        request.ik_request.timeout.sec = 1
        request.ik_request.attempts = 5

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().error_code.val == 1:
            return future.result().solution.joint_state.position[:7]
        else:
            self.get_logger().warn('IK solver failed or timed out')
            return None


def main(args=None):
    rclpy.init(args=args)
    node = FrankaOpenVLABridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

