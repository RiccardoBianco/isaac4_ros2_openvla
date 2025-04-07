import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation
import numpy as np
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState


class FrankaOpenVLABridge(Node):
    def __init__(self):
        super().__init__('franka_openvla_bridge')

        # ROS 2 interfaces
        self.pose_sub = self.create_subscription(
            PoseStamped,
            #'/cmd_pose',
            '/cartesian_impedance_example_controller/measured_pose',
            self.pose_callback,
            10)
    
        self.gripper_sub = self.create_subscription(
            JointState,
            '/gripper/joint_commands',
            self.gripper_callback,
            10)

        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_command',  # controller topic
            10)
        
        self.current_gripper_value = 1.0
        
        self.joint_angles = None

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        self.get_logger().info('Franka OpenVLA Bridge started.')

    def gripper_callback(self, msg: JointState):
        self.get_logger().info('Received gripper command')
        if len(msg.position) > 0:
            self.current_gripper_value = msg.position[0]
        else:
            self.current_gripper_value = 1.0

    def pose_callback(self, msg: PoseStamped):
        self.get_logger().info(f'Received full pose command: {msg}')
        self.process_target_pose(msg.pose)

    def process_target_pose(self, pose: Pose):
        #self.get_logger().info('Processing target pose')
        joint_angles = self.compute_ik(pose)
        #self.get_logger().info('IK computation done')

        if joint_angles is None:
            self.get_logger().warn('IK solution failed.')
            return
        #self.get_logger().info(f'IK solution found: {joint_angles}')
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
            'panda_finger_joint1', 'panda_finger_joint2'
        ]
        joint_msg.position = list(joint_angles) + [self.current_gripper_value] * 2

        self.joint_pub.publish(joint_msg) # TODO da inserire, rimosso per test
        #self.get_logger().info(f'Joint pose after IK: {joint_msg.position}')
        #self.get_logger().info('Published joint command')



    def compute_ik(self, pose: Pose):
        request = GetPositionIK.Request()
        request.ik_request.group_name = "panda_arm"
        request.ik_request.pose_stamped.header.frame_id = "panda_link0"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()

        request.ik_request.pose_stamped.pose = pose
        request.ik_request.timeout.sec = 2
        request.ik_request.avoid_collisions = True

        joint_state = JointState()
        joint_state.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        #joint_state.position = [-0.5, 0.3, 0.0, -1.5, 1.0, 1.2, 0.5]
        #joint_state.position = [2.294370409706764, -0.5127072893924473, -0.043720035001156524, -0.9431964410628216, -2.897205191313485, 3.5048637136569343, -0.17314318219185293]  # posizione iniziale neutra
        if self.joint_angles is not None:
            joint_state.position = self.joint_angles
        else:
            joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # posizione iniziale neutra

        request.ik_request.robot_state = RobotState(joint_state=joint_state, is_diff=False)

        future = self.ik_client.call_async(request)
        future.add_done_callback(self.handle_ik_response)
        return self.joint_angles

    def handle_ik_response(self, future):
        if future.result() and future.result().error_code.val == 1:
            #self.get_logger().info(f'future result: {future.result()}')
            self.joint_angles = future.result().solution.joint_state.position[:7]
        else:
            self.get_logger().info('IK solver failed or timed out')
            self.joint_angles = None
        self.get_logger().info(f'Joint new: {self.joint_angles}')


def main(args=None):
    rclpy.init(args=args)
    node = FrankaOpenVLABridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

