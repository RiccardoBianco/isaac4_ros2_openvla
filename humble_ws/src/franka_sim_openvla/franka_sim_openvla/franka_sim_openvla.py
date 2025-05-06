from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import RobotState
#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, Pose
from cv_bridge import CvBridge
import cv2
import requests
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage
import json_numpy
import yaml
import time
import threading

# Import moveit_commander for motion planning
import moveit_commander

# Patch json_numpy for serialization
json_numpy.patch()

# Topics and parameters
joint_cmd_topic = '/joint_command'
camera_topic = '/rgb'

# Cone trajectory parameters
res_list = []
radius = 0.05         # radius of the circle at the cone base
height = 0.05         # height of the cone (defines the vertical displacement)
num_points = 10       # number of points around the circle
gripper = 1.0         # constant gripper value

for i in range(num_points):
    theta = 2 * np.pi * i / num_points  # angle (in radians)
    dx = radius * np.cos(theta)
    dy = radius * np.sin(theta)
    dz = -height / 2 + (height * i / (num_points - 1))  # from bottom to top of the cone
    droll = 0.0
    dpitch = 0.0
    dyaw = theta  # rotate the end-effector around its z-axis a little
    res_list.append([dx, dy, dz, dpitch, droll, dyaw, gripper])

print(f'res_list: {res_list}')

class FrankaSimOpenVLA(Node):
    def __init__(self):
        super().__init__('franka_sim_openvla')
        
        # Initialize MoveIt commander (in ROS2 this works if moveit_commander is available)
        moveit_commander.roscpp_initialize(sys.argv)
        # Create a MoveGroupCommander for the "panda_arm" group
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        
        self.server_url = self.set_server_url("config.yaml")
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(Image, camera_topic, self.image_callback, 1)
        # You can still publish to joint command if needed, but now our motion planning will drive the robot.
        self.joint_pub = self.create_publisher(JointState, joint_cmd_topic, 1)
        
        self.max_gripper = 0.04
        self.cnt = 0
        self.processing = False

        # Store the current joint state and end-effector pose.
        self.current_joint_state = None  # will be updated below
        self.current_ee_pose = None
        self.new_ee_pose = None

        # Remove direct FK/IK service usage because we now rely on MoveIt planning.
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        while not self.fk_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('Waiting for /compute_fk service...')
        while not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('Waiting for /compute_ik service...')
        self.get_logger().info('Franka Sim OpenVLA Bridge started.')
        
        # Neutral pose (joint angles for the panda_arm) and gripper command
        self.neutral_pose = [0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, np.pi/4]
        self.neutral_gripper = [1.0, 1.0]
        
        # Move the robot to its neutral pose using the planner
        self.move_to_neutral_pose()

    def move_to_neutral_pose(self):
        self.get_logger().info('Moving to neutral pose using MoveIt planner...')
        # Set the joint target for the arm
        self.move_group.set_joint_value_target(self.neutral_pose)
        # Optionally, you can set planning time and number of attempts if needed:
        self.move_group.set_planning_time(5)
        self.move_group.set_num_planning_attempts(10)
        
        plan = self.move_group.plan()
        if plan and plan.joint_trajectory.points:
            self.get_logger().info('Neutral plan found, executing trajectory...')
            self.move_group.execute(plan, wait=True)
            # Update current_joint_state with the new state from the planner.
            arm_state = self.move_group.get_current_joint_values()  # returns list of 7 joint angles
            self.current_joint_state = arm_state + self.neutral_gripper
        else:
            self.get_logger().warn('Failed to plan to neutral pose.')
        input('Press Enter to continue...')

    def set_server_url(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        ip_address = config["ip_address"]
        port = config["port"]
        return f'http://{ip_address}:{port}/act'
    
    # We keep the FK service for debugging (to check end-effector pose)
    def compute_fk(self):
        done_event = threading.Event()
        result_holder = {}

        def _callback(future):
            if future.result() and future.result().error_code.val == 1:
                self.get_logger().info("FK succeeded")
                result_holder['pose'] = future.result().pose_stamped[0]
            else:
                self.get_logger().warn("FK failed")
                result_holder['pose'] = None
            done_event.set()

        request = GetPositionFK.Request()
        request.header.frame_id = "panda_link0"
        request.fk_link_names = ["panda_link8"]

        joint_msg = JointState()
        joint_msg.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]
        # Use the current_joint_state from the planner (first 7 values are for the arm)
        joint_msg.position = self.current_joint_state[:7] if self.current_joint_state else []
        joint_msg.header.stamp = self.get_clock().now().to_msg()

        robot_state = RobotState()
        robot_state.joint_state = joint_msg
        request.robot_state = robot_state

        future = self.fk_client.call_async(request)
        future.add_done_callback(_callback)
        done_event.wait(timeout=3.0)
        return result_holder.get('pose', None)

    def image_callback(self, msg):
        if self.processing or self.current_joint_state is None:
            return
        self.processing = True
        threading.Thread(target=self.process_image, args=(msg,)).start()

    def process_image(self, msg):
        # Update current end-effector pose using FK (for debugging)
        self.current_ee_pose = self.compute_fk()
        if self.current_ee_pose is None:
            self.get_logger().warn('FK computation failed, skipping image processing')
            self.processing = False
            return

        try:
            # Convert the ROS Image to a CV image and then to PIL format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            image_array = np.array(pil_image)

            # Prepare payload for OpenVLA
            payload = {
                'image': image_array,  # JSON-serializable array
                'instruction': 'Pick up the object',
                'unnorm_key': 'ucsd_kitchen_dataset_converted_externally_to_rlds'
            }
            response = requests.post(self.server_url, json=payload)
            if response.status_code != 200:
                self.get_logger().error(f'Error from server: {response.status_code}')
                self.processing = False
                return

            res = response.json()
            print(f'✅ Server response: {res}')
            # For testing, use the predetermined delta from res_list cyclically
            res = res_list[self.cnt]
            self.cnt = (self.cnt + 1) % len(res_list)
            print(f'cnt: {self.cnt}, res: {res}')

            # Compute the new target end-effector pose by applying a delta change
            self.new_ee_pose = self.apply_delta(self.current_ee_pose, res[:6])
            if self.new_ee_pose is None:
                self.get_logger().warn('Failed to apply pose delta.')
                self.processing = False
                return

            # Plan and execute a trajectory to the new pose using MoveIt
            self.move_group.set_start_state_to_current_state()
            # Set the target pose; use new_ee_pose.pose if new_ee_pose is a PoseStamped
            self.move_group.set_pose_target(self.new_ee_pose.pose)
            self.move_group.set_planning_time(5)
            self.move_group.set_num_planning_attempts(10)
            plan = self.move_group.plan()

            if plan and plan.joint_trajectory.points:
                self.get_logger().info("Plan found, executing trajectory...")
                self.move_group.execute(plan, wait=True)
                # Update current_joint_state after execution.
                arm_state = self.move_group.get_current_joint_values()  # arm joints only
                # Append gripper values (you can further adjust based on your application)
                self.current_joint_state = arm_state + [res[6] * self.max_gripper] * 2
            else:
                self.get_logger().warn("Planning failed!")
                self.processing = False
                return

        except Exception as e:
            self.get_logger().error(f'Error during processing: {e}')
        
        # For debugging, compute FK after execution to check the new pose
        updated_pose = self.compute_fk()
        if updated_pose:
            print(f'Post-planning FK pose: {updated_pose.pose}\n\n')
        time.sleep(4.0)
        self.processing = False

    def apply_delta(self, pose_stamped: PoseStamped, delta):
        pose = pose_stamped.pose
        self.get_logger().info(f'Pose before delta: {pose}')
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w])
        R = Rotation.from_quat(orientation).as_matrix()
        # Compute delta in the world frame
        world_delta = R @ np.array(delta[:3])
        new_position = position + world_delta

        # Apply rotation delta (Euler angles)
        delta_rot = Rotation.from_euler('xyz', delta[3:6])
        new_orientation = (Rotation.from_matrix(R @ delta_rot.as_matrix())).as_quat()

        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.header.frame_id = "panda_link0"
        new_pose.pose.position.x, new_pose.pose.position.y, new_pose.pose.position.z = new_position
        new_pose.pose.orientation.x, new_pose.pose.orientation.y, new_pose.pose.orientation.z, new_pose.pose.orientation.w = new_orientation

        self.get_logger().info(f'Pose after delta: {new_pose.pose}')
        return new_pose

def main(args=None):
    rclpy.init(args=args)
    node = FrankaSimOpenVLA()
    
    # Spin the node using a MultiThreadedExecutor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    # Shutdown MoveIt commander
    moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
