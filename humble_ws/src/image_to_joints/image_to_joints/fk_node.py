#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import tf_transformations


class DirectKinematics(Node):
    def __init__(self):
        super().__init__('direct_kinematics_node')

        # Subscriber to joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            1
        )

        # Publisher for pose
        self.publisher = self.create_publisher(PoseStamped, '/cartesian_impedance_example_controller/measured_pose', 1)
        self.get_logger().info("Direct kinematics node initialized.")

    def dh_params(self, joint_variables):
        M_PI = math.pi
        dh = [
            [0,       0,       0.333,   joint_variables[0]],
            [-M_PI/2, 0,       0,       joint_variables[1]],
            [M_PI/2,  0,       0.316,   joint_variables[2]],
            [M_PI/2,  0.0825,  0,       joint_variables[3]],
            [-M_PI/2, -0.0825, 0.384,   joint_variables[4]],
            [M_PI/2,  0,       0,       joint_variables[5]],
            [M_PI/2,  0.088,   0.107,   joint_variables[6]]
        ]
        return dh

    def tf_matrix(self, i, dh):
        alpha, a, d, q = dh[i]
        TF = np.array([
            [math.cos(q), -math.sin(q), 0, a],
            [math.sin(q)*math.cos(alpha), math.cos(q)*math.cos(alpha), -math.sin(alpha), -math.sin(alpha)*d],
            [math.sin(q)*math.sin(alpha), math.cos(q)*math.sin(alpha), math.cos(alpha), math.cos(alpha)*d],
            [0, 0, 0, 1]
        ])
        return TF

    def joint_state_callback(self, msg):
        if len(msg.position) < 7:
            self.get_logger().warn("JointState message does not contain 7 joints.")
            return
        #self.get_logger().info(f"Received joint states: {msg.position}")
        joint_variables = msg.position[:7]
        dh = self.dh_params(joint_variables)

        T = np.eye(4)
        for i in range(7):
            T = np.dot(T, self.tf_matrix(i, dh))

        # Extract translation and rotation
        translation = tf_transformations.translation_from_matrix(T)
        quaternion = tf_transformations.quaternion_from_matrix(T)

        # Build PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'  # Change if using different base frame
        pose_msg.pose.position.x = translation[0]
        pose_msg.pose.position.y = translation[1]
        pose_msg.pose.position.z = translation[2]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        self.publisher.publish(pose_msg)
        self.get_logger().info(f"Published pose: {pose_msg}")


def main(args=None):
    rclpy.init(args=args)
    node = DirectKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
