import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import requests
import numpy as np
import time

class ImageToJointsClient(Node):
    def __init__(self):
        super().__init__('image_to_joints_client')

        # Init CV Bridge
        self.bridge = CvBridge()

        # Image subscriber
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10)

        # Joint command publisher
        self.publisher_ = self.create_publisher(JointState, 'joint_command', 10)

        # Flag to avoid parallel requests
        self.processing = False

        # Joint names (must match robot definition)
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
            "panda_finger_joint1", "panda_finger_joint2"
        ]

        # Server endpoint
        self.server_url = 'http://localhost:5000/process_image'  # ✅ Assicurati che questo sia corretto


    def image_callback(self, msg):
        if self.processing:
            return

        self.processing = True

        try:
            # Convert image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Encode image as PNG
            _, img_encoded = cv2.imencode('.png', cv_image)

            # Prepare request payload
            files = {'image': ('image.png', img_encoded.tobytes(), 'image/png')}
            data = {'description': 'Current robot view'}

            # Send request
            response = requests.post(self.server_url, files=files, data=data)
            response.raise_for_status()

            # Parse response
            joint_positions = response.json().get('joint_position')
            if joint_positions and len(joint_positions) == len(self.joint_names):
                joint_state = JointState()
                joint_state.header.stamp = self.get_clock().now().to_msg()
                joint_state.name = self.joint_names
                joint_state.position = joint_positions
                self.publisher_.publish(joint_state)
                self.get_logger().info('✅ Nuova posizione inviata al robot')

            else:
                self.get_logger().warn('⚠️ Risposta non valida dal server o numero errato di joint')

        except Exception as e:
            self.get_logger().error(f'Errore nella richiesta HTTP: {e}')

        # Attendi prima di riattivare (simulazione robot richiede tempo per muoversi)
        time.sleep(0.5)
        self.processing = False

def main(args=None):
    rclpy.init(args=args)
    node = ImageToJointsClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
