import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import requests
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from PIL import Image as PILImage
import json_numpy

json_numpy.patch()

class ImageToPoseClient(Node):
    def __init__(self):
        super().__init__('image_to_pose_client')

        # CV bridge
        self.bridge = CvBridge()

        # Subscriber immagini
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.image_callback,
            10
        )

        # Subscriber posizione attuale EEF
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/franka/end_effector_pose',
            self.ee_pose_callback,
            10
        )

        # Publisher per target EEF
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/franka/end_effector_pose_cmd',
            10
        )

        self.processing = False
        self.current_ee_pose = None  # Aggiornata da callback

        self.server_url = 'http://129.132.39.85:8000/act'

    def ee_pose_callback(self, msg):
        self.current_ee_pose = msg

    def image_callback(self, msg):
        if self.processing:# or self.current_ee_pose is None: # TODO remove
            return
        self.get_logger().info('Immagine ricevuta, inizio elaborazione...')
        self.processing = True

        try:
            # Converti immagine
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            image_array = np.array(pil_image)

            # Prepara payload per OpenVLA
            payload = {
                'image': image_array,  # JSON serializzabile
                'instruction': 'Pick up the object',
                'unnorm_key': 'ucsd_kitchen_dataset_converted_externally_to_rlds'
            }

            # Invia richiesta
            response = requests.post(self.server_url, json=payload)
            if response.status_code != 200:
                self.get_logger().error(f'Errore nella risposta dal server: {response.status_code}')
                self.processing = False
                return

            # Parse della risposta
            res = response.json()
            self.get_logger().info(f'✅ Risposta dal server: {res}')

            # TODO add a print of the response to see the content
            dx, dy, dz = res[0], res[1], res[2]
            r, p, y = res[3], res[4], res[5]
            gripper = res[6]  # se vuoi usarlo dopo

            if self.current_ee_pose is None:
                self.get_logger().error('❌ La posizione attuale dell EEF non è disponibile.')
                self.processing = False
                return
            
            pose_cmd = self.get_pose_cmd(dx, dy, dz, r, p, y)
            
            self.pose_pub.publish(pose_cmd)
            self.get_logger().info(f'✅ Nuova pose inviata al robot:\n{pose_cmd.pose}')
            

        except Exception as e:
            self.get_logger().error(f'❌ Errore durante l elaborazione: {e}')

        time.sleep(1)
        self.processing = False


    def get_pose_cmd(self, dx, dy, dz, r, p, y):
        # Costruisci T_delta
        T_delta = np.eye(4)
        T_delta[:3, :3] = R.from_euler('xyz', [r, p, y]).as_matrix()
        T_delta[:3, 3] = [dx, dy, dz]

        # Costruisci T_current dalla posa attuale
        cp = self.current_ee_pose.pose.position
        cq = self.current_ee_pose.pose.orientation
        pos = np.array([cp.x, cp.y, cp.z])
        quat = np.array([cq.x, cq.y, cq.z, cq.w])
        T_current = np.eye(4)
        T_current[:3, :3] = R.from_quat(quat).as_matrix()
        T_current[:3, 3] = pos

        # Calcola T_target = T_current @ T_delta
        T_target = T_current @ T_delta
        target_pos = T_target[:3, 3]
        target_quat = R.from_matrix(T_target[:3, :3]).as_quat()

        # Pubblica nuova posa su /franka/end_effector_pose_cmd
        pose_cmd = PoseStamped()
        pose_cmd.header.stamp = self.get_clock().now().to_msg()
        pose_cmd.header.frame_id = "base_link"  # o il tuo frame di riferimento
        pose_cmd.pose.position.x = target_pos[0]
        pose_cmd.pose.position.y = target_pos[1]
        pose_cmd.pose.position.z = target_pos[2]
        pose_cmd.pose.orientation.x = target_quat[0]
        pose_cmd.pose.orientation.y = target_quat[1]
        pose_cmd.pose.orientation.z = target_quat[2]
        pose_cmd.pose.orientation.w = target_quat[3]



def main(args=None):
    rclpy.init(args=args)
    node = ImageToPoseClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
