import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, Pose
from cv_bridge import CvBridge
import cv2
import requests
import numpy as np
import time
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage
import json_numpy
import yaml
# from ikpy.utils import plot

json_numpy.patch()

simulation = True # TODO manage gripper correctly


ee_pose_topic = '/cartesian_impedance_example_controller/measured_pose'
ee_pose_cmd_topic = '/cmd_pose'
gripper_cmd_topic = '/gripper/joint_commands'
camera_topic = '/camera/color/image_raw'


class ImageToPoseClient(Node):
    def __init__(self):
        super().__init__('image_to_pose_client')

        # CV bridge
        self.bridge = CvBridge()
        self.cnt = 0
        # Subscriber immagini
        self.subscription = self.create_subscription(Image, camera_topic, self.image_callback, 1)
        # Subscriber posizione attuale EEF
        self.pose_sub = self.create_subscription(PoseStamped, ee_pose_cmd_topic, self.ee_pose_callback, 1)


        # Publisher per target EEF
        self.pose_pub = self.create_publisher(PoseStamped,ee_pose_cmd_topic, 1)
        # Publisher per gripper
        self.gripper_pub = self.create_publisher(JointState, gripper_cmd_topic, 1)
    

        if simulation:
            self.joint_names = [
                "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                "panda_joint5", "panda_joint6", "panda_joint7",
                "panda_finger_joint1", "panda_finger_joint2"
            ]
        self.max_gripper = 0.04  # Max apertura pinze


            # self.robot_chain.links = self.robot_chain.links[:8]

        self.processing = False
        self.current_ee_pose = None  # Aggiornata da callback

        self.server_url = self.set_server_url(".config.yaml") # TODO fix


    def set_server_url(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ip_address = config["ip_address"]
        port = config["port"]
        server_url = f'http://{ip_address}:{port}/act'
        return server_url

    def ee_pose_callback(self, msg): # TODO check if this is correct
        # Aggiorna la posa corrente dell EEF
        if self.processing:
            return
        self.current_ee_pose = msg


    def image_callback(self, msg):
        if self.processing or self.current_ee_pose is None: # TODO remove
            return
        
        print('Immagine ricevuta, inizio elaborazione...')
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
            print(f'✅ Risposta dal server: {res}')
   
            
            pose_cmd = self.apply_delta(self.current_ee_pose, res[:6])
            self.pose_pub.publish(pose_cmd)

            gripper_cmd = self.get_gripper_cmd(res[6])
            self.gripper_pub.publish(gripper_cmd)
                
        
        except Exception as e:
            self.get_logger().error(f'❌ Errore durante l elaborazione: {e}')

        time.sleep(2)
        self.processing = False

    def apply_delta(self, pose_stamped: PoseStamped, delta):
        pose = pose_stamped.pose
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w])
        R = Rotation.from_quat(orientation).as_matrix()
        world_delta = R @ np.array(delta[:3])
        new_position = position + world_delta

        delta_rot = Rotation.from_euler('xyz', delta[3:6])
        new_orientation = (Rotation.from_matrix(R @ delta_rot.as_matrix())).as_quat()

        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.header.frame_id = "panda_link0"
        new_pose.pose.position.x, new_pose.pose.position.y, new_pose.pose.position.z = new_position
        new_pose.pose.orientation.x, new_pose.pose.orientation.y, new_pose.pose.orientation.z, new_pose.pose.orientation.w = new_orientation

        # TODO check gripper
        #self.current_gripper_value = np.clip(self.current_gripper_value + delta[6], 0.0, 1.0)
        return new_pose

    def get_gripper_cmd(self, gripper_value):
        gripper_position = self.max_gripper * gripper_value

        gripper_cmd = JointState()
        gripper_cmd.header.stamp = self.get_clock().now().to_msg()
        gripper_cmd.position =  [gripper_position] 
        return gripper_cmd
    


def main(args=None):
    rclpy.init(args=args)
    node = ImageToPoseClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
