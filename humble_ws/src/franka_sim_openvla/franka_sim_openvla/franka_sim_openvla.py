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
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from moveit_msgs.msg import RobotState
import time
import threading

json_numpy.patch()

joint_cmd_topic = '/joint_command'
camera_topic = '/rgb'

res_list = []
radius = 0.05         # raggio del cerchio alla base del cono
height = 0.05         # altezza del cono (quanto si alza/scende il centro della traiettoria)
num_points = 10       # punti lungo il cerchio
gripper = 1.0         # valore costante per il gripper

for i in range(num_points):
    theta = 2 * np.pi * i / num_points  # angolo in rad
    dx = radius * np.cos(theta)
    dy = radius * np.sin(theta)
    dz = -height / 2 + (height * i / (num_points - 1))  # vai dal fondo alla cima del cono
    droll = 0.0
    dpitch = 0.0
    dyaw = theta  # facciamo ruotare un po’ l'EE attorno al proprio asse z

    res_list.append([dx, dy, dz, dpitch, droll, dyaw, gripper])

print(f'res_list: {res_list}')

class FrankaSimOpenVLA(Node):
    def __init__(self):
        super().__init__('franka_sim_openvla')

        self.server_url = self.set_server_url("config.yaml")
        self.bridge = CvBridge()


        self.subscription = self.create_subscription(Image, camera_topic, self.image_callback, 1)
        self.joint_pub = self.create_publisher(JointState, joint_cmd_topic, 1)


        self.max_gripper = 0.04
        self.cnt = 0
        self.processing = False

        self.current_joint_state = None
        self.current_ee_pose = None
        self.new_ee_pose = None
        self.new_joint_state = None


        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')

        while not self.fk_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('Waiting for /compute_fk service...')
        while not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().info('Waiting for /compute_ik service...')
        self.get_logger().info('Franka Sim OpenVLA Bridge started.')


        self.neutral_pose = [0.0, 0.0, 0.0, - np.pi/2, 0.0, np.pi/2, np.pi/4]  # Neutral pose
        self.neutral_gripper = [1.0, 1.0]

        self.move_to_neutral_pose()


    def move_to_neutral_pose(self):
        self.get_logger().info('Moving to neutral pose')
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
            'panda_finger_joint1', 'panda_finger_joint2'
         ] 
        
        joint_msg.position = self.neutral_pose + self.neutral_gripper
        self.joint_pub.publish(joint_msg)

        self.current_joint_state = self.neutral_pose + self.neutral_gripper

        input('Press Enter to continue...')  # Wait for user input before proceeding
        

    def set_server_url(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ip_address = config["ip_address"]
        port = config["port"]
        return f'http://{ip_address}:{port}/act'




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
        joint_msg.position = self.current_joint_state[:7]
        joint_msg.header.stamp = self.get_clock().now().to_msg()

        robot_state = RobotState()
        robot_state.joint_state = joint_msg
        request.robot_state = robot_state

        future = self.fk_client.call_async(request)
        future.add_done_callback(_callback)

        done_event.wait(timeout=3.0)  # Wait max 3 seconds
        return result_holder.get('pose', None)

    def image_callback(self, msg):
        
        # self.get_logger().info(f'Image callback called: Processing {self.processing}, Current ee pose {self.current_ee_pose}')
        if self.processing or self.current_joint_state is None: 
            return
        self.processing = True
        threading.Thread(target=self.process_image, args=(msg,)).start()
    
    def process_image(self, msg):
        # print(f"Current joint state: {self.current_joint_state}")
        self.current_ee_pose = self.compute_fk()

        if self.current_ee_pose is None:
            self.get_logger().warn('FK computation failed, skipping image processing')
            self.processing = False
            return

        # self.get_logger().info(f'Immagine ricevuta, inizio elaborazione')
        # self.get_logger().info(f'Posa corrente EEF aggiornata: {self.current_ee_pose}')


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
            '''
            # Invia richiesta
            response = requests.post(self.server_url, json=payload)
            if response.status_code != 200:
                self.get_logger().error(f'Errore nella risposta dal server: {response.status_code}')
                self.processing = False
                return

            # Parse della risposta
            res = response.json()
            print(f'✅ Risposta dal server: {res}')
            '''
            # if (self.cnt // 3) % 2 == 0: 
            #     sgn = 1
            # else:
            #     sgn = -1
            # self.cnt += 1
            # print(f'\n\ncnt: {self.cnt}, sgn: {sgn}')

            # res = [0.0, 0.0, sgn*0.05, 0 , 0 , 0, 0.5] # should move 5 cm in x direction and open gripper
            
            res = res_list[self.cnt]
            self.cnt += 1
            print(f'cnt: {self.cnt}, res: {res}')
            if self.cnt >= len(res_list):
                self.cnt = 0


            self.new_ee_pose = self.apply_delta(self.current_ee_pose, res[:6])
            if self.new_ee_pose is None:
                self.get_logger().warn('Pose delta application failed.')
                self.processing = False
                return
            
            self.new_joint_state = self.compute_ik(self.new_ee_pose.pose)
            if self.new_joint_state is None:
                self.get_logger().warn('IK solution failed.')
                self.processing = False
                return
            

            
            
            self.new_joint_state = list(self.new_joint_state[:7]) + [res[6] * self.max_gripper] * 2

            self.publish_joint_command(self.new_joint_state)

                
        
        except Exception as e:
            self.get_logger().error(f'❌ Errore durante l elaborazione: {e}')

        
        self.current_ee_pose = self.new_ee_pose
        self.current_joint_state = self.new_joint_state
        test_pose = self.compute_fk()
        print(f'Pose POST FK applicata dopo IK: {test_pose.pose}\n\n')
        time.sleep(4.0)
        self.processing = False



    def publish_joint_command(self, joint_angles):
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
            'panda_finger_joint1', 'panda_finger_joint2'
        ]
        joint_msg.position = list(joint_angles)
        self.joint_pub.publish(joint_msg)
        # self.get_logger().info(f'Published new joint pose: {joint_msg.position}')
        print(f'\n\nPublished NEW JOINTS: {joint_angles}\n\n')


    def compute_ik(self, pose: Pose):
        done_event = threading.Event()
        result_holder = {}

        def _callback(future):
            if future.result() and future.result().error_code.val == 1:
                self.get_logger().info('IK succeeded')
                result_holder['joints'] = future.result().solution.joint_state.position[:7]
            else:
                self.get_logger().warn('IK solver failed or timed out')
                result_holder['joints'] = None
            done_event.set()

        request = GetPositionIK.Request()
        request.ik_request.group_name = "panda_arm"
        request.ik_request.pose_stamped.header.frame_id = "panda_link0"
        request.ik_request.ik_link_name = "panda_link8"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = pose
        request.ik_request.timeout.sec = 2
        request.ik_request.avoid_collisions = True

        joint_state = JointState()
        joint_state.name = [
            'panda_joint1', 'panda_joint2', 'panda_joint3',
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
        ]
        joint_state.position = self.current_joint_state[:7]
        request.ik_request.robot_state = RobotState(joint_state=joint_state, is_diff=False)

        future = self.ik_client.call_async(request)
        future.add_done_callback(_callback)

        done_event.wait(timeout=3.0)  # Wait max 3 seconds
        return result_holder.get('joints', None)

    def apply_delta(self, pose_stamped: PoseStamped, delta):
        pose = pose_stamped.pose
        self.get_logger().info(f'Pose pre delta: {pose}')

        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w
        ])

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

        self.get_logger().info(f'Pose post delta: {new_pose.pose}')
        return new_pose

def main(args=None):
    rclpy.init(args=args)
    node = FrankaSimOpenVLA()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

