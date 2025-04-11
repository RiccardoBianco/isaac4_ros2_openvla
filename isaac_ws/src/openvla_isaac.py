"""
    # Usage
    isaac_lab/isaaclab.sh -p src/openvla_isaac.py  --enable_cameras --save

"""

OPENVLA_INSTRUCTION = "Pick up the yellow box"
OPENVLA_UNNORM_KEY = "bridge_orig"
MAX_GRIPPER_POSE = 1.0  # TODO check if this is correct


import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--renderer", type=str, default="RayTracedLighting", help="Renderer to use. Options: 'RayTracedLighting', 'PathTracing'.")
parser.add_argument("--anti_aliasing", type=int, default=3, help="Anti-aliasing level. Options: 0 (off), 1 (FXAA), 2 (TAA).")
parser.add_argument("--save", action="store_true", default=False, help="Save the data from camera at index specified by ``--camera_id``.",)
parser.add_argument("--camera_id", type=int, choices={0, 1}, default=0, help=("The camera ID to use for displaying points or saving the camera data. Default is 0." " The viewport will always initialize with the perspective of camera 0."),)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import json_numpy
import yaml
import requests


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils import convert_dict_to_backend
import omni.replicator.core as rep
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip
from isaaclab.sensors.camera import CameraCfg


# Apply patch for handling numpy arrays in JSON
json_numpy.patch()

# Define the URL of the server endpoint
def set_server_url():
    # if user is "wanghan"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"

    if user == "wanghan":
        server_url = "http://0.0.0.0:8000/act"
    else:
        print("Current working directory:", os.getcwd())

        config_path = os.path.abspath("src/config.yaml")  # assuming you are in /root/isaac_ws folder
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ip_address = config["ip_address"]
        port = config["port"]
        server_url = f'http://{ip_address}:{port}/act'
        print(f"Server URL: {server_url}")

    return server_url


SERVER_URL = set_server_url()



def send_request(payload):

    # Send POST request to the server
    response = requests.post(SERVER_URL, json=payload)

    # Check the response
    if response.status_code == 200:
        print("Response from server:", response.json())
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None


def apply_delta(position, orientation, delta):
    
    position = position.squeeze(0)
    orientation = orientation.squeeze(0)

    pos_list = list(position)
    orient = list(orientation)
    print(f"Prev position in base frame: {pos_list[0]:.3}, {pos_list[1]:.3}, {pos_list[2]:.3}")
    print(f"Prev orientation in base frame: {orient[0]:.3}, {orient[1]:.3}, {orient[2]:.3}, {orient[3]:.3}")
    # print(f"prev orientation: {orientation}")
    print(f"Delta position in ee frame: {delta[:3]}")
    print(f"Delta orientation in ee frame: {delta[3:6]}")


    R = Rotation.from_quat(orientation).as_matrix()
    print("X local:", R[:, 0])
    print("Y local:", R[:, 1])
    print("Z local:", R[:, 2])

    # Compute delta in the world frame
    world_delta = R @ np.array(delta[:3])
    new_position = position + world_delta


    # Apply rotation delta (Euler angles)
    delta_rot = Rotation.from_euler('zyx', delta[3:6]) # TODO change xyz
    new_orientation = (Rotation.from_matrix(R @ delta_rot.as_matrix())).as_quat()


    new_pos_list = list(new_position)
    new_orient_list = list(new_orientation)
    print(f"New position in base frame: {new_pos_list[0]:.3}, {new_pos_list[1]:.3}, {new_pos_list[2]:.3}")
    print(f"New orientation in base frame: {new_orient_list[0]:.3}, {new_orient_list[1]:.3}, {new_orient_list[2]:.3}, {new_orient_list[3]:.3}")
    # print(f"new_orientation: {new_orientation}")
    R_ee_to_world = R
    R_world_to_ee = R_ee_to_world.T  # Inversa della rotazione
    displacement_in_ee = R_world_to_ee @ (new_position - position)
    print("Displacement in EE frame:", displacement_in_ee)

    new_pose = np.concatenate([new_position, new_orientation])  # shape (7,)


    return new_pose


def take_image(camera_index, camera, rep_writer):
    if args_cli.save:
        # Save images from camera at camera_index
        # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )

        # Extract the other information
        single_cam_info = camera.data.info[camera_index]

        # Pack data back into replicator format to save them using its writer
        rep_output = {"annotators": {}}
        for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
            if info is not None:
                rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
            else:
                rep_output["annotators"][key] = {"render_product": {"data": data}}
        # Save images
        # Note: We need to provide On-time data for Replicator to save the images.
        rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
        rep_writer.write(rep_output)

        # Extract the image data (assuming 'rgb' is the key)
        image_data = single_cam_data.get('rgb')
        
        if image_data is not None:
            image_data = image_data.astype(np.uint8)
            pil_image = Image.fromarray(image_data)
            image_array = np.array(pil_image)
            return image_array

    return None 


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(

        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/thor_table.usd", scale=(1.5, 1.5, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    sugar_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SugarBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd", scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.3, 0.0)),
    )

    craker_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CrakerBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    )

    tomato_can = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TomatoCan",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd", scale=(1.0, 1.0, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, -0.3, 0.0)),
    )

    camera = CameraCfg(
        prim_path="/World/CameraSensor",
        update_period=0,
        height=1080,
        width=1920,
        data_types=[
            "rgb",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            #focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            focal_length=16.0,         # Wider view
            focus_distance=1000.0,     # Farther focus (everything is sharp)
            horizontal_aperture=30.0,  # Wider aperture = more stuff in view, but can reduce blur too
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    
    #######################
    # CAMERA STUFF - Start
    #######################

    # Get the camera
    camera = scene["camera"]

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[1.5, 1.5, 1.5]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    # These orientations are in ROS-convention, and will position the cameras to view the origin
    camera.set_world_poses_from_view(camera_positions, camera_targets)
    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    ###################
    # CAMERA STUFF - End
    ###################

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    ee_goal_deltas = [
        # [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # 10 cm x
        # [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # 10 cm y
        # [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # 10 cm z
        # [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # -10 cm x
        # [0.0, -0.1, 0.0, 0.0, 0.0, 0.0],  # -10 cm y
        # [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],  # -10 cm z
        # [0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0],  # 90° x
        # [0.0, 0.0, 0.0, np.pi/2, 0.0, 0.0], # -90° x
        # [0.0, 0.0, 0.0, 0.0, -np.pi/2, 0.0],  # 90° y
        # [0.0, 0.0, 0.0, 0.0, np.pi/2, 0.0], # -90° y
        # [0.0, 0.0, 0.0, 0.0, 0.0, -np.pi/2],  # 90° z
        # [0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2], # -90° z
        [0.0, 0.0, 0.0, -np.pi/2, -np.pi/2, 0.0], # 90° x and y
        #[0.0, 0.0, 0.0, np.pi/2, np.pi/2, 0.0], # -90° x and y
        [0.0, 0.0, 0.0, -np.pi/2, 0.0, -np.pi/2], # 90° x and z
        [0.0, 0.0, 0.0, np.pi/2, 0.0, np.pi/2], # -90° x and z
    ]

    ee_goal_deltas = torch.tensor(ee_goal_deltas, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = torch.tensor([0.5, 0.0, 0.7, 0, 1, 0, 0], device=sim.device) # TODO check if necessary

    # Specify robot-specific parameters
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop



    goal_reached = True

    while simulation_app.is_running():

        if count == 0:

            # Initialization - move to home position
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # set gripper position
            gripper_pos_des = torch.tensor([[1.0, 1.0]], device=sim.device)


        if goal_reached and count != 0:
            # nuovo goal

            ####################################
            ###### SEND REQUEST TO SERVER ######
            ####################################

            # take image
            image_array = take_image(camera_index, camera, rep_writer)

            payload = {
                "image": image_array,  # Sending as numpy array, no conversion to list
                "instruction": OPENVLA_INSTRUCTION,
                "unnorm_key": OPENVLA_UNNORM_KEY  # Add the unnorm_key to the payload
            }

            #Send request to the server
            res = send_request(payload)

            if res is None:
                print("Error in sending request to OpenVLA.")
                continue
            
            #####################################


            delta = res[:6]
            gripper_pos_des = torch.tensor([[res[6]*MAX_GRIPPER_POSE, res[6]*MAX_GRIPPER_POSE]], device=sim.device)
            ee_goal = apply_delta(ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), delta)
            #print(f"✅ Nuovo goal: {current_goal_idx}")
            #delta = ee_goal_deltas[current_goal_idx]
            #ee_goal = apply_delta(ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), delta.cpu().numpy())
            ee_goal = torch.tensor(ee_goal, device=sim.device).unsqueeze(0)
            ik_commands[:] = ee_goal
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            goal_reached = False
            
            current_goal_idx = (current_goal_idx + 1) % len(ee_goal_deltas)
            

        # get current state
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # posizione dell'end-effector relativa al root
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # calcolo comando IK

        

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # apply actions
        joint_pos_des = torch.cat((joint_pos_des, gripper_pos_des), dim=1).to(dtype=torch.float32)
        robot.set_joint_position_target(joint_pos_des, joint_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        scene.write_data_to_sim()
        # perform step
        sim.step()

        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        goal_reached = check_goal_reached(ik_commands, ee_pose_w)
        
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]

        # TODO remove this two lines
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

        # update marker positions 
        #draw_markers(ee_pose_w, ik_commands, scene, ee_marker, goal_marker)


def draw_markers(ee_pose_w, ik_commands, scene, ee_marker, goal_marker):
    quat_correction_np = Rotation.from_euler('x', 180, degrees=True).as_quat()  # TODO check if we need to rotate the marker
    quat_correction = torch.tensor(quat_correction_np, dtype=torch.float32, device=ee_pose_w.device)

    def apply_marker_rotation(quat_batch, quat_corr):
        """Applica rotazione correttiva (composizione di quaternioni)"""
        # Converti in numpy per usare scipy
        quat_batch_np = quat_batch.cpu().numpy()  # (N, 4)
        quat_corr_np = quat_corr.cpu().numpy()    # (4,)

        # Composizione: R_correction * R_original
        r_orig = Rotation.from_quat(quat_batch_np)
        r_corr = Rotation.from_quat(quat_corr_np)
        r_combined = r_corr * r_orig

        # Torna a torch tensor
        return torch.tensor(r_combined.as_quat(), dtype=torch.float32, device=quat_batch.device)

    # ---Apply correction to quaternions ---
    corrected_ee_quat = apply_marker_rotation(ee_pose_w[:, 3:7], quat_correction)
    corrected_goal_quat = apply_marker_rotation(ik_commands[:, 3:7], quat_correction)

    # --- Markers with corrected visualization  ---
    ee_marker.visualize(ee_pose_w[:, 0:3], corrected_ee_quat)
    goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, corrected_goal_quat)


def check_goal_reached(ik_commands, ee_pose_w, position_threshold=0.0005, angle_threshold=0.1):
    goal_pos = ik_commands[:, 0:3]
    goal_quat = ik_commands[:, 3:7]
    current_pos = ee_pose_w[:, 0:3]
    current_quat = ee_pose_w[:, 3:7]

    # errore posizione
    position_error = torch.norm(goal_pos - current_pos, dim=1)

    # errore orientamento (angular distance tra quaternioni)
    quat_dot = torch.abs(torch.sum(goal_quat * current_quat, dim=1))  # q1 · q2
    quat_dot = torch.clamp(quat_dot, -1.0, 1.0)  # clamp per stabilità numerica
    angle_error = 2 * torch.acos(quat_dot)

    # controllo soglia
    if position_error.item() < position_threshold and angle_error.item() < angle_threshold:
        angle_deg = np.degrees(angle_error.item())
        print(f"🎯 Goal raggiunto! Pos err: {position_error.item():.4f} m | Ang err: {angle_deg:.2f}°")
        return True
    
    return False


def clear_img_folder():
    if os.path.exists("./isaac_ws/src/output/camera"):
        for file in os.listdir("./isaac_ws/src/output/camera"):
            if file.startswith("rgb") and file.endswith(".png"):
                os.remove(os.path.join("./isaac_ws/src/output/camera", file))

def main():
    """Main function."""
    clear_img_folder()
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
