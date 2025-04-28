"""
    # Usage
    connect to the computer in the lab through ssh

    # Move to the right directory and activate the conda environment
    cd ~/Desktop/PLRItalians/isaac4_ros2_openvla
    conda activate env_isaaclab

    # run the script
    ~/IsaacLab/isaaclab.sh -p isaac_ws/src/isaac_data_collection.py --enable_cameras --save --headless
"""

# Scipy -> quaternion -> scalar last order [x, y, z, w]
# Iaaclab -> quaternion -> scalar first order [w, x, y, z]

OPENVLA_INSTRUCTION = "Pick up the red box" # TODO CAMI - task instruction
OPENVLA_UNNORM_KEY = "bridge_orig"
MAX_GRIPPER_POSE = 1.0 
VISUALIZE_MARKERS = True


import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--renderer", type=str, default="RayTracedLighting", help="Renderer to use. Options: 'RayTracedLighting', 'PathTracing'.")
# parser.add_argument("--anti_aliasing", type=int, default=3, help="Anti-aliasing level. Options: 0 (off), 1 (FXAA), 2 (TAA).")
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
import time
import sys
import signal
import json

SAVE_DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "plr_openvla_dataset")

def save_step_npz(image_array, joint_angles, ee_pose, camera_pose, instruction, step_id):
    os.makedirs(SAVE_DATASET_DIR, exist_ok=True)
    save_dict = {
        "observation/image_primary": image_array.astype(np.uint8),
        "observation/proprio": joint_angles.astype(np.float32),  # shape: (7,)
        "observation/camera_pose": camera_pose.astype(np.float32),  # shape: (7,)
        "action": ee_pose.astype(np.float32),  # shape: (7,)
        "task/language_instruction": instruction,
        "dataset_name": "plr_openvla_finetuning_dataset",
    }
    np.savez_compressed(os.path.join(SAVE_DATASET_DIR, f"step_{step_id:06d}.npz"), **save_dict)


# Apply patch for handling numpy arrays in JSON
json_numpy.patch()


def scalar_first_to_last(q):
    w, x, y, z = q
    return [x, y, z, w]


def scalar_last_to_first(q):
    x, y, z, w = q
    return [w, x, y, z]


def apply_delta(position, orientation, delta):
    
    position = position.squeeze(0)
    orientation = orientation.squeeze(0)


    R = Rotation.from_quat(scalar_first_to_last(orientation)).as_matrix()

    # Compute delta in the world frame
    world_delta = R @ np.array(delta[:3])
    new_position = position + world_delta


    # Apply rotation delta (Euler angles)
    r_x = Rotation.from_euler('x', delta[3])
    r_y = Rotation.from_euler('y', delta[4])
    r_z = Rotation.from_euler('z', delta[5])

    # Apply the rotation to the current orientation
    delta_rot = r_x * r_y * r_z # first apply x, then y, then z
    # delta_rot = r_z* r_y * r_x # first apply z, then y, then x


    new_orientation = (Rotation.from_matrix(R @ delta_rot.as_matrix())).as_quat()
    new_orientation = scalar_last_to_first(new_orientation) # Needed for isaac sim

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

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/flat_plane.usd", scale=(1.0, 1.0, 1.0), 
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # # # mount
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/thor_table.usd", scale=(1.5, 1.5, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),

    )

    cube = AssetBaseCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd", scale=(0.05, 0.1, 0.05)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
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
            focal_length=24.0,         # Wider view
            focus_distance=400.0,     # Farther focus (everything is sharp)
            horizontal_aperture=30.0,  # Wider aperture = more stuff in view, but can reduce blur too
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="/World/Robot")
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
    camera_positions = torch.tensor([[1.2, -0.2, 0.8]], device=sim.device)
    camera_targets = torch.tensor([[0.0, 0.0, -0.3]], device=sim.device)
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

    if VISUALIZE_MARKERS:
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    ee_goal_deltas = [
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # 10 cm x
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # 10 cm y
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # 10 cm z
        [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # -10 cm x
        [0.0, -0.1, 0.0, 0.0, 0.0, 0.0],  # -10 cm y
        [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],  # -10 cm z
    ]

    ee_goal_deltas = torch.tensor(ee_goal_deltas, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = torch.tensor([0.5, 0.0, 0.4, 0, 1, 0, 0], device=sim.device) # TODO check if necessary

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


            # take image
            image_array = take_image(camera_index, camera, rep_writer)
            #  - image_array -> image from camera
             

            print(f"✅ Nuovo goal: {current_goal_idx}")
            delta = ee_goal_deltas[current_goal_idx]
            ee_goal = apply_delta(ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), delta.cpu().numpy())
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
        # TODO CAMI - ee_pose_b, ee_quat_b -> position and orientation of the end-effector in the base frame
        # TODO CAMI - joint_pos -> joint position of the robot
        # Get joint angles, ee pose and camera pose
        joint_angles_np = joint_pos.squeeze(0).cpu().numpy()
        ee_pose_np = torch.cat([ee_pos_b, ee_quat_b], dim=1).squeeze(0).cpu().numpy()
        camera_pose_np = torch.cat([camera.data.pose[camera_index]["position"], camera.data.pose[camera_index]["orientation"]], dim=0).cpu().numpy()

        # Save to .npz format
        save_step_npz(
            image_array=image_array,
            joint_angles=joint_angles_np,
            ee_pose=ee_pose_np,
            camera_pose=camera_pose_np,
            instruction=OPENVLA_INSTRUCTION,
            step_id=count
        )




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


        # update marker positions 
        if VISUALIZE_MARKERS:
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

     

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

    # usd_path = os.path.join(os.getcwd(), "isaac_ws/src/table_parquet.usd")
    # omni.usd.get_context().open_stage(usd_path)
    # stage = omni.usd.get_context().get_stage()
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
