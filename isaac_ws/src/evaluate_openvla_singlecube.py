
# * ###########################################################################
# * Evaluation Script for Single-Cube Lift Environment
# * ###########################################################################

# ^ Configuration Parameters

OPENVLA_RESPONSE = True

GT_EPISODE_PATH = "./isaac_ws/src/output/random_camera_green/episode_0011.npy"
CONFIG_PATH = "./isaac_ws/src/output/multicube_yellow/multicube_yellow.json"
CUBE_COLOR_STR= "green" # "green", "blue", "yellow"

SAVE_STATS = True

CAMERA_X_RANGE = (-0.2, 0.2)
CAMERA_Y_RANGE = (-0.2, 0.2)
CAMERA_Z_RANGE = (-0.2, 0.2)

save_stats_file = "single_green_r_all_camerainrange.json"
save_stats_file = "single_green_r_all_rangeyout_-10.json"
CAMERA_Y_RANGE = (-0.3, -0.2)


SAVE_STATS_DIR = "./isaac_ws/src/stats/single_green_r_all_12+5%"

RANDOM_CAMERA = True
RANDOM_OBJECT = True
RANDOM_TARGET = True

if CUBE_COLOR_STR== "green":
    CUBE_COLOR = (0.0, 1.0, 0.0)
elif CUBE_COLOR_STR== "blue":
    CUBE_COLOR = (0.0, 0.0, 1.0)
elif CUBE_COLOR_STR== "yellow":
    CUBE_COLOR = (1.0, 1.0, 0.0)
else:
    raise ValueError("Invalid cube color. Choose from 'green', 'blue', or 'yellow'.")

if OPENVLA_RESPONSE:
    INIT_OBJECT_POS = [0.4, -0.1, 0.0]
    INIT_TARGET_POS = [0.4, 0.1, 0.0]  # Z must be 0 in OpenVLA inference script
else:
    import numpy as np
    episode = np.load(GT_EPISODE_PATH, allow_pickle=True)
    step = episode[0]
    if 'target_pose' in step:
        INIT_TARGET_POS = step['target_pose'][0, :3]
    if 'object_pose' in step:
        INIT_OBJECT_POS = step['object_pose'][0, :3]

INIT_ROBOT_POSE = [0.4, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0]

OPENVLA_UNNORM_KEY = "sim_data_custom_v0"
OPENVLA_INSTRUCTION = f"Pick the {CUBE_COLOR_STR} cube and place it on the red area. \n"

SEED = 777

CAMERA_HEIGHT = 1920
CAMERA_WIDTH = 1920
OPENVLA_CAMERA_HEIGHT = 256
OPENVLA_CAMERA_WIDTH = 256

CAMERA_POSITION = [0.9, -0.16, 0.6] #[1.0, -0.4, 0.9] video single_cube_random_all_static_camera #[0.9, -0.16, 0.6] original camera config # [1.2, -0.2, 0.9] for pretrained video
CAMERA_TARGET = [0.4, 0.0, 0.0]


CUBE_SIZE = [0.07, 0.03, 0.06]  # Cube size


if RANDOM_TARGET and OPENVLA_RESPONSE: # ABSOLUTE POSITION
    TARGET_X_RANGE = (-0.2 + INIT_TARGET_POS[0], 0.2 + INIT_TARGET_POS[0])
    TARGET_Y_RANGE = (-0.2 + INIT_TARGET_POS[1] , 0.2 + INIT_TARGET_POS[1])
    TARGET_Z_RANGE = (0.0 + INIT_TARGET_POS[2], 0.0 + INIT_TARGET_POS[2])
else:
    TARGET_X_RANGE = (INIT_TARGET_POS[0], INIT_TARGET_POS[0])
    TARGET_Y_RANGE = (INIT_TARGET_POS[1], INIT_TARGET_POS[1])
    TARGET_Z_RANGE = (INIT_TARGET_POS[2], INIT_TARGET_POS[2])

if RANDOM_OBJECT and OPENVLA_RESPONSE: # RELATIVE POSITION (TO INIT_OBJECT_POS)
    OBJECT_X_RANGE = (-0.2, 0.2)
    OBJECT_Y_RANGE = (-0.2, 0.2)
    OBJECT_Z_RANGE = (0.0, 0.0)
else:
    OBJECT_X_RANGE = (0.0, 0.0)
    OBJECT_Y_RANGE = (0.0, 0.0)
    OBJECT_Z_RANGE = (0.0, 0.0) 


EULER_NOTATION = "zyx" 

# ^ Imports and isaac lab app launch

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--camera_id", type=int, choices={0, 1}, default=0, help=("The camera ID to use for displaying points or saving the camera data. Default is 0." " The viewport will always initialize with the perspective of camera 0."),)
parser.add_argument("--renderer", type=str, default="RayTracedLighting", help="Renderer to use. Options: 'RayTracedLighting', 'PathTracing'.")
parser.add_argument("--anti_aliasing", type=int, default=3, help="Anti-aliasing level. Options: 0 (off), 1 (FXAA), 2 (TAA).")
parser.add_argument("--save", action="store_true", default=False, help="Save the data from camera at index specified by ``--camera_id``.",)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=args_cli.enable_cameras)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence
from PIL import Image


import numpy as np
import os
import shutil
from datetime import datetime
import json
from scipy.spatial.transform import Rotation

from PIL import Image
import json_numpy
import yaml
import requests
import random

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData


from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.env_eval_singlecube import LiftEnvCfg
import isaaclab.sim as sim_utils
import omni.replicator.core as rep
from isaaclab.utils.math import subtract_frame_transforms
##
# Pre-defined configs
##

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import omni.usd
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import convert_dict_to_backend



from pxr import UsdGeom, Usd, UsdShade

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.spawners import UsdFileCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

json_numpy.patch()

def set_server_url():
    """ Set the server URL based on the user environment."""
    # if user is "wanghan"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"

    if user == "wanghan":
        server_url = "http://0.0.0.0:8000/act"
    else:
        print("Current working directory:", os.getcwd())

        config_path = os.path.abspath("./isaac_ws/src/config.yaml")  # assuming you are in /root/isaac_ws folder
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ip_address = config["ip_address"]
        port = config["port"]
        server_url = f'http://{ip_address}:{port}/act'
        print(f"Server URL: {server_url}")

    return server_url


SERVER_URL = set_server_url()

def send_request(payload):
    """Send a POST request to the server with the given payload."""

    # Send POST request to the server
    response = requests.post(SERVER_URL, json=payload)

    # Check the response
    if response.status_code == 200:
        print("Response from server:", response.json())
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None


def scalar_first_to_last(q):
    """Convert quaternion from scalar-first to scalar-last format."""
    w, x, y, z = q
    return [x, y, z, w]


def scalar_last_to_first(q):
    """Convert quaternion from scalar-last to scalar-first format."""
    x, y, z, w = q
    return [w, x, y, z]


def compute_delta(ee_pose, next_ee_pose):
    """
    Compute the delta (expressed in EE frame) between two poses in the world frame,
    using quaternions internally.
    Input:
        - ee_pose, next_ee_pose: arrays of shape (8,) containing
            [x, y, z, roll, pitch, yaw, gripper_state]
    Output:
        - delta: array of shape (7,) containing
            [dx, dy, dz, dr, dp, dy, new_gripper_state]
    """
    # Decompose inputs
    pos1, rpy1, grip1 = ee_pose[:3], ee_pose[3:6], ee_pose[7]
    pos2, rpy2, grip2 = next_ee_pose[:3], next_ee_pose[3:6], next_ee_pose[7]

    # Convert to quaternions
    q1 = Rotation.from_euler(EULER_NOTATION, rpy1)
    q2 = Rotation.from_euler(EULER_NOTATION, rpy2)

    # Translation delta in world frame
    delta_pos_world = pos2 - pos1
    # Express translation in EE frame
    delta_pos_ee = q1.inv().apply(delta_pos_world)

    # Rotation delta as quaternion
    q_delta = q1.inv() * q2
    # Convert back to Euler only for output
    delta_euler = q_delta.as_euler(EULER_NOTATION)

    # Gripper state
    next_gripper = np.atleast_1d(grip2)

    # Assemble final delta
    delta = np.concatenate([delta_pos_ee, delta_euler, next_gripper]).astype(np.float32)
    return delta


def apply_delta(position, orientation, delta, env):
    """
    Apply a delta (in EE frame) to a pose in the world frame,
    using quaternions internally.
    Input:
        - position: (x, y, z) world
        - orientation: (w, x, y, z) quaternion world (scalar-first)
        - delta: array (7,) [dx, dy, dz, dr, dp, dy, gripper]
    Output:
        - new_pose: array (7,) [x, y, z, qw, qx, qy, qz]
    """
    # Build current rotation
    q_world = Rotation.from_quat(scalar_first_to_last(orientation))

    # Translate in world using EE-frame delta
    world_delta = q_world.apply(delta[:3])
    new_position = position + world_delta

    # Build rotation delta quaternion
    q_delta = Rotation.from_euler(EULER_NOTATION, delta[3:6])
    # Compose rotations
    q_new = q_world * q_delta

    # Convert back to Isaac format (scalar-first)
    new_orientation = scalar_last_to_first(q_new.as_quat())

    ee_pose =  np.concatenate([new_position, new_orientation])
    ee_pose = torch.tensor(ee_pose, device=env.unwrapped.device)
    return ee_pose


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    """Configuration for the Franka Cube Lift Environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]), 
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

                # Set the body name for the end effector
        self.commands.target_pose.body_name = "panda_hand"
        self.commands.target_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=TARGET_X_RANGE, pos_y=TARGET_Y_RANGE, pos_z=TARGET_Z_RANGE, roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        )

        self.events.reset_object_position.params["pose_range"] = {"x": OBJECT_X_RANGE, "y": OBJECT_Y_RANGE, "z": OBJECT_Z_RANGE}



        self.scene.plane = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/flat_plane.usd", scale=(1.0, 1.0, 1.0), 
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )

        self.scene.table = AssetBaseCfg(
            prim_path="/World/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/thor_table.usd", scale=(1.5, 1.5, 1.0)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        self.scene.light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )


        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=CUBE_SIZE,  # Dimensioni del cubo
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Proprietà fisiche
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Massa
                collision_props=sim_utils.CollisionPropertiesCfg(),  # Proprietà di collisione
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(CUBE_COLOR),  # Colore rosso
                    metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=INIT_OBJECT_POS,  # OVERWRITTEN BY THE COMMANDER
                rot=(1.0, 0.0, 0.0, 0.0)  # Orientamento iniziale (quaternione)
            ),
        )
        
        self.scene.box = RigidObjectCfg(
            prim_path="/World/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.005),  # Dimensioni del cubo
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Proprietà fisiche
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Massa
                collision_props=sim_utils.CollisionPropertiesCfg(),  # Proprietà di collisione
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),  # Colore rosso
                    metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=INIT_TARGET_POS,  # OVERWRITTEN BY THE COMMANDER
                rot=(1.0, 0.0, 0.0, 0.0)  # Orientamento iniziale (quaternione)
            ),
        )


    
                # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


        self.scene.camera = CameraCfg(
            prim_path="/World/CameraSensor",
            update_period=0,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
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



class GripperState:
    """States for the gripper."""
    OPEN = 1.0
    CLOSE = -1.0


def assign_material(object_path, material_path):
    """Assign a material to a prim in the USD stage."""
    stage = omni.usd.get_context().get_stage()

    # Get the object primitive
    object_prim = stage.GetPrimAtPath(object_path)

    # Get the existing material
    material_prim = stage.GetPrimAtPath(material_path)

    if object_prim and material_prim:
        material = UsdShade.Material(material_prim)
        UsdShade.MaterialBindingAPI(object_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        print("Material assigned successfully to ", object_path)
    else:
        print("Error: Primitive or material not found.")

def hide_prim(prim_path: str):
    """Hide a prim in the USD stage."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"✅ Hidden prim: {prim_path}")
    else:
        print(f"⚠️ Prim '{prim_path}' not found or invalid.")


def take_image(camera_index, camera):
    """
    Take an image from the camera and save it using the replicator writer.
    Args:
        camera_index: Index of the camera to use.
        camera: The camera object.
    """

    # Save images from camera at camera_index
    single_cam_data = convert_dict_to_backend(
        {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
    )
    image_data = single_cam_data.get('rgb')

    if image_data is not None:
        image_data = image_data.astype(np.uint8)
        high_res_image = Image.fromarray(image_data)

        low_res_image = high_res_image.resize((OPENVLA_CAMERA_HEIGHT, OPENVLA_CAMERA_WIDTH), Image.BICUBIC)

        return np.array(low_res_image)

    return None 

def get_init_des_state(env):
    """Get the initial desired state for the robot end-effector."""
    robot_init_pose = torch.tensor(INIT_ROBOT_POSE, device=env.unwrapped.device)
    gripper = torch.tensor([GripperState.OPEN], device=env.unwrapped.device)
    init_ee_pose = torch.cat([robot_init_pose, gripper], dim=-1) # shape: (8,) -> x, y, z, qw, qx, qy, qz, gripper_state
    return init_ee_pose.unsqueeze(0) # shape: (1, 8)

def get_current_state(robot, env):
    """Get the current state of the robot end-effector in the environment."""
    ee_pose_w = robot.data.body_state_w[:, 8, 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    current_gripper_state = robot.data.joint_pos.clone()[:, -1] 
    
    current_state = torch.cat([ee_pos_b, ee_quat_b, current_gripper_state.unsqueeze(-1)], dim=-1) # (1, 8)
    return current_state

def set_new_goal_pose(env):
    """Set a new goal pose for the box in the environment."""
    goal_pose = env.unwrapped.command_manager.get_command("target_pose")
    new_pos = goal_pose[..., :3].clone()
    new_pos[..., 2] = 0.0
    new_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=new_pos.device).expand(new_pos.shape[0], 4)

    root_state = torch.zeros((env.unwrapped.num_envs, 13), device=env.unwrapped.device)
    root_state[:, 0:3] = new_pos
    root_state[:, 3:7] = new_rot
    # Write the new pose to the simulation
    env.unwrapped.scene["box"].write_root_state_to_sim(root_state)

def get_openvla_res(camera_index, camera):
    """Get the OpenVLA response by sending the image and instruction to the server."""
    image_array = take_image(camera_index, camera)
    payload = {
        "image": image_array,  # Sending as numpy array, no conversion to list
        "instruction": OPENVLA_INSTRUCTION,
        "unnorm_key": OPENVLA_UNNORM_KEY  # Add the unnorm_key to the payload
    }
    #Send request to the server
    print("Sending request to OpenVLA...")
    res = send_request(payload)
    if res is None:
        print("Error in sending request to OpenVLA.")
        simulation_app.close()
    return res


if not OPENVLA_RESPONSE:
    episode = np.load(GT_EPISODE_PATH, allow_pickle=True)
    current_step_index = 0
def get_ground_truth_res():
    global current_step_index
    global episode
    finished_episode = False
    

    if current_step_index >= len(episode):
        print("No more steps available in the episode. Closing the simulation.")
        finished_episode = True
        return None, finished_episode

    step = episode[current_step_index]
    current_step_index += 1
    return step["action"], finished_episode

class ThresholdAdaptationState:
    def __init__(self, default_threshold=0.005, max_stuck_steps=10):
        self.default_threshold = default_threshold
        self.max_stuck_steps = max_stuck_steps
        self.position_threshold = default_threshold
        self.prev_position_error = None
        self.stuck_counter = 0
        self.stuck_reference_error = None  # nuovo

def adaptive_check_des_state_reached(current_state, desired_state, angle_threshold, adaptation_state):
    """
    Adaptive version of check_des_state_reached that updates position threshold
    only if the position error remains within 1 mm of a fixed value for several steps.
    """
    position_error = torch.norm(current_state[:, :3] - desired_state[:, :3], dim=1).item()
    
    quat_dot = torch.abs(torch.sum(current_state[:, 3:7] * desired_state[:, 3:7], dim=1)).clamp(-1.0, 1.0)
    angle_error = 2 * torch.acos(quat_dot).item()
    
    if desired_state[:, 7] == GripperState.CLOSE:
        gripper_correct = current_state[:, 7] <= CUBE_SIZE[1] / 2 + 0.001
    else:
        gripper_correct = current_state[:, 7] >= 0.04 - 0.001

    # Check if error is stuck within ±1mm of the initial one
    if adaptation_state.stuck_reference_error is None:
        adaptation_state.stuck_reference_error = position_error
        adaptation_state.stuck_counter = 1
    elif abs(position_error - adaptation_state.stuck_reference_error) < 0.001:
        adaptation_state.stuck_counter += 1
    else:
        adaptation_state.stuck_counter = 0
        adaptation_state.stuck_reference_error = position_error
        adaptation_state.position_threshold = adaptation_state.default_threshold

    if adaptation_state.stuck_counter >= adaptation_state.max_stuck_steps:
        adaptation_state.position_threshold = adaptation_state.stuck_reference_error + 0.002
        print(f"[ADAPTIVE] Threshold adapted to {adaptation_state.position_threshold:.4f} m")
        adaptation_state.stuck_counter = 0  # reset to avoid continuous readjustments
        adaptation_state.stuck_reference_error = None

    adaptation_state.prev_position_error = position_error

    if position_error < adaptation_state.position_threshold and angle_error < angle_threshold and gripper_correct:
        #print(f"REACHED des_state! Pos err: {position_error:.4f} m | Ang err: {angle_error:.4f}°")
        return True
    else:
        #print(f"NOT REACHED des_state! Pos err: {position_error:.4f} m | Ang err: {angle_error:.4f}°")
        return False




def check_des_state_reached(current_state, desired_state, position_threshold, angle_threshold):
    """
        Check if the current position is within the threshold of the desired position.
        Returns True if the goal is reached, False otherwise.
        
        state: [x, y, z, qw, qx, qy, qz, gripper_state]

    """
    position_error = torch.norm(current_state[:, :3] - desired_state[:, :3], dim=1)

    quat_dot = torch.abs(torch.sum(current_state[:, 3:7] * desired_state[:, 3:7], dim=1))  # q1 · q2
    quat_dot = torch.clamp(quat_dot, -1.0, 1.0)  # clamp per stabilità numerica
    angle_error = 2 * torch.acos(quat_dot)

    if desired_state[:, 7] == GripperState.CLOSE:
        gripper_correct = current_state[:, 7] <= CUBE_SIZE[1] / 2 + 0.001 
    else:
        gripper_correct = current_state[:, 7] >= 0.04 - 0.001


    if position_error.item() < position_threshold and angle_error.item() < angle_threshold and gripper_correct:
        #print(f"REACHED des_state! Pos err: {position_error.item():.4f} m | Ang err: {angle_error.item():.4f}°")
        return True
    else:
        #print(f"NOT REACHED des_state! Pos err: {position_error.item():.4f} m | Ang err: {angle_error.item():.4f}°")
        pass
    return False

def set_new_random_camera_pose(env, camera, x_range=(-0.2, 0.2), y_range=(-0.2, 0.2), z_range=(-0.2, 0.2)):
    """
    Set a new random camera position within the specified ranges for x, y, z.

    Args:
        env: simulation environment
        camera: camera object
        x_range, y_range, z_range: tuple with (min, max) for each axis
    """
    device = env.unwrapped.device

    base_camera_position = torch.tensor(CAMERA_POSITION, device=device)

    # Generate random offsets for each axis
    offset_x = torch.FloatTensor(1).uniform_(*x_range).to(device)
    offset_y = torch.FloatTensor(1).uniform_(*y_range).to(device)
    offset_z = torch.FloatTensor(1).uniform_(*z_range).to(device)

    random_offset = torch.cat([offset_x, offset_y, offset_z])

    # Apply the offset
    camera_position = base_camera_position + random_offset
    camera_position = camera_position.unsqueeze(0)  # (1, 3)

    camera_target = torch.tensor([CAMERA_TARGET], device=device)

    camera.set_world_poses_from_view(camera_position, camera_target)
    return camera_position.clone().squeeze(0).cpu().numpy().astype(np.float32)  


def check_task_completed(env, robot):
    """ Check if the task is completed based on the object and target poses."""
    current_object_pose = env.unwrapped.scene["object"].data.root_state_w[:, :7].clone().cpu().numpy().squeeze(0).astype(np.float32)
    current_target_pose = env.unwrapped.scene["box"].data.root_state_w[:, :7].clone().cpu().numpy().squeeze(0).astype(np.float32)
    distance_object_target = np.linalg.norm(current_object_pose[:3] - current_target_pose[:3])

    ee_pose_w = robot.data.body_state_w[:, 8, 0:7]
    distance_object_ee = np.linalg.norm(current_object_pose[:3] - ee_pose_w[:, :3].cpu().numpy().squeeze(0).astype(np.float32))
    if distance_object_target < 0.06 and distance_object_ee > 0.15:
        return True
    return False

def convert_numpy(obj):
    """Convert numpy objects to standard Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # es. np.float32, np.bool_
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    else:
        return obj

def save_stats(simulation_step, save_stats_file, initial_camera_pose, initial_target_pose, initial_object_pose, distance_object_target):
    """Save the statistics of the simulation step to a JSON file."""
    stats = {
        "simulation_step": simulation_step,
        "initial_camera_pose": initial_camera_pose,
        "initial_target_pose": initial_target_pose,
        "initial_object_pose": initial_object_pose,
        "distance_object_target": distance_object_target,
        "completed": distance_object_target < 0.07,
    }

    stats = convert_numpy(stats)

    if not os.path.exists(SAVE_STATS_DIR):
        os.makedirs(SAVE_STATS_DIR, exist_ok=True)

    save_stats_path = os.path.join(SAVE_STATS_DIR, save_stats_file)

    with open(save_stats_path, "a") as f:
        f.write(json.dumps(stats) + "\n") 


def get_dist_object_target(env):
    """Calculate the distance between the object and the target in the environment."""
    current_object_pose = env.unwrapped.scene["object"].data.root_state_w[:, :7].clone().cpu().numpy().squeeze(0).astype(np.float32)
    current_target_pose = env.unwrapped.scene["box"].data.root_state_w[:, :7].clone().cpu().numpy().squeeze(0).astype(np.float32)

    distance_object_target = np.linalg.norm(current_object_pose[:2] - current_target_pose[:2]) # only x, y
    return distance_object_target


def check_valid_task(env):
    """Check if the task is valid by ensuring the object is not too close to the target."""
    object_pos = env.unwrapped.scene["object"].data.root_state_w[:, :3].clone()
    target_pos = env.unwrapped.scene["box"].data.root_state_w[:, :3].clone()
    distance_object_target = torch.norm(object_pos - target_pos, dim=1)

    if distance_object_target.item() <= 0.08:
        print("Object too close to the target.")
        return False
    return True

def run_simulator(env, args_cli):
    """Run the simulator with the specified environment and arguments."""
   
    camera = env.unwrapped.scene["camera"]

    robot = env.unwrapped.scene["robot"]

    print("\n\nRUNNING SIMULATOR!\n\n")

    # Set the camera position and target (wrist camera is already attached to the robot in the config)
    camera_positions = torch.tensor([CAMERA_POSITION], device=env.unwrapped.device)
    camera_targets = torch.tensor([CAMERA_TARGET], device=env.unwrapped.device)
    camera.set_world_poses_from_view(camera_positions, camera_targets)
    camera_index = args_cli.camera_id

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    assign_material(object_path="/World/Table", material_path="/World/Table/Looks/Black")


    count = 0
    task_count = 0
    adaptation_state = ThresholdAdaptationState(default_threshold=0.005, max_stuck_steps=10)
    
    
    valid_task = True

    # if SAVE_STATS:
    #     save_stats_file = input("Specify the name of the file to save stats: ")

    goal_reached = False
    task_completed = False
    simulation_step = 600
    saved_stats_cnt = 0
    max_saved_stats = 104

    initial_camera_pose = np.array([CAMERA_POSITION])
    while simulation_app.is_running():

        with torch.inference_mode(): 
            if count == 0:
                valid_task = check_valid_task(env) 
                        
            distance_object_target = get_dist_object_target(env)
 

            task_completed = check_task_completed(env, robot)
            if task_completed and simulation_step==600:
                simulation_step = count

            if count == 0:
                initial_target_pose = env.unwrapped.scene["box"].data.root_state_w[:, :7].clone().cpu().numpy().squeeze(0).astype(np.float32)
                initial_object_pose = env.unwrapped.scene["object"].data.root_state_w[:, :7].clone().cpu().numpy().squeeze(0).astype(np.float32)
                des_state = get_init_des_state(env)
                ee_prev_pos = torch.tensor(des_state[:, :3], device=env.unwrapped.device)
                ee_prev_quat = torch.tensor(des_state[:, 3:7], device=env.unwrapped.device)
                
            # print("Getting curretn state...")
            current_state = get_current_state(robot, env)
            # print("Checking if goal is reached...")
            # goal_reached = check_des_state_reached(current_state, des_state, position_threshold=0.0153, angle_threshold=0.05)
            goal_reached = adaptive_check_des_state_reached(current_state, des_state, angle_threshold=0.05, adaptation_state=adaptation_state)

            
            if valid_task and goal_reached and count > 0 and not task_completed:
                print("Goal reached: ", goal_reached)
                if OPENVLA_RESPONSE:
                    res = get_openvla_res(camera_index, camera)
                    finished_episode = False
                    
                else:
                
                    res, finished_episode = get_ground_truth_res()

                
                if not finished_episode:
                    ee_des_pose = apply_delta(ee_prev_pos[0].cpu().numpy(), ee_prev_quat[0].cpu().numpy(), res, env)
                    ee_prev_pos = torch.tensor(ee_des_pose[:3], device=env.unwrapped.device).unsqueeze(0)
                    ee_prev_quat = torch.tensor(ee_des_pose[3:7], device=env.unwrapped.device).unsqueeze(0)
                    
                    if res[6] < 0.03: # open = 0.04
                        des_gripper_state = torch.tensor([GripperState.CLOSE], device=env.unwrapped.device)
                    else:
                        des_gripper_state = torch.tensor([GripperState.OPEN], device=env.unwrapped.device)
                
                    des_state = torch.cat([ee_des_pose.unsqueeze(0), des_gripper_state.unsqueeze(0)], dim=-1) # (1, 8)


            dones = env.step(des_state)[-2]

            camera.update(dt=env.unwrapped.sim.get_physics_dt())


            if dones.any():
                
                if SAVE_STATS and valid_task:
                    saved_stats_cnt += 1
                    save_stats(simulation_step, save_stats_file, initial_camera_pose, initial_target_pose, initial_object_pose, distance_object_target)
                    if saved_stats_cnt >= max_saved_stats:
                        print("Collected enough stats. Closing the simulation.")
                        simulation_app.close()

                print("\n\nRESETTING ENVIRONMENT...\n\n")
                if RANDOM_CAMERA:
                    initial_camera_pose = set_new_random_camera_pose(env, camera, x_range=CAMERA_X_RANGE, y_range=CAMERA_Y_RANGE, z_range=CAMERA_Z_RANGE) # set the new random camera position in simulation

                set_new_goal_pose(env) # set the new box (goal) position in simulation 
            
                count = 0
                global current_step_index
                current_step_index = 0
                task_count += 1
                task_completed = False
                simulation_step = 600

                continue

            count += 1

    # close the environment
    env.close()

def load_config(config_path = CONFIG_PATH):
    """Load the configuration from a JSON file and update global variables."""
    # Load the configuration file
    if config_path.endswith(".json"):
        print("Loading config from JSON file...")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Change Globals according to the loaded config
        for key, value in config.items():
            if key in globals() and key != "INIT_TARGET_POS" and key != "INIT_OBJECT_POS":
                globals()[key] = value
    else:
        print("Not loading config from JSON file. Using default values.")


def main():
    """Main function to run the Franka Cube Lift Environment."""
    
    #load_config()


    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.seed = SEED

    env_cfg.scene.ee_frame.visualizer_cfg.markers["frame"].enabled = False
    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    # Try to set the seed
    # reset environment at start
    env.unwrapped.sim.set_camera_view([1.0, 1.5, 1.5], [0.2, 0.0, 0.0])

    env.reset()

    hide_prim("/Visuals/Command/goal_pose")
    hide_prim("/Visuals/Command/body_pose")
    
    run_simulator(env, args_cli)
    

if __name__ == "__main__":
    main()
    simulation_app.close()