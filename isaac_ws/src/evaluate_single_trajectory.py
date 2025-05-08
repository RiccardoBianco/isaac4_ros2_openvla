"""
    # Usage
    ~/isaac_ws/isaac_lab/isaaclab.sh -p ~/isaac_ws/src/client_isaac_openvla.py  --enable_cameras --save

"""

# Scipy -> quaternion -> scalar last order [x, y, z, w]
# Iaaclab -> quaternion -> scalar first order [w, x, y, z]
PICK_AND_PLACE = True
if PICK_AND_PLACE: 
    OPENVLA_INSTRUCTION = "Pick the green cube and place it on the red area. \n"
else:
    OPENVLA_INSTRUCTION = "Pick the green cube and lift it. \n"


OPENVLA_UNNORM_KEY = "sim_data_custom_v0" # TODO check if this is correct -> sim_data_custom_v0
MAX_GRIPPER_POSE = 1.0  # TODO check if this is correct
VISUALIZE_MARKERS = False

OBJECT_POS = [0.5, 0, 0.055] # Must be equal to init object pose in isaac_data_collection.py
TARGET_POS = (0.4, -0.35, 0.0) # Must be equal to target range in lift_env_cfg_pers.py

CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256



CAMERA_POSITION = [1.2, -0.2, 0.8]
CAMERA_TARGET = [0.0, 0.0, -0.3]

# CAMERA_POSITION = [0.9, -0.4, 0.6] # IGNORE THIS
# CAMERA_TARGET = [0.3, 0.0, -0.2]


CUBE_MULTICOLOR = False # ^ Change this to True if you want to use the multicolor cube

OPENVLA_RESPONSE = True


EULER_NOTATION = "zyx" # or 'xyz' 
# -> zyx rotate first around x, then y, then z axis of the end_effector
# -> xyz rotate first around z, then y, then x axis of the end_effector


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

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
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

from pxr import Usd, UsdPhysics, UsdGeom, UsdShade
import omni.usd


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

        config_path = os.path.abspath("./isaac_ws/src/config.yaml")  # assuming you are in /root/isaac_ws folder
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ip_address = config["ip_address"]
        port = config["port"]
        server_url = f'http://{ip_address}:{port}/act'
        print(f"Server URL: {server_url}")

    return server_url


SERVER_URL = set_server_url()

#######################################

def add_rigid_body_api(usd_file_path):
    # Open the USD stage
    stage = Usd.Stage.Open(usd_file_path)

    # Get the default prim
    default_prim = stage.GetDefaultPrim()

    # Check if default prim exists and if RigidBodyAPI already exists
    if default_prim and not default_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        # Add RigidBodyAPI
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(default_prim)
        if rigid_body_api:
            print(f"Added RigidBodyAPI to {usd_file_path}")
            # Save the changes
            stage.GetRootLayer().Save()
            
    UsdPhysics.CollisionAPI.Apply(default_prim)
    stage.Save()

#######################################


def assign_material(object_path, material_path):
    stage = omni.usd.get_context().get_stage()

    # Prendi la primitiva della tabella
    object_prim = stage.GetPrimAtPath(object_path)
    
    # Prendi il materiale esistente
    material_prim = stage.GetPrimAtPath(material_path)

    if object_prim and material_prim:
        material = UsdShade.Material(material_prim)
        UsdShade.MaterialBindingAPI(object_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        print("Materiale assegnato correttamente a ", object_path)
    else:
        print("Errore: Primitiva o materiale non trovati.")


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


def scalar_first_to_last(q):
    w, x, y, z = q
    return [x, y, z, w]


def scalar_last_to_first(q):
    x, y, z, w = q
    return [w, x, y, z]


# def apply_delta(position, orientation, delta):
#     """
#     Apply a delta in ee frame to a position and orientation (quaternion) in the world frame.
#         Input: 
#             - position: (x, y, z) in world frame
#             - orientation: (w, x, y, z) in world frame
#             - delta: (dx, dy, dz, rx, ry, rz) in ee frame
#         Output:
#             - new_pose: (x, y, z, w, x, y, z) in world frame
#         Note on quaternion order:
#             - Isaac sim uses scalar first order for quaternions, while scipy uses scalar last order.
#     """
    
#     R = Rotation.from_quat(scalar_first_to_last(orientation)).as_matrix()

#     # Compute delta in the world frame
#     world_delta = R @ np.array(delta[:3])
#     new_position = position + world_delta


#     # Apply rotation delta (Euler angles)
#     r_x = Rotation.from_euler('x', delta[3])
#     r_y = Rotation.from_euler('y', delta[4])
#     r_z = Rotation.from_euler('z', delta[5])

#     # Apply the rotation to the current orientation
#     delta_rot = r_x * r_y * r_z # first apply x, then y, then z
#     # delta_rot = r_z* r_y * r_x # first apply z, then y, then x


#     new_orientation = (Rotation.from_matrix(R @ delta_rot.as_matrix())).as_quat()
#     new_orientation = scalar_last_to_first(new_orientation) # Needed for isaac sim

#     new_pose = np.concatenate([new_position, new_orientation])  # shape (7,)

#     return new_pose


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

# def compute_delta(ee_pose, next_ee_pose):
#     """
#         Compute the delta (expressed in ee frame) between two poses in the world frame.
#         Input: 
#             - pose: (x, y, z, roll, pitch, yaw, gripper_state) in world frame
#             - next_pose: (x, y, z, roll, pitch, yaw, new_gripper_state) in world frame
#         Output:
#             - delta: (dx, dy, dz, rx, ry, rz, new_gripper_state) in ee frame
#     """
#     # Decomponi le pose
#     # pos1, rpy1, grip1 = ee_pose[:3], ee_pose[3:6], ee_pose[6]
#     # pos2, rpy2, grip2 = next_ee_pose[:3], next_ee_pose[3:6], next_ee_pose[6] # no padding

#     pos1, rpy1, grip1 = ee_pose[:3], ee_pose[3:6], ee_pose[7]
#     pos2, rpy2, grip2 = next_ee_pose[:3], next_ee_pose[3:6], next_ee_pose[7]

#     # Rotazioni come oggetti Rotation
#     rot1 = Rotation.from_euler(EULER_NOTATION, rpy1)
#     rot2 = Rotation.from_euler(EULER_NOTATION, rpy2)

#     # Calcola la traslazione nel world frame
#     delta_pos_world = pos2 - pos1

#     # Riporta la traslazione nel frame dell'EE
#     delta_pos_ee = rot1.inv().apply(delta_pos_world)

#     # Calcola rotazione relativa: R_delta = R1^-1 * R2
#     delta_rot = rot1.inv() * rot2

#     # Estrai rotazione relativa in Euler angles
#     delta_euler = delta_rot.as_euler(EULER_NOTATION)  # RPY in radianti

#     next_gripper = np.atleast_1d(grip2)


#     # Combina il delta finale
#     delta = np.concatenate([delta_pos_ee, delta_euler, next_gripper]).astype(np.float32)  # shape (7,)

#     return delta


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


def apply_delta(position, orientation, delta):
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

    return np.concatenate([new_position, new_orientation])

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # # # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg(
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    # )
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

    # # mount
    table = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/thor_table.usd", scale=(1.5, 1.5, 1.0)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),

    )

    # cube = AssetBaseCfg(
    #     prim_path="/World/Cube",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd", scale=(0.05, 0.1, 0.05)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    # )
    if PICK_AND_PLACE:
        box = RigidObjectCfg(
                    prim_path="/World/Box",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.1, 0.1, 0.01),  # Dimensioni del cubo
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Proprietà fisiche
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Massa
                        collision_props=sim_utils.CollisionPropertiesCfg(),  # Proprietà di collisione
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),  # Colore rosso
                            metallic=0.0
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=TARGET_POS,  # Posizione iniziale
                        rot=(1.0, 0.0, 0.0, 0.0)  # Orientamento iniziale (quaternione)
                    ),
                )

    if CUBE_MULTICOLOR:
        object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(pos=OBJECT_POS, rot=[1, 0, 0, 0]), # must be within pos=[(0.2, 0.7)(-0.35, 0.35, 0.2, 0.2)]
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.8, 0.8, 0.8),
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                ),
            )
    else:
        object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.05),  # Dimensioni del cubo
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Proprietà fisiche
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Massa
                    collision_props=sim_utils.CollisionPropertiesCfg(),  # Proprietà di collisione
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),  # Colore rosso
                        metallic=0.0
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=OBJECT_POS,  # OVERWRITTEN BY THE COMMANDER
                    rot=(1.0, 0.0, 0.0, 0.0)  # Orientamento iniziale (quaternione)
                ),
            )
    
    # sugar_box = AssetBaseCfg(
    #     prim_path="/World/SugarBox",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd", scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.3, 0.0)),
    # )

    # craker_box = AssetBaseCfg(
    #     prim_path="/World/CrakerBox",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    # )

    # tomato_can = AssetBaseCfg(
    #     prim_path="/World/TomatoCan",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd", scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, -0.3, 0.0)),
    # )

    camera = CameraCfg(
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

    # banana = AssetBaseCfg(
    #     prim_path="/World/Banana",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(os.getcwd(), "isaac_ws/assets/011_banana.usd"), scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    # ) 

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="/World/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")


episode_path = "./isaac_ws/src/episode_0000.npy"
episode = np.load(episode_path, allow_pickle=True)
current_step_index = 0

def get_next_ground_truth_action():

    global current_step_index
    global episode

    # Controlla se ci sono ancora step disponibili
    if current_step_index >= len(episode):
        print("No more steps available in the episode. Closing the simulation.")
        simulation_app.close()
        exit(0)

    # Ottieni lo step corrente
    step = episode[current_step_index]

    # Incrementa il contatore per la prossima chiamata
    current_step_index += 1

    # Ritorna il campo "action"
    return step
    

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
    camera_positions = torch.tensor([CAMERA_POSITION], device=sim.device)
    camera_targets = torch.tensor([CAMERA_TARGET], device=sim.device)
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
        # [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # 10 cm x
        # [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # 10 cm y
        # [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # 10 cm z
        # [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # -10 cm x
        # [0.0, -0.1, 0.0, 0.0, 0.0, 0.0],  # -10 cm y
        # [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],  # -10 cm z
        # [0.0, 0.0, 0.0, np.pi/2, 0.0, 0.0],  # 90° x
        # [0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0], # -90° x
        # [0.0, 0.0, 0.0, 0.0, np.pi/2, 0.0],  # 90° y
        # [0.0, 0.0, 0.0, 0.0, -np.pi/2, 0.0], # -90° y
        # [0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2],  # 90° z
        # [0.0, 0.0, 0.0, 0.0, 0.0, -np.pi/2], # -90° z
        # [0.0, 0.0, 0.0, np.pi/2, np.pi/2, 0.0], # 90° x and y
        # [0.0, 0.0, 0.0, -np.pi/2, -np.pi/2, 0.0], # -90° x and y
        # [0.0, 0.0, 0.0, np.pi/2, np.pi/2, np.pi/2], # 90° y and z
        [0.0, 0.0, 0.0, -np.pi/2, -np.pi, -np.pi/2], # 90° x and z
        [0.0, 0.0, 0.0, np.pi/2, 0.0, np.pi/2], # -90° x and z
    ]

    ee_goal_deltas = torch.tensor(ee_goal_deltas, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    # init_ee_pos = [0.5, 0.0, 0.4, 0, 1, 0, 0]

    # init_ee_pos = [ 4.5067e-01, -1.8113e-05,  3.9752e-01, 0.0086, 0.9218, 0.0204, 0.3871]
    quat = Rotation.from_euler(EULER_NOTATION, [-3.4807291e-02, 6.9246048e-01, 3.1373665e+00]).as_quat()
    quat_isaac = scalar_last_to_first(quat)

    init_ee_pos = [3.7324736e-01,  1.6673391e-04,  4.3809804e-01] + quat_isaac

    

    # INIT JOINT POS [-2.9808e-05, -6.4019e-01, -8.9470e-06, -2.8247e+00,  2.5376e-05, 2.9797e+00,  7.4100e-01,  4.0000e-02,  4.0000e-02]

    ik_commands[:] = torch.tensor(init_ee_pos, device=sim.device) # TODO check if necessary

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

    # Create BANANA
    folder_path = os.getcwd()
    add_rigid_body_api(usd_file_path=os.path.join(folder_path, "isaac_ws/assets/011_banana.usd"))
    
    assign_material(object_path="/World/Table", material_path="/World/Table/Looks/Black")
    assign_material(object_path="/World/Cube", material_path="/World/Robot/panda_leftfinger/visuals/Looks/RubberRed")

    goal_reached = True

    while simulation_app.is_running():

        if count == 0:

            # Initialization - move to home position
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            
            joint_pos = torch.tensor([-2.9808e-05, -6.4019e-01, -8.9470e-06, -2.8247e+00,  2.5376e-05, 2.9797e+00,  7.4100e-01,  4.0000e-02,  4.0000e-02], device=sim.device)
            # print("\n\nINITIAL JOINT POS", joint_pos) # [ 0.0000, -0.5690,  0.0000, -2.8100,  0.0000,  3.0370,  0.7410,  0.0400, 0.0400]
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
            if OPENVLA_RESPONSE:
                # take image
                image_array = take_image(camera_index, camera, rep_writer)

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
                    continue
            # else:
                step = get_next_ground_truth_action()
                #res = step["action"]
                gt_state = step["state"]
                print("\nGT STATE: ", gt_state)

            
            #####################################

            ############
            delta = res[:6] #
            gripper_pos_des = torch.tensor([[res[6]*MAX_GRIPPER_POSE, res[6]*MAX_GRIPPER_POSE]], device=sim.device) #

            euler_orientation_pose = Rotation.from_quat(scalar_first_to_last(ee_quat_b.cpu().numpy().squeeze(0))).as_euler(EULER_NOTATION)
            ee_pose_eul = np.concatenate([ee_pos_b.cpu().numpy().squeeze(0), euler_orientation_pose, gripper_pos_des.cpu().numpy().squeeze(0)])
            
            ee_goal = apply_delta(ee_pos_b.cpu().numpy().squeeze(0), ee_quat_b.cpu().numpy().squeeze(0), delta) #

            euler_orientation_goal = Rotation.from_quat(scalar_first_to_last(ee_goal[3:7])).as_euler(EULER_NOTATION)
            ee_goal_eul = np.concatenate([ee_goal[:3], euler_orientation_goal, gripper_pos_des.cpu().numpy().squeeze(0)])
            recomputed_delta = compute_delta(ee_pose_eul, ee_goal_eul)
            print("CURRENT STATE: ", ee_pose_eul, "\n")
            print("\n\nDELTA position: ", delta[:3])
            print("RECOMPUTED DELTA position: ", recomputed_delta[:3], "\n")
            print("DELTA orientation: ", delta[3:6])
            print("RECOMPUTED DELTA orientation: ", recomputed_delta[3:6], "\n\n")
            ###########

            # print(f"✅ Nuovo goal: {current_goal_idx}") #
            # delta = ee_goal_deltas[current_goal_idx] #

            # euler_orientation_pose = Rotation.from_quat(scalar_first_to_last(ee_quat_b.cpu().numpy().squeeze(0))).as_euler(EULER_NOTATION)
            # ee_pose_eul = np.concatenate([ee_pos_b.cpu().numpy().squeeze(0), euler_orientation_pose, gripper_pos_des.cpu().numpy().squeeze(0)])
            
            # ee_goal = apply_delta(ee_pos_b.cpu().numpy().squeeze(0), ee_quat_b.cpu().numpy().squeeze(0), delta.cpu().numpy()) #

            # euler_orientation_goal = Rotation.from_quat(scalar_first_to_last(ee_goal[3:7])).as_euler(EULER_NOTATION)
            # ee_goal_eul = np.concatenate([ee_goal[:3], euler_orientation_goal, gripper_pos_des.cpu().numpy().squeeze(0)])
            # recomputed_delta = compute_delta(ee_pose_eul, ee_goal_eul)
            # print("\n\nDELTA position: ", delta[:3])
            # print("RECOMPUTED DELTA position: ", recomputed_delta[:3], "\n")
            # print("DELTA orientation: ", delta[3:6])
            # print("RECOMPUTED DELTA orientation: ", recomputed_delta[3:6], "\n\n")
            # #############

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


        # update marker positions 
        if VISUALIZE_MARKERS:
            ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

     

def check_goal_reached(ik_commands, ee_pose_w, position_threshold=0.00005, angle_threshold=0.001):
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
