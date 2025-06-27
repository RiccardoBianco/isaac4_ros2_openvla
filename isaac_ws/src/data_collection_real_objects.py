
# * ###########################################################################
# * Data Collection Script for Real Objects Environment
# * ###########################################################################

OBJECT_TO_PICK = "object1"  # The object to pick

RANDOM_CAMERA = True
RANDOM_OBJECT = True
RANDOM_TARGET = False

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
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

# ^ Isaac Sim imports

import gymnasium as gym
import torch
from collections.abc import Sequence
from PIL import Image

import numpy as np
import os
import shutil
from datetime import datetime
import json

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
from isaaclab_tasks.manager_based.manipulation.lift.env_data_real_objects import LiftEnvCfg
import isaaclab.sim as sim_utils
import omni.replicator.core as rep
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import omni.usd
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import convert_dict_to_backend

from pxr import UsdGeom, Usd, UsdShade

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.spawners import UsdFileCfg
from pxr import Usd, UsdPhysics, UsdGeom, UsdShade
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


# ^ Configuration for the environment

if RANDOM_OBJECT:
    object_random_range = (-0.1, 0.1)
else:
    object_random_range = (0.0, 0.0)

objects = {
    "cracker_box": {
        "path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "init_pos": [0.45, 0.05, 0.0], # [0.65, 0.10, 0.0]
        "init_rot": [-0.7071, 0.7071, 0, 0.0],  # Quaternions for rotation
        "specific_offset_ee": 0.09,  # Specific offset for the object
        "offset_on_goal": [-0.15, 0.0, 0.0],
        "x_range": object_random_range,  # APPLIED TO EVERY OBJECT
        "y_range": object_random_range,
        "z_range": (0.0, 0.0),
        "scale": (0.6, 0.6, 0.6)  # Scale the object to fit better in the scene
    },
    "tomato_can": {
        "path": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd", # -> third object
        "init_pos": [0.45, 0.30, 0.0],
        "init_rot": [-0.7071, 0.7071, 0.0, 0.0],  # Quaternions for rotation
        "specific_offset_ee": 0.04,
        "offset_on_goal": [0.15, 0.0, 0.0],
        "x_range": object_random_range,  # APPLIED TO EVERY OBJECT
        "y_range": object_random_range,
        "z_range": (0.0, 0.0),
        "scale": (0.8, 0.8, 0.8)  # Scale the object to fit better in the scene
    },
    "pencil_case": {
        "path":  os.path.abspath("./isaac_ws/assets/Pinwheel_Pencil_Case/meshes/_converted/model_obj.usd"),
        "init_pos": [0.70, 0.30, 0.0],
        "init_rot": [0.9659, 0, 0, 0.2588],  # Quaternions for rotation
        "specific_offset_ee": 0.01,
        "offset_on_goal": [0.0, 0.0, 0.0],
        "x_range": (-0.05, 0.05),  # APPLIED TO EVERY OBJECT
        "y_range": (0.0, 0.1),
        "z_range": (0.0, 0.0),
        "scale": (0.7, 0.7, 0.7)  # Scale the object to fit better in the scene
    },
    "thomas_train": {
        "path": os.path.abspath("./isaac_ws/assets/Thomas_Friends_Woodan_Railway_Henry/meshes/_converted/model_obj.usd"),
        "init_pos": [0.40, -0.15, 0.0],
        "init_rot": [0.9659, 0, 0, -0.2588],  # Quaternions for rotation
        "specific_offset_ee": 0.0,
        "offset_on_goal": [0.0, 0.0, 0.0],
        "x_range": (-0.1, 0.1),  # APPLIED TO EVERY OBJECT
        "y_range": (0.0, 0.0),
        "z_range": (0.0, 0.0),
        "scale": (1.1, 1.1, 1.1)  # Scale the object to fit better in the scene
    },
    "dog": {
        "path": os.path.abspath("./isaac_ws/assets/Dog/meshes/_converted/model_obj.usd"),
        "init_pos": [0.65, -0.15, 0.0],
        "init_rot": [0.643, 0, 0, 0.766],  # Quaternions for rotation
        "specific_offset_ee": 0.02,
        "offset_on_goal": [0.0, 0.0, 0.0],
        "x_range": (-0.1, 0.1),  # APPLIED TO EVERY OBJECT
        "y_range": (0.0, 0.0),
        "z_range": (0.0, 0.0),
        "scale": (0.6, 0.6, 0.6)  # Scale the object to fit better in the scene
    },
    "school_bus": {
        "path": os.path.abspath("./isaac_ws/assets/SCHOOL_BUS/meshes/_converted/model_obj.usd"), # -> secondo oggetto
        "init_pos": [0.70, 0.10, 0.0],
        "init_rot": [0.7071, 0, 0, 0.7071],  # Quaternions for rotation
        "specific_offset_ee": 0.0,
        "offset_on_goal": [0.0, 0.0, 0.0],
        "x_range": object_random_range,  # APPLIED TO EVERY OBJECT
        "y_range": object_random_range,
        "z_range": (0.0, 0.0),
        "scale": (1.1, 1.1, 1.1)  # Scale the object to fit better in the scene
    },
}

OBJECTS = {
    "object1": "cracker_box",
    "object2": "tomato_can",
    "object3": "school_bus",
    "object4": "dog",
    "object5": "pencil_case",
    "object6": "thomas_train",
}

SAMPLES_PER_OBJECT = 700 # Number of samples per object

INIT_TARGET_POS = [0.50, -0.35, 0.0]  # Z must be 0 in OpenVLA inference script
INIT_ROBOT_POSE = [0.4, 0.0, 0.35, 0.0, 1.0, 0.0, 0.0]

OPENVLA_INSTRUCTION = f"Pick the {OBJECTS[OBJECT_TO_PICK]} and place it in the brown box. \n" 

SEED = 50


SAVE = True

CAMERA_HEIGHT = 1920
CAMERA_WIDTH = 1920
OPENVLA_CAMERA_HEIGHT = 256
OPENVLA_CAMERA_WIDTH = 256

CAMERA_POSITION = [1.0, -0.18, 0.6]
CAMERA_TARGET = [0.4, 0.0, 0.0]


# OFFSET_EE = OBJECT_SIZE[2] / 2 - 0.01 + 0.107 # 0.01 settato a mano a naso
SPECIFIC_OFFSET_EE = objects[OBJECTS[OBJECT_TO_PICK]]["specific_offset_ee"]  # Specific offset for the object
NATURAL_OFFSET_EE = 0.107  # Natural offset for the end effector
OFFSET_EE = SPECIFIC_OFFSET_EE + NATURAL_OFFSET_EE # 0.01 settato a mano a naso

ABOVE_TARGET_OFFSET = 0.15
ABOVE_OBJECT_OFFSET = 0.15

CAMERA_X_RANGE = (-0.05, 0.05)
CAMERA_Y_RANGE = (-0.05, 0.05)
CAMERA_Z_RANGE = (-0.05, 0.05)


if RANDOM_TARGET: # ABSOLUTE POSITION
    TARGET_X_RANGE = (-0.2 + INIT_TARGET_POS[0], 0.3 + INIT_TARGET_POS[0])
    TARGET_Y_RANGE = (-0.3 + INIT_TARGET_POS[1] , 0.3 + INIT_TARGET_POS[1])
    TARGET_Z_RANGE = (0.0 + INIT_TARGET_POS[2], 0.0 + INIT_TARGET_POS[2])
else:
    TARGET_X_RANGE = (INIT_TARGET_POS[0], INIT_TARGET_POS[0])
    TARGET_Y_RANGE = (INIT_TARGET_POS[1], INIT_TARGET_POS[1])
    TARGET_Z_RANGE = (INIT_TARGET_POS[2], INIT_TARGET_POS[2])

EULER_NOTATION = "zyx" 


def scalar_first_to_last(q):
    """Convert quaternion from scalar-first to scalar-last format."""
    w, x, y, z = q
    return [x, y, z, w]


def scalar_last_to_first(q):
    """Convert quaternion from scalar-last to scalar-first format."""
    x, y, z, w = q
    return [w, x, y, z]

import numpy as np
from scipy.spatial.transform import Rotation

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


        self.scene.object1 = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=objects[OBJECTS["object1"]]["init_pos"], rot=objects[OBJECTS["object1"]]["init_rot"]),
                spawn=UsdFileCfg(
                    usd_path=objects[OBJECTS["object1"]]["path"],
                    scale=objects[OBJECTS["object1"]]["scale"],
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
            
        self.scene.object2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=objects[OBJECTS["object2"]]["init_pos"], rot=objects[OBJECTS["object2"]]["init_rot"]),
            spawn=UsdFileCfg(
                usd_path=objects[OBJECTS["object2"]]["path"],
                scale=objects[OBJECTS["object2"]]["scale"],
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

        self.scene.object3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=objects[OBJECTS["object3"]]["init_pos"], rot=objects[OBJECTS["object3"]]["init_rot"]),
            spawn=UsdFileCfg(
                usd_path=objects[OBJECTS["object3"]]["path"],
                scale=objects[OBJECTS["object3"]]["scale"],
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

        self.scene.object4 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object4",
            init_state=RigidObjectCfg.InitialStateCfg(pos=objects[OBJECTS["object4"]]["init_pos"], rot=objects[OBJECTS["object4"]]["init_rot"]),
            spawn=UsdFileCfg(
                usd_path=objects[OBJECTS["object4"]]["path"],
                scale=objects[OBJECTS["object4"]]["scale"],
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
        self.scene.object5 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object5",
            init_state=RigidObjectCfg.InitialStateCfg(pos=objects[OBJECTS["object5"]]["init_pos"], rot=objects[OBJECTS["object5"]]["init_rot"]),
            spawn=UsdFileCfg(
                usd_path=objects[OBJECTS["object5"]]["path"],
                scale=objects[OBJECTS["object5"]]["scale"],
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

        self.scene.object6 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object6",
            init_state=RigidObjectCfg.InitialStateCfg(pos=objects[OBJECTS["object6"]]["init_pos"], rot=objects[OBJECTS["object6"]]["init_rot"]),
            spawn=UsdFileCfg(
                usd_path=objects[OBJECTS["object6"]]["path"],
                scale=objects[OBJECTS["object6"]]["scale"],
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

        self.events.reset_object1_position.params["pose_range"] = {"x": objects[OBJECTS["object1"]]["x_range"], "y": objects[OBJECTS["object1"]]["y_range"], "z": objects[OBJECTS["object1"]]["z_range"]}
        self.events.reset_object2_position.params["pose_range"] = {"x": objects[OBJECTS["object2"]]["x_range"], "y": objects[OBJECTS["object2"]]["y_range"], "z": objects[OBJECTS["object2"]]["z_range"]}
        self.events.reset_object3_position.params["pose_range"] = {"x": objects[OBJECTS["object3"]]["x_range"], "y": objects[OBJECTS["object3"]]["y_range"], "z": objects[OBJECTS["object3"]]["z_range"]}
        self.events.reset_object4_position.params["pose_range"] = {"x": objects[OBJECTS["object4"]]["x_range"], "y": objects[OBJECTS["object4"]]["y_range"], "z": objects[OBJECTS["object4"]]["z_range"]}
        self.events.reset_object5_position.params["pose_range"] = {"x": objects[OBJECTS["object5"]]["x_range"], "y": objects[OBJECTS["object5"]]["y_range"], "z": objects[OBJECTS["object5"]]["z_range"]}
        self.events.reset_object6_position.params["pose_range"] = {"x": objects[OBJECTS["object6"]]["x_range"], "y": objects[OBJECTS["object6"]]["y_range"], "z": objects[OBJECTS["object6"]]["z_range"]}


        # Base of the container
        self.scene.box = RigidObjectCfg(
            prim_path="/World/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.2, 0.0025),  # Larger base: 50x20 cm
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.1, 0.05),
                    metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=INIT_TARGET_POS,
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )

        # Left wall
        self.scene.box_wall_left = RigidObjectCfg(
            prim_path="/World/BoxWall/WallLeft",
            spawn=sim_utils.CuboidCfg(
                size=(0.005, 0.2, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.1, 0.05),
                    metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(INIT_TARGET_POS[0] - 0.25, INIT_TARGET_POS[1], INIT_TARGET_POS[2]),  # Half of the new width
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )

        # Right wall
        self.scene.box_wall_right = RigidObjectCfg(
            prim_path="/World/BoxWall/WallRight",
            spawn=sim_utils.CuboidCfg(
                size=(0.005, 0.2, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.1, 0.05),
                    metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(INIT_TARGET_POS[0] + 0.25, INIT_TARGET_POS[1], INIT_TARGET_POS[2]),
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )

        # Front wall
        self.scene.box_wall_front = RigidObjectCfg(
            prim_path="/World/BoxWall/WallFront",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.005, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.1, 0.05),
                    metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(INIT_TARGET_POS[0], INIT_TARGET_POS[1] - 0.0975, INIT_TARGET_POS[2]),  # Half of the new depth
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )

        # Back wall
        self.scene.box_wall_back = RigidObjectCfg(
            prim_path="/World/BoxWall/WallBack",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.005, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=False
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.1, 0.05),
                    metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(INIT_TARGET_POS[0], INIT_TARGET_POS[1] + 0.0975, INIT_TARGET_POS[2]),
                rot=(1.0, 0.0, 0.0, 0.0)
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

        self.scene.wrist_camera = CameraCfg(
            prim_path="/World/envs/env_0/Robot/panda_hand/WristCameraSensor",
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
             offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707), convention="ros"),
        )

class GripperState:
    """States for the gripper."""
    OPEN = 1.0
    CLOSE = -1.0

class SmState:
    """States for the pick state machine."""

    ROBOT_INIT_POSE = 0
    APPROACH_ABOVE_OBJECT = 1
    APPROACH_OBJECT = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4
    PLACE_ABOVE_GOAL = 5
    PLACE_ON_GOAL = 6
    RELEASE_OBJECT = 7
    MOVE_ABOVE_GOAL = 8
    TERMINAL_STATE = 9 


class SmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    ROBOT_INIT_POSE = 1.0
    APPROACH_ABOVE_OBJECT = 0.8
    APPROACH_OBJECT = 0.6
    GRASP_OBJECT = 0.3
    LIFT_OBJECT = 0.5
    PLACE_ABOVE_GOAL = 1.0 
    PLACE_ON_GOAL = 0.6
    RELEASE_OBJECT = 0.4
    MOVE_ABOVE_GOAL = 0.5
    TERMINAL_STATE = 1.0 



def check_pose_reached(current_state, desired_state, position_threshold, angle_threshold):
    """
        Check if the current position is within the threshold of the desired position.
        Returns True if the goal is reached, False otherwise.
        state: [x, y, z, qw, qx, qy, qz]

    """
    position_error = torch.norm(current_state[:, :3] - desired_state[:, :3], dim=1)

    quat_dot = torch.abs(torch.sum(current_state[:, 3:7] * desired_state[:, 3:7], dim=1))  # q1 · q2
    quat_dot = torch.clamp(quat_dot, -1.0, 1.0)  # clamp for numerical stability
    angle_error = 2 * torch.acos(quat_dot)


    if position_error.item() < position_threshold and angle_error.item() < angle_threshold:
        angle_deg = np.degrees(angle_error.item())
        return True
    return False

class StateMachine:
    """A simple state machine to pick and lift an object."""

    def __init__(self, dt: float, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.sm_wait_time = 0.0
        self.device = device

        self.sm_dt = torch.tensor(self.dt, device=self.device)
        self.sm_state = SmState.ROBOT_INIT_POSE
        self.sm_wait_time = SmWaitTime.ROBOT_INIT_POSE
    
    def reset(self):
        """Reset the state machine."""
        self.sm_wait_time = 0.0
        self.sm_state = SmState.ROBOT_INIT_POSE

    def get_des_pose(self, ee_current_pose, init_pose, initial_object_pose, target_pose, offset_ee):
        """
        Get the desired end-effector pose based on the current state and target.
        """
        
        init_pose = init_pose.clone()
        initial_object_pose = initial_object_pose.clone()
        target_pose = target_pose.clone()
        init_pose[:, 2] += NATURAL_OFFSET_EE
        initial_object_pose[:, 2] += offset_ee
        target_pose[:, 2] += offset_ee 

        # around x of 180 ->  [-x, w, -z, y]
        # around y of 180 -> [-y, z, w, -x]
        quat_down = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=initial_object_pose.device)  # shape [1, 4]

        initial_object_pose[:, 3:7] = quat_down
        target_pose[:, 3:7] = quat_down

        above_initial_object_pose = initial_object_pose.clone()
        above_target_pose = target_pose.clone()
        above_target_pose[:, 2] += ABOVE_TARGET_OFFSET
        above_initial_object_pose[:, 2] += ABOVE_OBJECT_OFFSET


        if self.sm_state == SmState.ROBOT_INIT_POSE:
            des_ee_pose = init_pose
            gripper_state = GripperState.OPEN
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) and self.sm_wait_time >= SmWaitTime.ROBOT_INIT_POSE:
                self.sm_state = SmState.APPROACH_ABOVE_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.APPROACH_ABOVE_OBJECT:
            des_ee_pose = above_initial_object_pose
            gripper_state = GripperState.OPEN
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.APPROACH_ABOVE_OBJECT:
                self.sm_state = SmState.APPROACH_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.APPROACH_OBJECT:
            des_ee_pose = initial_object_pose
            gripper_state = GripperState.OPEN
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.APPROACH_OBJECT:
                self.sm_state = SmState.GRASP_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.GRASP_OBJECT:
            des_ee_pose = initial_object_pose
            gripper_state = GripperState.CLOSE
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) and self.sm_wait_time >= SmWaitTime.GRASP_OBJECT:
                self.sm_state = SmState.LIFT_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.LIFT_OBJECT:
            des_ee_pose = above_initial_object_pose
            gripper_state = GripperState.CLOSE
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.LIFT_OBJECT:
                self.sm_state = SmState.PLACE_ABOVE_GOAL
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.PLACE_ABOVE_GOAL:
            des_ee_pose = above_target_pose
            gripper_state = GripperState.CLOSE
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.PLACE_ABOVE_GOAL:
                self.sm_state = SmState.PLACE_ON_GOAL
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.PLACE_ON_GOAL:
            des_ee_pose = target_pose
            gripper_state = GripperState.CLOSE
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.PLACE_ON_GOAL:
                self.sm_state = SmState.RELEASE_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.RELEASE_OBJECT:
            des_ee_pose = target_pose
            gripper_state = GripperState.OPEN
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.02, 0.02) and self.sm_wait_time >= SmWaitTime.RELEASE_OBJECT:
                self.sm_state = SmState.MOVE_ABOVE_GOAL
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.MOVE_ABOVE_GOAL:
            des_ee_pose = above_target_pose
            gripper_state = GripperState.OPEN
            if check_pose_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.MOVE_ABOVE_GOAL:
                self.sm_state = SmState.TERMINAL_STATE
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.TERMINAL_STATE:
            des_ee_pose = above_target_pose
            gripper_state = GripperState.OPEN

        self.sm_wait_time += self.sm_dt.item() 

        des_gripper_state = torch.tensor([gripper_state], device=self.device)
        return torch.cat([des_ee_pose, des_gripper_state.unsqueeze(-1)], dim=-1)



def assign_material(object_path, material_path):
    """Assign a material to a USD object."""
    stage = omni.usd.get_context().get_stage()

    # Take the prim of the object
    object_prim = stage.GetPrimAtPath(object_path)

    # Take the existing material
    material_prim = stage.GetPrimAtPath(material_path)

    if object_prim and material_prim:
        material = UsdShade.Material(material_prim)
        UsdShade.MaterialBindingAPI(object_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        print("Material assigned successfully to ", object_path)
    else:
        print("Error: Primitive or material not found.")

def take_image(camera_index, camera, camera_type, sim_num):
    """
    Take an image from the camera and save it using the replicator writer.
    Args:
        camera_index: Index of the camera to use.
        camera: The camera object.
        rep_writer: The replicator writer object.
    """
    if args_cli.save:
        # Save images from camera at camera_index
        # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
        single_cam_data = convert_dict_to_backend(
            {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        )

        # Extract the image data (assuming 'rgb' is the key)
        image_data = single_cam_data.get('rgb')
        
        if image_data is not None:
            image_data = image_data.astype(np.uint8)
            high_res_image = Image.fromarray(image_data)

            low_res_image = high_res_image.resize((OPENVLA_CAMERA_HEIGHT, OPENVLA_CAMERA_WIDTH), Image.BICUBIC)

            if sim_num <= 10:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                file_name = f"{camera_type}_{timestamp}.png"

                folder_dir = f"./isaac_ws/src/output/camera/simulation_{sim_num}"
                os.makedirs(folder_dir, exist_ok=True)
                file_path = os.path.join(folder_dir, file_name)

                # Save image as RGB PNG
                low_res_image.save(file_path)
                
            return np.array(low_res_image)

    return None 

def get_current_state(robot):
    """Get the current state of the robot's end-effector in the world frame."""
    joint_pos = robot.data.joint_pos.clone()
    ee_pose_w = robot.data.body_state_w[:, 8, 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]

    # Position of the end-effector relative to the root
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    gripper_state = joint_pos[:, -1].unsqueeze(-1)  # shape: (1, 1)

    quat_np = scalar_first_to_last(ee_quat_b[0].cpu().numpy())  # shape (4,)
    R_euler_np = Rotation.from_quat(quat_np).as_euler(EULER_NOTATION) # shape (3,)
    R_euler = torch.tensor(R_euler_np, device=ee_pos_b.device).unsqueeze(0)  # shape (1, 3)
    pad = torch.tensor([[0.0]], device=ee_pos_b.device)
    current_state = torch.cat([ee_pos_b, R_euler, pad, gripper_state], dim=-1)  # shape: (1, 8)
    return current_state
                

def save_episode_stepwise(episode_steps, save_dir="isaac_ws/src/output/episodes"):
    """
    Save a list of timestep dictionaries into a progressively numbered .npy file.
    
    Args:
        episode_steps (List[dict]): Each step must include keys like "state", "action", "image", etc.
        save_dir (str): Directory where .npy episodes are stored.
    """
    # folder_name = "episodes"
    # save_task_dir = os.path.join(SAVE_DATASET_DIR, folder_name)

    # Check on the final state of the object wrt the target
    distance_object_target = np.linalg.norm(episode_steps[-1]["object_pose"][:, :3] - episode_steps[-1]["target_pose"][:, :3])
    if distance_object_target > 0.08:
        print("Episode not saved: object is too far from the target.")
        return 0
    
    distance_object_target_start = np.linalg.norm(episode_steps[0]["object_pose"][:, :3] - episode_steps[0]["target_pose"][:, :3])
    if distance_object_target_start < 0.10:
        print("Episode not saved: object is too close to the target at the start.")
        return 0
            
    for i in range(len(episode_steps)-1):
        episode_steps[i]["action"]= compute_delta(episode_steps[i]["state"], episode_steps[i+1]["state"])
    
    episode_steps[-1]["action"] = compute_delta(episode_steps[-1]["state"], episode_steps[-1]["state"])

    os.makedirs(save_dir, exist_ok=True)

    # Get next available episode number
    existing = [f for f in os.listdir(save_dir) if f.startswith("episode_") and f.endswith(".npy")]
    episode_nums = [int(f.split("_")[1].split(".")[0]) for f in existing if "_" in f]
    next_num = max(episode_nums) + 1 if episode_nums else 0
    if next_num > 6000: 
        simulation_app.close()
        print("Maximum number of episodes reached. Exiting...")
        exit(0)

    filename = f"episode_{next_num:04d}.npy"
    filepath = os.path.join(save_dir, filename)

    # Save
    np.save(filepath, episode_steps, allow_pickle=True)
    print(f"✅ Saved episode with {len(episode_steps)} steps to {filepath}")
    return next_num

    
def is_significant_change(delta, grasped_bool_vec, pos_th, rot_th, sm_state):
    """
    Check if the change in state is significant enough to warrant a new episode.
    """
    grasp_start_saved, grasp_end_saved, release_start_saved, release_end_saved = grasped_bool_vec

    updated_vec = grasped_bool_vec
    if sm_state == SmState.GRASP_OBJECT:
        if not grasp_start_saved:
            updated_vec = [True, grasp_end_saved, release_start_saved, release_end_saved]
            return True, updated_vec
        else:
            return False, updated_vec

    elif sm_state == SmState.LIFT_OBJECT:
        if grasp_start_saved and not grasp_end_saved:
            updated_vec = [grasp_start_saved, True, release_start_saved, release_end_saved]
            return True, updated_vec
        else:
            return False, updated_vec

    elif sm_state == SmState.RELEASE_OBJECT:
        if not release_start_saved:
            updated_vec = [grasp_start_saved, grasp_end_saved, True, release_end_saved]
            return True, updated_vec
        else:
            return False, updated_vec

    elif sm_state == SmState.MOVE_ABOVE_GOAL:
        if release_start_saved and not release_end_saved:
            updated_vec = [grasp_start_saved, grasp_end_saved, release_start_saved, True]
            return True, updated_vec


    dx, dy, dz, droll, dpitch, dyaw, gripper = delta
    pos_change = np.linalg.norm([dx, dy, dz]) > pos_th
    rot_change = np.linalg.norm([droll, dpitch, dyaw]) > rot_th
    return pos_change or rot_change, updated_vec


def get_current_ee(robot):
    """Get the current end-effector pose in the body frame."""
    ee_pose_w = robot.data.body_state_w[:, 8, 0:7]
    root_pose_w = robot.data.root_state_w[:, 0:7]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    current_state = torch.cat([ee_pos_b, ee_quat_b], dim=-1) # (1, 7)
    return current_state


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
    camera_pose_to_save = torch.cat([camera_position, camera_target], dim=-1)
    return camera_pose_to_save


def set_new_target_pose(env):
    """ Set a new target pose for the box in the simulation environment."""
    goal_pose = env.unwrapped.command_manager.get_command("target_pose")
    new_pos = goal_pose[..., :3].clone()
    new_pos[..., 2] = 0.0
    new_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=new_pos.device).expand(new_pos.shape[0], 4)

    root_state = torch.zeros((env.unwrapped.num_envs, 13), device=env.unwrapped.device)
    root_state[:, 0:3] = new_pos
    root_state[:, 3:7] = new_rot
    
    # Write the new pose to the simulation
    env.unwrapped.scene["box"].write_root_state_to_sim(root_state)

def save_config_file():
    """
    Create a configuration file for the current run.
    """
    ### Create the config file with the Global Parameters defined in the current run
    config = {
        "OPENVLA_INSTRUCTION": OPENVLA_INSTRUCTION,
        "SEED": SEED,
        "RANDOM_CAMERA": RANDOM_CAMERA,
        "RANDOM_OBJECT": RANDOM_OBJECT,
        "RANDOM_TARGET": RANDOM_TARGET,
        "SAVE": SAVE,
        "CAMERA_HEIGHT": CAMERA_HEIGHT,
        "CAMERA_WIDTH": CAMERA_WIDTH,
        "OPENVLA_CAMERA_HEIGHT": OPENVLA_CAMERA_HEIGHT,
        "OPENVLA_CAMERA_WIDTH": OPENVLA_CAMERA_WIDTH,
        "CAMERA_POSITION": CAMERA_POSITION,
        "CAMERA_TARGET": CAMERA_TARGET,
        "CAMERA_X_RANGE": CAMERA_X_RANGE,
        "CAMERA_Y_RANGE": CAMERA_Y_RANGE,
        "CAMERA_Z_RANGE": CAMERA_Z_RANGE,
        "OFFSET_EE": OFFSET_EE,
        "ABOVE_TARGET_OFFSET": ABOVE_TARGET_OFFSET,
        "ABOVE_OBJECT_OFFSET": ABOVE_OBJECT_OFFSET,
        "INIT_TARGET_POS": INIT_TARGET_POS,
        "INIT_ROBOT_POSE": INIT_ROBOT_POSE,
        "TARGET_X_RANGE": TARGET_X_RANGE,
        "TARGET_Y_RANGE": TARGET_Y_RANGE,
        "TARGET_Z_RANGE": TARGET_Z_RANGE,
        "CAMERA_X_RANGE": CAMERA_X_RANGE,
        "CAMERA_Y_RANGE": CAMERA_Y_RANGE,
        "CAMERA_Z_RANGE": CAMERA_Z_RANGE,
        "EULER_NOTATION": EULER_NOTATION,
        "OBJECT_TO_PICK": OBJECT_TO_PICK,
        "OBJECTS": OBJECTS,
        "objects_dict": objects,
    }

    # Create output directory
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(config_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"config_{timestamp}.json"
    config_file_path = os.path.join(config_dir, filename)

    # Write to JSON
    with open(config_file_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    print(f"✅ Config saved to: {config_file_path}")

def check_valid_task(env):
    """
    Check if the current task is valid.
    """
    object_pos = env.unwrapped.scene["object1"].data.root_state_w[:, :3].clone()
    object2_pos = env.unwrapped.scene["object2"].data.root_state_w[:, :3].clone()
    object3_pos = env.unwrapped.scene["object3"].data.root_state_w[:, :3].clone()
    target_pos = env.unwrapped.scene["box"].data.root_state_w[:, :3].clone()
    distance_object_object2 = torch.norm(object_pos - object2_pos, dim=1)
    distance_object_object3 = torch.norm(object_pos - object3_pos, dim=1)
    distance_object_target = torch.norm(object_pos - target_pos, dim=1)
    distance_object2_target = torch.norm(object2_pos - target_pos, dim=1)
    distance_object3_target = torch.norm(object3_pos - target_pos, dim=1)

    if OBJECT_TO_PICK == "object1":
        distance_main_object = distance_object_target
    elif OBJECT_TO_PICK == "object2":
        distance_main_object = distance_object2_target
    elif OBJECT_TO_PICK == "object3":
        distance_main_object = distance_object3_target


    if distance_object_object2.item() < 0.065 or distance_object_object3.item() < 0.065:
        print("Object too close to another object.")
        return False
    if distance_main_object.item() < 0.08: #or distance_object2_target.item() < 0.08 or distance_object3_target.item() < 0.08:
        print("Object too close to the target.")
        return False
    return True

def add_rigid_body_live(prim_path):
    """Add RigidBodyAPI and CollisionAPI to a prim in the USD stage."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print(f"❌ Prim not found at {prim_path}")
        return

    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
        print(f"✅ RigidBodyAPI applied to {prim_path}")
    
    UsdPhysics.CollisionAPI.Apply(prim)
    print(f"✅ CollisionAPI applied to {prim_path}")

def run_simulator(env, env_cfg, args_cli):
    """
    Main function that runs the Isaac Sim simulator, creates the environment, the state machine, etc.
    """

    save_config_file()

    camera = env.unwrapped.scene["camera"]
    wrist_camera = env.unwrapped.scene["wrist_camera"]
    robot = env.unwrapped.scene["robot"]

    print("\n\nRUNNING SIMULATOR!\n\n")

    episode_data = []

    # Set the camera position and target (wrist camera is already attached to the robot in the config)
    camera_positions = torch.tensor([CAMERA_POSITION], device=env.unwrapped.device)
    camera_targets = torch.tensor([CAMERA_TARGET], device=env.unwrapped.device)
    camera_pose_to_save = torch.cat((camera_positions, camera_targets), dim=-1) # (1, 6)
    camera.set_world_poses_from_view(camera_positions, camera_targets)
    camera_index = args_cli.camera_id

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0

    # create state machine
    sm = StateMachine(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.device)

    assign_material(object_path="/World/Table", material_path="/World/Table/Looks/Black")

    count = 0
    task_count = 0
    restarted = True
    openvla_instruction = OPENVLA_INSTRUCTION
    offset_ee = OFFSET_EE
    object_to_pick = OBJECT_TO_PICK
    next_num = 0

    robot_init_pose = torch.tensor(INIT_ROBOT_POSE, device=env.unwrapped.device).unsqueeze(0) # (1, 7) ->  [x, y, z, qw, qx, qy, qz] # towards down
    robot_init_pose[:, 2] -= NATURAL_OFFSET_EE

    grasped_bool_vec= (False, False, False, False)
    valid_task = True
    while simulation_app.is_running():
        
    
        if valid_task and task_count!=0 and not restarted and sm.sm_state != SmState.ROBOT_INIT_POSE and sm.sm_state != SmState.TERMINAL_STATE:
            if SAVE:

                current_state = get_current_state(robot) # shape: (1, 8) # x, y, z, roll, pitch, yaw, pad, gripper

                should_save = False

                if len(episode_data) == 0:
                    should_save = True
                else:   
                    delta_steps = compute_delta(episode_data[-1]["state"], current_state.clone().cpu().squeeze().numpy().astype(np.float32)) 

                    should_save, grasped_bool_vec = is_significant_change(delta_steps,grasped_bool_vec, pos_th=0.08, rot_th=0.1, sm_state=sm.sm_state )

                if should_save:
                    table_image_array = take_image(camera_index, camera, camera_type="table", sim_num=task_count-1)
                    wrist_image_array = take_image(camera_index, wrist_camera, camera_type="wrist", sim_num=task_count-1)
                    step_data = {
                        "state": current_state.clone().cpu().squeeze().numpy().astype(np.float32),
                        "image": table_image_array.astype(np.uint8),
                        "wrist_image": wrist_image_array.astype(np.uint8),
                        "language_instruction": openvla_instruction,
                        "object_pose": current_object_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "target_pose": target_pose.clone().cpu().numpy().astype(np.float32),
                        "camera_pose": camera_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "initial_object_pose": initial_object_pose.clone().cpu().numpy().astype(np.float32),
                        "object1_pose": current_object1_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "object2_pose": current_object2_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "object3_pose": current_object3_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "object4_pose": current_object4_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "object5_pose": current_object5_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "object6_pose": current_object6_pose_to_save.clone().cpu().numpy().astype(np.float32),
                    }

                    # Include additional object poses if using multiple cubes
                    episode_data.append(step_data)
                else:
                    # print("Not saving step")
                    pass


        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            if count == 0:
                # valid_task = check_valid_task(env)
                valid_task = True # NOTE modified
                
            current_object_pose_to_save = env.unwrapped.scene[object_to_pick].data.root_state_w[:, 0:7].clone()
            current_object1_pose_to_save = env.unwrapped.scene["object1"].data.root_state_w[:, :7].clone()
            current_object2_pose_to_save = env.unwrapped.scene["object2"].data.root_state_w[:, :7].clone()
            current_object3_pose_to_save = env.unwrapped.scene["object3"].data.root_state_w[:, :7].clone()
            current_object4_pose_to_save = env.unwrapped.scene["object4"].data.root_state_w[:, :7].clone()
            current_object5_pose_to_save = env.unwrapped.scene["object4"].data.root_state_w[:, :7].clone()
            current_object6_pose_to_save = env.unwrapped.scene["object4"].data.root_state_w[:, :7].clone()
            if restarted == True:
                initial_object_pose = env.unwrapped.scene[object_to_pick].data.root_state_w[:, :7].clone()

                
                target_pose = env.unwrapped.command_manager.get_command("target_pose")[..., :7].clone()
                target_pose[:, 0] += objects[OBJECTS[object_to_pick]]["offset_on_goal"][0]
                target_pose[:, 1] += objects[OBJECTS[object_to_pick]]["offset_on_goal"][1]
                target_pose[:, 2] += objects[OBJECTS[object_to_pick]]["offset_on_goal"][2]
                restarted = False

            # advance state machine
            if valid_task:
                des_pose = sm.get_des_pose(
                    get_current_ee(robot), # shape (1, 7)
                    robot_init_pose,       # shape (1, 7)
                    initial_object_pose,   # shape (1, 7)
                    target_pose,           # shape (1, 7)
                    offset_ee
                )
            else:
                gripper_state = torch.tensor(GripperState.OPEN, device=env.unwrapped.device).unsqueeze(0)  # shape: (1, 1)
                robot_pose = torch.tensor(INIT_ROBOT_POSE, device=env.unwrapped.device).unsqueeze(0)  # shape: (1, 7)
                des_pose = torch.cat([robot_pose, gripper_state.unsqueeze(-1)], dim=-1)
                
        
            dones = env.step(des_pose)[-2]

            camera.update(dt=env.unwrapped.sim.get_physics_dt())

            # reset state machine
            if dones.any():
                grasped_bool_vec = [False, False, False, False]
                
                if valid_task and task_count != 0:
                    next_num = save_episode_stepwise(episode_data)
                    episode_data = []
                count = 0
               
                if RANDOM_CAMERA:
                    camera_pose_to_save = set_new_random_camera_pose(env, camera, x_range=CAMERA_X_RANGE, y_range=CAMERA_Y_RANGE, z_range=CAMERA_Z_RANGE)
                   
                set_new_target_pose(env)

                restarted = True
                sm.reset()
                task_count += 1
                if next_num >= 2*SAMPLES_PER_OBJECT:
                    object_to_pick = "object3"
                elif next_num >= SAMPLES_PER_OBJECT:
                    object_to_pick = "object2"
                
                if next_num > 3 * SAMPLES_PER_OBJECT:
                    print("All tasks completed. Exiting...")
                    simulation_app.close()
                    exit(0)

                openvla_instruction = f"Pick the {OBJECTS[object_to_pick]} and place it in the brown box. \n"
                specific_offset_ee = objects[OBJECTS[object_to_pick]]["specific_offset_ee"]  # Specific offset for the object
                offset_ee = specific_offset_ee + NATURAL_OFFSET_EE # 0.01 settato a mano a naso 
                valid_task = True
                continue

            count += 1

    # close the environment
    env.close()

def clear_img_folder():
    """Clear the image folder where the camera images are saved."""
    if os.path.exists("./isaac_ws/src/output/camera"):
        shutil.rmtree("./isaac_ws/src/output/camera")
    if os.path.exists("./isaac_ws/src/output/episodes"):
        shutil.rmtree("./isaac_ws/src/output/episodes")
    os.mkdir("./isaac_ws/src/output/camera")
    os.mkdir("./isaac_ws/src/output/episodes")



def hide_prim(prim_path: str):
    """Hide a prim in the USD stage."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"✅ Hidden prim: {prim_path}")
    else:
        print(f"⚠️ Prim '{prim_path}' not found or invalid.")


def main():
    """Main function to create the environment and run the simulator."""
    # # parse configuration
    clear_img_folder()

    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if SEED is not None:
        env_cfg.seed = SEED

    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)

    env.unwrapped.sim.set_camera_view([1.0, 1.0, 1.0], [0.3, 0.0, 0.0])

    # Remove markers after the environment has created them automatically

    env.reset()

    hide_prim("/Visuals/Command/goal_pose")
    hide_prim("/Visuals/Command/body_pose")

    run_simulator(env, env_cfg, args_cli)
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()