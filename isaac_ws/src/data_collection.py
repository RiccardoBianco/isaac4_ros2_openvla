
OPENVLA_INSTRUCTION = "Pick the green cube and place it on the red area. \n"

SEED = 42

RANDOM_CAMERA = False
RANDOM_OBJECT = False
RANDOM_TARGET = False

SAVE_EVERY_ITERATIONS = 6
SAVE = True

CAMERA_HEIGHT = 1920
CAMERA_WIDTH = 1920
OPENVLA_CAMERA_HEIGHT = 256
OPENVLA_CAMERA_WIDTH = 256

CAMERA_POSITION = [0.9, -0.16, 0.6]
CAMERA_TARGET = [0.4, 0.0, 0.0]

CUBE_SIZE = [0.07, 0.03, 0.06]  # Dimensioni del cubo

OFFSET_EE = CUBE_SIZE[2] / 2 - 0.01 + 0.107 # 0.01 settato a mano a naso
ABOVE_TARGET_OFFSET = 0.15
ABOVE_OBJECT_OFFSET = 0.15

INIT_OBJECT_POS = [0.4, -0.1, 0.0]
INIT_TARGET_POS = [0.4, 0.1, 0.0]  # Z must be 0 in OpenVLA inference script
INIT_ROBOT_POS = [0.4, 0.0, 0.35]


if RANDOM_TARGET: # ABSOLUTE POSITION
    TARGET_X_RANGE = (-0.2 + INIT_TARGET_POS[0], 0.2 + INIT_TARGET_POS[0])
    TARGET_Y_RANGE = (-0.2 + INIT_TARGET_POS[1] , 0.2 + INIT_TARGET_POS[1])
    TARGET_Z_RANGE = (0.0 + INIT_TARGET_POS[2], 0.0 + INIT_TARGET_POS[1])
else:
    TARGET_X_RANGE = (INIT_TARGET_POS[0], INIT_TARGET_POS[0])
    TARGET_Y_RANGE = (INIT_TARGET_POS[1], INIT_TARGET_POS[1])
    TARGET_Z_RANGE = (INIT_TARGET_POS[2], INIT_TARGET_POS[2])

if RANDOM_OBJECT: # RELATIVE POSITION (TO INIT_OBJECT_POS)
    OBJECT_X_RANGE = (-0.2, 0.2)
    OBJECT_Y_RANGE = (-0.2, 0.2)
    OBJECT_Z_RANGE = (0.0, 0.0)
else:
    OBJECT_X_RANGE = (0.0, 0.0)
    OBJECT_Y_RANGE = (0.0, 0.0)
    OBJECT_Z_RANGE = (0.0, 0.0) 


CUBE_MULTICOLOR = False

EULER_NOTATION = "zyx" 

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
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg_pers import LiftEnvCfg
import isaaclab.sim as sim_utils
import omni.replicator.core as rep
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.managers import SceneEntityCfg
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

import numpy as np

# ^ Import Utils class
from sim_utils.utils import Utils

@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
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


        if CUBE_MULTICOLOR:
            self.scene.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(pos=INIT_OBJECT_POS, rot=[1, 0, 0, 0]),
                spawn=UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                    scale=(0.7, 0.7, 0.7),
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
            self.scene.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=sim_utils.CuboidCfg(
                    size=CUBE_SIZE,  # Dimensioni del cubo
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # Proprietà fisiche
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # Massa
                    collision_props=sim_utils.CollisionPropertiesCfg(),  # Proprietà di collisione
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),  # Colore rosso
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
    PLACE_ABOVE_GOAL = 0.1 # 0.1 molto fluido -> diretto al goal 
    PLACE_ON_GOAL = 1.0
    RELEASE_OBJECT = 0.3
    MOVE_ABOVE_GOAL = 0.5
    TERMINAL_STATE = 1.0 

def print_sm_state(state):
    if state == SmState.ROBOT_INIT_POSE:
        return "ROBOT_INIT_POSE"
    elif state == SmState.APPROACH_ABOVE_OBJECT:
        return "APPROACH_ABOVE_OBJECT"
    elif state == SmState.APPROACH_OBJECT:
        return "APPROACH_OBJECT"
    elif state == SmState.GRASP_OBJECT:
        return "GRASP_OBJECT"
    elif state == SmState.LIFT_OBJECT:
        return "LIFT_OBJECT"
    elif state == SmState.PLACE_ABOVE_GOAL:
        return "PLACE_ABOVE_GOAL"
    elif state == SmState.PLACE_ON_GOAL:
        return "PLACE_ON_GOAL"
    elif state == SmState.RELEASE_OBJECT:
        return "RELEASE_OBJECT"
    elif state == SmState.MOVE_ABOVE_GOAL:
        return "MOVE_ABOVE_GOAL"
    elif state == SmState.TERMINAL_STATE:
        return "TERMINAL_STATE"
    else:
        return "UNKNOWN_STATE"

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

    def get_des_pose(self, ee_current_pose, init_pose, initial_object_pose, target_pose):
        
        init_pose = init_pose.clone()
        initial_object_pose = initial_object_pose.clone()
        target_pose = target_pose.clone()
        init_pose[:, 2] += OFFSET_EE
        initial_object_pose[:, 2] += OFFSET_EE
        target_pose[:, 2] += OFFSET_EE + 0.03

        
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
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) and self.sm_wait_time >= SmWaitTime.ROBOT_INIT_POSE:
                self.sm_state = SmState.APPROACH_ABOVE_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.APPROACH_ABOVE_OBJECT:
            des_ee_pose = above_initial_object_pose
            gripper_state = GripperState.OPEN
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.APPROACH_ABOVE_OBJECT:
                self.sm_state = SmState.APPROACH_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.APPROACH_OBJECT:
            des_ee_pose = initial_object_pose
            gripper_state = GripperState.OPEN
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.APPROACH_OBJECT:
                self.sm_state = SmState.GRASP_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.GRASP_OBJECT:
            des_ee_pose = initial_object_pose
            gripper_state = GripperState.CLOSE
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) and self.sm_wait_time >= SmWaitTime.GRASP_OBJECT:
                self.sm_state = SmState.LIFT_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.LIFT_OBJECT:
            des_ee_pose = above_initial_object_pose
            gripper_state = GripperState.CLOSE
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.LIFT_OBJECT:
                self.sm_state = SmState.PLACE_ON_GOAL
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.PLACE_ON_GOAL:
            des_ee_pose = target_pose
            gripper_state = GripperState.CLOSE
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.PLACE_ON_GOAL:
                self.sm_state = SmState.RELEASE_OBJECT
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.RELEASE_OBJECT:
            des_ee_pose = target_pose
            gripper_state = GripperState.OPEN
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) and self.sm_wait_time >= SmWaitTime.RELEASE_OBJECT:
                self.sm_state = SmState.MOVE_ABOVE_GOAL
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.MOVE_ABOVE_GOAL:
            des_ee_pose = above_target_pose
            gripper_state = GripperState.OPEN
            if Utils.check_des_state_reached(ee_current_pose, des_ee_pose, 0.005, 0.005) or self.sm_wait_time >= SmWaitTime.MOVE_ABOVE_GOAL:
                self.sm_state = SmState.TERMINAL_STATE
                self.sm_wait_time = 0.0
        elif self.sm_state == SmState.TERMINAL_STATE:
            des_ee_pose = above_target_pose
            gripper_state = GripperState.OPEN

        self.sm_wait_time += self.sm_dt.item() # TODO CHECK

        des_gripper_state = torch.tensor([gripper_state], device=self.device)
        return torch.cat([des_ee_pose, des_gripper_state.unsqueeze(-1)], dim=-1)

def update_save_every_iterations(state):
    if state == SmState.APPROACH_ABOVE_OBJECT: 
        return SAVE_EVERY_ITERATIONS
    elif state == SmState.APPROACH_OBJECT:
        return SAVE_EVERY_ITERATIONS
    elif state == SmState.GRASP_OBJECT:
        return 3
    elif state == SmState.LIFT_OBJECT:
        return SAVE_EVERY_ITERATIONS
    elif state == SmState.PLACE_ABOVE_GOAL:
        return 2
    elif state == SmState.PLACE_ON_GOAL:
        return SAVE_EVERY_ITERATIONS
    elif state == SmState.RELEASE_OBJECT:
        return 3
    elif state == SmState.MOVE_ABOVE_GOAL:
        return 3
    elif state == SmState.TERMINAL_STATE:
        return SAVE_EVERY_ITERATIONS
    else:
        return SAVE_EVERY_ITERATIONS

def save_config_file():
        ### Create the config file with the Global Parameters defined in the current run
    config = {
        "OPENVLA_INSTRUCTION": OPENVLA_INSTRUCTION,
        "SEED": SEED,
        "RANDOM_CAMERA": RANDOM_CAMERA,
        "RANDOM_OBJECT": RANDOM_OBJECT,
        "RANDOM_TARGET": RANDOM_TARGET,
        "SAVE_EVERY_ITERATIONS": SAVE_EVERY_ITERATIONS,
        "SAVE": SAVE,
        "CAMERA_HEIGHT": CAMERA_HEIGHT,
        "CAMERA_WIDTH": CAMERA_WIDTH,
        "OPENVLA_CAMERA_HEIGHT": OPENVLA_CAMERA_HEIGHT,
        "OPENVLA_CAMERA_WIDTH": OPENVLA_CAMERA_WIDTH,
        "CAMERA_POSITION": CAMERA_POSITION,
        "CAMERA_TARGET": CAMERA_TARGET,
        "CUBE_SIZE": CUBE_SIZE,
        "OFFSET_EE": OFFSET_EE,
        "ABOVE_TARGET_OFFSET": ABOVE_TARGET_OFFSET,
        "ABOVE_OBJECT_OFFSET": ABOVE_OBJECT_OFFSET,
        "INIT_OBJECT_POS": INIT_OBJECT_POS,
        "INIT_TARGET_POS": INIT_TARGET_POS,
        "INIT_ROBOT_POS": INIT_ROBOT_POS,
        "TARGET_X_RANGE": TARGET_X_RANGE,
        "TARGET_Y_RANGE": TARGET_Y_RANGE,
        "TARGET_Z_RANGE": TARGET_Z_RANGE,
        "OBJECT_X_RANGE": OBJECT_X_RANGE,
        "OBJECT_Y_RANGE": OBJECT_Y_RANGE,
        "OBJECT_Z_RANGE": OBJECT_Z_RANGE,
        "CUBE_MULTICOLOR": CUBE_MULTICOLOR,
        "EULER_NOTATION": EULER_NOTATION,
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



def run_simulator(env, env_cfg, args_cli):

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

    Utils.assign_material(object_path="/World/Table", material_path="/World/Table/Looks/Black")

    count = 0
    task_count = 0
    restarted = True


    robot_init_pose = torch.tensor([INIT_ROBOT_POS[0], INIT_ROBOT_POS[1], INIT_ROBOT_POS[2]-OFFSET_EE, 0.0, 1.0, 0.0, 0.0], device=env.unwrapped.device).unsqueeze(0) # [x, y, z, qw, qx, qy, qz] # towards down


    printed = False
    while simulation_app.is_running():
        
        #########################
        # NOTE Just to check -> to be removed
        if not printed:
            if sm.sm_state != SmState.ROBOT_INIT_POSE:
                printed = True
            joint_pos = robot.data.joint_pos.clone()
            print("\n\nROBOT_INIT_POSE JOINT POSITION: ", joint_pos) #  [ 0.0000, -0.5690,  0.0000, -2.8100,  0.0000,  3.0370,  0.7410,  0.0400, 0.0400]
            ee_pose_w = robot.data.body_state_w[:, 8, 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            print("ROBOT_INIT_POSE POS: ee_pos_b: ", ee_pos_b) # [ 4.4507e-01, -1.7705e-05,  4.0302e-01]
            print("ORIENTATION POS: ee_quat_b: ", ee_quat_b) # [0.0086, 0.9218, 0.0204, 0.3871]

        #########################

        save_every_iterations = update_save_every_iterations(sm.sm_state)


        if count % save_every_iterations == 0 and task_count!=0 and not restarted and sm.sm_state != SmState.ROBOT_INIT_POSE and sm.sm_state != SmState.TERMINAL_STATE:
            if SAVE:

                current_state = Utils.get_current_state(robot) # shape: (1, 8) # x, y, z, roll, pitch, yaw, pad, gripper

                should_save = False

                if len(episode_data) == 0:
                    should_save = True
                else:
                    
                    delta_steps = Utils.compute_delta(episode_data[-1]["state"], current_state.clone().cpu().squeeze().numpy().astype(np.float32)) 
                    delta_gripper = abs(episode_data[-1]["state"][-1] - current_state.clone().cpu().squeeze().numpy().astype(np.float32)[-1])
                    should_save = Utils.is_significant_change(delta_steps, delta_gripper, pos_th=0.05, rot_th=0.05, gripper_th=0.013)

                if should_save:
                    print("Saving step")
                    table_image_array = Utils.take_image(camera_index, camera, camera_type="table", sim_num=task_count-1)
                    wrist_image_array = Utils.take_image(camera_index, wrist_camera, camera_type="wrist", sim_num=task_count-1)

                    step_data = {
                        "state": current_state.clone().cpu().squeeze().numpy().astype(np.float32),
                        "image": table_image_array.astype(np.uint8),
                        "wrist_image": wrist_image_array.astype(np.uint8),
                        "language_instruction": OPENVLA_INSTRUCTION,
                        "object_pose": current_object_pose_to_save.clone().cpu().numpy().astype(np.float32),
                        "target_pose": target_pose.clone().cpu().numpy().astype(np.float32),
                        "camera_pose": camera_pose_to_save.clone().cpu().numpy().astype(np.float32),
                    }

                    episode_data.append(step_data)
                else:
                    print("Not saving step")


        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            
            current_object_pose_to_save = env.unwrapped.scene["object"].data.root_state_w[:, 0:7].clone()
            if restarted == True:
                initial_object_pose = env.unwrapped.scene["object"].data.root_state_w[:, :7].clone()
                target_pose = env.unwrapped.command_manager.get_command("target_pose")[..., :7].clone()
                restarted = False

            # advance state machine
            des_pose = sm.get_des_pose(
                Utils.get_current_ee(robot), # shape (1, 7)
                robot_init_pose,       # shape (1, 7)
                initial_object_pose,   # shape (1, 7)
                target_pose            # shape (1, 7)
            )
        
            dones = env.step(des_pose)[-2]

            camera.update(dt=env.unwrapped.sim.get_physics_dt())

            # reset state machine
            if dones.any():
                printed = False
        
                if task_count != 0:
                    Utils.save_episode_stepwise(episode_data)
                    episode_data = []
                count = 0
               
                if RANDOM_CAMERA:
                    camera_pose_to_save = Utils.set_new_random_camera_pose(env, camera)
                   
                Utils.set_new_target_pose(env)

                restarted = True
                sm.reset()
                task_count += 1
                continue

            count += 1

    # close the environment
    env.close()

def main():
    # # parse configuration
    Utils.clear_img_folder()

    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.seed = SEED

    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)

    env.unwrapped.sim.set_camera_view([1.0, 1.0, 1.0], [0.3, 0.0, 0.0])

    # Rimuovi marker dopo che l'ambiente li ha creati automaticamente

    env.reset()

    Utils.hide_prim("/Visuals/Command/goal_pose")
    Utils.hide_prim("/Visuals/Command/body_pose")

    run_simulator(env, env_cfg, args_cli)
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()