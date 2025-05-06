# TODO cambiare oggetto -> deve essere un rigid body altrimenti non funziona
"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaac_ws/isaac_lab/isaaclab.sh -p isaac_ws/src/sm_pick.py --enable_cameras --save

"""

"""Launch Omniverse Toolkit first."""

PICK_AND_PLACE = True # set to False to only pick and lift the object, bringing it back to the goal pose

OPENVLA_INSTRUCTION = "Pick and place the object in the red goal pose. \n"

RANDOM_CAMERA = False

SAVE_EVERY_ITERATIONS = 10
SAVE = True

CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
CAMERA_POSITION = [0.9, -0.4, 0.6]
CAMERA_TARGET = [0.3, 0.0, -0.2]

INIT_OBJECT_POS = [0.5, 0, 0.055]



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

import warp as wp
import numpy as np
import os
import shutil


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
##
# Pre-defined configs
##

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import omni.usd
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import convert_dict_to_backend



from pxr import UsdGeom, Usd, UsdShade
# initialize warp
wp.init()


from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sim.spawners import UsdFileCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

SAVE_DATASET_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "plr_openvla_dataset")

def save_step_npz(table_image_array, wrist_image_array, joint_angles, joint_velocities, instruction, step_id = 0, task_count=0):
    """
    Save a step of the simulation to a .npz file
    # TODO NEED TO FINISH THE FUNCTION
    """
    folder_name = f"simulation_{task_count:03d}"
    save_task_dir = os.path.join(SAVE_DATASET_DIR, folder_name)
    os.makedirs(save_task_dir, exist_ok=True)
    save_dict = {
        "image": table_image_array.astype(np.uint8),
        "wrist_image": wrist_image_array.astype(np.uint8), 
        "state": joint_angles.cpu().numpy().astype(np.float32),  # shape: (9,)
        "action": joint_velocities.cpu().numpy().astype(np.float32),  # shape: (9,)
        "language_instruction": instruction,
    }
    np.savez_compressed(os.path.join(save_task_dir, f"step_{step_id:06d}.npz"), **save_dict)



def scalar_first_to_last(q):
    w, x, y, z = q
    return [x, y, z, w]


def scalar_last_to_first(q):
    x, y, z, w = q
    return [w, x, y, z]

import numpy as np
from scipy.spatial.transform import Rotation

# def compute_delta(ee_pose, next_ee_pose, gripper_state):
#     # Decomponi le pose
#     pos1, quat1 = ee_pose[:3], ee_pose[3:]
#     pos2, quat2 = next_ee_pose[:3], next_ee_pose[3:]

#     # Converti i quaternioni in rotazioni (convertendo l'ordine per scipy)
#     rot1 = Rotation.from_quat(scalar_first_to_last(quat1))
#     rot2 = Rotation.from_quat(scalar_first_to_last(quat2))

#     # Calcola la traslazione nel world frame
#     delta_pos_world = pos2 - pos1

#     # Riporta la traslazione nel frame dell'EE
#     delta_pos_ee = rot1.inv().apply(delta_pos_world)

#     # Calcola rotazione relativa: R_delta = R1^-1 * R2
#     delta_rot = rot1.inv() * rot2

#     # Estrai rotazione relativa in Euler angles
#     delta_euler = delta_rot.as_euler('xyz')  # RPY in radianti

#     # Combina in unico array
#     delta = np.concatenate([delta_pos_ee, delta_euler, gripper_state])  # shape (7,)
#     return delta
def compute_delta(ee_pose, next_ee_pose):
    # Decomponi le pose
    # pos1, rpy1, grip1 = ee_pose[:3], ee_pose[3:6], ee_pose[6]
    # pos2, rpy2, grip2 = next_ee_pose[:3], next_ee_pose[3:6], next_ee_pose[6] # no padding

    pos1, rpy1, grip1 = ee_pose[:3], ee_pose[3:6], ee_pose[7]
    pos2, rpy2, grip2 = next_ee_pose[:3], next_ee_pose[3:6], next_ee_pose[7]

    # Rotazioni come oggetti Rotation
    rot1 = Rotation.from_euler('xyz', rpy1)
    rot2 = Rotation.from_euler('xyz', rpy2)

    # Calcola la traslazione nel world frame
    delta_pos_world = pos2 - pos1

    # Riporta la traslazione nel frame dell'EE
    delta_pos_ee = rot1.inv().apply(delta_pos_world)

    # Calcola rotazione relativa: R_delta = R1^-1 * R2
    delta_rot = rot1.inv() * rot2

    # Estrai rotazione relativa in Euler angles
    delta_euler = delta_rot.as_euler('xyz')  # RPY in radianti

    next_gripper = np.atleast_1d(grip2)


    # Combina il delta finale
    delta = np.concatenate([delta_pos_ee, delta_euler, next_gripper]).astype(np.float32)  # shape (7,)

    return delta





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
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

                # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"


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


        # TODO understand how to set the object different from this cube
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=INIT_OBJECT_POS, rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
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

        # self.scene.box = AssetBaseCfg(
        #     prim_path="/World/CrakerBox",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", scale=(1.0, 1.0, 1.0)
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.4, 0.0)),
        # )
    
        # self.scene.box = RigidObjectCfg(
        #     prim_path="/World/Box",
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", # TODO check if rigid body props are needed
        #         scale=(1.0, 1.0, 0.2),
        #         rigid_props=RigidBodyPropertiesCfg(),  # default va bene per ora
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.4, 0.0)),
        # )

        # self.scene.box = AssetBaseCfg(
        #     prim_path="/World/Box",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd", scale=(0.1, 0.1, 0.01)
        #     ),
        #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.4, 0.0)),
        # )
        if PICK_AND_PLACE:
            self.scene.box = RigidObjectCfg(
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
                    pos=(0.5, 0.4, 0.0),  # OVERWRITTEN BY THE COMMANDER
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

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


    PLACE_ON_GOAL = wp.constant(5)
    PLACE_BELOW_GOAL = wp.constant(6)
    RELEASE_OBJECT = wp.constant(7)
    RETURN_HOME = wp.constant(8)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(1.5)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(0.5)

    PLACE_ON_GOAL = wp.constant(0.4)
    PLACE_BELOW_GOAL = wp.constant(0.4)
    RELEASE_OBJECT = wp.constant(0.4)
    RETURN_HOME = wp.constant(0.5)


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    above_object_pose: wp.array(dtype=wp.transform),
    below_goal_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.PLACE_ON_GOAL # TODO: LIFT_OBJECT if we want to lift first
            sm_wait_time[tid] = 0.0
    # elif state == PickSmState.LIFT_OBJECT:
    #     des_ee_pose[tid] = above_object_pose[tid]
    #     gripper_state[tid] = GripperState.CLOSE
    #     if distance_below_threshold(
    #         wp.transform_get_translation(ee_pose[tid]),
    #         wp.transform_get_translation(des_ee_pose[tid]),
    #         position_threshold,
    #     ):
    #         # wait for a while
    #         if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
    #             # move to next state and reset wait time
    #             sm_state[tid] = PickSmState.PLACE_ABOVE_GOAL
    #             sm_wait_time[tid] = 0.0
    elif state == PickSmState.PLACE_ON_GOAL:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.PLACE_ON_GOAL:
                # move to next state and reset wait time
                if PICK_AND_PLACE:
                    sm_state[tid] = PickSmState.PLACE_BELOW_GOAL
                else:
                    sm_state[tid] = PickSmState.PLACE_ON_GOAL
                sm_wait_time[tid] = 0.0
                
    elif state == PickSmState.PLACE_BELOW_GOAL and PICK_AND_PLACE:
        des_ee_pose[tid] = below_goal_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.PLACE_BELOW_GOAL:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.RELEASE_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.RELEASE_OBJECT and PICK_AND_PLACE:
        des_ee_pose[tid] = below_goal_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.RELEASE_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.RETURN_HOME
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.RETURN_HOME and PICK_AND_PLACE:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.RETURN_HOME:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.RETURN_HOME
                sm_wait_time[tid] = 0.0



    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor, above_object_pose: torch.Tensor, below_goal_pose: torch.Tensor) -> torch.Tensor:
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        above_object_pose = above_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        below_goal_pose = below_goal_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        above_object_pose_wp = wp.from_torch(above_object_pose.contiguous(), wp.transform)
        below_goal_pose_wp = wp.from_torch(below_goal_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                above_object_pose_wp,
                below_goal_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,

            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

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

def take_image(camera_index, camera, rep_writer):
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

def get_current_state(robot):
    joint_pos = robot.data.joint_pos.clone()
    ee_pose_w = robot.data.body_state_w[:, 8, 0:7] # TODO fix robot_entity_cfg.body_ids[0] = 8
    root_pose_w = robot.data.root_state_w[:, 0:7]

    # posizione dell'end-effector relativa al root
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    # print("joint_pos: ", joint_pos)
    # print("joint_pose shape: ", joint_pos.shape)
    gripper_state = joint_pos[:, -1].unsqueeze(-1)  # shape: (1, 1)
    # print("gripper_state: ", gripper_state.shape)
    # print("gripper_state: ", gripper_state)
    quat_np = scalar_first_to_last(ee_quat_b[0].cpu().numpy())  # shape (4,)
    R_xyz_np = Rotation.from_quat(quat_np).as_euler('xyz') # shape (3,)
    R_xyz = torch.tensor(R_xyz_np, device=ee_pos_b.device).unsqueeze(0)  # shape (1, 3)
    pad = torch.tensor([[0.0]], device=ee_pos_b.device)
    current_state = torch.cat([ee_pos_b, R_xyz, pad, gripper_state], dim=-1)  # shape: (1, 8) # TODO probabilmente va aggiunto il padding solo allo state
    #current_state = torch.cat([ee_pos_b, R_xyz, gripper_state], dim=-1) # shape: (1, 7)
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
    for i in range(len(episode_steps)-1):
        episode_steps[i]["action"]= compute_delta(episode_steps[i]["state"], episode_steps[i+1]["state"])
        # print("Step: ", i)
        # print("Action: ", episode_steps[i]["action"]) # dx, dy, dz, droll, dpitch, dyaw, next_gripper
        # print("State: ", episode_steps[i]["state"]) # x, y, z, roll, pitch, yaw, gripper
    
    episode_steps[-1]["action"] = compute_delta(episode_steps[-1]["state"], episode_steps[-1]["state"])

    os.makedirs(save_dir, exist_ok=True)

    # Get next available episode number
    existing = [f for f in os.listdir(save_dir) if f.startswith("episode_") and f.endswith(".npy")]
    episode_nums = [int(f.split("_")[1].split(".")[0]) for f in existing if "_" in f]
    next_num = max(episode_nums) + 1 if episode_nums else 0

    filename = f"episode_{next_num:03d}.npy"
    filepath = os.path.join(save_dir, filename)

    # Save
    np.save(filepath, episode_steps, allow_pickle=True)
    print(f"✅ Saved episode with {len(episode_steps)} steps to {filepath}")

def run_simulator(env, env_cfg, args_cli):
    camera = env.unwrapped.scene["camera"]
    wrist_camera = env.unwrapped.scene["wrist_camera"]

    robot = env.unwrapped.scene["robot"]

    print("\n\nRUNNING SIMULATOR!\n\n")

    # ^ Temporary data structure for saving ["state", "action", "image", "wrist_image", "language_instruction"] for each step
    episode_data = []

    rep_writer = rep.BasicWriter(
        output_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera"), # don't save the first sim
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    # Set the camera position and target (wrist camera is already attached to the robot in the config)
    camera_positions = torch.tensor([CAMERA_POSITION], device=env.unwrapped.device)
    camera_targets = torch.tensor([CAMERA_TARGET], device=env.unwrapped.device)
    camera.set_world_poses_from_view(camera_positions, camera_targets)
    camera_index = args_cli.camera_id

    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    # desired object orientation (we only do position control of object)
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0

    # create state machine
    pick_sm = PickAndLiftSm(
        env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device, position_threshold=0.01
    )

    assign_material(object_path="/World/Table", material_path="/World/Table/Looks/Black")

    count = 0
    task_count = 0
    restarted = True

    while simulation_app.is_running():

        # if pick_sm.sm_state[0].item() == PickSmState.REST:
        #     joint_pos = robot.data.joint_pos.clone()
        #     print("\n\nREST JOINT POSITION: ", joint_pos) #  [ 0.0000, -0.5690,  0.0000, -2.8100,  0.0000,  3.0370,  0.7410,  0.0400, 0.0400]
        #     ee_frame_sensor = env.unwrapped.scene["ee_frame"]
        #     tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
        #     tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

        #     print("REST POS: tcp_rest_position: ", tcp_rest_position) # [ 4.4507e-01, -1.7705e-05,  4.0302e-01]
        #     print("ORIENTATION POS: tcp_rest_orientation: ", tcp_rest_orientation) # [0.0086, 0.9218, 0.0204, 0.3871]





        if count % SAVE_EVERY_ITERATIONS == 0 and task_count!=0 and not restarted and pick_sm.sm_state[0].item() != PickSmState.REST:
            table_image_array = take_image(camera_index, camera, rep_writer)
            wrist_image_array = take_image(camera_index, wrist_camera, rep_writer)
            # if count >= 10: # NOTE added ric to avoi saving too many images locally
            #     if os.path.exists("./isaac_ws/src/output/camera"):
            #         shutil.rmtree("./isaac_ws/src/output/camera")
            #     os.mkdir("./isaac_ws/src/output/camera", exist_ok=True)

            
            # joint_vel = robot.data.joint_vel.clone()

            # print("Joint Position: ", joint_pos)
            # print("Joint Velocity: ", joint_vel)
            if SAVE:
                current_state = get_current_state(robot) # shape: (1, 8) # x, y, z, roll, pitch, yaw, pad, gripper
                
                step_data = {
                    "state": current_state.clone().cpu().squeeze().numpy().astype(np.float32),  # shape: (8,)
                    "image": table_image_array.astype(np.uint8),
                    "wrist_image": wrist_image_array.astype(np.uint8),
                    "language_instruction": OPENVLA_INSTRUCTION,
                    "object_pose": object_pose_to_save.clone().cpu().numpy().astype(np.float32),
                    "goal_pose": goal_pose_to_save.clone().cpu().numpy().astype(np.float32),
                }

                # Add step to episode_data
                episode_data.append(step_data)

        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            

            # observations
            # -- end-effector frame
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = env.unwrapped.scene["object"].data
            object_pose_to_save = object_data.root_state_w[:, 0:7].clone()
            # print("OBJECT POSE: ", object_pose_to_save)
            goal_pose_to_save = env.unwrapped.command_manager.get_command("object_pose")[..., :7].clone()
            # print("GOAL POSE: ", goal_pose_to_save)


            object_position = object_data.root_pos_w - env.unwrapped.scene.env_origins
            # -- target object frame
            desired_position = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            if restarted == True:
                above_object_position = object_position.clone()
                above_object_position[:, 2] += 0.1  # 10 cm sopra oggetto

                below_goal_position = desired_position.clone()
                below_goal_position[:, 2] -= 0.18  # 10 cm sopra goal (dato che goal è già a +20 cm)
                restarted = False

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, desired_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
                torch.cat([above_object_position, desired_orientation], dim=-1),
                torch.cat([below_goal_position, desired_orientation], dim=-1),
            )
        
            dones = env.step(actions)[-2]

            camera.update(dt=env.unwrapped.sim.get_physics_dt())

            # reset state machine
            if dones.any():
                # ^ Create the .npy file with the data of the current episode
                if task_count != 0:
                    save_episode_stepwise(episode_data)
                    # ^ Reset the data structures
                    episode_data = []
                count = 0
                rep_writer = rep.BasicWriter(
                    output_dir=get_next_simulation_folder(),
                    frame_padding=0,
                    colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
                    colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
                    colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
                )
                if RANDOM_CAMERA:
                    # Base position
                    base_camera_position = torch.tensor(CAMERA_POSITION, device=env.unwrapped.device)
                    
                    # Random offset in [-0.3, 0.3]
                    random_offset = (torch.rand(3, device=env.unwrapped.device) - 0.5) * 0.6

                    # Final camera position
                    camera_positions = base_camera_position + random_offset
                    camera_positions = camera_positions.unsqueeze(0)  # shape: (1, 3)
                    camera_targets = torch.tensor([CAMERA_TARGET], device=env.unwrapped.device)
                    camera.set_world_poses_from_view(camera_positions, camera_targets)


                goal_pose = env.unwrapped.command_manager.get_command("object_pose")

                # Calcola posizione della box
                if PICK_AND_PLACE:
                    new_pos = goal_pose[..., :3].clone()
                    new_pos[..., 2] = 0.0

                    new_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=new_pos.device).expand(new_pos.shape[0], 4)

                    root_state = torch.zeros((env.unwrapped.num_envs, 13), device=env.unwrapped.device)
                    root_state[:, 0:3] = new_pos
                    root_state[:, 3:7] = new_rot

                    # Scrive la nuova pose alla simulazione
                    env.unwrapped.scene["box"].write_root_state_to_sim(root_state)

                restarted = True
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
                task_count += 1
                continue

            count += 1

    # close the environment
    env.close()

def clear_img_folder():
    if os.path.exists("./isaac_ws/src/output/camera"):
        shutil.rmtree("./isaac_ws/src/output/camera")
    if os.path.exists("./isaac_ws/src/output/plr_openvla_dataset"):
        shutil.rmtree("./isaac_ws/src/output/plr_openvla_dataset")
    if os.path.exists("./isaac_ws/src/output/episodes"):
        shutil.rmtree("./isaac_ws/src/output/episodes")
    os.mkdir("./isaac_ws/src/output/camera")
    os.mkdir("./isaac_ws/src/output/plr_openvla_dataset")
    os.mkdir("./isaac_ws/src/output/episodes")




def get_next_simulation_folder(base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")):
    i = 0
    while os.path.exists(os.path.join(base_path, f"simulation_{i}")):
        i += 1
    # Create the new directory
    os.makedirs(os.path.join(base_path, f"simulation_{i}"))
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera", f"simulation_{i}")
    return output_dir

def remove_prim(prim_path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        stage.RemovePrim(prim_path)
        print(f"✅ Removed prim at path: {prim_path}")
    else:
        print(f"⚠️  No prim found at path: {prim_path}")

def hide_prim(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"✅ Hidden prim: {prim_path}")
    else:
        print(f"⚠️ Prim '{prim_path}' not found or invalid.")


def main():
    # # parse configuration
    clear_img_folder()

    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    env_cfg.scene.ee_frame.visualizer_cfg.markers["frame"].enabled = False
    # create environment
    env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
    # reset environment at start
    env.unwrapped.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Rimuovi marker dopo che l'ambiente li ha creati automaticamente

    env.reset()

    hide_prim("/Visuals/Command/goal_pose")
    hide_prim("/Visuals/Command/body_pose")

    

    run_simulator(env, env_cfg, args_cli)
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
