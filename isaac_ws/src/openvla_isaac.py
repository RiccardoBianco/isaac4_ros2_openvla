# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py --num_envs 1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# parser.add_argument(
#     "--save",
#     action="store_true",
#     default=False,
#     help="Save the data from camera at index specified by ``--camera_id``.",
# )

# parser.add_argument(
#     "--camera_id",
#     type=int,
#     choices={0, 1},
#     default=0,
#     help=(
#         "The camera ID to use for displaying points or saving the camera data. Default is 0."
#         " The viewport will always initialize with the perspective of camera 0."
#     ),
# )

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os
import numpy as np
from scipy.spatial.transform import Rotation


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
import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep

from isaaclab.utils import convert_dict_to_backend
##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip

from isaaclab.sensors.camera import Camera, CameraCfg


        # ee_pose = actions["ee_pose"]
        # current_pos = current_ee[:3] # position
        # pos_delta = ee_pose[:3]  # delta[:3]
        # target_pos = current_pos + pos_delta

        # current_orient_xyzw = current_ee[3:]  # current orientation as x, y, z, w
        # orientation_delta_euler = ee_pose[
        #     3:
        # ]  # Delta in orientation as predicted by the model in euler angles

        # # Convert relative orientation to absolute orientation
        # current_orientation = R.from_quat(current_orient_xyzw)
        # orientation_delta = R.from_euler("xyz", orientation_delta_euler)

        # target_orientation = current_orientation * orientation_delta
        # target_orientation_xyzw = target_orientation.as_quat()



        # x,y,z = target_pos

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
    delta_rot = Rotation.from_euler('xyz', delta[3:6])
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

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
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
        #[0.1, 0.1, 0.1, 30 * np.pi / 180, 30 * np.pi / 180, 30 * np.pi / 180], # +10° x
        #[-0.1, -0.1, -0.1, -30 * np.pi / 180, -30 * np.pi / 180, -30 * np.pi / 180], # -10° x
    ]

    # ee_goal_deltas = [
    #     [0.0, 0.0, 0.0, np.pi/2, 0.0, 0.0],  # 90° x
    #     [0.0, 0.0, 0.0, -np.pi/2, 0.0, 0.0], # -90° x
    #     [0.0, 0.0, 0.0, 0.0, np.pi/2, 0.0],  # 90° y
    #     [0.0, 0.0, 0.0, 0.0, -np.pi/2, 0.0], # -90° y
    #     [0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2],  # 90° z
    #     [0.0, 0.0, 0.0, 0.0, 0.0, -np.pi/2], # -90° z
    # ]

    ee_goal_deltas = torch.tensor(ee_goal_deltas, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = torch.tensor([0.5, 0.0, 0.8, 0, 0, 0, 1], device=sim.device) # TODO check if necessary

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
    position_threshold = 0.0005  
    angle_threshold = 0.1

    while simulation_app.is_running():

        if count == 0:
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)


        if goal_reached and count != 0:
            # nuovo goal

            # prendo immagine e invio a OpenVLA
            # delta = res[:6]
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

        # calcolo comando IK

        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)



        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time

        count += 1
        # update buffers
        scene.update(sim_dt)

        goal_reached = check_goal_reached(ik_commands, ee_pose_w, position_threshold, angle_threshold)
        

        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions

        draw_markers(ee_pose_w, ik_commands, scene, ee_marker, goal_marker)


def draw_markers(ee_pose_w, ik_commands, scene, ee_marker, goal_marker):
    quat_correction_np = Rotation.from_euler('x', 180, degrees=True).as_quat()  # [x, y, z, w]
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

    # --- Applica correzione ai quaternioni ---
    corrected_ee_quat = apply_marker_rotation(ee_pose_w[:, 3:7], quat_correction)
    corrected_goal_quat = apply_marker_rotation(ik_commands[:, 3:7], quat_correction)

    # --- Visualizza i marker con orientamento corretto ---
    ee_marker.visualize(ee_pose_w[:, 0:3], corrected_ee_quat)
    goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, corrected_goal_quat)


def check_goal_reached(ik_commands, ee_pose_w, position_threshold, angle_threshold):
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



def main():
    """Main function."""
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
