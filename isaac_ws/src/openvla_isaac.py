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


def apply_delta(position, orientation, delta):
    
    position = position.squeeze(0)
    orientation = orientation.squeeze(0)

    R = Rotation.from_quat(orientation).as_matrix()
    # Compute delta in the world frame
    world_delta = R @ np.array(delta[:3])
    new_position = position + world_delta

    # Apply rotation delta (Euler angles)
    delta_rot = Rotation.from_euler('xyz', delta[3:6])
    new_orientation = (Rotation.from_matrix(R @ delta_rot.as_matrix())).as_quat()

    print(f"new_position: {new_position}")
    print(f"new_orientation: {new_orientation}")

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

    # banana = AssetBaseCfg(https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac
    #     prim_path="{ENV_REGEX_NS}/Banana",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/011_banana.usd", scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
    # )
    

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
    # room = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Room",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Room/simple_room.usd", scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 7.0)),
    # )
    
    # floor = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Floor",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Room/Props/Towel_Room01_floor_bottom.usd", scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    # )


    # room = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Room",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/root/isaac_ws/scene1.usd", scale=(1.0, 1.0, 1.0)
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    # )

    # camera = CameraCfg(
    #     prim_path="/World/CameraSensor",
    #     update_period=0,
    #     height=480,
    #     width=640,
    #     data_types=[
    #         "rgb",
    #         "distance_to_image_plane",
    #         "normals",
    #         "semantic_segmentation",
    #         "instance_segmentation_fast",
    #         "instance_id_segmentation_fast",
    #     ],
    #     colorize_semantic_segmentation=True,
    #     colorize_instance_id_segmentation=True,
    #     colorize_instance_segmentation=True,
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    # )



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
    # camera = scene["camera"]

    # # Create replicator writer
    # output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    # rep_writer = rep.BasicWriter(
    #     output_dir=output_dir,
    #     frame_padding=0,
    #     colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
    #     colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
    #     colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    # )

    # # Camera positions, targets, orientations
    # camera_positions = torch.tensor([2.5, 2.5, 2.5], device=sim.device)
    # camera_targets = torch.tensor([0.0, 0.0, 0.0], device=sim.device)
    # camera.set_world_poses_from_view(camera_positions, camera_targets)
    # camera_index = args_cli.camera_id

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the arm
    ee_goal_deltas = [
        [0.050,  0.020,  0.100,   0.05,  0.02,  0.00],  # 5 cm x, 10 cm z, 5° rx
        [0.055,  0.025,  0.110,   0.06,  0.02,  0.01],  # leggera variazione
        [0.060,  0.030,  0.120,   0.07,  0.03,  0.01],
        [0.065,  0.035,  0.125,   0.08,  0.03,  0.02],
        [0.070,  0.040,  0.130,   0.09,  0.04,  0.03],
        [0.075,  0.045,  0.140,   0.10,  0.05,  0.04],  # 7.5 cm x, 14 cm z, 10° rx
    ]

    ee_goal_deltas = [
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # 10 cm x
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],  # 10 cm y
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],  # 10 cm z
        [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # -10 cm x
        [0.0, -0.1, 0.0, 0.0, 0.0, 0.0],  # -10 cm y
        [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],  # -10 cm z
        [0.1, 0.1, 0.1, 10 * np.pi / 180, 10 * np.pi / 180, 10 * np.pi / 180], # +10° x
        [-0.1, -0.1, -0.1, -10 * np.pi / 180, -10 * np.pi / 180, -10 * np.pi / 180], # -10° x
    ]

    ee_goal_deltas = torch.tensor(ee_goal_deltas, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = torch.tensor([0.5, 0.5, 0.7, 0.707, 0, 0.707, 0], device=sim.device) # TODO check if necessary

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
    position_threshold = 0.005  
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
            delta = ee_goal_deltas[current_goal_idx]
            ee_goal = apply_delta(ee_pos_b.cpu().numpy(), ee_quat_b.cpu().numpy(), delta.cpu().numpy())
            ee_goal = torch.tensor(ee_goal, device=sim.device).unsqueeze(0)
            ik_commands[:] = ee_goal
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            goal_reached = False
            print(f"✅ Nuovo goal: {current_goal_idx}")
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
        # camera.update(dt=sim.get_physics_dt())

        count += 1
        # update buffers
        scene.update(sim_dt)

        goal_reached = check_goal_reached(ik_commands, ee_pose_w, position_threshold, angle_threshold)
        

        # if "rgb" in camera.data.output.keys():
        #     print("Received shape of rgb image        : ", camera.data.output["rgb"].shape)
        # # Extract camera data
        # if args_cli.save:
        #     # Save images from camera at camera_index
        #     # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
        #     single_cam_data = convert_dict_to_backend(
        #         {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
        #     )

        #     # Extract the other information
        #     single_cam_info = camera.data.info[camera_index]

        #     # Pack data back into replicator format to save them using its writer
        #     rep_output = {"annotators": {}}
        #     for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
        #         if info is not None:
        #             rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
        #         else:
        #             rep_output["annotators"][key] = {"render_product": {"data": data}}
        #     # Save images
        #     # Note: We need to provide On-time data for Replicator to save the images.
        #     rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
        #     rep_writer.write(rep_output)
        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


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
