# TODO : add docstrings to all functions
# TODO : polish the code
# TODO : split the Utils class into multiple classes (by functionality)
##############################################################################
# ^ General imports
##############################################################################
import torch
from PIL import Image
import os
import json
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import shutil

##############################################################################
# ^ Isaac imports
##############################################################################
import omni.usd
from pxr import UsdGeom, Usd, UsdShade
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.math import subtract_frame_transforms

###############################################################################
# & Utils Class
###############################################################################
class Utils:
    @staticmethod
    def scalar_first_to_last(q):
        w, x, y, z = q
        return [x, y, z, w]

    @staticmethod
    def scalar_last_to_first(q):
        x, y, z, w = q
        return [w, x, y, z]

    @staticmethod
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

    @staticmethod
    def check_des_state_reached(current_state, desired_state, position_threshold, angle_threshold):
        """
            Check if the current position is within the threshold of the desired position.
            Returns True if the goal is reached, False otherwise.
            state: [x, y, z, qw, qx, qy, qz]

        """
        position_error = torch.norm(current_state[:, :3] - desired_state[:, :3], dim=1)

        quat_dot = torch.abs(torch.sum(current_state[:, 3:7] * desired_state[:, 3:7], dim=1))  # q1 · q2
        quat_dot = torch.clamp(quat_dot, -1.0, 1.0)  # clamp per stabilità numerica
        angle_error = 2 * torch.acos(quat_dot)


        if position_error.item() < position_threshold and angle_error.item() < angle_threshold:
            angle_deg = np.degrees(angle_error.item())
            print(f"REACHED des_state! Pos err: {position_error.item():.4f} m | Ang err: {angle_deg:.4f}°")
            return True
        return False

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        R_euler_np = Rotation.from_quat(quat_np).as_euler(EULER_NOTATION) # shape (3,)
        R_euler = torch.tensor(R_euler_np, device=ee_pos_b.device).unsqueeze(0)  # shape (1, 3)
        pad = torch.tensor([[0.0]], device=ee_pos_b.device)
        current_state = torch.cat([ee_pos_b, R_euler, pad, gripper_state], dim=-1)  # shape: (1, 8) # TODO probabilmente va aggiunto il padding solo allo state
        #current_state = torch.cat([ee_pos_b, R_euler, gripper_state], dim=-1) # shape: (1, 7)
        return current_state

    @staticmethod
    def save_episode_stepwise(episode_steps, save_dir="isaac_ws/src/output/episodes"):
        """
        Save a list of timestep dictionaries into a progressively numbered .npy file.
        
        Args:
            episode_steps (List[dict]): Each step must include keys like "state", "action", "image", etc.
            save_dir (str): Directory where .npy episodes are stored.
        """

        # Check on the final state of the object wrt the target
        distance_object_target = np.linalg.norm(episode_steps[-1]["object_pose"][:, :3] - episode_steps[-1]["target_pose"][:, :3])
        if distance_object_target > 0.05:
            print("Episode not saved: object is too far from the target.")
            return
        
        for i in range(len(episode_steps)-1):
            episode_steps[i]["action"] = compute_delta(episode_steps[i]["state"], episode_steps[i+1]["state"])
            # print("Step: ", i)
            # print("Action: ", episode_steps[i]["action"]) # dx, dy, dz, droll, dpitch, dyaw, next_gripper
            # print("State: ", episode_steps[i]["state"]) # x, y, z, roll, pitch, yaw, gripper
        
        episode_steps[-1]["action"] = compute_delta(episode_steps[-1]["state"], episode_steps[-1]["state"])

        os.makedirs(save_dir, exist_ok=True)

        # Get next available episode number
        existing = [f for f in os.listdir(save_dir) if f.startswith("episode_") and f.endswith(".npy")]
        episode_nums = [int(f.split("_")[1].split(".")[0]) for f in existing if "_" in f]
        next_num = max(episode_nums) + 1 if episode_nums else 0
        if next_num > 4500: 
            simulation_app.close()
            print("Maximum number of episodes reached. Exiting...")
            exit(0)

        filename = f"episode_{next_num:04d}.npy"
        filepath = os.path.join(save_dir, filename)

        # Save to .npy file
        np.save(filepath, episode_steps, allow_pickle=True)
        print(f"✅ Saved episode with {len(episode_steps)} steps to {filepath}")

    @staticmethod
    def get_current_ee(robot):
        ee_pose_w = robot.data.body_state_w[:, 8, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        current_state = torch.cat([ee_pos_b, ee_quat_b], dim=-1) # (1, 7)
        return current_state

    @staticmethod
    def is_significant_change(delta, delta_gripper, pos_th, rot_th, gripper_th):
        dx, dy, dz, droll, dpitch, dyaw, gripper = delta
        pos_change = np.linalg.norm([dx, dy, dz]) > pos_th
        # print("pos_change: ", pos_change)
        rot_change = np.linalg.norm([droll, dpitch, dyaw]) > rot_th
        # print("rot_change: ", rot_change)
        grip_change = delta_gripper > gripper_th
        # print("grip_change: ", grip_change)
        return pos_change or rot_change or grip_change

    @staticmethod
    def set_new_random_camera_pose(env, camera):
        # Base position
        base_camera_position = torch.tensor(CAMERA_POSITION, device=env.unwrapped.device)
        
        # Random offset in [-0.3, 0.3]
        random_offset = (torch.rand(3, device=env.unwrapped.device) - 0.5) * 0.6

        # Final camera position
        camera_positions = base_camera_position + random_offset
        camera_positions = camera_positions.unsqueeze(0)  # shape: (1, 3)
        camera_targets = torch.tensor([CAMERA_TARGET], device=env.unwrapped.device)
        camera.set_world_poses_from_view(camera_positions, camera_targets)
        camera_pose_to_save = torch.cat([camera_positions, camera_targets], dim=-1)
        return camera_pose_to_save

    @staticmethod
    def set_new_target_pose(env):
        goal_pose = env.unwrapped.command_manager.get_command("target_pose")
        new_pos = goal_pose[..., :3].clone()
        new_pos[..., 2] = 0.0
        new_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=new_pos.device).expand(new_pos.shape[0], 4)

        root_state = torch.zeros((env.unwrapped.num_envs, 13), device=env.unwrapped.device)
        root_state[:, 0:3] = new_pos
        root_state[:, 3:7] = new_rot
        # Scrive la nuova pose alla simulazione
        env.unwrapped.scene["box"].write_root_state_to_sim(root_state)
    
    @staticmethod
    def clear_img_folder():
        if os.path.exists("./isaac_ws/src/output/camera"):
            shutil.rmtree("./isaac_ws/src/output/camera")
        if os.path.exists("./isaac_ws/src/output/episodes"):
            shutil.rmtree("./isaac_ws/src/output/episodes")
        os.mkdir("./isaac_ws/src/output/camera")
        os.mkdir("./isaac_ws/src/output/episodes")

    @staticmethod
    def hide_prim(prim_path: str):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            print(f"✅ Hidden prim: {prim_path}")
        else:
            print(f"⚠️ Prim '{prim_path}' not found or invalid.")