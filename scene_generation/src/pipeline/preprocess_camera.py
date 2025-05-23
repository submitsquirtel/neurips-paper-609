import os
import json
import argparse
import numpy as np
import yaml
import cv2
from src.utils.transforms import pose_array_quat_2_matrix, pose_array_euler_2_pose_array_quat, pose_array_quat_2_pose_array_euler,matrix_2_pose_array_quat,pose_array_rotvec_2_matrix
from src.utils.camera_params_bridge import CameraParams
import pickle
import torch
import math
import json
import glob
import random

def bound_to_0_2pi(num): return (num + np.pi * 2 if num < 0 else num)

def tcp_as_q(tcp: np.ndarray):
    assert tcp.shape == (6,), tcp
    tcp_q = pose_array_euler_2_pose_array_quat(tcp[0:6])
    return np.concatenate((tcp_q, tcp[6:]), axis=0)

def tcp_preprocessor(robot: str, tcp: np.ndarray):
    if robot == "flexiv": 
        return tcp
    elif robot == "franka": 
        return pose_array_quat_2_pose_array_euler(matrix_2_pose_array_quat(pose_array_rotvec_2_matrix(tcp) @ np.array([[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 0], [0., 0., 0., 1.]], dtype=np.float64)))
    elif robot == "kuka":
        return np.array([tcp[0] / 1000., tcp[1] / 1000., tcp[2] / 1000., bound_to_0_2pi(tcp[3]), bound_to_0_2pi(tcp[4]), bound_to_0_2pi(tcp[5])], dtype=np.float64)  
class CameraCalibration:
    def __init__(self, data_type, data_dir, cfg, camera_to_use, depth_folder, 
                 color_folder, scene_name, raw_path, scene_names=None):
        self.data_dir = data_dir
        self.data_type = data_type
        self.camera_to_use = camera_to_use
        self.depth_folder = depth_folder
        self.color_folder = color_folder
        self.scene_name = scene_name
        self.raw_path = raw_path
        self.scene_names = scene_names
        self.cfg = cfg
    
    def process_rh20t_scenes(self):
        for scene_name in self.scene_names:
            self.scene_name = scene_name
            self.process_rh20t()
    
    def process_rh20t(self):
        # check if there is a folder with the scene name
        if self.scene_name not in os.listdir(self.data_dir):
            # Get the scene
            scene = os.path.join(self.data_dir, self.scene_name)
            os.makedirs(scene, exist_ok=True)
            camera = f"cam_{self.camera_to_use}"
            original_rgb = os.path.join(self.raw_path, f"RH20T_cfg{self.cfg}", self.scene_name, camera, "color.mp4")
            original_depth = os.path.join(self.raw_path, f"RH20T_cfg{self.cfg}_depth", self.scene_name, camera, "depth.mp4")
            joint_path = os.path.join(self.raw_path, f"RH20T_cfg{self.cfg}", self.scene_name, "transformed", "joint.npy")
            gripper_path = os.path.join(self.raw_path, f"RH20T_cfg{self.cfg}", self.scene_name, "transformed", "gripper.npy")
            metadata = os.path.join(self.raw_path, f"RH20T_cfg{self.cfg}", self.scene_name, "metadata.json")
            metadata = json.load(open(metadata, 'r'))
            camera_calib = metadata["calib"]
            camera_folder = os.path.join(self.raw_path, f"RH20T_cfg{self.cfg}", "calib", f"{camera_calib}")
            ori_intrinsics = os.path.join(camera_folder, "intrinsics.npy")
            ori_extrinsics = os.path.join(camera_folder, "extrinsics.npy")
            ori_tcp = os.path.join(camera_folder, "tcp.npy")
            config = os.path.join(self.raw_path, f"RH20T_cfg{self.cfg}", "calib", "configs.json")
            
            # copy the color and depth files
            os.system(f"cp {original_rgb} {os.path.join(scene, 'color.mp4')}")
            os.system(f"cp {original_depth} {os.path.join(scene, 'depth.mp4')}")
            os.system(f"cp {joint_path} {os.path.join(scene, 'joint.npy')}")
            os.system(f"cp {gripper_path} {os.path.join(scene, 'gripper.npy')}")
            os.system(f"cp {ori_intrinsics} {os.path.join(scene, 'intrinsics.npy')}")
            os.system(f"cp {ori_extrinsics} {os.path.join(scene, 'extrinsics.npy')}")
            os.system(f"cp {ori_tcp} {os.path.join(scene, 'tcp.npy')}")
            os.system(f"cp {config} {os.path.join(scene, 'configs.json')}")
            print(f"Scene {self.scene_name} copied successfully")

        for folder in os.listdir(self.data_dir):
            if not folder == self.scene_name:
                continue
            intrinsics_file = os.path.join(self.data_dir, folder, "intrinsics.npy")
            extrinsics_file = os.path.join(self.data_dir, folder, "extrinsics.npy")

            if not (np.load(intrinsics_file,allow_pickle=True).shape == (3, 3)):
                json_file = os.path.join(self.data_dir, folder, "configs.json")
                with open(json_file, 'r') as f:
                    data = json.load(f)
                robot =  data['robot']
                
                calib_tcp = np.load(os.path.join(self.data_dir, folder, "tcp.npy")) if robot == "ur5" else tcp_preprocessor(robot, np.load(os.path.join(self.data_dir, folder, "tcp.npy")))
                
                intrinsics = np.load(intrinsics_file, allow_pickle=True).item()
                extrinsics = np.load(extrinsics_file, allow_pickle=True).item()

                if len(calib_tcp) == 6:
                    calib_tcp = tcp_as_q(calib_tcp)
            
                in_hand_serials = data['in_hand']
                tcp_camera_mat = np.array(data['tc_mat'])
                align_mat_base = np.array(data['align_mat_base'])
                
                base_world_mat = (
                    np.linalg.inv(extrinsics[in_hand_serials[0]])
                    @ tcp_camera_mat
                    @ np.linalg.inv(pose_array_quat_2_matrix(calib_tcp))
                )
                extrinsics_dict = {
                    k: extrinsics[k] @ base_world_mat @ align_mat_base
                    for k in extrinsics
                }
                # read intrinsics and extrinsics
                intrinsic = intrinsics[self.camera_to_use]
                extrinsic = extrinsics_dict[self.camera_to_use]
                # save intrinsics and extrinsics to json
                extrinsics_npy = os.path.join(self.data_dir, folder, "extrinsics.npy")
                np.save(extrinsics_npy, extrinsic[0])
                
                intrinsics = np.array([
                    [0.5 * intrinsic[0, 0], 0, 0.5 * intrinsic[0, 2]],
                    [0, 0.5 * intrinsic[1, 1], 0.5 * intrinsic[1, 2]],
                    [0, 0, 1]
                ])
                intrinsics_npy = os.path.join(self.data_dir, folder, "intrinsics.npy")
                np.save(intrinsics_npy, intrinsics)
                
            self.process_normal()
        
    def process_normal(self):      
        # Process Depth and RGB images
        cap = cv2.VideoCapture(os.path.join(self.data_dir, self.scene_name, "color.mp4"))
        frame_index = 0
        color_folder = os.path.join(self.data_dir, self.scene_name, self.color_folder)
        os.makedirs(color_folder, exist_ok=True)
        depth_folder = os.path.join(self.data_dir, self.scene_name, self.depth_folder)
        os.makedirs(depth_folder, exist_ok=True)
        
        width, height = None, None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  
            width, height = frame.shape[1], frame.shape[0]
            frame_path = os.path.join(color_folder, f"{frame_index}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_index += 1
        cap.release()

    
        cap = cv2.VideoCapture(os.path.join(self.data_dir, self.scene_name, "depth.mp4"))

        cnt = 0
        if self.data_type == "rh20t":
            min_depth_m=0.1
            max_depth_m=3.0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if self.data_type == "rh20t":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray1 = np.array(gray[:height, :]).astype(np.int32)
                    gray2 = np.array(gray[height:, :]).astype(np.int32)
                    gray = np.array(gray2 * 256 + gray1).astype(np.float32)
                    depth = cv2.resize(gray, (int(width ), int(height )))
                    depth /= 1000.0
                    depth[depth < min_depth_m] = 0
                    depth[depth > max_depth_m] = 0
                    # save depth as npy
                    depth_path = os.path.join(depth_folder, f"{cnt}.npy")
                    np.save(depth_path, depth)
                cnt += 1
            else:
                break
        cap.release()
    
    def preprocess(self):
        if self.data_type == "rh20t":
            self.process_rh20t_scenes()
        elif self.data_type == "bridge":
            self.process_bridge()
        else:
            self.process_normal()

    def process_bridge(self, save_video=True):
        data_dir = self.data_dir
        for folder in os.listdir(data_dir):
            if folder.startswith("scene"):
                # if alread have preprocessed, skip
                if os.path.exists(os.path.join(data_dir, folder, "intrinsics.npy")) and os.path.exists(os.path.join(data_dir, folder, "extrinsics.npy")):
                    continue
                if os.path.exists(os.path.join(data_dir, folder, "video.mp4")):
                    continue
                frames_path = os.path.join(data_dir, folder, "images0")
                frames = []
                frame_files = sorted(os.listdir(frames_path), key=lambda x: int(x.split("_")[1].split(".")[0]))
                for frame in frame_files:
                    old_path = os.path.join(frames_path, frame)
                    frame_id = int(frame.split("_")[1].split(".")[0])
                    new_name = f"{frame_id}.jpg"
                    new_path = os.path.join(frames_path, new_name)
                    img = cv2.imread(os.path.join(frames_path, frame))
                    # rename the file from `im_0.jpg` to `0.jpg`
                    frames.append(img)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                
                # make a video from the frames
                if frames != [] and save_video:
                    height, width, _ = frames[0].shape
                    output_path = os.path.join(data_dir, folder, "video.mp4")
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    print(f"Video saved to {output_path}")                    
                
def main():
    parser = argparse.ArgumentParser(description="Camera Calibration Script")
    parser.add_argument("--data_type", type=str, default = "rh20t")
    parser.add_argument("--data_dir", type=str, default = "./data/scene")
    parser.add_argument("--camera_to_use", type=str, default = "036422060215")
    parser.add_argument("--depth_folder", type=str, default = "depth")
    parser.add_argument("--color_folder", type=str, default = "images0")
    parser.add_argument("--scene_name", type=str, default = "scene0")
    parser.add_argument("--raw_path", type=str)
    parser.add_argument("--depth", type=int, default = 4)
    parser.add_argument("--num_scenes", type=int, default = 5)
    parser.add_argument("--cfg", type=int, default = 5)
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    if args.data_type == "bridge":
        # Create a list of args.depth wildcards
        wildcards = ['*'] * args.depth
        pattern = os.path.join(args.raw_path, *wildcards, 'raw', 'traj_group*', 'traj0*')
        traj_paths = glob.glob(pattern)
        random.shuffle(traj_paths)
        traj_paths = traj_paths[:args.num_scenes]
        
        # if is args.data_dir is empty, no subfolders
        if len(os.listdir(args.data_dir)) == 0:
            base_count = 0
        else:
            base_count = max([int(x.split("scene")[1]) for x in os.listdir(args.data_dir) if x.startswith("scene")]) + 1
            
        for traj_path in traj_paths:
            os.system(f"cp -r {traj_path} {args.data_dir}")
            # rename from traj0 to scene{base_count}
            os.system(f"mv {args.data_dir}/traj0 {args.data_dir}/scene{base_count}")
            base_count += 1
    
    scene_names = []
    if args.data_type == "rh20t":
        rh20t_scenes = os.path.join(args.raw_path, f"RH20T_cfg{args.cfg}")
        for i, folder in enumerate(os.listdir(rh20t_scenes)):
            if folder.startswith("task") and( not folder.endswith("human")):
                scene_names.append(folder)
            if i >= args.num_scenes:
                break

    calibration = CameraCalibration(args.data_type, args.data_dir, args.cfg, args.camera_to_use, 
                                    args.depth_folder, args.color_folder, args.scene_name, 
                                    args.raw_path, scene_names)
    calibration.preprocess()
    
if __name__ == "__main__":
    main()
