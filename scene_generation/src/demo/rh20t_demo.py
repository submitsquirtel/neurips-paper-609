
import numpy as np
from src.sim.simulator_genesis import SimulatorGenesis
import genesis as gs
import cv2
from transforms3d.axangles import mat2axangle
from transforms3d.euler import euler2axangle, euler2mat, euler2quat
from transforms3d.quaternions import axangle2quat, mat2quat, quat2axangle, quat2mat
import os
import yaml
import argparse
import numpy as np
import json
import pickle
import torch

def main(scene_name, extrinsics,proprioception_data,intrinsics, background_path, asset_folder, 
        object_position, object_properties, robot_name = "ur5", bypass_sam = True, asset_position = None, 
        robot_position = None, robot_proproception = None):
    '''
    This function is an exampele funtion that shows how to use the simulator to generate a video of the robot moving
    We first load the proprioceptive data from the pickle file and then we use the simulator to move the robot to the
    desired position and then we render the image from the camera and save it to a list. We then use the animate function
    from the genesis library to create a video from the list of images.
    '''
    simulator = SimulatorGenesis(robot_name, 1, show_viewer=False, add_robot=True)
    simulator.start_sim()
    
    for key, attributes in object_position.items():
        asset_file = os.path.join(asset_folder, key + ".glb") 
        count = int(key.split("_")[1])
        if object_properties:
            physics = object_properties[count]
        else:
            physics = None
        if os.path.exists(asset_file):  
            pos = attributes["translation"]
            pos = np.array(pos)
            pos[-1] += 0.02
            scale = attributes["scale"]
            rotation = mat2quat(np.array(attributes["rotation"]))
            simulator.asset_addtion(asset_file, pos=np.array(pos), scale=scale, quat=rotation, physics=physics)
        else:
            print(f"Warning: File not found - {asset_file}")
    
    T = np.array(extrinsics)
    estimated_transform = T
    estimated_transform =  np.linalg.inv(T)
    Rotations_correction =  np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    estimated_transform[:3, :3] =  estimated_transform[:3, :3] @ Rotations_correction
    
    target_image = cv2.imread(background_path)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]  # Extract focal lengths from intrinsic matrix

    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    W, H = int(2 * cx), int(2 * cy) 
    target_image = cv2.resize(target_image, (W, H))
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    camera = simulator.scene.add_camera(
        res=(W,H),
        pos=(-0.095, -0.154387184491962615, 1.6611819729252184),
        lookat=(0.44, 0.20, 0.20),
        fov=fov_y,
        GUI=False,
    )
    camera_1 = simulator.scene.add_camera(
        res=(W,H),
        pos=(2, 1, 1),
        lookat=(0.44, 0.20, 0.20),
        fov=80,
        GUI=False,
    )
    simulator.scene.build()
    
    # set physics properties

    for _, value in simulator.get_asset_ID().items():
        asset = value[0]
        mass = value[1]
        friction = value[2]
        if mass is not None:
            asset.set_mass(mass)
        if friction is not None:
            asset.set_friction(friction)

    camera.set_pose(estimated_transform)
    video = []
    images =[]
    video_1 = []
    for key,value in proprioception_data.items():
        rgb1,_,_,_ = camera_1.render()
        video_1.append(rgb1)
        simulator.robot.set_qpos(np.array(value))
        for i in range(5):
            simulator.step()
        simulator.step()
        rgb, depth, seg,_ = camera.render(rgb=True, depth=True, segmentation=True)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        seg_binary = np.where(seg > 0, 255, 0).astype(np.uint8)
        segmented_part = cv2.bitwise_and(rgb, rgb, mask=seg_binary)
        mask_inv = cv2.bitwise_not(seg_binary)
        target_background = cv2.bitwise_and(target_image, target_image, mask=mask_inv)
        result = cv2.addWeighted(segmented_part, 1, target_background, 1, 0)
        images.append(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        video.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    os.makedirs(f'./data/output/{scene_name}', exist_ok=True)
    gs.tools.animate(images, f'./data/output/{scene_name}/demo_x.mp4', fps=30)
    gs.tools.animate(video, f'./data/output/{scene_name}/real_x.mp4', fps=30)
    gs.tools.animate(video_1, f'./data/output/{scene_name}/real1_x.mp4', fps=30)

def load_config(config_path="config_rh20t.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def loader():
    parser = argparse.ArgumentParser(description='Processing Scene Generation')
    parser.add_argument('--config', type=str, default="configs/rh20t_config.yaml", help='Path to the YAML config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # hardcoding for testing
    if config["data_type"] == "bridge":
        base_folder = "./data/bridge"
    else:
        base_folder = "./data/scene"

    extrinsics = np.load(f'{base_folder}/{config["scene_name"]}/extrinsics.npy', allow_pickle=True)
    intrinsics = np.load(f'{base_folder}/{config["scene_name"]}/intrinsics.npy', allow_pickle=True)

    
    # If exists, load the object position from the config
    if "object_position" in config:
        with open(config["object_position"], 'r') as f:
            object_positions = json.load(f)
    else:
        # initialize object_positions with default values
        object_positions = None
    proprioception_data = np.load(config['proprioception_path'], allow_pickle=True)
    if isinstance(proprioception_data, np.ndarray) and proprioception_data.dtype == object:
        proprioception_data = proprioception_data.item()
    if isinstance(intrinsics, np.ndarray) and intrinsics.dtype == object:
        intrinsics = intrinsics.item()
    background_path = f'./data/scene_background/{config["scene_name"]}/background.png'
    asset_path = f'./data/assets/{config["scene_name"]}'
    physic_properties_path = os.path.join(base_folder, config["scene_name"], "physical_properties.json")
    
    # assert os.path.exists(physic_properties_path), f"Physical properties file not found at {physic_properties_path}"
    object_properties = None
    physic_properties = None
    if os.path.exists(physic_properties_path):
        with open(physic_properties_path, 'r') as f:
            physic_properties = json.load(f)
    # Run main function
    main(
        config["scene_name"],
        np.array(extrinsics),
        proprioception_data[config['camera_to_use']],
        np.array(intrinsics),
        background_path,
        asset_path,
        object_positions,
        physic_properties
    )

if __name__ == "__main__":
    loader()