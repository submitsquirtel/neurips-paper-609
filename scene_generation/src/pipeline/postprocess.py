from src.utils.dataloader_utils import SamGemini
from src.utils.physics_utils import query_physical_properties_from_object_list
from src.utils.matching_utils import remove_noise
import os
import json
import numpy as np
import open3d as o3d

def reproject_to_plane(X_world ,K, T_world2cam, z_plane):
    """
    Project an arbitrary 3-D world point onto a given z-plane (default z = 0)
    while keeping its pixel location unchanged.

    Parameters
    ----------
    X_world : (3,) array_like
        Original world coordinate [X, Y, Z]^T.
    K : (3, 3) array_like
        Camera intrinsic matrix.
    T_world2cam : (4, 4) array_like
        Extrinsic matrix that converts homogeneous world coordinates
        to homogeneous camera coordinates, i.e.  [R | t] on top row,
        with the last row [0 0 0 1].
        X_cam_h = T_world2cam @ X_world_h
    z_plane : float, optional
        Target plane's z value. Default 0 (ground).
    Returns
    -------
    X_new : (3,) ndarray
        New world coordinate that lies on the specified z-plane and
        projects to the same pixel position as X_world.
    """
    # ---------- Cast inputs ----------
    X_world = np.asarray(X_world).reshape(3)
    K = np.asarray(K).reshape(3, 3)
    T_world2cam = np.asarray(T_world2cam).reshape(4, 4)
    # Split R and t out of the 4×4 for convenience
    R = T_world2cam[:3, :3]
    t = T_world2cam[:3, 3]
    # ---------- Step 1: world point → image pixel ----------
    X_cam = R @ X_world + t
    u_h = K @ X_cam
    u = u_h[:2] / u_h[2]
    # ---------- Step 2: back-project pixel to a camera-frame ray ----------
    u_homo = np.array([u[0], u[1], 1.0])
    ray_cam = np.linalg.inv(K) @ u_homo
    ray_world = R.T @ ray_cam
    # Camera centre in world frame
    cam_center_world = -R.T @ t
    # ---------- Step 3: intersect ray with the z-plane ----------
    if abs(ray_world[2]) < 1e-12:
        raise ValueError("Ray is parallel to the z-plane; intersection undefined.")
    t_scalar = (z_plane - cam_center_world[2]) / ray_world[2]
    X_new = cam_center_world + t_scalar * ray_world

    Z_orig = (R @ X_world + t)[2]
    Z_new  = (R @ X_new  + t)[2]
    scale_factor = Z_new / Z_orig
    return X_new, scale_factor


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', type=str, help='Path to the configuration file')
    parser.add_argument("--use_bypass", type=str2bool, default=False)
    parser.add_argument("--dataset", type=str, default="bridge")
    args = parser.parse_args()
    folder = os.path.join(args.scene_name)
    print(f'Processing folder: {folder}')
    if args.dataset == "bridge":
        config_path = 'configs/config_bridge.yaml'
    elif args.dataset == "rh20t":
        config_path = 'configs/config_rh20t.yaml'
    else:
        config_path = 'configs/config_DROID.yaml'
    
    sam = SamGemini(folder, use_bypass=args.use_bypass, dataset=args.dataset, config_path=config_path)
    pcd_folder = sam.pcd_dir
    transformation_file = sam.mask_dir
    position_file = os.path.join(sam.data_dir_task, "masks", "transformations.json")

    transformations = None
    with open(position_file, 'r') as f:
        transformations = json.load(f)
    new_json = position_file.replace(".json", "_new.json")
    new_dict = {}
    intrinsics = sam.intrinsic
    extrinsics = sam.extrinsic
    for key, attributes in transformations.items():
        count = key.split("_")[1]
        pcd_path = os.path.join(pcd_folder, count+"_frame1.ply")
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = remove_noise(pcd)
        points = np.asarray(pcd.points)
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        z_actual = (z_max - z_min) / 2
        # if args.dataset == "bridge":
        #     z_actual = 0.02
        rotation = attributes["rotation"]
        pos = attributes["translation"]
        pos = np.array(pos)
        pos_new, scale_factor = reproject_to_plane(pos, intrinsics, extrinsics, z_actual)
        scale = attributes["scale"] * scale_factor
        new_dict[key] = {
            "translation": pos_new.tolist(),
            "scale": float(scale),
            "rotation": rotation
        }
    with open(new_json, 'w') as f:
        json.dump(new_dict, f, indent=4)
