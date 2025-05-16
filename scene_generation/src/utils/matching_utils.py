import cv2
import numpy as np
import os
import open3d as o3d
class Unprojector:
    def __init__(self, intrinsics, extrinsics, depth, keypoints,pair,output_path):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.depth_npy = depth
        self.keypoints = keypoints
        self.pair = pair
        self.output_path = output_path
    
    def project_pixel_to_point(self, u, v, depth):
        fx, fy, cx, cy = self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        points = np.array([x, y, z, 1])
        extrinsics = np.linalg.inv(self.extrinsics)
        points = extrinsics @ points  # Apply extrinsics transformation
        return points[:3]

    def save_point_cloud(self, point_cloud, filename):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud(filename, pcl)

    # def unproject_to_point_cloud(self,X,Z,h,w,ww,hh,image_path):
    def unproject_to_point_cloud(self,X,Z,image_path):
        if self.pair == 0:
            H, W = self.depth_npy.shape
            point_cloud = []
            for v in range(H):
                for u in range(W):
                    depth = self.depth_npy[v, u]
                    if depth > 0: 
                        point = self.project_pixel_to_point(u, v, depth)
                        point_cloud.append(point)
            point_cloud = np.array(point_cloud)
        
        image = cv2.imread(image_path)
        points_2d = []
        for u, v in self.keypoints:
            x, y = int(u), int(v)
            if self.pair == 1:
                y += X
                x += Z
            points_2d.append([x,y])
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        
        if self.pair == 1:
            self.keypoints = points_2d
        
        keypoints_3d = []
        keypoints_3d_npy = []
        
        if self.pair == 0:
            for u, v in self.keypoints:
                depth = self.depth_npy[int(v), int(u)] 
                if depth <= 0:
                    keypoints_3d_npy.append([-1, -1, -1])
                if depth > 0: 
                    point_3d = self.project_pixel_to_point(u, v, depth)
                    keypoints_3d.append(point_3d)
                    keypoints_3d_npy.append(point_3d)

            keypoints_3d = np.array(keypoints_3d)
            keypoints_3d_npy = np.array(keypoints_3d_npy)
        
            np.save(os.path.join(self.output_path, f"keypoints3d{self.pair}.npy"), keypoints_3d_npy)
            self.save_point_cloud(keypoints_3d_npy, os.path.join(self.output_path, f"keypoints3d{self.pair}.ply"))
        

def apply_transformation(pcd, R, t):
    """
    Apply transformation (R, t) to an Open3D point cloud.
    """
    pcd_np = np.asarray(pcd.points)
    transformed_points = (R @ pcd_np.T).T + t
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    return pcd

def compute_rigid_transform(source_corr, target_corr):
    """
    Compute the optimal rigid transformation (R, t) from source to target using SVD.
    """
    assert source_corr.shape == target_corr.shape, "Point sets must have the same shape"

    # Compute centroids
    centroid_source = np.mean(source_corr, axis=0)
    centroid_target = np.mean(target_corr, axis=0)

    # Center the points
    source_centered = source_corr - centroid_source
    target_centered = target_corr - centroid_target

    # Compute cross-covariance matrix
    H = source_centered.T @ target_centered

    # Compute SVD
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T  # Rotation

    # Ensure a proper rotation matrix (det(R) = 1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_target - R @ centroid_source

    return R, t

def look_at_camera(camera_pos, look_at, up):
    L = look_at - camera_pos
    L_normalized = L / np.linalg.norm(L)
    s = np.cross(L_normalized, up)
    s_normalized = s / np.linalg.norm(s)
    u_prime = np.cross(s_normalized, L_normalized)

    R = np.vstack([s_normalized, u_prime, -L_normalized])  # Transpose to match the required format
    
    # Step 7: Compute the translation vector t = -R * camera_pos
    t = -np.dot(R, camera_pos)
    # Construct the 4x4 extrinsics matrix
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t
    return extrinsics

import cv2
import numpy as np

def crop_and_resize(image, padding=40, output_path=None):
    # Convert image to HSV color space
    coords = np.argwhere(np.any(image != [0, 0, 0], axis=-1)) 
    # Compute the bounding box around all non-black pixels
    x_min = coords[:, 1].min()
    y_min = coords[:, 0].min()
    x_max = coords[:, 1].max()
    y_max = coords[:, 0].max()
    
    # Apply padding while ensuring the crop does not exceed image boundaries
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.shape[1])
    y_max = min(y_max + padding, image.shape[0])

    # Crop the image based on the calculated bounding box
    cropped = image[y_min:y_max, x_min:x_max]

    # Save the cropped image if an output path is provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    return y_min, x_min, cropped


def get_uncropped_keypoints(keypoints,X,Z):
    points_2d = []
    for (u, v) in keypoints:
        x, y = int(u), int(v)
        # y =int(y * w/ww)
        # x =int( x * h/hh)
        y += X
        x += Z
        points_2d.append([x,y])
    return np.array(points_2d)

def project_pixel_to_point(u, v, depth, intrinsics, extrinsics=None): 
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    points = np.array([x, y, z, 1])
    extrinsics = np.linalg.inv(extrinsics)
    points = extrinsics @ points  # Apply extrinsics transformation
    return points[:3]

params = {
    "nb_neighbors": 80,
    "std_ratio": 0.5,
    "distance_threshold": 0.03,
    "ransac_n": 3,
    "num_iterations": 500
}
def remove_noise_neigh(pointcloud):
    clean, _ = pointcloud.remove_statistical_outlier(
        nb_neighbors=params["nb_neighbors"], std_ratio=params["std_ratio"]
    )
    return clean

def remove_noise(pointcloud):
    pointcloud = remove_noise_neigh(pointcloud)
    plane_model, inliers = pointcloud.segment_plane(
        params["distance_threshold"], params["ransac_n"], params["num_iterations"]
    )
    return pointcloud.select_by_index(inliers)
