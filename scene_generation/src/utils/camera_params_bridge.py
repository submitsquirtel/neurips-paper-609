import matplotlib.pyplot as plt
import numpy as np
import torch
import os
'''
Calcualte camera Pose from 2D points and 3D points
Calculate the scale and translation from the 3D points and 2D points

'''
class CameraParams:
    def __init__(self, video, end_traj, depth_frames, K, gripper_points):
        
        self.video = video
        self.end_traj = end_traj
        self.depth_frames = depth_frames
        self.K = K
        self.gripper_points = gripper_points
        

    def get_points(self, frame):
        # Display the current frame
        query_points = [] 
        query_points_color = [] 
        query_count = 0
        img = frame
        plt.imshow(img)
        plt.title("Select 3 Points")
        
        # Let the user select 3 points
        selected_points = plt.ginput(3, timeout=0, show_clicks=True)
        
        plt.close()  # Close the plot after selecting points

        for point in selected_points:
            print(point)
            x, y = int(point[0]), int(point[1])
            print(f"You selected {(x, y, 0)}")

            # Add the selected point to the query points list
            query_points.append((x, y, 0))

            # Choose the color for the point from the colormap
            color = plt.cm.get_cmap("gist_rainbow")(query_count % 20 / 20)
            query_points_color.append(color)

            # Draw the point on the current frame for visualization
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.scatter(x, y, s=50, c=[color], edgecolors="black", linewidths=1)
            ax.axis("off")
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)

            # Update the query count
            query_count += 1

        return (
            img,  # Updated frame for preview
            query_points,  # Updated query points
            query_points_color,  # Updated query points color
            query_count  # Updated query count
        )

    def get_moge_video(self, pkl_path):
        import pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def track(self, video_preview, query_points, visualise=True):
        from base64 import b64encode
        query_points = torch.tensor(query_points)
        query_points[:, [0, -1]] = query_points[:, [-1, 0]]
        query_points = query_points.to(torch.float)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from cotracker.predictor import CoTrackerPredictor
        import os
        model = CoTrackerPredictor(
            checkpoint=os.path.join(
                './co-tracker/checkpoints/scaled_offline.pth'
            )
        )
        video = torch.from_numpy(np.array(self.video)).permute(0, 3, 1, 2)[None].float()
        if torch.cuda.is_available():
            query_points = query_points.cuda()   
            model = model.cuda()
            video = video.cuda()

        pred_tracks, pred_visibility = model(video, queries=query_points[None])
        tracks = pred_tracks.squeeze(0,2)
        tracks = tracks.cpu().detach().numpy()

        if visualise:
            from cotracker.utils.visualizer import Visualizer, read_video_from_path
            vis = Visualizer(save_dir='./videos',
            linewidth=6,
            mode='cool',
            tracks_leave_trace=-1)
            vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser');
            print("Tracking Visualisation saved at ./videos/teaser.mp4")
        return tracks
    
    def get_camera_pose(self, points_2d):
        points_3d = self.end_traj
        points_3d = np.array(points_3d, dtype=np.float32)
        
        points_3d += np.array([0, 0, 1])
        points_2d = np.array(points_2d, dtype=np.float32)
        K = self.K
        distCoeffs = np.zeros((4, 1), dtype=np.float32)

        import cv2
        from scipy.optimize import least_squares
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, distCoeffs, 
            iterationsCount=1000000, reprojectionError=10, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, distCoeffs)
            projected_points = projected_points.reshape(-1, 2)
            
            # Compute the reprojection ersror
            errors = np.linalg.norm(projected_points - points_2d, axis=1)
            
            # Sort points by reprojection error (ascending order)
            sorted_indices = np.argsort(errors)
            
            # Select the top 10 points with the least reprojection error
            top_indices = sorted_indices[:10]
            top_points_3d = points_3d[top_indices]
            top_points_2d = points_2d[top_indices]
            
            # Recalculate pose using the top 10 points
            success_refined, rvec_refined, tvec_refined = cv2.solvePnP(
                top_points_3d, top_points_2d, K, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success_refined:
                # Convert the refined rotation vector to a rotation matrix
                R_refined, _ = cv2.Rodrigues(rvec_refined)
                
                # Print the refined rotation matrix and translation vector
                print("Refined Rotation Matrix (R):", R_refined)
                print("Refined Translation Vector (t):", tvec_refined)
                
                # Project the points again to compute the mean reprojection error
                refined_projected_points, _ = cv2.projectPoints(
                    points_3d, rvec_refined, tvec_refined, K, distCoeffs
                )
                refined_projected_points = refined_projected_points.reshape(-1, 2)
                refined_errors = np.linalg.norm(refined_projected_points - points_2d, axis=1)
                mean_refined_error = np.mean(refined_errors)
                print("Mean Refined Reprojection Error:", mean_refined_error)
            else:
                print("Refined SolvePnP failed.")
        else:
            print("Initial SolvePnP failed.")        # print("Rotation Matrix (R):", R)
        # print("Translation Vector (t):")
        def reprojection_error(params, points_2d, points_3d, K):
            # Extract rotation (rvec), translation (tvec) and 3D points from params
            rvec = params[:3]
            tvec = params[3:6]
            points_3d_new = params[6:].reshape(-1, 3)

            # Convert rvec to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Project the 3D points using the current camera pose
            projected_points, _ = cv2.projectPoints(points_3d_new, rvec, tvec, K, np.zeros(4))

            # Flatten projected points for comparison
            projected_points = projected_points.reshape(-1, 2)
            
            # Compute the error between the observed and projected points
            error = projected_points - points_2d
            return error.flatten()

        def bundle_adjustment(points_3d, points_2d, K, rvec_init, tvec_init):
            # Flatten initial guess for optimization
            init_params = np.hstack((rvec_init.flatten(), tvec_init.flatten(), points_3d.flatten()))

            # Perform the optimization using least squares
            result = least_squares(reprojection_error, init_params, args=(points_2d, points_3d, K))

            # Extract the optimized parameters
            optimized_params = result.x
            rvec_opt = optimized_params[:3]
            tvec_opt = optimized_params[3:6]
            points_3d_opt = optimized_params[6:].reshape(-1, 3)

            # Return optimized results
            return rvec_opt, tvec_opt, points_3d_opt

        def rotation_error(R1, R2):
            R_error = R1 @ R2.T
            print("RE",R_error)
            cos_theta = (np.trace(R_error) - 1) / 2
            cos_theta = np.clip(cos_theta, -1, 1)  # Clamp for numerical stability
            return np.arccos(cos_theta)  # In radians

        def translation_error(t1, t2):
            return np.linalg.norm(t1 - t2)

        def extrinsic_error(T1, T2):
            # Extract rotation and translation
            R1, t1 = T1[:3, :3], T1[:3, 3]
            R2, t2 = T2[:3, :3], T2[:3, 3]
            # Compute errors
            rot_err = rotation_error(R1, R2)
            trans_err = translation_error(t1, t2)

            return rot_err, trans_err

        rvec_opt, tvec_opt, points_3d_opt = bundle_adjustment(points_3d, points_2d, K, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec_opt)

        T = np.eye(4)  # Initialize as 4x4 identity
        T[:3, :3] = R  # Set rotation
        T[:3, 3] = tvec_opt.flatten()  # Set translation

        estimated_transform = np.eye(4)
        estimated_transform =  np.linalg.inv(T)
        Rotations_correction =  np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        estimated_transform[:3, :3] =  estimated_transform[:3, :3] @ Rotations_correction

        return T
    
    def recover_scale_from_video(self, average_trajectory, R, tvec):
        
        intrinsic = self.K
        points_3d = self.end_traj
        non_metric_points = []
        real_depth = []
        depths = self.depth_frames
        
        for i, point in enumerate(average_trajectory):
            u = int(point[0])
            v = int(point[1])
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            Z = depths[i][int(point[0]), int(point[1])]
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            # X, Y, Z = R @ np.array([X, Y, Z]) + tvec
            non_metric_points.append(np.array([X, Y, Z]))

        non_metric_points = np.array(non_metric_points)
        point_3d_camera = []
        
        for point in points_3d:
            X, Y, Z = R @ np.array([point[0], point[1], point[2]]) + tvec
            point_3d_camera.append(np.array([X,Y,Z]))
        point_3d_camera = np.array(point_3d_camera)

        def ransac(points_1, points_2, max_trials=1000, threshold=0.01):
            best_scale = 0
            best_inliers = 0
            
            for _ in range(max_trials):
                # Randomly select a point
                idx = np.random.randint(0, len(points_1))
                p1, p2 = points_1[idx], points_2[idx]
                scale = p2 / p1 if p1 != 0 else 1
                
                residuals = np.abs(points_2 - scale * points_1)
                inliers = residuals < threshold
                num_inliers = np.sum(inliers)
                
                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    best_scale = scale
                    
            return best_scale
        
        scale = ransac(non_metric_points[:,2],point_3d_camera[:,2])
        return scale
    
    def find_base_position(self):
        query_points = self.gripper_points
        print("Taking manually selected points")
        

        # query_points = self.get_points(self.video[0])
        tracks  = self.track(self.video,query_points)
        average_trajectory = np.mean(tracks, axis=1)
        T0 = self.get_camera_pose(average_trajectory)
        # scale = self.recover_scale_from_video(average_trajectory, T0[:3,:3],T0[:3,3])
        print("Estimated Transform", T0)
        # print("Estimated Scale", scale)
        return T0
    