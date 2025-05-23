import genesis as gs
import numpy as np
import torch
import trimesh
from configs import robot_config

def process_surface(surface):
    match surface:
        case "Default":
            return gs.surfaces.Default()
        case "Glass":
            return gs.surfaces.Glass()
        case "Water":
            return gs.surfaces.Water()
        case "Emission":
            return gs.surfaces.Emission()
        case "Plastic":
            return gs.surfaces.Plastic()
        case "Rough":
            return gs.surfaces.Rough()
        case "Smooth":
            return gs.surfaces.Smooth()
        case "Reflective":
            return gs.surfaces.Reflective()
        case "Metal":
            return gs.surfaces.Metal()
        case "Iron":
            return gs.surfaces.Iron()
        case "Aluminium":
            return gs.surfaces.Aluminium()
        case "Copper":
            return gs.surfaces.Copper()
        case "Gold":
            return gs.surfaces.Gold()
        case _:
            return gs.surfaces.Default()
        # case _:
        #     raise Exception(f"Unknown surface: {surface}")


class SimulatorGenesis:
    def __init__(self, robot_name, envs, add_robot = None, device = None, show_viewer =False, seed=None):
        self.robot_name = robot_name
        self.robot = None
        self.envs = envs
        self.seed = seed
        self.asset_path = None
        self.device = device
        self.show_viewer = show_viewer
        self.robot_addition = add_robot
        self.asset_ID = {}
    
    def get_asset_ID(self):
        return self.asset_ID

    def asset_addtion(self, asset_path, pos = None, 
                      quat = None, scale = None, 
                      physics = None, fixed = False):
        # TODO : Add right Position and Quat for the assests
        if pos is None:
            pos = (1, 0.5, 1.0)
        if quat is None:
            quat = (0.717, 0.717, 0, 0)
        if scale is None:
            scale = 0.12
        if physics is None:
            asset = (self.scene.add_entity(
            gs.morphs.Mesh(file=asset_path,
                        fixed=fixed,
                        pos= pos,
                        quat= quat,
                        scale= scale,
                        ),
            ),
            None,
            None,
            )
        else:
            if "material" in physics:
                surface = process_surface(physics["material"])
            else:
                surface = process_surface(physics["surface"])
            asset = (self.scene.add_entity(
            gs.morphs.Mesh(file=asset_path,
                        fixed=fixed,
                        pos= pos,
                        quat= quat,
                        scale= scale,
                        convexify= True,
                        ),
                surface = surface
            ),
            physics["mass"],
            physics["friction"],
            )
        self.asset_ID[scale] = asset
        return quat

    def set_scene(self):
        viewer_options = gs.options.ViewerOptions(
                camera_pos=(3, -1, 2.5),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=30,
                max_FPS=60,
            )

        self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt= 0.005,
                ),
                viewer_options=viewer_options,
                vis_options=gs.options.VisOptions(
                    show_world_frame=False,
                    shadow=False,
                    ambient_light=(0.4, 0.4, 0.4),
                ),
                rigid_options=gs.options.RigidOptions(
                    dt=0.005,
                ),
                show_viewer=self.show_viewer,
                show_FPS=False,
            )
        # block = self.scene.add_entity(
        #     gs.morphs.Box(
        #         fixed=True,
        #         size=(0.4, 0.4, 0.4),
        #         pos=(0.8, 0.0, -0.2),
        #     ),
        # )
        plane = self.scene.add_entity(
            gs.morphs.Plane(
                fixed=True,
                visualization=False,
            ),
        # gs.morphs.URDF(file='urdf/plane/plane.urdf', 
        #                fixed=True,
        #                visualization=False,
        #                # visualization=True,
        #                ),
            )
        return self.scene
    
    def add_robot(self):
        robot_path = robot_config[self.robot_name]["path"]
        position = robot_config[self.robot_name]["position"]
        quaternion = robot_config[self.robot_name]["quaternion"]
        
        import os
        ext = os.path.splitext(robot_path)[1].lower()  # Get file extension

        if ext == ".xml":
            self.robot = self.scene.add_entity(gs.morphs.MJCF(file=robot_path,
                # fixed=True,
                # merge_fixed_links=True,
                # links_to_keep =  [robot_config[self.robot_name]["ee_link"]] ,
                pos= (0,0,0.0),
                #euler= (0,0,1.57),
                ),)
        elif ext == ".urdf":
            self.robot = self.scene.add_entity(gs.morphs.URDF(file=robot_path,
                fixed=True,
                merge_fixed_links=False,
                links_to_keep =  [robot_config[self.robot_name]["ee_link"]] ,
                pos= (0,0,0.0),),)
        else:
            return "Unknown file type"
        
        # hand = self.scene.add_entity(
        #     gs.morphs.URDF(file="urdf/panda_bullet/hand.urdf", merge_fixed_links=False, fixed=True),
        # )
        
        # self.scene.link_entities(self.robot, hand, "tool0", "hand")
        return self.robot
        
    def gs_transform_by_quat(self, pos, quat):
        qw, qx, qy, qz = quat.unbind(-1)

        rot_matrix = torch.stack([
            1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
        ], dim=-1).reshape(*quat.shape[:-1], 3, 3)

        rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

        return rotated_pos

    def transform_point_cloud(self, point_cloud, translation, quaternion):
        rotated_points  = self.gs_transform_by_quat(point_cloud, quaternion)
        translated_points = rotated_points  + translation
        return translated_points


    def get_point_cloud(self):
        res = []
        self.latest_pc = []
        for link in self.robot.links:
            link_points = []
            for mesh in [g.get_trimesh() for g in link.vgeoms]:
                points, face_indices = trimesh.sample.sample_surface_even(mesh, count=1000)
                link_points.append(points)
            res.append(np.concatenate(link_points, axis=0))
        
        pos = self.robot.get_links_pos()# (B, n_links, 3)
        quat = self.robot.get_links_quat()  # (B, n_links, 4)
        extra_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=gs.device)
        extra_quat = torch.tensor([0.7071, 0, 0, 0.7071], dtype=torch.float32, device=gs.device)
        for i in range(len(res)):
            if (i == 0) | (i == 1) | (i == 2)  :
                rotation_matrix = np.array([
                    [0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]
                ])

                res[i] = np.dot(res[i], rotation_matrix.T)
            elif (i == 3) :
                rotation_matrix = np.array([
                    [0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]
                ])

                res[i] = np.dot(res[i], rotation_matrix.T)
                res[i] = res[i] + np.array([-0.00, 0.0, 0.0])
            elif (i == 4) :
                rotation_matrix = np.array([
                    [0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]
                ])

                res[i] = np.dot(res[i], rotation_matrix.T)
                res[i] = res[i] + np.array([-0.0, 0.0, 0.0])
            elif (i == 5) :
                rotation_matrix = np.array([
                    [0, 1, 0],
                    [-1,  0, 0],
                    [0,  0, 1]
                ])


                res[i] = np.dot(res[i], rotation_matrix)
                res[i] = res[i] + np.array([-0.02, 0.0, 0.0])
            elif (i == 6) :
                rotation_matrix = np.array([
                    [0, 1, 0],
                    [-1,  0, 0],
                    [0,  0, 1]
                ])


                res[i] = np.dot(res[i], rotation_matrix)
                res[i] = res[i] + np.array([-0.069, 0.0, 0.0])
            elif (i == 7) :
                rotation_matrix = np.array([
                    [-1,  0,  0],
                    [ 0, -1,  0],
                    [ 0,  0,  1]
                ])
                res[i] = np.dot(res[i], rotation_matrix)
                res[i] = res[i] + np.array([0.00, 0.0, 0.0])
            elif (i == 8):

                rotation_matrix = np.array([
                    [-1,  0,  0],
                    [ 0,  1,  0],
                    [ 0,  0, -1]
                ])
                res[i] = np.dot(res[i], rotation_matrix)
                
                res[i] = res[i] + np.array([0.00, 0.0, 0.0])

                
            t = self.transform_point_cloud(torch.tensor(res[i], dtype=torch.float32).to(gs.device), pos[i], quat[i])
            # t = self.transform_point_cloud(t, extra_pos, extra_quat)

            res[i] = t.cpu().numpy()
            self.latest_pc.append(t)
        return res

    def control_ee_pose(self, target_pos, target_quat):
        q = self.robot.inverse_kinematics(
            link= self.robot.get_link(robot_config[self.robot_name]["ee_link"]),
            pos= target_pos,
            quat= (0,0,0,1),
        )

        self.robot.control_dofs_position(q)

    def start_sim(self):
        gs.init(backend=gs.cpu if self.device is not None else gs.gpu)
        self.set_scene()
        if self.robot_addition:
            self.add_robot()
        
    def set_camera(self):
        cam_0 = self.scene.add_camera(
        pos=(2.5, -0.15, 2.42),
        lookat=(0.5, 0.5, 0.1),
        res=(1280, 720),
        fov=30,
        GUI=False,
        )

    def visualize_pc(self):
        self.scene.clear_debug_objects()
        for i in range(len(self.latest_pc)):
            self.scene.draw_debug_spheres(poss=self.latest_pc[i], radius=0.001, color=(0.8, 0.8, 0, 0.7))

    def step(self):
        self.scene.step()
