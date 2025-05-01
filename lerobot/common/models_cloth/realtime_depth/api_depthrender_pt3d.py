import os, sys, time
import cv2, torch
import numpy as np
import open3d as o3d
import random
import pyrealsense2 as rs

# PyTorch3D
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    FoVPerspectiveCameras,
    look_at_view_transform,
    TexturesVertex,)
from pytorch3d.renderer.mesh.shader import ShaderBase

class CustomDepthShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        # Obtain the z-buffer from the rasterizer output.
        zbuf = fragments.zbuf[..., 0]  # Shape: [N, H, W]

        # Define the depth range for your object and the desired target range.
        zmin, zmax = 0.70, 0.82
        # zmin, zmax = 0.0, 1.5
        target_min, target_max = 0.1, 0.8

        # Apply normalization for depth values within the range.
        depth_norm = ((zbuf - zmin) / (zmax - zmin)) * (target_max - target_min) + target_min

        # For pixels with no valid geometry, the z-buffer is usually inf or a large constant.
        # Create a mask for valid geometry.
        valid_mask = torch.isfinite(zbuf) & (zbuf >= zmin) & (zbuf <= zmax)
        depth_norm[~valid_mask] = 1.0  # Background as white (1.0)

        # Clamp values to ensure the output is within the [0, 1] domain.
        depth_norm = torch.clamp(depth_norm, 0.0, 1.0)

        # Expand to a 3-channel (grayscale) image.
        image = depth_norm.unsqueeze(-1).expand(-1, -1, -1, 3)
        return image


class DepthRendering():
    def __init__(self,
                 predictor,
                 width,
                 height,
                 pipeline,
                 decimation,
                 device
                 ):
        self.predictor = predictor
        self.pipeline = pipeline
        self.decimation = decimation
        self.device = device

        self.width = width
        self.height = height

        self.is_init = False
        self.prompt_points = np.array([[1000, 600]], dtype=np.float32)
        self.prompt_labels = np.array([1], dtype=np.int32)

        # camera setup
        R, T = look_at_view_transform(dist=0.8, 
                                      elev=0, 
                                      azim=0)
        
        self.cameras = FoVPerspectiveCameras(device=self.device, 
                                             R=R, 
                                             T=T, 
                                             fov=40.0, 
                                             znear=0.01, 
                                             zfar=100.0)
        
        # render the mesh to get the depth
        self.depth_shader = CustomDepthShader()
        self.raster_settings = RasterizationSettings(image_size=512, 
                                                     blur_radius=0.0001, 
                                                     faces_per_pixel=20)
        
        self.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras, 
                                                          raster_settings=self.raster_settings),
                                     shader=CustomDepthShader(device=self.device, 
                                                          cameras=self.cameras))
        
    def get_one_frame(self):
        # get the frames
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Didn't get frame")
            return None
        
        # get the pcl
        depth_frame_decimated = self.decimation.process(depth_frame)
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame_decimated)

        color_np = np.asanyarray(color_frame.get_data())
        color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        return color_np, points

    def get_mask_sam2(self, frame):
        if not self.is_init:
            # initialize mask tracking
            self.predictor.load_first_frame(frame)
            _, obj_ids, mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, 
                obj_id=1, 
                points=self.
                prompt_points, 
                labels=self.prompt_labels
                )
        else:
            # keep tracking by input new frame
            obj_ids, mask_logits = self.predictor.track(frame)

        # get mask by checking percentage
        mask = (mask_logits[0] > 0.0).permute(1, 2, 0)
        return mask

    def get_pcl_with_mask(self, mask, points):
        # pcl in camera coordinate
        pcl_np = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # (N,) array of rs_vertex
        pcl = torch.from_numpy(pcl_np).to(self.device)   # Convert to (N, 3) tensor

        # pcl in pixel coordinate | with value from 0 - 1
        pcl_coords_np = np.array(points.get_texture_coordinates())  # (N,) 
        pcl_coords_np = np.column_stack((pcl_coords_np['f0'], pcl_coords_np['f1']))  # shape (N, 2)
        pcl_coords = torch.from_numpy(pcl_coords_np).to(self.device)

        # Convert pcl_coords from [0, 1] to pixel coords
        u = (pcl_coords[:, 0] * self.width).int().clamp(0, self.width - 1)
        v = (pcl_coords[:, 1] * self.height).int().clamp(0, self.height - 1)

        mask_pcl = mask[v, u, :].view(-1).bool()
        filtered_pcl = pcl[mask_pcl]
        return filtered_pcl

    def centralize_mesh_torch(self, pred_vertices):
        """
        Centralizes a 3D mesh using PyTorch tensors by calculating its bounding box
        and shifting its vertices to the origin.

        Args:
            pred_vertices (torch.Tensor): A tensor of shape (N, 3), representing the vertices of the mesh.

        Returns:
            centralized_vertices (torch.Tensor): The centralized vertices tensor.
            bbox_min (torch.Tensor): The minimum coordinates of the bounding box.
            bbox_max (torch.Tensor): The maximum coordinates of the bounding box.
            bbox_center (torch.Tensor): The center of the bounding box.
        """
        
        # inverse z-axis
        centralized_vertices = pred_vertices.clone()

        # Compute the bounding box
        # print(f"shape of pred_vetices: {pred_vertices.shape}")
        bbox_min = torch.min(pred_vertices, dim=0).values  # Minimum x, y
        bbox_max = torch.max(pred_vertices, dim=0).values  # Maximum x, y
        # Compute the center of the bounding box
        bbox_center = (bbox_min + bbox_max) / 2.0

        # # Shift vertices to centralize the mesh
        # centralized_vertices[:, 0:1] = pred_vertices[:, 0:1] - bbox_center[0:1]
        centralized_vertices = pred_vertices - bbox_center
        centralized_vertices[:, 2] = -centralized_vertices[:, 2]

        return centralized_vertices
    
    def pcl_to_mesh_o3d(self, pcl_np, pcd_o3d):
        # convert o3d mesh to pytorch3d mesh

        pcd_o3d.points = o3d.utility.Vector3dVector(pcl_np)
        # pcd = pcd.voxel_down_sample(voxel_size=0.01)

        pcd_o3d.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        pcd_o3d.estimate_normals()
        pcd_o3d.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pcd, world_frame], point_show_normal=True)

        distances = pcd_o3d.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances) # 0.003

        # create a mesh from the point cloud using ball pivoting
        radii = [avg_dist/2, avg_dist, avg_dist*2, avg_dist*2.5]
        tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd_o3d, o3d.utility.DoubleVector(radii))
        # o3d.visualization.draw_geometries([pcd, tri_mesh])
        pcd_o3d.points = o3d.utility.Vector3dVector(pcl_np)
        return tri_mesh, pcd_o3d
    
    def render_depth_pytorch3d(self, tri_mesh):
        # convert to pytorch3d mesh
        vertices_np = np.array(tri_mesh.vertices, dtype=np.float32)
        faces_np = np.array(tri_mesh.triangles, dtype=np.int64)
        vertices_torch = torch.tensor(vertices_np).to(self.device)
        faces_torch = torch.tensor(faces_np).to(self.device)
        vertex_colors = torch.zeros_like(vertices_torch)[None]  # Assign a simple color (black color here, shape=[1, V, 3])
        pytorch3d_mesh = Meshes(verts=[vertices_torch], faces=[faces_torch], textures=TexturesVertex(verts_features=vertex_colors))

        images = self.renderer(pytorch3d_mesh)
        image_np = images[0, ..., :3].cpu().numpy()
        return image_np

    def rendering_pipeline(self, pcd_o3d):
        # get one frame from realsense pipeline
        t0 = time.time()
        color_frame, points = self.get_one_frame()      # color_frame: h * w *3
        t1 = time.time()
        # get the mask of cloth using sam2
        mask = self.get_mask_sam2(color_frame)          # mask: h * w | 0 or 1
        t2 = time.time()
        # get the cloth pcl using mask as numpy array
        pcl = self.get_pcl_with_mask(mask, points)
        t3 = time.time()
        # get the cloth mesh using pcl
        pcl_np = self.centralize_mesh_torch(pcl).cpu().numpy()
        t4 = time.time()
        # transfer pcl to mesh using o3d
        mesh_o3d, pcd_o3d = self.pcl_to_mesh_o3d(pcl_np, pcd_o3d)
        t5 = time.time()
        # render depth image using pytorch3d
        image = self.render_depth_pytorch3d(mesh_o3d)
        t6 = time.time()
        print(f'get image: {t1-t0}s, get mask: {t2-t1}s, filter pcl: {t4-t2}s, get mesh: {t5-t4}s, render depth: {t6-t5}s')
        return image, color_frame, mask ,mesh_o3d, pcd_o3d
