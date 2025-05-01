import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time
import cv2
import torch
import matplotlib.pyplot as plt

# PyTorch3D
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    FoVPerspectiveCameras,
    look_at_view_transform,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.shader import ShaderBase

def centralize_mesh_torch(pred_vertices):
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
    centralized_vertices[:, 2] = -pred_vertices[:, 2]

    # Compute the bounding box
    # print(f"shape of pred_vetices: {pred_vertices.shape}")
    bbox_min = torch.min(pred_vertices, dim=0).values  # Minimum x, y
    bbox_max = torch.max(pred_vertices, dim=0).values  # Maximum x, y
    # Compute the center of the bounding box
    bbox_center = (bbox_min + bbox_max) / 2.0

    # Shift vertices to centralize the mesh
    centralized_vertices = pred_vertices - bbox_center

    return centralized_vertices

def get_pcd(pipeline, pcd, device, color_w, color_h):
    """
    Acquire frames from RealSense, convert to point cloud,
    then filter by both a rectangular ROI and a 'black' color threshold,
    using a vectorized approach (no Python for-loop over every point).
    """
    # Wait for frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None
    # Apply decimation to reduce depth resolution
    depth_frame_decimated = decimation.process(depth_frame)

    # Use the built-in PointCloud class to obtain the point cloud
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame_decimated)

    pcl_np = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # (N,) array of rs_vertex
    pcl = torch.from_numpy(pcl_np).to(device)   # Convert to (N, 3) tensor

    # ------------------------------------------------
    # ------------------- ROI mask -------------------
    # ------------------------------------------------
    # Retrieve texture coordinates (u,v) in range [0..1]
    pcl_coords_np = np.array(points.get_texture_coordinates())  # (N,) 
    pcl_coords_np = np.column_stack((pcl_coords_np['f0'], pcl_coords_np['f1']))  # shape (N, 2)
    pcl_coords = torch.from_numpy(pcl_coords_np).to(device)
    
    # Convert pcl_coords from [0, 1] to pixel coords
    u = (pcl_coords[:, 0] * color_w).int()
    v = (pcl_coords[:, 1] * color_h).int()
    mask_ROI = ((u >= 660) & (u < 1260) & 
                (v >= 240) & (v < 840))  # shape (N,)
    u_valid = u[mask_ROI]
    v_valid = v[mask_ROI]   # shape (M,) filter out invalid pixel coords

    # ------------------------------------------------
    # ------------------ color mask ------------------
    # ------------------------------------------------
    # Get the color image as torch tensor
    tex_np = np.asanyarray(color_frame.get_data())
    tex = torch.from_numpy(tex_np).to(device)  # shape (H, W, 3)
    # visualizing the color image
    # cv2.imshow("color", tex)
    # cv2.waitKey(1)

    black_threshold = 0.15
    mean_tex = torch.mean(tex[v_valid, u_valid, :]/255.0, dim=-1)  # shape (H, W)
    mask_black = (mean_tex < black_threshold)   # shape (H, W)
    
    # For each point, get its color mask value
    final_mask = torch.zeros_like(mask_ROI, dtype=torch.bool)
    final_mask[mask_ROI] = mask_black
    filtered_pcl = pcl[final_mask]  # shape (M, 3)

    # centralize and convert to numpy
    centralized_pcl = centralize_mesh_torch(filtered_pcl).cpu().numpy()

    return centralized_pcl

class CustomDepthShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        # Obtain the z-buffer from the rasterizer output.
        zbuf = fragments.zbuf[..., 0]  # Shape: [N, H, W]

        # Define the depth range for your object and the desired target range.
        # zmin, zmax = 0.70, 0.82
        zmin, zmax = 0.0, 1.5
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

if __name__ == '__main__':
    color_w = 1920
    color_h = 1080
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, 30)
    pipeline.start(config)

    # Create the decimation filter
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 3)
    name_decimation = "de3"

    # find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Visualization
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # # Create a dummy OpenCV window for keyboard pollingbuyong
    # cv2.namedWindow("Key", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Key", 100, 100)

    pcd = o3d.geometry.PointCloud()
    num_frame = 0
    num_save = 0
    t_total = 0
    try:
        while True:
            t0 = time.time()
            pcd_np = get_pcd(pipeline, pcd, device, color_w, color_h)

            # convert a numpy array to a point cloud
            pcd.points = o3d.utility.Vector3dVector(pcd_np)

            pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            pcd.estimate_normals()
            pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, world_frame], point_show_normal=True)

            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            # print("")
            # print(f"Average distance: {avg_dist}")
            # print(f"num_points: {len(pcd_np)}")

            # create a mesh from the point cloud using ball pivoting
            radii = [avg_dist/2, avg_dist, avg_dist*2, avg_dist*2.5]
            tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            # o3d.visualization.draw_geometries([pcd, tri_mesh])

            # convert to pytorch3d mesh
            vertices_np = np.array(tri_mesh.vertices, dtype=np.float32)
            faces_np = np.array(tri_mesh.triangles, dtype=np.int64)
            vertices_torch = torch.tensor(vertices_np).to(device)
            faces_torch = torch.tensor(faces_np).to(device)
            vertex_colors = torch.zeros_like(vertices_torch)[None]  # Assign a simple color (black color here, shape=[1, V, 3])
            pytorch3d_mesh = Meshes(verts=[vertices_torch], faces=[faces_torch], textures=TexturesVertex(verts_features=vertex_colors))

            # camera setup
            R, T = look_at_view_transform(dist=0.8, elev=0, azim=0)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=40.0, znear=0.01, zfar=100.0)
            
            # render the mesh to get the depth
            raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0001, faces_per_pixel=20)
            renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),shader=CustomDepthShader(device=device, cameras=cameras))
            images = renderer(pytorch3d_mesh)
            image_np = images[0, ..., :3].cpu().numpy() # Extract RGBA image and convert to CPU numpy array

            t1 = time.time()
            t_total += t1 - t0
            num_frame += 1
            print(f"Time to get pcd: {t1 - t0:.2f}s")
            print(f"FPS: {1 / (t1 - t0):.2f}")
            print(f"avg FPS: {num_frame / t_total:.2f}")

            # # Display
            image_np = np.mean(image_np, axis=-1)
            plt.imshow(image_np, cmap='gray', vmin=0, vmax=1)
            plt.colorbar()
            plt.show()
            # plt.imsave("depth.png", image_np, cmap='gray', vmin=0, vmax=1)
            # break

            
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # vis.destroy_window()
        # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
        pipeline.stop()