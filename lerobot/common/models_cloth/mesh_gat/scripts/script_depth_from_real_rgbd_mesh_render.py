import torch
import numpy as np
import open3d as o3d
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
    bbox_min = torch.min(pred_vertices[:, 0:2], dim=0).values  # Minimum x, y
    bbox_max = torch.max(pred_vertices[:, 0:2], dim=0).values  # Maximum x, y

    # print(f'Bbox min: {bbox_min}, Bbox max: {bbox_max}')

    # Compute the center of the bounding box
    bbox_center = (bbox_min + bbox_max) / 2.0

    # print(f'Bbox min: {bbox_min}, Bbox max: {bbox_max}, Bbox center: {bbox_center}')

    # Shift vertices to centralize the mesh
    centralized_vertices[:, 0:2] = pred_vertices[:, 0:2] - bbox_center

    bbox_min = torch.min(centralized_vertices, dim=0).values  # Minimum x, y, z
    bbox_max = torch.max(centralized_vertices, dim=0).values  # Maximum x, y, z

    # Compute the center of the bounding box
    bbox_center = (bbox_min + bbox_max) / 2.0

    print(f'Bbox min: {bbox_min}, Bbox max: {bbox_max}, Bbox center: {bbox_center}')

    return centralized_vertices

def depth_to_white_bg(depth_map: torch.Tensor):
    """
    Convert a PyTorch3D depth map to a [0..1] grayscale image,
    making background (inf or -1) appear as white.
    
    Args:
      depth_map: (H, W) float tensor with depth values in camera or NDC space.
                 Background pixels may be +inf or negative (e.g., -1).
      invert: If True, invert so that larger depths become darker, etc.
    Returns:
      gray: (H, W) float32 in [0..1], with background = 1.0 (white).
    """
    depth = depth_map.clone()
    out_min=0.2
    out_max=0.8
    
    # 1. Identify background
    #    PyTorch3D might use +inf or -1 for "no geometry".
    is_inf = torch.isinf(depth)
    is_neg = depth < 0  # e.g. -1

    # 2. We'll treat these as "background" and fill them after we find min/max
    valid_mask = ~(is_inf | is_neg)
    if not torch.any(valid_mask):
        # If no valid points, just return white.
        white_img = torch.ones_like(depth)
        return white_img

    # 3. Compute min/max of valid depths
    d_min = depth[valid_mask].min()
    d_max = depth[valid_mask].max()

    # 4. Assign background = d_max so it becomes "far" => white
    depth[~valid_mask] = d_max

    # Now scale everything to [out_min..out_max]
    scale = (depth - d_min) / (d_max - d_min)  # in [0..1]
    scaled_depth = out_min + (out_max - out_min) * scale
    scaled_depth[~valid_mask] = d_max

    return scaled_depth

class CustomDepthShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        # Obtain the z-buffer from the rasterizer output.
        zbuf = fragments.zbuf[..., 0]  # Shape: [N, H, W]

        # Define the depth range for your object and the desired target range.
        zmin, zmax = 0.70, 0.82
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    decimation_name = 'de4'
    num_data = 50
    for i in range(num_data):
        
        # load a point cloud given the address, centralize it and convert to pytorch3d pointclouds
        pcl_address = f"/home/ktang/ws/realsense/test_generated/real/pcl/{decimation_name}/real_pcl_{decimation_name}_{i:06d}.ply"
        
        # load .ply file
        pcl = o3d.io.read_point_cloud(pcl_address)
        pcl = np.asarray(pcl.points)
        pcl_torch = torch.tensor(pcl[:, :3], dtype=torch.float, device=device)
        pcl_centralized = centralize_mesh_torch(pcl_torch)
        pcl_np = pcl_centralized.cpu().numpy()

        # Create Open3D point cloud
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcl_np)
        
        # estimate normal and align them with the z-axis
        o3d_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        o3d_pcd.estimate_normals()
        o3d_pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

        # visualize the point cloud with normals and coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([o3d_pcd, world_frame], point_show_normal=True)

        distances = o3d_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        print("")
        print(f"num_data: {i}")
        print(f"Average distance: {avg_dist}")
        print(f"num_points: {len(pcl_np)}")

        # create a mesh from the point cloud using ball pivoting
        radii = [avg_dist/2, avg_dist, avg_dist*2, avg_dist*2.5]
        tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            o3d_pcd, o3d.utility.DoubleVector(radii))
        # o3d.visualization.draw_geometries([o3d_pcd, tri_mesh])

        # convert to pytorch3d mesh
        vertices_np = np.array(tri_mesh.vertices, dtype=np.float32)
        faces_np = np.array(tri_mesh.triangles, dtype=np.long)
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
        
        # Display
        image_np = np.mean(image_np, axis=-1)
        # plt.imshow(image_np, cmap='gray', vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        plt.imsave(f"/home/ktang/ws/realsense/test_generated/result/real/{decimation_name}/mesh_render/depth_{i:06d}.png", image_np, cmap='gray', vmin=0, vmax=1)
