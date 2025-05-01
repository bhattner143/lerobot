import os
import sys
import torch
import math
import pickle
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

def styled_message(message, fg_color="37", bg_color="44", bold=True):
    """
    Prints a styled message with customizable colors and bold effects.

    Args:
    - message (str): The message to display.
    - fg_color (str): ANSI code for foreground color (default: white = "37").
    - bg_color (str): ANSI code for background color (default: blue = "44").
    - bold (bool): Whether to make the text bold (default: True).
    """
    style = f"\033[{'1;' if bold else ''}{fg_color};{bg_color}m"
    reset = "\033[0m"
    print(f"\n{style}{message}{reset}\n")

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
    
    # print(f"shape of pred_vetices: {pred_vertices.shape}")
    centralized_vertices = pred_vertices.clone()

    # inverse z-axis
    centralized_vertices[:, 2] = pred_vertices[:, 2]

    # Compute the bounding box
    bbox_min = torch.min(pred_vertices[:, 0:3], dim=0).values  # Minimum x, y
    bbox_max = torch.max(pred_vertices[:, 0:3], dim=0).values  # Maximum x, y
    centralized_vertices[:, 2] = pred_vertices[:, 2] - bbox_min[2]
    
    # Compute the center of the bounding box
    bbox_center = (bbox_min + bbox_max) / 2.0

    # Shift vertices to centralize the mesh
    centralized_vertices[:, 0:2] = pred_vertices[:, 0:2] - bbox_center[0:2]

    # Compute the bounding box again for verification
    # bbox_min = torch.min(centralized_vertices, dim=0).values  # Minimum x, y, z
    # bbox_max = torch.max(centralized_vertices, dim=0).values  # Maximum x, y, z
    # bbox_center = (bbox_min + bbox_max) / 2.0
    # print(f'Bbox min: {bbox_min}, Bbox max: {bbox_max}, Bbox center: {bbox_center}')

    return centralized_vertices

# Utility Functions
def read_mesh_vertices(file_path, device):
    """
    Reads a mesh vertices file and converts it into a PyTorch tensor.
    """
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            vertex = list(map(float, line.strip().split()))
            vertices.append(vertex)

    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)

    # Ensure tensor shape is valid
    if vertices_tensor.shape[1] != 3:
        raise ValueError(f"Expected tensor shape (N, 3), but got {vertices_tensor.shape}")
    
    # vertices_tensor = vertices_tensor.unsqueeze(0)  # Add batch dimension
    return vertices_tensor

class CustomDepthShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        # Obtain the z-buffer from the rasterizer output.
        zbuf = fragments.zbuf[..., 0]  # Shape: [N, H, W]

        # Define the depth range for your object and the desired target range.
        zmin, zmax = 0.65, 0.82
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

    # Parameters
    batch_size = 100
    use_obj = True
    fabric_name = 't_shirt_l3'
    name_folder = 't_shirt_l3_candidate_3'

    # List all .simu_mesh.obj files in the directory
    data_address = f"/home/ktang/ws/data/mesh_gat/{fabric_name}/source/{name_folder}"
    if use_obj:
        all_files_obj = sorted([f for f in os.listdir(data_address) if f.endswith('.simu_mesh.obj')])
    else:
        template_info = pickle.load(open(os.path.join(data_address , f"template_{fabric_name}.pickle"), 'rb'))
        all_files_obj = sorted([f for f in os.listdir(data_address) if f.endswith('.simu_pcl.txt')])
    num_data = len(all_files_obj)
    num_batches = math.ceil(num_data / batch_size)

    for batch_idx in range(num_batches):

        # Calculate start and end indices for the current batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_data)
        current_batch_size = end_idx - start_idx

        # Lists to accumulate mesh data
        verts_list = []
        faces_list = []
        colors_list = []

        styled_message(f"Batch {batch_idx+1}/{num_batches}: processing meshes {start_idx} to {end_idx-1}", fg_color="37", bg_color="44", bold=True)
        
        # -------------------------------------------------------------------
        # ----------------------- Mesh Loading as Batch----------------------
        # -------------------------------------------------------------------
        for i in range(start_idx, end_idx):

            # Load mesh from file using o3d
            if use_obj:
                mesh_address = os.path.join(data_address, all_files_obj[i])
                mesh_o3d = o3d.io.read_triangle_mesh(mesh_address)

                # Get vertices and faces from o3d mesh
                vertices_np = np.asarray(mesh_o3d.vertices)
                faces_np = np.asarray(mesh_o3d.triangles)

                # Convert numpy arrays to torch tensors.
                vertices_torch = torch.tensor(vertices_np, dtype=torch.float32, device=device)
                faces_torch = torch.tensor(faces_np, dtype=torch.long, device=device)

            else:
                # load vertices from .txt and load faces from template.obj
                vertices_address = os.path.join(data_address, all_files_obj[i])
                vertices_torch = read_mesh_vertices(vertices_address, device)

                # Load faces from template file
                # face_address = os.path.join(data_address, 'template.obj')
                # face_mesh_o3d = o3d.io.read_triangle_mesh(face_address)
                # faces_np = np.asarray(face_mesh_o3d.triangles)
                # vertice_template_np = np.asarray(face_mesh_o3d.vertices)
                # vertice_template_torch = torch.tensor(vertice_template_np, dtype=torch.float32, device=device)
                # faces_torch = torch.tensor(faces_np, dtype=torch.long, device=device)
                
                faces_torch    = torch.tensor(template_info['face_idx'], dtype=torch.long).to(device)

            # Centralize the mesh vertices.
            vertices_centralized_torch = centralize_mesh_torch(vertices_torch)

            # Create a vertex color tensor. Note: TexturesVertex expects a tensor of shape (V, 3)
            vertex_colors_torch = torch.zeros(vertices_centralized_torch.shape, device=device)
            
            # Append to list. 
            # Each element of verts_list and faces_list can have a different number of vertices/faces.
            verts_list.append(vertices_centralized_torch)
            faces_list.append(faces_torch)
            colors_list.append(vertex_colors_torch)

        meshes = Meshes(
            verts=verts_list, 
            faces=faces_list, 
            textures=TexturesVertex(verts_features=colors_list)
        )
            
        # -------------------------------------------------------------------
        # -------------------- Camera Setting as Batch ----------------------
        # -------------------------------------------------------------------
        # We use the same camera parameters for all meshes in the batch.
        R, T = look_at_view_transform(dist=0.8, elev=0, azim=0)
        R_batch = R.repeat(current_batch_size, 1, 1)  # (current_batch_size, 3, 3)
        T_batch = T.repeat(current_batch_size, 1)     # (current_batch_size, 3)
        cameras = FoVPerspectiveCameras(
            device=device,
            R=R_batch,
            T=T_batch,
            fov=40.0,
            znear=0.01,
            zfar=100.0,
        )

        # -------------------------------------------------------------------
        # ---------------------- Rendering Setting --------------------------
        # -------------------------------------------------------------------
        # Define renderer settings.
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0001,
            faces_per_pixel=20,
        )
        
        # Instantiate the renderer with the batched cameras.
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=CustomDepthShader(device=device)
        )

        # -------------------------------------------------------------------
        # ------------------------ Render Batch -----------------------------
        # -------------------------------------------------------------------
        images = renderer(meshes)
        # Convert the rendered images to single channel depth by averaging over RGB channels.
        depth_images = torch.mean(images, dim=-1).cpu().numpy()

        # Save each rendered depth image with a global index.
        for batch_offset, depth_img in enumerate(depth_images):
            global_idx = start_idx + batch_offset
            
            # # Display
            # plt.imshow(depth_img, cmap='gray', vmin=0, vmax=1)
            # plt.colorbar()
            # plt.show()

            # save in the same location, remove the end .simu_mesh.obj and add render_depth.png
            if use_obj:
                mesh_name = all_files_obj[global_idx]
                depth_name = mesh_name[:-13] + 'render_depth.png'
                save_address = os.path.join(data_address, depth_name)
            else:
                mesh_name = all_files_obj[global_idx]
                depth_name = mesh_name[:-12] + 'render_depth.png'
                save_address = os.path.join(data_address, depth_name)
            plt.imsave(save_address, depth_img, cmap='gray', vmin=0, vmax=1)




