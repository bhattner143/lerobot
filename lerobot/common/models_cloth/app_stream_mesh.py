import os, sys, time
import cv2, torch
import numpy as np
import open3d as o3d
import random
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# utils
from utils.configs_utils import get_args

# Depth render by PyTorch3D
from realtime_depth.api_depthrender_pt3d import DepthRendering

# SAM2
from realtime_depth.sam2_realtime.sam2.build_sam import build_sam2_camera_predictor

# Mesh GAT
from mesh_gat.api_mesh_gat import API_Mesh_GAT

# Camera API
from realsense.api_realsense import api_realsense

def main():
    # load configs
    stream_cfg = get_args()

    # initialize sam2
    sam2_checkpoint = "./realtime_depth/sam2_realtime/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    # realsense camera
    pipeline, decimation = api_realsense(stream_cfg.w_rgb,
                             stream_cfg.h_rgb,
                             stream_cfg.w_depth,
                             stream_cfg.h_depth,
                             stream_cfg.desired_exposure,
                             stream_cfg.desired_white_balance,
                             stream_cfg.desired_gain,
                             stream_cfg.deci_ratio
                             )

    # find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize depth render
    depthrender = DepthRendering(predictor,
                                 stream_cfg.w_rgb,
                                 stream_cfg.h_rgb,
                                 pipeline,
                                 decimation,
                                 device
                                 )
    
    # initialize mesh_gat
    mesh_gat = API_Mesh_GAT(stream_cfg, device)

    # pcl visualization
    if stream_cfg.vis_pcl:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    
        # create windows
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # add a coordinate frame
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        vis.add_geometry(world_frame)
        time.sleep(0.1)
        is_init = True

    # create a point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    
    num_frame = 0
    
    try:
        while True:
            
            # get depth
            t_0 = time.time()
            frame, origin, mask, mesh_o3d, pcd_o3d = depthrender.rendering_pipeline(pcd_o3d)
            t_1 = time.time()
            print(f"used time for one frame: {t_1-t_0} s. FPS: {1/(t_1-t_0)}")

            # get mesh
            mesh = mesh_gat.predict(frame)

            # pcl visualization
            if stream_cfg.vis_pcl:
                if is_init:            
                    vis.add_geometry(pcd_o3d)
                    is_init = False
                else:
                    vis.update_geometry(pcd_o3d)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)

            # mask visualization
            if stream_cfg.vis_mask:
                # mask post process
                mask = mask.cpu().numpy().astype(np.uint8) * 255
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

                # image post process
                origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
                origin = cv2.addWeighted(origin, 1, mask, 0.3, 0)
                origin = cv2.resize(origin, (stream_cfg.w_rgb//3, stream_cfg.h_rgb//3))
                cv2.imshow("origin", origin)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # depth visualization
            if stream_cfg.vis_depth:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("frame", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # mesh visualization
            # if stream_cfg.vis_mesh:
            #     mesh_o3d.compute_vertex_normals()
            #     mesh_o3d.paint_uniform_color([0.1, 0.1, 0.7])
            #     mesh_o3d.compute_triangle_normals()
            #     mesh_o3d.triangle_normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            #     mesh_o3d.orient_triangles()
            #     mesh_o3d.compute_triangle_normals()

            #     if is_init:
            #         vis.add_geometry(mesh_o3d)
            #         is_init = False
            #     else:
            #         vis.update_geometry(mesh_o3d)
            #     vis.poll_events()
            #     vis.update_renderer()
            #     time.sleep(0.1)

            print(f"The shape of the mesh is {mesh.shape}.") # (1, 520, 3)
            # transfer the mesh from torch to numpy
            mesh = mesh.squeeze(0).cpu().numpy()
            mesh = mesh.reshape(-1, 3)
            # save the mesh as txt file
            mesh_file = f"mesh_{num_frame}.txt"
            # np.savetxt(mesh_file, mesh, fmt='%.6f')
            # exit()

            


    except KeyboardInterrupt:
        print("Got error.")
    finally:
        pipeline.stop()
        if stream_cfg.vis_pcl:
            vis.destroy_window()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

if __name__ == '__main__':
    main()

    
    
    