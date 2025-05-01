import os, sys, time
import cv2, torch
import numpy as np
import open3d as o3d
import random
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# Depth render by PyTorch3D
from api_depthrender_pt3d import DepthRendering

# SAM2
from sam2_realtime.sam2.build_sam import build_sam2_camera_predictor

if __name__ == '__main__':
    # initialize sam2
    sam2_checkpoint = "./sam2_realtime/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    # initialize camera
    width = 1920
    height = 1080
    
    # realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # Create the decimation filter
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 8)

    # find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depthrender = DepthRendering(predictor,
                                 width,
                                 height,
                                 pipeline,
                                 decimation,
                                 device
                                 )
    
    # Visualization
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    vis.add_geometry(world_frame)

    pcd_o3d = o3d.geometry.PointCloud()
    
    time.sleep(0.1)
    num_frame = 0
    is_init = True
    try:
        while True:
            
            t_0 = time.time()
            frame, origin, mask, mesh_o3d, pcd_o3d = depthrender.rendering_pipeline(pcd_o3d)
            t_1 = time.time()
            print(f"used time for one frame: {t_1-t_0} s. FPS: {1/(t_1-t_0)}")

            if is_init:            
                vis.add_geometry(pcd_o3d)
                is_init = False
            else:
                vis.update_geometry(pcd_o3d)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)

            # mask post process
            mask = mask.cpu().numpy().astype(np.uint8) * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

            # image post process
            origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
            origin = cv2.addWeighted(origin, 1, mask, 0.3, 0)
            origin = cv2.resize(origin, (width//3, height//3))
            
            mask = cv2.resize(mask, (width//3, height//3))
            cv2.imshow("mask", mask)
            cv2.imshow("origin", origin)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    except KeyboardInterrupt:
        print("Got error.")
    finally:
        pipeline.stop()
        vis.destroy_window()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)