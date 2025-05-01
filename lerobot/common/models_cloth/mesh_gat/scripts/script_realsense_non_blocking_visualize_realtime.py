import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import time
import cv2

should_exit = False  # global flag for exit

def get_pcd(pipeline, pcd):
    """
    Acquire frames from RealSense, convert to point cloud,
    then filter by both a rectangular ROI and a 'black' color threshold,
    using a vectorized approach (no Python for-loop over every point).
    """
    # 1. Wait for frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None
    # Apply decimation to reduce depth resolution
    depth_frame_decimated = decimation.process(depth_frame)

    # 2. Use the built-in PointCloud class
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame_decimated)

    # 3. Convert the point cloud to NumPy arrays
    vtx = np.asanyarray(points.get_vertices())        # (N,) array of rs_vertex
    pts = np.array([[p[0], p[1], p[2]] for p in vtx])

    # 4. Get the color image as NumPy
    tex = np.asanyarray(color_frame.get_data())  # shape (H, W, 3)
    tex_h, tex_w, _ = tex.shape

    # 5. Retrieve texture coordinates (u,v) in range [0..1]
    coords = np.array(points.get_texture_coordinates())  # shape (N, 2)
    tex_coords = np.column_stack((coords['f0'], coords['f1']))

    # -- Define the rectangle region of interest (ROI) --
    #    Example: A rectangle centered in the image, half image width, ~0.8 * image height
    rect_width = tex_w // 1.5
    rect_height = tex_h // 1.2

    rect_left = (tex_w - rect_width) // 2
    rect_right = rect_left + rect_width
    rect_top = (tex_h - rect_height) // 1.3
    rect_bottom = rect_top + rect_height

    # -- Define black threshold (average of R,G,B < this value considered black)
    black_threshold = 0.15

    # =========================================================================
    #  Vectorized approach:
    #    1) Convert (u,v) from [0..1] → pixel coords
    #    2) Mask out invalid pixel coords & outside-ROI coords
    #    3) Gather the colors from the image
    #    4) Compute color 'darkness' mask
    #    5) Combine masks & build final pcd
    # =========================================================================

    # 1) Convert tex_coords from [0..1] to pixel coords
    u = (tex_coords[:, 0] * tex_w).astype(np.int32)
    v = (tex_coords[:, 1] * tex_h).astype(np.int32)

    # 2) Create a mask for valid pixel coords & inside rectangle
    #    valid range in the image
    mask_valid_xy = ((u >= 0) & (u < tex_w) & 
                     (v >= 0) & (v < tex_h))

    #    within the rectangle
    mask_roi = ((u >= rect_left) & (u < rect_right) &
                (v >= rect_top) & (v < rect_bottom))

    # Combine them
    mask_valid = mask_valid_xy & mask_roi

    # 3) Gather colors from the image using advanced indexing
    #    Only gather from valid pixel coords to avoid out-of-bounds
    u_valid = u[mask_valid]
    v_valid = v[mask_valid]
    colors_raw = tex[v_valid, u_valid, :] / 255.0  # shape (M, 3), M <= N

    # 4) Compute "darkness" (average of R, G, B) and compare with threshold
    mean_colors = np.mean(colors_raw, axis=1)
    mask_black = (mean_colors < black_threshold)

    # 5) Combine the rectangle mask and black mask
    #    But note: we only have color info for already "mask_valid" points.
    #    So let's incorporate "mask_black" on top of that.
    #    final_mask in the original indexing space (size N)
    final_mask = np.zeros_like(mask_valid, dtype=bool)
    # We place the black check results back into final_mask:
    #   final_mask[mask_valid] = mask_black
    # so that final_mask has True only where both ROI AND black threshold are satisfied
    final_mask[mask_valid] = mask_black

    # Now gather final points & colors
    valid_points = pts[final_mask]
    valid_colors = colors_raw[mask_black]

    # align valid_point to xy plane by calculating the mean of z
    z_mean = np.mean(valid_points[:, 2])
    valid_points[:, 2] = valid_points[:, 2] - z_mean

    # Create the Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)

    # # save the points as a txt file
    # np.savetxt("000000.simu_mesh.txt", valid_points, fmt="%.6f")

    # # save as a ply file
    # o3d.io.write_point_cloud("000000.simu_mesh.ply", pcd)

    # exit()

    return pcd


if __name__ == '__main__':
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipeline.start(config)
    print("Streaming... Press Ctrl+C to exit at any time.")

    # 4. Create the decimation filter
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 3)
    name_decimation = "de1"

    # Visualization
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create a dummy OpenCV window for keyboard polling
    cv2.namedWindow("Key", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Key", 100, 100)

    pcd = o3d.geometry.PointCloud()
    num_frame = 0
    num_save = 0

    try:
        while True:
            t0 = time.time()
            pcd = get_pcd(pipeline, pcd)
            t1 = time.time()
            if pcd is not None:
                if num_frame == 0:
                    vis.add_geometry(pcd)
                else:
                    vis.update_geometry(pcd)
                    # print(f"Frame {num_frame}: {len(pcd.points)} points")
                vis.poll_events()
                vis.update_renderer()
                num_frame += 1
            t2 = time.time()
            # print(f"Time to get pcd: {t1 - t0:.2f}s, Time to visualize: {t2 - t1:.2f}s")
            # print(f"FPS: {1 / (t2 - t0):.2f}")

            # Use OpenCV to poll for keyboard events
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                filepath = f"/home/ktang/ws/realsense/test_generated/real/pcl/{name_decimation}/real_pcl_{name_decimation}_{num_save:06d}.ply"
                print(f"Saving point cloud to {filepath}")
                o3d.io.write_point_cloud(filepath, pcd)
                num_save += 1
            elif key == 27:  # ESC key
                print("Exiting visualization.")
                break
            
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        vis.destroy_window()
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
        pipeline.stop()