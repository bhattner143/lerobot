import pyrealsense2 as rs
import numpy as np
import open3d as o3d

def main():
    # 1. Configure the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 2. Start streaming
    pipeline.start(config)
    print("Streaming... Press Ctrl+C to exit at any time.")

    try:
        while True:
            # 3. Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 4. Use pyrealsense2's built-in PointCloud class
            pc = rs.pointcloud()
            # Map the point cloud to the color frame to get color for each point
            pc.map_to(color_frame)
            # Generate the point cloud from the depth frame
            points = pc.calculate(depth_frame)

            # 5. Convert the point cloud to a numpy array of shape (N, 3)
            vtx = np.asanyarray(points.get_vertices())  # dtype is usually float32
            pts = np.array([[v[0], v[1], v[2]] for v in vtx])

            # 6. Convert the color data to numpy array of shape (N, 3)
            tex = np.asanyarray(color_frame.get_data())
            tex_h, tex_w, _ = tex.shape

            # Use built-in texture coordinates to map each 3D vertex to a color
            tex_coords = np.asanyarray(points.get_texture_coordinates())
            colors = []
            for uv in tex_coords:
                # u,v range from 0..1, map them to pixel coordinates
                u = int(uv[0] * tex_w)
                v = int(uv[1] * tex_h)
                # Ensure the mapped coords are in valid range
                if 0 <= u < tex_w and 0 <= v < tex_h:
                    color = tex[v, u, :] / 255.0
                else:
                    color = [0, 0, 0]
                colors.append(color)
            colors = np.array(colors)

            # 7. Create an Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 8. Visualize the point cloud in a window
            o3d.visualization.draw_geometries([pcd])

            # If you want continuous updates in the same window,
            # you could integrate a custom visualization loop or
            # stop after the first capture:
            # break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print("Streaming stopped.")

if __name__ == "__main__":
    main()