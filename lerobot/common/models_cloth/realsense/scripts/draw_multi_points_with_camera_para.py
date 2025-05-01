import pyrealsense2 as rs
import cv2
import yaml
import numpy as np
import os
import pickle
from pprint import pprint

# -----------------------------
# Part 1. Load the YAML file with camera parameters.
# -----------------------------
save_dir = "./realsense/parameters/" 
yaml_filepath = os.path.join(save_dir, "l515_parameters.yaml")

with open(yaml_filepath, "r") as f:
    data = yaml.safe_load(f)

cam_params = data["camera_parameters"]

# Extract the intrinsic parameters.
w_rgb = cam_params["w_rgb"]
h_rgb = cam_params["h_rgb"]
w_depth = cam_params["w_depth"]
h_depth = cam_params["h_depth"]
fx = cam_params["fx"]
fy = cam_params["fy"]
ppx = cam_params["ppx"]
ppy = cam_params["ppy"]


# Construct the intrinsic matrix K.
K = np.array([
    [fx, 0,   ppx],
    [0,  fy,  ppy],
    [0,  0,    1]
])

print("Loaded camera intrinsic matrix K:")
print(K)

# -----------------------------
# Part 2. Read 3D points from a text file.
# -----------------------------
# Specify the path to the text file containing the 3D points.
# Each line in the file should contain three float values (X Y Z) separated by whitespace.
points_file = "mesh_0.txt"  # Adjust this path if necessary

points_3d = []
if os.path.exists(points_file):
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue  # skip empty lines
            parts = line.split()
            if len(parts) != 3:
                print(f"Skipping line (invalid format): {line}")
                continue
            try:
                coords = [float(val) for val in parts]
                points_3d.append(np.array(coords))
            except ValueError:
                print(f"Skipping line (could not convert to float): {line}")
else:
    print(f"Points file {points_file} not found.")
    exit(1)

print(f"Loaded {len(points_3d)} points from {points_file}")

# for each point, z axis add 0.7
for i in range(len(points_3d)):
    points_3d[i][2] += 0.7

# -----------------------------
# Part 4. Load template
# -----------------------------
filename = './mesh_gat/configs/template_t_shirt_l3.pickle'

with open(filename, 'rb') as file:
    data = pickle.load(file)

print("Contents of the pickle file:")
pprint(data)
kp_idx = data['keypoint_idx']


# -----------------------------
# Part 4. Define a function to project a 3D point to the 2D image plane.
# -----------------------------
def project_point(point_3d, intrinsic_matrix):
    """
    Projects a 3D point in camera coordinates onto the image plane.
    
    Args:
        point_3d (np.array): The 3D point as [X, Y, Z]. Z must be > 0.
        intrinsic_matrix (np.array): 3x3 camera intrinsic matrix.
    
    Returns:
        tuple or None: (u, v) pixel coordinates as integers, or None if invalid.
    """
    X, Y, Z = point_3d
    # Ensure the point is in front of the camera.
    if Z <= 0:
        return None
    # Compute projected coordinates using the pinhole camera model.
    u = (fx * X / Z) + ppx
    v = (fy * Y / Z) + ppy
    return int(round(u)), int(round(v))

# -----------------------------
# Part 5. Set up the RealSense pipeline for streaming.
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

# Use the camera parameters for the color stream.
config.enable_stream(rs.stream.color, w_rgb, h_rgb, rs.format.bgr8, 30)

# For the depth stream, use a supported resolution for the L515.
# If not needed, you can comment this out.
config.enable_stream(rs.stream.depth, w_depth, h_depth, rs.format.z16, 30)

# Start the RealSense pipeline.
pipeline.start(config)
print("Streaming started. Press 'q' to exit.")

# -----------------------------
# Part 5. Stream the image and overlay the projected points.
# -----------------------------
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the color frame to a numpy array.
        color_image = np.asanyarray(color_frame.get_data())

        # iterate the point in points_3d, if the index is in kp_idx, draw the point with different color
        for i, point in enumerate(points_3d):
            projected = project_point(point, K)
            if projected is not None:
                u, v = projected
                # Check if point lies within the image boundaries.
                if 0 <= u < w_rgb and 0 <= v < h_rgb:
                    # Draw a circle at the projected point.
                    if i in kp_idx:
                        cv2.circle(color_image, (u, v), radius=3, color=(255, 0, 0), thickness=-1)
                    else:
                        cv2.circle(color_image, (u, v), radius=3, color=(0, 0, 255), thickness=-1)

        # Display the image with overlaid points.
        cv2.imshow("RGB Stream with Projected Points", color_image)

        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Streaming stopped and resources released.")