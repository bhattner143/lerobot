import pyrealsense2 as rs
import cv2
import yaml
import numpy as np
import os

# -----------------------------
# Step 1. Load the YAML file with camera parameters.
# -----------------------------

# Set the directory where the YAML file is stored
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

# Build the intrinsic matrix K.
K = np.array([
    [fx, 0,   ppx],
    [0,  fy,  ppy],
    [0,  0,    1]
])

print("Loaded camera intrinsic matrix K:")
print(K)

# -----------------------------
# Step 2. Define the function to project a 3D point to the image plane.
# -----------------------------
def project_point(point_3d, K):
    """
    Projects a 3D point in the camera coordinate system to 2D image coordinates.
    
    Args:
        point_3d (np.array): The 3D point as [X, Y, Z] (Z must be > 0).
        K (np.array): The 3x3 intrinsic matrix.
    
    Returns:
        tuple: The projected pixel coordinates (u, v) as integers.
    """
    X, Y, Z = point_3d
    if Z <= 0:
        return None
    u = (fx * X / Z) + ppx
    v = (fy * Y / Z) + ppy
    return int(round(u)), int(round(v))

# -----------------------------
# Step 3. Set a 3D point (in the camera coordinate system) to be projected.
# -----------------------------
# Example point: a point directly in front of the camera 1 meter away.
point_cam = np.array([0, 0.1, 0.7])  # Change [X, Y, Z] values as needed.

# -----------------------------
# Step 4. Set up the RealSense pipeline.
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, w_rgb, h_rgb, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, w_depth, h_depth, rs.format.z16, 30)

pipeline.start(config)
print("Streaming started. Press 'q' to exit.")

try:
    while True:
        # Retrieve the latest set of frames.
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert the color frame to a numpy array.
        color_image = np.asanyarray(color_frame.get_data())

        # Project the 3D point to 2D image coordinates.
        projected = project_point(point_cam, K)
        if projected is not None:
            u, v = projected

            # Draw the projected point on the color image if it's within the image boundaries.
            if 0 <= u < w_rgb and 0 <= v < h_rgb:
                cv2.circle(color_image, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.putText(color_image, f"({u},{v})", (u + 10, v),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Display the resulting image.
        cv2.imshow("RGB Stream with Projected Point", color_image)

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Streaming stopped and resources released.")