import pyrealsense2 as rs
import os
import yaml

# Set your desired save directory (change the path as needed)
save_dir = "./realsense/parameters"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
yaml_filepath = os.path.join(save_dir, "l515_parameters.yaml")
w_rgb = 1920
h_rgb = 1080
w_depth = 1024
h_depth = 768

# Initialize the RealSense pipeline and stream configuration
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Retrieve a frame to extract stream profiles
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()

# Get the intrinsic parameters for the color stream
intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

# Print the intrinsic parameters
print("Intrinsic Parameters:")
print("Width_rgb: {}, Height_rgb: {}".format(intrinsics.width, intrinsics.height))
print("Width_depth: {}, Height_depth: {}".format(w_depth, h_depth))
print("fx: {}, fy: {}".format(intrinsics.fx, intrinsics.fy))
print("ppx: {}, ppy: {}".format(intrinsics.ppx, intrinsics.ppy))
print("Distortion model: {}".format(intrinsics.model))
print("Distortion coefficients: {}".format(intrinsics.coeffs))

# Stop the pipeline
pipeline.stop()

# Prepare the parameters dictionary for saving
params = {
    'camera_parameters': {
        'w_rgb': intrinsics.width,
        'h_rgb': intrinsics.height,
        'w_depth': w_depth,
        'h_depth': h_depth,
        'fx': intrinsics.fx,
        'fy': intrinsics.fy,
        'ppx': intrinsics.ppx,
        'ppy': intrinsics.ppy,
        'distortion_coefficients': list(intrinsics.coeffs)  # Convert tuple to list for YAML serialization
    }
}

# Save the parameters to a YAML file
with open(yaml_filepath, "w") as file:
    yaml.dump(params, file, default_flow_style=False, sort_keys=False)

print(f"\nCamera parameters have been saved to: {yaml_filepath}")