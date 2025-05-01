import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

def main():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and enable the streams you want
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    print("Streaming started... Press 'Esc' to quit.")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            # In case color frame is not ready, skip
            if not depth_frame:
                continue

            # Convert image to numpy array
            depth_image = np.asanyarray(depth_frame.get_data())

            # Display color frame using OpenCV
            cv2.imshow("Depth Stream", depth_image)

            # Break on 'Esc' key press
            if cv2.waitKey(1) & 0xFF == 27:  # 27 is ASCII for Esc
                break

    except Exception as e:
        print("Error occurred:", e)
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Streaming stopped.")

if __name__ == "__main__":
    main()