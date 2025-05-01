import pyrealsense2 as rs
import numpy as np
import cv2
import pprint

def main():
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and enable the streams you want
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
    print("Streaming started... Press 'Esc' to quit.")

    # Get the device from the pipeline profile
    device = profile.get_device()

  # Iterate over sensors to find the RGB camera
    for sensor in device.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        if sensor_name == 'RGB Camera':
            print(f"Configuring sensor: {sensor_name}")
            
            # Disable auto exposure
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                print("Disabled auto exposure.")

            # Disable auto white balance
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, 0)
                print("Disabled auto white balance.")
            
            # Set manual exposure (in microseconds)
            if sensor.supports(rs.option.exposure):
                desired_exposure = 50  # adjust based on your camera's range
                sensor.set_option(rs.option.exposure, desired_exposure)
                print(f"Set exposure to {desired_exposure} microseconds.")

            # Set manual white balance (typically in Kelvin)
            if sensor.supports(rs.option.white_balance):
                desired_white_balance = 5600  # adjust as needed
                sensor.set_option(rs.option.white_balance, desired_white_balance)
                print(f"Set white balance to {desired_white_balance}.")

            # (Optional) Set manual gain if available (example value)
            if sensor.supports(rs.option.gain):
                desired_gain = 50  # example gain value; adjust based on your needs
                sensor.set_option(rs.option.gain, desired_gain)
                print(f"Set gain to {desired_gain}.")

            break  # Stop once we have configured the desired sensor

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            # In case color frame is not ready, skip
            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Display color frame using OpenCV
            cv2.imshow("Color Stream", color_image)

            # Break on 'Esc' key press
            if cv2.waitKey(1) & 0xFF == 27:  # 27 is ASCII for Esc
                break
        
        # save the last frame
        cv2.imwrite("color_frame.jpg", color_image)

    except Exception as e:
        print("Error occurred:", e)
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Streaming stopped.")

if __name__ == "__main__":
    main()