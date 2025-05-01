import pyrealsense2 as rs
import numpy as np

def api_realsense(w_rgb = 1920, 
                  h_rgb = 1080, 
                  w_depth = 1024, 
                  h_depth = 768, 
                  desired_exposure = 50, 
                  desired_white_balance = 5600, 
                  desired_gain = 50, 
                  deci_ratio = 8):
    
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and enable the streams you want
    config = rs.config()
    config.enable_stream(rs.stream.depth, w_depth, h_depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, w_rgb, h_rgb, rs.format.bgr8, 30)

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
                sensor.set_option(rs.option.exposure, desired_exposure)
                print(f"Set exposure to {desired_exposure} microseconds.")

            # Set manual white balance
            if sensor.supports(rs.option.white_balance):
                sensor.set_option(rs.option.white_balance, desired_white_balance)
                print(f"Set white balance to {desired_white_balance}.")

            # Set manual gain
            if sensor.supports(rs.option.gain):
                sensor.set_option(rs.option.gain, desired_gain)
                print(f"Set gain to {desired_gain}.")

            break 

    
    # Create the decimation filter
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, deci_ratio)
    
    return pipeline, decimation