import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pyrealsense2 as rs


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time

sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# cap = cv2.VideoCapture("./notebooks/videos/aquarium/aquarium.mp4")

if_init = False


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and enable the streams you want
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
t_0 =time.time()

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break
    t_1 = time.time()
    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()
    frame = np.asanyarray(frame.get_data())

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t_2 = time.time() - t_1

    width, height = frame.shape[:2][::-1]
    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started


        ##! add points, `1` means positive click and `0` means negative click
        points = np.array([[1000, 600]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        )

        # ! add bbox
        # bbox = np.array([[1000, 600], [1100, 610]], dtype=np.float32)
        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
        # )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        
    # for point in points:
    #     cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 0), thickness=-1)
    t_3 = time.time()-t_2-t_1

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width//2, height//2))
    cv2.imshow("frame", frame)

    print(f"get image: {t_2} s, get mask: {t_3} s")
    print(f"FPS: get image: {1/t_2} Hz, get mask: {1/t_3} Hz")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
