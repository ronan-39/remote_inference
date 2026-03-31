import select
import logging
import os
import time

import requests
import rerun as rr
import pyrealsense2 as rs
import numpy as np
from PIL import Image

from camera_manager import select_camera
from inference.sam import Sam3Client
from inference.depth_pro import DepthProClient
from inference.sam3d import Sam3DClient
import utils

rr.init("ml_demos", spawn=True)

camera = select_camera()

logger = logging.getLogger("rr_logger")
logger.addHandler(rr.LoggingHandler("logs"))
logger.setLevel(logging.INFO)

if 'REMOTE_IP' in os.environ:
    ip_addr = os.environ['REMOTE_IP']
else:
    logger.info("No IP address specified. Using localhost")
    ip_addr = "localhost"

# client = Sam3Client()
# client = DepthProClient()
client = Sam3DClient()

term_reader = utils.TerminalInputReader()
print("Waiting for inference queries...")
query_count = 0
static = True

while True:
    time.sleep(0.1)
    frame = camera.get_frame()
    rr.log("realsense/rgb/image", rr.Image(frame), static=static) 

    term_input = term_reader.poll()
    if not term_input:
        continue

    args = term_input.split(" ")
    match args[0]:
        case 'q' | 'quit':
            logger.into("Quitting demo.py")
            print("Exiting")
            break
        case 'sam' | 'segment':
            if not len(args) > 1:
                logger.info("Must supply SAM with a prompt")
                continue
            
            prompt = " ".join(args[1:])
            logger.info(f"Running SAM3 with prompt {prompt}")
            color_image_pil = Image.fromarray(frame)
            color_image_pil.thumbnail((224, 224))

            masks = client(color_image_pil, prompt)

            if len(masks) == 0:
                logger.info(f"No mask found for prompt {prompt}")
                continue

            mask = masks[0] # TODO use more than just the first mask

            rr.log(
                'segmentation_result',
                rr.Image(color_image_pil),
                rr.SegmentationImage(mask.numpy())
            )
        case 'depth' | 'depthpro':
            logger.info("Running DepthPro")
            color_image_pil = Image.fromarray(frame)
            color_image_pil.thumbnail((224, 224))
            depth_img = client(color_image_pil)

            rr.log(
                'depth_result',
                rr.DepthImage(depth_img)
            )
        case 'sam3d' | 'pose':
            logger.info("Running SAM3D Body")
            color_image_pil = Image.fromarray(frame)
            color_image_pil.thumbnail((512, 512))
            result = client(color_image_pil)

            if result is None:
                logger.info("No people found in image!")
                continue

            result_img = Image.fromarray(np.uint8(result.numpy()))

            rr.log(
                'pose_result',
                rr.Image(result_img)
            )
