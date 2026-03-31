# select a camera input and provide methods for getting images from it
from typing import List
import platform
import sys
from abc import ABC, abstractmethod

import numpy as np
import pyrealsense2 as rs
import cv2
from cv2_enumerate_cameras import enumerate_cameras, camera_info

def get_generic_cameras(): # excludes realsense, since its awkward to read from them with cv2
    match platform.system():
        case 'Windows':
            backend = cv2.CAP_MSMF
        case 'Linux':
            backend = cv2.CAP_V4L2
        case _:
            raise NotImplementedError("Haven't implemented for MacOS")

    return list(filter(
        lambda x: "RealSense" not in str(x),
        enumerate_cameras(backend)
    ))

def select_camera():        
        ctx = rs.context()
        rs_devices = ctx.query_devices()
        if len(rs_devices) > 1:
            raise NotImplementedError

        generic_cameras = get_generic_cameras()

        num_devices = len(rs_devices) + len(generic_cameras)
        if num_devices == 0:
            print("No cameras found. Exiting")
            sys.exit()

        print("Found devices:")
        for i, d in enumerate(rs_devices):
            print(f'{i}: {d.get_info(rs.camera_info.name)}')
        
        for i, d in enumerate(generic_cameras):
            print(f'{i+len(rs_devices)}: {d.name}')

        if num_devices > 1:
            device_idx = int(input(f"Select device (0-{num_devices-1}): "))
        else:
            device_idx = 0

        if device_idx < len(rs_devices):
            return RealSenseCamera()
        else:
            return GenericCamera(generic_cameras[device_idx-len(rs_devices)])

class Camera(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        pass

class RealSenseCamera(Camera):
    def __init__(self):
        super().__init__()
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start() # TODO: is this line needed?

    def get_frame(self):
        frames = self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

class GenericCamera(Camera):
    def __init__(self, cam_info: camera_info.CameraInfo):
        super().__init__()
        self.cam = cv2.VideoCapture(cam_info.index, cam_info.backend)

    def get_frame(self):
        for i in range(10):
            ok, frame = self.cam.read()
            if not ok:
                continue

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raise Exception("No frame returned from camera")

if __name__ == "__main__":
    camera = select_camera()
    frame = camera.get_frame()
    print(type(frame))