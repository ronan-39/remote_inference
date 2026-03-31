import os 
import sys
from pathlib import Path
import io
from typing import List, Literal
from importlib import import_module

import cv2
import numpy as np
import torch
from PIL import Image
import requests
import base64
import litserve as ls

class Sam3DLocal():
    def __init__(self):
        file_path = Path(os.path.dirname(os.path.realpath(__file__)))
        repo_path = file_path.parent.parent
        sys.path.insert(1, str(repo_path / "sam-3d-body"))

        from notebook.utils import setup_sam_3d_body
        self.visualize_sample_together = import_module("tools.vis_utils").visualize_sample_together
        self.estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

    @torch.no_grad()
    def __call__(
        self,
        images: Image.Image,
    ):
        outputs = self.estimator.process_one_image(np.array(images))

        if len(outputs) == 0:
            return None

        rend_img = self.visualize_sample_together(np.array(images), outputs, self.estimator.faces)

        return torch.tensor(rend_img)

class Sam3DServer(ls.LitAPI):
    def setup(self, device):
        self.model = Sam3DLocal()

    def decode_request(self, request):
        return {
            'images': request['images'].file
        }

    def predict(self, x):
        return self.model(Image.open(x['images']))

    def encode_response(self, x):
        buffer = io.BytesIO()
        torch.save(x, buffer)
        tensor_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return tensor_str

class Sam3DClient():
    def __init__(
        self,
        ip: str = 'localhost',
        port: int = 8000
    ):
        self.url = f'http://{ip}:{port}/predict'

    def __call__(
        self,
        images: Image.Image | List[Image.Image]
    ):
        img_byte_arr = io.BytesIO()
        images.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        response = requests.post(self.url, files={'images': img_byte_arr})
        if response.status_code == 200:
            tensor_bytes = base64.b64decode(response.content)
            return torch.load(io.BytesIO(tensor_bytes))
        else:
            print(f'Sam3DClient Error: Response with status code {response.status_code} - {response.text}')

def test_local():
    sam3dlocal = Sam3DLocal()

    image = Image.open('/home/ronan/Pictures/football_player.png')
    res = sam3dlocal(image)

    print(type(res))

def serve(port=8000):
    print("Starting SAM3D Body inference server")
    server = ls.LitServer(Sam3DServer())
    server.run(port=port, generate_client_file=False)

def main(
    cmd: Literal["test", "serve"],
    /,
    port: int = 8000
):
    if cmd == "test":
        test_local()
    else:
        serve(port)

if __name__ == "__main__":
    import tyro
    tyro.cli(main)