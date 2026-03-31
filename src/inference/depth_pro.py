from typing import List, Literal
import dataclasses
import io
import requests
import base64

import litserve as ls
import torch
from PIL import Image

class DepthProLocal():
    def __init__(self):
        from transformers import DepthProImageProcessor, DepthProForDepthEstimation

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(self.device)
        self.processor = DepthProImageProcessor.from_pretrained("apple/DepthPro-hf")

    @torch.no_grad()
    def __call__(
        self,
        images: Image.Image | List[Image.Image],
    ):
        if type(images) is List and len(images) > 1:
            raise NotImplementedError("havent tested with more than 1 image")

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(images.height, images.width)]
        )

        depth = results[0]['predicted_depth'].cpu()
        return depth

class DepthProServer(ls.LitAPI):
    def setup(self, device):
        self.model = DepthProLocal()

    def decode_request(self, request):
        return {
            'images': request['images'].file,
        }

    def predict(self, x):
        return self.model(Image.open(x['images']))

    def encode_response(self, x):
        buffer = io.BytesIO()
        torch.save(x, buffer)
        tensor_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return tensor_str

class DepthProClient():
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
            print(f'DepthProClient Error: Response with status code {response.status_code} - {response.text}')

def serve(port=8000):
    print("Starting DepthPro inference server")
    server = ls.LitServer(DepthProServer())
    server.run(port=port, generate_client_file=False)

def main(
    cmd: Literal["test", "serve"],
    /,
    port: int = 8000
):
    if cmd == "test":
        raise NotImplementedError
    else:
        serve(port)

if __name__ == "__main__":
    import tyro
    tyro.cli(main)