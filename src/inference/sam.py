from typing import List, Literal
import dataclasses
import io
import requests
import base64

import litserve as ls
import torch
from PIL import Image

class Sam3Local():
    def __init__(self):
        from transformers import Sam3Processor, Sam3Model

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    @torch.no_grad()
    def __call__(
        self,
        images: Image.Image | List[Image.Image],
        prompts: str | List[str],
        confidence_threshold: float = 0.0
    ) -> torch.Tensor:
        inputs = self.processor(images=images, text=prompts, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=inputs.get("original_sizes").tolist()
        )

        valid_masks = torch.max(outputs['pred_logits'], dim=1).values > confidence_threshold
        results = [results[i].cpu() for i, valid in enumerate(valid_masks) if valid]

        if len(results) == 0:
            return torch.tensor([])
        else:
            return torch.stack(results)

class Sam3Server(ls.LitAPI):
    def setup(self, device):
        self.segmenter = Sam3Local()

    def decode_request(self, request):
        return {
            'images': request['images'].file,
            'prompts': request['prompts'],
            'confidence_threshold': float(request['confidence_threshold'])
        }

    def predict(self, x):
        return self.segmenter(
            Image.open(x['images']),
            x['prompts'],
            x['confidence_threshold']
        )

    def encode_response(self, x):
        buffer = io.BytesIO()
        torch.save(x, buffer)
        tensor_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return tensor_str

class Sam3Client():
    def __init__(
        self,
        ip: str = 'localhost',
        port: int = 8000
    ):
        self.url = f'http://{ip}:{port}/predict'

    def __call__(
        self,
        images: Image.Image | List[Image.Image],
        prompts: str | List[str],
        confidence_threshold: float = 0.0
    ):
        img_byte_arr = io.BytesIO()
        images.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        data = {
            'prompts': prompts,
            'confidence_threshold': str(confidence_threshold)
        }

        response = requests.post(self.url, data=data, files={'images': img_byte_arr})
        if response.status_code == 200:
            tensor_bytes = base64.b64decode(response.content)
            return torch.load(io.BytesIO(tensor_bytes))
        else:
            print(f'Sam3Client Error: Response with status code {response.status_code} - {response.text}')


def remote_inference_example(port=8000):
    from threading import Thread
    import time

    def start_server():
        print("starting server")
        server = ls.LitServer(Sam3Server())
        server.run(port=port, generate_client_file=False)

    thread = Thread(target=start_server)
    thread.start()

    time.sleep(5) # give time for the server to load the model. it can take a while, especially the first time.

    client = Sam3Client()
    print("created client")

    dummy_image = Image.open('./notebooks/media/arm1.jpg')
    dummy_image.thumbnail((224,224))
    dummy_prompt = "test"

    client(dummy_image, dummy_prompt)

def serve(port=8000):
    print("Starting SAM3 inference server")
    server = ls.LitServer(Sam3Server())
    server.run(port=port, generate_client_file=False)

def main(
    cmd: Literal["test", "serve"],
    /,
    port: int = 8000
):
    if cmd == "test":
        remote_inference_example(port)
    else:
        serve(port)

if __name__ == "__main__":
    import tyro
    tyro.cli(main)