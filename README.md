Use Python 3.11.

## Sever
*Install:*
1. Install torch and torchvision with CUDA.
2. `pip install -r requirements_server.txt`

For SAM3D Body support, follow the instructions here:
1. Follow installation instructions here: https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md
    - Include the optional MoGe (`pip install git+https://github.com/microsoft/MoGe.git`)
2. `git submodule add https://github.com/facebookresearch/sam-3d-body.git`

*Use:*
`inference/sam.py serve --port <optional>`

## Client
*Install:*
1. Install torch and torchvision, CUDA is optional.
2. `pip install -r requirements_client.txt`

*Use:*
`python src/demo.py`

