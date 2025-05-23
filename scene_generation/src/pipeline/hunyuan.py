# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Hunyuan3D-2'))

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline


def image_to_3d(image_path, save_path):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    image = Image.open(image_path)

    if image.mode == 'RGB':
        image = rembg(image)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    mesh = pipeline(image=image, num_inference_steps=45, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        mesh = pipeline(mesh, image=image)
        mesh.export(save_path)
    except Exception as e:
        print(e)
        print('Please try to install requirements by following README.md')

def text_to_3d(prompt='a car'):
    rembg = BackgroundRemover()
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
    model_path = 'tencent/Hunyuan3D-2'
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    image = t2i(prompt)
    image = rembg(image)
    mesh = i23d(image, num_inference_steps=30, mc_algo='mc')[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('t2i_demo.glb')

if __name__ == '__main__':
    import argparse
    import os
    import re
    import json
    import yaml
    import cv2
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder")
    args = parser.parse_args()
    last_folder = os.path.basename(os.path.normpath(args.folder))
    os.makedirs(os.path.join("data/assets/",last_folder), exist_ok=True)
    save_path = os.path.join("data/assets/",last_folder)
    for root,_,files in os.walk(args.folder):
        for file in files:
            if "enhanced" in root and (file.endswith('.png') or file.endswith('.jpg')):
                print("Processing: ", os.path.join(root,file))
                print("Creating 3D model")
                mesh_name = re.sub(r"\.(png|jpg)$", ".glb", file, flags=re.IGNORECASE)
                image_to_3d(os.path.join(root,file), os.path.join(save_path, mesh_name))