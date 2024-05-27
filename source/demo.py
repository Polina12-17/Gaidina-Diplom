import argparse
import glob
import os

import cv2
import numpy as np
import rawpy
import torch
import torchvision
from torch import nn


from model import UnetTMO
from source.currentDebayer import debayer


class newMO(nn.Module):

    def forward(self, x):
        return x, x

map_location = torch.device('cpu')


def read_image(path):
    raw = rawpy.imread(path)
    raw_image = raw.raw_image_visible.copy()
    raw_image = raw_image.astype(np.float32) / (2 ** 14 - 1)
    bayer_batch = torch.from_numpy(raw_image).unsqueeze(0).unsqueeze(0)
    return bayer_batch


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="..\\pretrained\\203.ckpt")
parser.add_argument("--input_dir", type=str, default="samples")
parser.add_argument("--output_dir", type=str, default="output")

args = parser.parse_args()

model = UnetTMO()
state_dict = read_pytorch_lightning_state_dict(torch.load(args.checkpoint, map_location))
model.load_state_dict(state_dict)
model.eval()
model.cpu()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

input_images = glob.glob(os.path.join(args.input_dir, "*"))
for path in input_images:
    print("Processing:", path)
    img = read_image(path)
    with torch.no_grad():
        a, _ =model(img)
        a =debayer(a).squeeze(0)
    print(f"a: {a.shape}")
    print(f"a min: {a.min()}")
    print(f"a max: {a.max()}")
    torchvision.utils.save_image(a, path.replace(args.input_dir,args.output_dir).replace(".CR2",".png"))