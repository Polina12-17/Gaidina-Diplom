import argparse
import glob
import os

import cv2
import numpy as np
import rawpy
import torch
from torch import nn

from model import UnetTMO


class newMO(nn.Module):

    def forward(self, x):
        return x, x


map_location = torch.device('cpu')


def read_image(path):
    if not path.lower().endswith(".arw, .cr2, .cr3, .nef, .raw, .raf"):
        raise Exception("illegal file extension for {}".format(path.strip(".")[-1]))
    raw = rawpy.imread(path)
    raw_image = raw.raw_image_visible.copy()
    raw_image = raw_image.astype(np.float32) / (2 ** 14 - 1)
    bayer_batch = torch.from_numpy(raw_image).unsqueeze(0).unsqueeze(0)
    return bayer_batch, raw


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="..\\pretrained\\8.ckpt")
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
    img, raw = read_image(path)
    with torch.no_grad():
        img, _ = model(img)
        img = (img * (2 ** 14 - 1)).unsqueeze(0).unsqueeze(0).detach().numpy().astype(np.uint16)
        print(f"img: {img.shape}")
        raw.raw_image_visible[:] = img
        img = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)
        print(f"img post shape: {img.shape}")
        img = cv2.resize(img, (3072, 4608))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        a = img
    print(type(a))
    print(f"a: {a.shape}")
    print(f"a min: {a.min()}")
    print(f"a max: {a.max()}")
    cv2.imwrite(path.replace(args.input_dir, args.output_dir).replace(".NEF", ".png").replace(".CR2", ".png"), a,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
