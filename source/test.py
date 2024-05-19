import argparse
import glob
import math
import os

import cv2
import numpy as np
import torch
import torchvision
from model import UnetTMO
import rawpy


def read_image(path, is_raw=False):
    # print("path ", path)

    raw = rawpy.imread(path)
    print(raw.color_desc )
    print(raw.raw_pattern  )
    print(type(raw))
    im = raw.raw_image
    print(type(im))

    raw.close()

    assert im is not None, path

    im = im / 255.0

    im = torch.from_numpy(im).float()

#    im = im.permute(2, 0, 1)
    return im


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="..\\pretrained\\my_afifi_14_epoh_loss_240.ckpt")
parser.add_argument("--input_dir", type=str, default="samples")
parser.add_argument("--output_dir", type=str, default="output")

args = parser.parse_args()

model = UnetTMO()
state_dict = read_pytorch_lightning_state_dict(torch.load(args.checkpoint))
model.load_state_dict(state_dict)
model.eval()
model.cuda()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

input_images = glob.glob(os.path.join(args.input_dir, "*"))
for path in input_images:
    print(path)
    image = read_image(path).cuda()
    # with torch.no_grad():
    #     output, _ = model(image)
    #torchvision.utils.save_image(image, path.replace(args.input_dir, args.output_dir) + ".jpg")
