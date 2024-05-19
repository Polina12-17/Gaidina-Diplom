import argparse
import glob
import math
import os

import cv2
import torch
import torchvision
from model import UnetTMO
import rawpy
from debayer import Debayer5x5
from debayer import Debayer5x5, Layout

f = Debayer5x5(layout=Layout.GRBG).cuda()


def read_image(path, is_raw=True):
    # print("path ", path)

    if path.lower().endswith('.jpg') or path.lower().endswith('.png'):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    elif is_raw:
        raw = rawpy.imread(path)
        im = raw.raw_image.copy()
        raw.close()
    else:
        raw = rawpy.imread(path)
        rgb = raw.postprocess()
        im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        raw.close()

    assert im is not None, path

    if is_raw:
        im = im / (math.pow(2, 14) - 1)
    else:
        im = im / 255.0

    im = torch.from_numpy(im).float()
    if is_raw:
        h, w = im.shape
        im = im.reshape((1, 1, h, w))
    else:
        im = im.permute(2, 0, 1)
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
    print("Process:", path)
    image = read_image(path).cuda()
    with torch.no_grad():
        test_img = f(image)
        torchvision.utils.save_image(test_img, path.replace(args.input_dir, args.output_dir).replace(".CR2", ".jpg_test.jpg"))

        output, _ = model(image)
        output = f(output)
    torchvision.utils.save_image(output, path.replace(args.input_dir, args.output_dir).replace(".CR2", ".jpg"))
