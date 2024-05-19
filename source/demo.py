import argparse
import glob
import math
import os
import rawpy
import torch
import torchvision
from debayer import Debayer5x5, Layout, Debayer3x3, DebayerSplit

from model import UnetTMO

f = DebayerSplit(Layout.RGGB).cuda()

def read_image(path, is_raw=True):
    # print("path ", path)

    raw = rawpy.imread(path)
    im = raw.raw_image.copy()

    raw.close()

    assert im is not None, path

    im = im / (math.pow(2, 14) - 1)

    im = torch.from_numpy(im).float()
    h, w = im.shape
    im = im.reshape((1, 1, h, w))
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
parser.add_argument("--checkpoint", type=str, default="..\\pretrained\\with_rawpy_postprocess_loss_283.ckpt")
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
        res = f(image)

        torchvision.utils.save_image(res,
                                     path.replace(args.input_dir, args.output_dir).replace(".CR2", ".jpg_test.jpg"))

        output, _ = model(image)
        output = f(output)
    torchvision.utils.save_image(output, path.replace(args.input_dir, args.output_dir).replace(".CR2", ".jpg"))
