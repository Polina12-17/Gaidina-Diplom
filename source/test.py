import argparse
import glob
import os

import cv2
import torch
import torchvision
from model import UnetTMO
import rawpy





def read_image(path):
    if (path.lower().endswith('.jpg')):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    else:
        raw = rawpy.imread(path)  # access to the RAW image

        # print(raw.color_desc) выводит порядок цветов в изображении

        raw_image = raw.raw_image.copy()
        print(raw_image[0][0])

        # print(raw.shape)
        colors = raw.raw_colors
        print("shape", raw_image.shape)
        print("raw image type", type(raw_image))
        raw.close()

        # help(raw)

        rgb = raw_image  # a numpy RGB array

        print(rgb.shape)

        # help(rgb)
        # img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:, :, ::-1]
        img = rgb

    # img = cv2.imread(path)[:, :, ::-1]
    img = img / 4095.0
    img = torch.from_numpy(img).float().unsqueeze(0)
    return img


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="..\\workdirs\\afifi\\version_10\checkpoints\\last.ckpt")
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
        output, _ = model(image)
    torchvision.utils.save_image(output, path.replace(args.input_dir, args.output_dir).replace(".CR2", ".jpg"))
