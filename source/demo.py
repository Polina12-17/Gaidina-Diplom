import argparse
import glob
import os

import cv2
import numpy as np
import rawpy
import torch
from torch import nn


#from model import UnetTMO

class newMO(nn.Module):

    def forward(self, x):
        return x, x

map_location = torch.device('cpu')


def read_image(path, is_raw=True):
    raw = rawpy.imread(path)
    # Получаем сырое изображение в формате Bayer
    raw_image = raw.raw_image_visible.astype(np.float32) / np.iinfo(raw.raw_image_visible.dtype).max
    raw_image = np.expand_dims(raw_image, axis=0)  # Добавляем канал
    bayer_batch = torch.from_numpy(raw_image).unsqueeze(0)  # (1, 1, H, W)
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
parser.add_argument("--checkpoint", type=str, default="..\\pretrained\\with_rawpy_postprocess_loss_283.ckpt")
parser.add_argument("--input_dir", type=str, default="samples")
parser.add_argument("--output_dir", type=str, default="output")

args = parser.parse_args()

model = newMO()
state_dict = read_pytorch_lightning_state_dict(torch.load(args.checkpoint, map_location))
#model.load_state_dict(state_dict)
model.eval()
model.cpu()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

input_images = glob.glob(os.path.join(args.input_dir, "*"))
for path in input_images:
    print("Processing:", path)
    image, raw = read_image(path)
    img = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                           demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)
    bgr_out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path.replace(args.input_dir, args.output_dir).replace(".CR2", "_test.png"), bgr_out,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    with torch.no_grad():
        output, _ = model(image)
    print(type(output))
    output_image = output[0, 0].cpu().numpy()  # Первый элемент кортежа и извлекаем первый канал

    # Преобразование выходного тензора обратно в формат Bayer для rawpy
    output_image = (output_image * np.iinfo(np.uint16).max).astype(np.uint16)  # Приведение к типу uint16

    raw.raw_image_visible[:] = output_image  # Заменяем сырые данные на выход модели

    # Выполнение пост-обработки с использованием rawpy
    post_processed_image = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                           demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)

    # Преобразование пост-обработанного изображения в формат подходящий для сохранения
    post_processed_image = post_processed_image.astype(np.float32) / np.iinfo(post_processed_image.dtype).max

    # Сохранение результата
    rgb_out = torch.from_numpy(post_processed_image).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    rgb_out = torch.clip(rgb_out[0].permute(1, 2, 0), 0.0, 1.0).numpy() * 255.0
    rgb_out = rgb_out.astype(np.uint8)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path.replace(args.input_dir, args.output_dir).replace(".CR2", ".png"), bgr_out,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
