import cv2
import numpy as np
import rawpy
import torch


def debayer(output_image, raw):
    output_image = output_image[0, 0].cpu().detach().numpy()
    output_image = (output_image * np.iinfo(np.uint16).max).astype(np.uint16)
    startH, startW = output_image.shape
    h, w = raw.raw_image_visible.shape
    output_image = cv2.resize(output_image, (w, h))
    raw.raw_image_visible[:] = output_image
    post_processed_image = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                           demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)
    post_processed_image = post_processed_image.astype(np.float32) / np.iinfo(post_processed_image.dtype).max
    post_processed_image = cv2.resize(post_processed_image, (startH, startW))
    rgb_out = torch.from_numpy(post_processed_image).permute(2, 0, 1).unsqueeze(0)
    rgb_out = torch.clip(rgb_out[0].permute(1, 2, 0), 0.0, 1.0).numpy() * 255.0
    rgb_out = rgb_out.astype(np.uint8)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    bgr_out = torch.from_numpy(bgr_out).type(torch.float)

    bgr_out /= 255.0
    bgr_out = bgr_out.unsqueeze(0)
    bgr_out.requires_grad = True
    return bgr_out
