import cv2
import numpy as np
import rawpy
import torch
import torchvision
from torch import nn

from source.data import InMemoryDataset
from source.currentDebayer import debayer

# class newMO(nn.Module):
#
#     def forward(self, x):
#         return x, x
#
#
# def get_label_fn(path):
#     # Заменить папку "INPUT_IMAGES" на "GT_IMAGES"
#     gt_path = path.replace("samples", "output")
#     # Удалить приписку в конце имени файла, если она существует, и заменить на "_gt" с сохранением формата
#     gt_path = gt_path.replace(".CR2", ".png")
#     return gt_path
#
#
# dataset = InMemoryDataset("", "samples/*.*", get_label_fn,
#                           return_name=False)
#
#
# counter = 0
#
# model = newMO()
# model.eval()
# model.cpu()
#
#
#
#
# for sample in dataset:
#     counter += 1
#     img, gt = sample
#     output_image, _ = model(img)
#     with torch.no_grad():
#         bgr_out = debayer(output_image)
#     bgr_out = bgr_out.numpy()
#
#     print(f"bgr_out: {bgr_out.shape}")
#     print(f"bgr_out min: {bgr_out.min()}")
#     print(f"bgr_out max: {bgr_out.max()}")
#
#     cv2.imwrite("test_output/" + str(counter) + ".png", bgr_out,
#                 [cv2.IMWRITE_PNG_COMPRESSION, 0])
#
#
#     gt = gt.detach().numpy()
#
#
#     print(f"gt: {gt.shape}")
#     print(f"gt min: {gt.min()}")
#     print(f"gt max: {gt.max()}")
#
#     cv2.imwrite(f"test_output/{counter}_gt.png", gt, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# path = ".\\output\\1.png"
# savePath = ".\\test_output\\2.png"
# image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# image = image / 255.0
#
# image = torch.from_numpy(image).float()
#
# image = image.permute(2, 0, 1).unsqueeze(0)
#
# print(f"image shape: {image.shape}")
# torchvision.utils.save_image(image, savePath)

#############################################

path = ".\\samples\\1.CR2"
savePath = ".\\test_output\\1.png"
raw = rawpy.imread(path)
raw_image = raw.raw_image_visible.copy()
raw_image = raw_image.astype(np.float32) / (2**14-1)
bayer_batch = torch.from_numpy(raw_image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
bayer_batch.requires_grad = True

print(f"batch_shape: {bayer_batch.shape}")
print(f"batch_shape min: {bayer_batch.min()}")
print(f"batch_shape max: {bayer_batch.max()}")
with torch.no_grad():
    raw.raw_image_visible[:] = bayer_batch.squeeze(0).squeeze(0).detach().numpy()
    a = raw.postprocess()
    #a =debayer(bayer_batch).squeeze(0)
b = torch.from_numpy(a)
print(f"b: {b.shape}")
print(f"b min: {b.min()}")
print(f"b max: {b.max()}")
# print(f"a min r: {a[0].min()}")
# print(f"a max r: {a[0].max()}")
# print(f"a min g: {a[1].min()}")
# print(f"a max g: {a[1].max()}")
# print(f"a min b: {a[2].min()}")
# print(f"a max b: {a[2].max()}")


torchvision.utils.save_image(b, savePath)
