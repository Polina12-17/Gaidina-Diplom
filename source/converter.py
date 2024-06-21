import glob

import rawpy
import torch
import torchvision

path = "..\\data_root\\cr2_afifi\\training\\INPUT_IMAGES\\"


def read_image(path):

    raw = rawpy.imread(path)
    im = raw.postprocess( use_camera_wb=True,
                         demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR)

    raw.close()

    assert im is not None, path

    im = im / 255.0

    im = torch.from_numpy(im).float()

    im = im.permute(2, 0, 1)
    return im


data_list = sorted(glob.glob(path + "*.*", recursive=True))
for p in data_list:
    print(p)
    im = read_image(p)
    np = p.replace("INPUT_IMAGES", "GT_IMAGES").replace("NEF", "png")
    torchvision.utils.save_image(im, np)
