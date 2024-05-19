import glob
import math
import os
import rawpy
import cv2
import torch
import re
from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import DataLoader, Dataset



class NoGTDataset(Dataset):
    def __init__(self, root_folder, pattern, get_label_fn, resize=None, return_name=False):
        super().__init__()
        self.data_list = sorted(glob.glob(root_folder + pattern, recursive=True))
        self.gt_list = [get_label_fn(p) for p in self.data_list]
        self.resize = resize
        self.return_name = return_name
        print("Total data samples:", len(self.data_list))

    def read_image(self, path, is_raw=False):
        #print("path ", path)

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
        if self.resize is not None:
            im = cv2.resize(im, (self.resize, self.resize))

        if is_raw:
            im = im / (math.pow(2, 14) - 1)
        else:
            im = im / 255.0

        im = torch.from_numpy(im).float()
        if is_raw:
            h, w = im.shape
            im = im.reshape((1, h, w))
        else:
            im = im.permute(2, 0, 1)
        return im

    def __getitem__(self, index):
        input_path = self.data_list[index]
        input_im = self.read_image(input_path, is_raw=True)

        gt_im = self.read_image(self.gt_list[index], is_raw=True)
        if self.return_name:
            return input_im, gt_im, os.path.join(*input_path.split("/")[-2:])
        return input_im, gt_im

    def __len__(self):
        return len(self.data_list)


@DATAMODULE_REGISTRY
class AfifiDataModule(LightningDataModule):
    def __init__(self, data_root, train_batch_size, val_batch_size, num_workers):
        super().__init__()

        def get_label_fn(path):
            # Заменить папку "INPUT_IMAGES" на "GT_IMAGES"
            gt_path = path.replace("INPUT_IMAGES", "GT_IMAGES")
            # Удалить приписку в конце имени файла, если она существует, и заменить на "_gt" с сохранением формата
            gt_path = re.sub(r'\s*\(\d+\)?(\.[^.]+)$', r'_gt\1', gt_path)
            return gt_path

        self.train_data = NoGTDataset(data_root , "training/INPUT_IMAGES/*.*", get_label_fn, resize=255,
                                      return_name=False)
        self.train_loader = DataLoader(self.train_data, batch_size=train_batch_size, shuffle=True,
                                       num_workers=num_workers)

        self.val_data = NoGTDataset(data_root , "validation/INPUT_IMAGES/*.*", get_label_fn, resize=512,
                                    return_name=False)
        self.val_loader = DataLoader(self.val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        self.test_data = NoGTDataset(data_root , "testing/INPUT_IMAGES/*.*", get_label_fn, resize=None,
                                     return_name=True)
        self.test_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
