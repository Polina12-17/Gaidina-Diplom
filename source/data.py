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


class InMemoryDataset(Dataset):
    def __init__(self, root_folder, pattern, get_label_fn, resize=None, return_name=False):
        super().__init__()
        self.data_list = sorted(glob.glob(root_folder + pattern, recursive=True))
        self.resize = resize
        self.return_name = return_name
        self.data = []
        for path in self.data_list:
            im = self.read_image(path, resize)
            im_gt = self.read_image(get_label_fn(path), resize, True)
            self.data.append((im, im_gt))
        print("Total data samples:", len(self.data))

    @staticmethod
    def read_image(path, resize, is_Target=False):
        print("path ", path)

        if path.lower().endswith('.jpg') or path.lower().endswith('.png'):
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        elif not is_Target:
            raw = rawpy.imread(path)
            im = raw.raw_image.copy()
            raw.close()
        else:
            raw = rawpy.imread(path)
            rgb = raw.postprocess()
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:, :, ::-1]
            raw.close()

        assert im is not None, path
        if resize is not None:
            im = cv2.resize(im, (resize, resize))

        if not is_Target:
            im = im / (math.pow(2, 14) - 1)
        else:
            im = im / 255.0

        im = torch.from_numpy(im).float()
        if not is_Target:
            h, w = im.shape
            im = im.reshape((1, h, w))
        else:
            im = im.permute(2, 0, 1)
        return im

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


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

        self.train_data = InMemoryDataset(data_root, "training/INPUT_IMAGES/*.*", get_label_fn, resize=256,
                                          return_name=False)
        self.train_loader = DataLoader(self.train_data, batch_size=train_batch_size, shuffle=True,
                                       num_workers=num_workers)

        self.val_data = InMemoryDataset(data_root, "validation/INPUT_IMAGES/*.*", get_label_fn, resize=512,
                                        return_name=False)
        self.val_loader = DataLoader(self.val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        self.test_data = InMemoryDataset(data_root, "testing/INPUT_IMAGES/*.*", get_label_fn, resize=None,
                                         return_name=True)
        self.test_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
