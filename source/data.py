import glob

import cv2
import numpy as np
import rawpy
import torch
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
        for index, path in enumerate(self.data_list):
            im = self.read_raw(path, resize)
            im_gt = self.read_png_to_matrix(get_label_fn(path), resize)
            self.data.append((im, im_gt))
        print("Total data samples:", len(self.data))

    @staticmethod
    def read_raw(path, resize):
        print("path ", path)
        raw = rawpy.imread(path)
        # Получаем сырое изображение в формате Bayer
        raw_image = raw.raw_image_visible.copy().astype(np.float32) / (2**14-1)
        if (resize is not None):
            raw_image = cv2.resize(raw_image, (resize, resize))
        assert raw_image is not None, path
        im = np.expand_dims(raw_image, axis=0)  # Добавляем канал
        bayer_batch = torch.from_numpy(raw_image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        bayer_batch.requires_grad = True
        return bayer_batch

    @staticmethod
    def read_png_to_matrix(path, resize):
        print("path ", path)

        # Чтение изображения
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image / 255.0

        if (resize is not None):
            image = cv2.resize(image, (resize, resize))
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1).unsqueeze(0)
        image.requires_grad = True
        return image

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
            gt_path = gt_path.replace(".CR2", ".png")
            return gt_path

        self.train_data = InMemoryDataset(data_root, "training/INPUT_IMAGES/*.*", get_label_fn, resize=256,
                                          return_name=False)
        self.train_loader = DataLoader(self.train_data, batch_size=train_batch_size, shuffle=True,
                                       num_workers=num_workers)

        self.val_data = InMemoryDataset(data_root, "validation/INPUT_IMAGES/*.*", get_label_fn,
                                        return_name=False)
        self.val_loader = DataLoader(self.val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        self.test_data = InMemoryDataset(data_root, "testing/INPUT_IMAGES/*.*", get_label_fn,
                                         return_name=True)
        self.test_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
