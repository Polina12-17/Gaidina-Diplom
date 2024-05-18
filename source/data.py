import glob  # для работы с папочками и файликами
import math
import os
import rawpy

import cv2
import torch
from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils import data
from torch.utils.data import DataLoader


class NoGTDataset(data.Dataset):  # подгрузка данных
    def __init__(self, root_folder, pattern, resize=None, return_name=False):
        super().__init__()
        self.data_list = sorted(glob.glob(root_folder + pattern, recursive=True))  # грузится список всех фалов
        self.resize = resize
        self.return_name = return_name
        print("Total data samples:", len(self.data_list))

    def __getitem__(self, index):  # получают по индексу сам файл
        path = self.data_list[index]
        if (path.lower().endswith('.jpg')):
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        else:
            raw = rawpy.imread(path)  # access to the RAW image

            # print(raw.color_desc) выводит порядок цветов в изображении

            raw_image = raw.raw_image.copy()

            raw.close()


            rgb = raw_image  # a numpy RGB array

            # print(rgb.shape)

            im = rgb

        # img = cv2.imread(path)[:, :, ::-1]
        if self.resize is not None:
            im = cv2.resize(im, (self.resize, self.resize))
        im = im / (math.pow(2,14) - 1)

        if(im.max() > 1.0):
            print("im min = ", im.min())
            print("im max = ", im.max())

        im = torch.from_numpy(im).float()

        h, w = im.shape
        im = im.reshape((1, h, w))
        # print(im.shape)

        # print("тест ")
        if self.return_name:
            return im, os.path.join(*path.split("/")[-2:])
        return im

    def __len__(self):  # общее количество элементов в датасете
        return len(self.data_list)


class PairedDataset(data.Dataset):
    def __init__(self, root_folder, pattern, get_label_fn, resize=None, return_name=False):
        super().__init__()
        print(root_folder)
        self.data_list = sorted(glob.glob(root_folder + pattern, recursive=True))
        self.gt_list = [get_label_fn(p) for p in self.data_list]
        self.resize = resize
        self.return_name = return_name
        print("Total data samples:", len(self.data_list))

    def read_image(self, path):
        if (path.lower().endswith('.jpg') or path.lower().endswith('.png')):
            im = cv2.imread(path)
        else:
            raw = rawpy.imread(path)  # access to the RAW image
            rgb = raw.postprocess()  # a numpy RGB array
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        assert im is not None, path
        im = im[:, :, ::-1]
        if self.resize is not None:
            im = cv2.resize(im, (self.resize, self.resize))
        im = im / 255.0
        im = torch.from_numpy(im).float().permute(2, 0, 1)
        return im

    def __getitem__(self, index):
        input_path = self.data_list[index]
        input_im = self.read_image(input_path)
        gt_im = self.read_image(self.gt_list[index])
        if self.return_name:
            return input_im, gt_im, os.path.join(*input_path.split("/")[-2:])
        return (
            input_im,
            gt_im,
        )

    def __len__(self):
        return len(self.data_list)


@DATAMODULE_REGISTRY  # регистрируем класс
class AfifiDataModule(LightningDataModule):
    def __init__(self, data_root, train_batch_size, val_batch_size, num_workers):
        super().__init__()
        train_data = NoGTDataset(data_root, "training/INPUT_IMAGES/*.*", resize=256, return_name=False)
        self.train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

        val_data = NoGTDataset(data_root, "validation/INPUT_IMAGES/*.*", resize=512, return_name=False)
        self.val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

        def get_label_fn(path):
            gt_path = path.replace("INPUT_IMAGES", "expert_e_testing_set")
            gt_path = gt_path[:-9] + "*"
            gt_path = glob.glob(gt_path)[0]
            return gt_path

        test_data = PairedDataset(data_root, "testing/INPUT_IMAGES/*.*", get_label_fn, resize=None, return_name=True)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


@DATAMODULE_REGISTRY
class SICEDataModule(LightningDataModule):
    def __init__(self, data_root, train_batch_size, num_workers):
        super().__init__()
        train_data = NoGTDataset(data_root, "train_data/*", resize=256, return_name=False)
        self.train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

        def get_label_fn(path):
            dirname = os.path.dirname(path)
            dirname, img_idx = os.path.split(dirname)
            out = glob.glob(os.path.join(dirname, "Label", img_idx + ".*"))
            return out[0]

        test_data = PairedDataset(data_root, "Dataset_Part2/[0-9]*/*.*", get_label_fn, resize=None, return_name=True)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader


@DATAMODULE_REGISTRY
class LOLDataModule(LightningDataModule):
    def __init__(self, data_root, num_workers):
        super().__init__()

        def get_label_fn(path):
            return path.replace("low", "high")

        test_data = PairedDataset(data_root, "*/low/*.*", get_label_fn, resize=None, return_name=True)
        self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)

    def test_dataloader(self):
        return self.test_loader
