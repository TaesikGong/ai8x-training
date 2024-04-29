import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from utils.data_reshape import data_reshape

initial_image_size = 512


class Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories with 101,000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
            self,
            root: Union[str, Path],
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


import os

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import ai8x
import math


def food101_get_datasets(data, load_train=True, load_test=True,
                         input_size=224, target_size=64, target_channel=3):
    (data_dir, args) = data
    if load_train:
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            data_reshape(target_size, target_channel),
            # transforms.Resize(target_size)
        ])

        train_dataset = Food101(
            os.path.join(data_dir, 'Food101'),
            split='train',
            transform=train_transform,
            download=True,
        )
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            # transforms.Resize(int(input_size / 0.875)),  # 224/256 = 0.875
            # transforms.CenterCrop(input_size),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            data_reshape(target_size, target_channel),
            # transforms.Resize(target_size)
        ])

        test_dataset = Food101(
            root=os.path.join(data_dir, 'Food101'),
            split='test',
            transform=test_transform,
            download=True,
        )

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def food101_get_datasets_3x32x32(data, load_train=True, load_test=True,
                                 input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=32, target_channel=3)


def food101_get_datasets_12x32x32(data, load_train=True, load_test=True,
                                  input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=32, target_channel=12)


def food101_get_datasets_48x32x32(data, load_train=True, load_test=True,
                                  input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=32, target_channel=48)


def food101_get_datasets_64x32x32(data, load_train=True, load_test=True,
                                  input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=32, target_channel=64)


def food101_get_datasets_3x64x64(data, load_train=True, load_test=True,
                                 input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=64, target_channel=3)


def food101_get_datasets_12x64x64(data, load_train=True, load_test=True,
                                  input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=64, target_channel=12)


def food101_get_datasets_48x64x64(data, load_train=True, load_test=True,
                                  input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=64, target_channel=48)


def food101_get_datasets_3x112x112(data, load_train=True, load_test=True,
                                   input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=112, target_channel=3)


def food101_get_datasets_12x112x112(data, load_train=True, load_test=True,
                                    input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=112, target_channel=12)


def food101_get_datasets_48x112x112(data, load_train=True, load_test=True,
                                    input_size=initial_image_size):
    return food101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                target_size=112, target_channel=48)


datasets = [
    {
        'name': 'Food101_3x32x32',
        'input': (3, 32, 32),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_3x32x32
    },
    {
        'name': 'Food101_12x32x32',
        'input': (12, 32, 32),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_12x32x32
    },
    {
        'name': 'Food101_48x32x32',
        'input': (48, 32, 32),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_48x32x32
    },
    {
        'name': 'Food101_64x32x32',
        'input': (64, 32, 32),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_64x32x32
    },
    {
        'name': 'Food101_3x64x64',
        'input': (3, 64, 64),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_3x64x64
    },
    {
        'name': 'Food101_12x64x64',
        'input': (12, 64, 64),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_12x64x64
    },
    {
        'name': 'Food101_48x64x64',
        'input': (48, 64, 64),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_48x64x64
    },
    {
        'name': 'Food101_3x112x112',
        'input': (3, 112, 112),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_3x112x112
    },
    {
        'name': 'Food101_12x112x112',
        'input': (12, 112, 112),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_12x112x112
    },
    {
        'name': 'Food101_48x112x112',
        'input': (48, 112, 112),
        'output': list(map(str, range(101))),
        'loader': food101_get_datasets_48x112x112
    },
]
