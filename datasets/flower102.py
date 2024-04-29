from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets import VisionDataset


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
            self,
            root: Union[str, Path],
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

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

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)


import os

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import ai8x
import math


class data_reshape:
    """
    Fold data to increase the number of channels. An interlaced approach used in this folding
    as explained in [1].

    [1] https://arxiv.org/pdf/2203.16528.pdf
    """

    def __init__(self, target_size, target_channel):
        self.target_size = target_size
        self.target_channel = target_channel

    def __call__(self, img):
        current_num_channel = img.shape[0]
        if self.target_channel == current_num_channel:
            return img
        fold_ratio = int(math.sqrt(self.target_channel / current_num_channel))
        img_reshaped = None
        for i in range(fold_ratio):
            for j in range(fold_ratio):
                img_subsample = img[:, i::fold_ratio, j::fold_ratio]
                if img_reshaped is not None:
                    img_reshaped = torch.cat((img_reshaped, img_subsample), dim=0)
                else:
                    img_reshaped = img_subsample

        return img_reshaped


def flower102_get_datasets(data, load_train=True, load_test=True,
                           input_size=224, target_size=64, target_channel=3):
    (data_dir, args) = data
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            data_reshape(target_size, target_channel),
            transforms.Resize(target_size)
        ])

        train_dataset = Flowers102(
            os.path.join(data_dir, 'Flower102'),
            split='train',
            transform=train_transform,
            download=True,
        )
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize(int(input_size / 0.875)),  # 224/256 = 0.875
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            data_reshape(target_size, target_channel),
            transforms.Resize(target_size)
        ])

        test_dataset = Flowers102(
            root=os.path.join(data_dir, 'Flower102'),
            split='test',
            transform=test_transform,
            download=True,
        )

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def flower102_get_datasets_3x32x32(data, load_train=True, load_test=True,
                                   input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=32, target_channel=3)


def flower102_get_datasets_12x32x32(data, load_train=True, load_test=True,
                                    input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=32, target_channel=12)


def flower102_get_datasets_48x32x32(data, load_train=True, load_test=True,
                                    input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=32, target_channel=48)


def flower102_get_datasets_3x64x64(data, load_train=True, load_test=True,
                                   input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=64, target_channel=3)


def flower102_get_datasets_12x64x64(data, load_train=True, load_test=True,
                                    input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=64, target_channel=12)


def flower102_get_datasets_48x64x64(data, load_train=True, load_test=True,
                                    input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=64, target_channel=48)


def flower102_get_datasets_3x112x112(data, load_train=True, load_test=True,
                                     input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=112, target_channel=3)


def flower102_get_datasets_12x112x112(data, load_train=True, load_test=True,
                                      input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=112, target_channel=12)


def flower102_get_datasets_48x112x112(data, load_train=True, load_test=True,
                                      input_size=224):
    return flower102_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                  target_size=112, target_channel=48)


datasets = [
    {
        'name': 'Flower102_3x32x32',
        'input': (3, 32, 32),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_3x32x32
    },
    {
        'name': 'Flower102_12x32x32',
        'input': (12, 32, 32),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_12x32x32
    },
    {
        'name': 'Flower102_48x32x32',
        'input': (48, 32, 32),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_48x32x32
    },
    {
        'name': 'Flower102_3x64x64',
        'input': (3, 64, 64),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_3x64x64
    },
    {
        'name': 'Flower102_12x64x64',
        'input': (12, 64, 64),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_12x64x64
    },
    {
        'name': 'Flower102_48x64x64',
        'input': (48, 64, 64),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_48x64x64
    },
    {
        'name': 'Flower102_3x112x112',
        'input': (3, 112, 112),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_3x112x112
    },
    {
        'name': 'Flower102_12x112x112',
        'input': (12, 112, 112),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_12x112x112
    },
    {
        'name': 'Flower102_48x112x112',
        'input': (48, 112, 112),
        'output': list(map(str, range(102))),
        'loader': flower102_get_datasets_48x112x112
    },
]
