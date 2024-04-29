import os
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

            .. warning::

                To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
    """

    def __init__(
            self,
            root: Union[str, Path],
            target_type: Union[List[str], str] = "category",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        )

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self) -> int:
        return len(self.index)

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )
        download_and_extract_archive(
            "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m",
            self.root,
            filename="Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91",
        )

    def extra_repr(self) -> str:
        return "Target type: {target_type}".format(**self.__dict__)


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


class CustomSubsetDataset(Dataset): #required to apply transform to subset datasets
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def caltech101_get_datasets(data, load_train=True, load_test=True,
                            input_size=224, target_size=64, target_channel=3):
    (data_dir, args) = data

    full_train_dataset = Caltech101(
        os.path.join(data_dir, 'Caltech101'),
        download=True,
    )

    # Split the dataset into training (80%) and validation (20%)
    num_train = int(0.8 * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [num_train, num_val])

    if load_train:
        train_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),  # Convert grayscale to RGB if needed
            # transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            data_reshape(target_size, target_channel),
            transforms.Resize(target_size)
        ])

        train_dataset = CustomSubsetDataset(
            train_dataset, transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),  # Convert grayscale to RGB if needed
            transforms.Resize(int(input_size / 0.875)),  # 224/256 = 0.875
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            data_reshape(target_size, target_channel),
            transforms.Resize(target_size)
        ])

        test_dataset = CustomSubsetDataset(
            val_dataset, transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def caltech101_get_datasets_3x32x32(data, load_train=True, load_test=True,
                                    input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=32, target_channel=3)


def caltech101_get_datasets_12x32x32(data, load_train=True, load_test=True,
                                     input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=32, target_channel=12)


def caltech101_get_datasets_48x32x32(data, load_train=True, load_test=True,
                                     input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=32, target_channel=48)


def caltech101_get_datasets_3x64x64(data, load_train=True, load_test=True,
                                    input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=64, target_channel=3)


def caltech101_get_datasets_12x64x64(data, load_train=True, load_test=True,
                                     input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=64, target_channel=12)


def caltech101_get_datasets_48x64x64(data, load_train=True, load_test=True,
                                     input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=64, target_channel=48)


def caltech101_get_datasets_3x112x112(data, load_train=True, load_test=True,
                                      input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=112, target_channel=3)


def caltech101_get_datasets_12x112x112(data, load_train=True, load_test=True,
                                       input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=112, target_channel=12)


def caltech101_get_datasets_48x112x112(data, load_train=True, load_test=True,
                                       input_size=224):
    return caltech101_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   target_size=112, target_channel=48)


datasets = [
    {
        'name': 'Caltech101_3x32x32',
        'input': (3, 32, 32),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_3x32x32
    },
    {
        'name': 'Caltech101_12x32x32',
        'input': (12, 32, 32),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_12x32x32
    },
    {
        'name': 'Caltech101_48x32x32',
        'input': (48, 32, 32),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_48x32x32
    },
    {
        'name': 'Caltech101_3x64x64',
        'input': (3, 64, 64),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_3x64x64
    },
    {
        'name': 'Caltech101_12x64x64',
        'input': (12, 64, 64),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_12x64x64
    },
    {
        'name': 'Caltech101_48x64x64',
        'input': (48, 64, 64),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_48x64x64
    },
    {
        'name': 'Caltech101_3x112x112',
        'input': (3, 112, 112),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_3x112x112
    },
    {
        'name': 'Caltech101_12x112x112',
        'input': (12, 112, 112),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_12x112x112
    },
    {
        'name': 'Caltech101_48x112x112',
        'input': (48, 112, 112),
        'output': list(map(str, range(101))),
        'loader': caltech101_get_datasets_48x112x112
    },
]
