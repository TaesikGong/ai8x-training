
import torchvision
from functools import partial
import os
from torchvision import transforms
import ai8x
from datasets.custom_dataset import CustomSubsetDataset
from utils.data_reshape import data_reshape
import torch

initial_image_size = 360

import pathlib
from typing import Any, Callable, Optional, Tuple, Union

from PIL import Image

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset


class StanfordCars(VisionDataset):
    """Stanford Cars  Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    The original URL is https://ai.stanford.edu/~jkrause/cars/car_dataset.html, but it is broken.

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): This parameter exists for backward compatibility but it does not
            download the dataset, since the original URL is not available anymore. The dataset
            seems to be available on Kaggle so you can try to manually download it using
            `these instructions <https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616>`_.
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. Try to manually download following the instructions in "
                "https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616."
            )

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()

    def download(self):
        raise ValueError(
            "The original URL is broken so the StanfordCars dataset is not available for automatic "
            "download anymore. You can try to download it manually following "
            "https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616, "
            "and set download=False to avoid this error."
        )


def stanfordcars_get_datasets(data, load_train=True, load_test=True,
                              input_size=224, target_size=64, target_channel=3):
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            data_reshape(target_size, target_channel),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            # transforms.Resize(target_size)
        ])

        train_dataset = StanfordCars(
            os.path.join(data_dir),
            split='train',
            transform=train_transform,
            download=False,  # automatic download link is broken. must download and place the folder manually
        )
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            # transforms.Resize(int(input_size / 0.875)),  # 224/256 = 0.875
            # transforms.CenterCrop(input_size),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            data_reshape(target_size, target_channel),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ai8x.normalize(args=args),
            # transforms.Resize(target_size)
        ])

        test_dataset = StanfordCars(
            root=os.path.join(data_dir),
            split='test',
            transform=test_transform,
            download=False,
        )

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset
datasets = []

for size in [32]:
    for channel in [3, 12, 48, 64]:
        dic = {}
        dic['name'] = f'StanfordCars_{channel}x{size}x{size}'
        dic['input'] = (channel, size, size)
        dic['output'] = list(map(str, range(196)))
        dic['loader'] = partial(stanfordcars_get_datasets, load_train=True, load_test=True, input_size=initial_image_size,
                                target_size=size, target_channel=channel)
        datasets.append(dic)
