###################################################################################################
#
# Copyright (C) 2018-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
ImageNet Dataset (using PyTorch's ImageNet and ImageFolder classes)
"""
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


def imagenette_get_datasets(data, load_train=True, load_test=True,
                            input_size=224, target_size=64, target_channel=3, folder=True, augment_data=True):
    """
    Load the ImageNet 2012 Classification dataset.

    The original training dataset is split into training and validation sets.
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-128/128, +127/128]

    Data augmentation: 4 pixels are padded on each side, and a 224x224 crop is randomly sampled
    from the padded image or its horizontal flip.
    """
    (data_dir, args) = data
    if augment_data:
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

            if not folder:
                train_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'Imagenette'),
                    split='train',
                    transform=train_transform,
                )
            else:
                train_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'Imagenette', 'train'),
                    transform=train_transform,
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

            if not folder:
                test_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'Imagenette'),
                    split='val',
                    transform=test_transform,
                )
            else:
                test_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'Imagenette', 'val'),
                    transform=test_transform,
                )

            if args.truncate_testset:
                test_dataset.data = test_dataset.data[:1]
        else:
            test_dataset = None

    else:
        if load_train:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.ToTensor(),
                ai8x.normalize(args=args),
                data_reshape(target_size, target_channel),
                transforms.Resize(target_size)
            ])

            if not folder:
                train_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'Imagenette'),
                    split='train',
                    transform=train_transform,
                )
            else:
                train_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'Imagenette', 'train'),
                    transform=train_transform,
                )
        else:
            train_dataset = None

        if load_test:
            test_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.ToTensor(),
                ai8x.normalize(args=args),
                data_reshape(target_size, target_channel),
                transforms.Resize(target_size)
            ])

            if not folder:
                test_dataset = torchvision.datasets.ImageNet(
                    os.path.join(data_dir, 'Imagenette'),
                    split='val',
                    transform=test_transform,
                )
            else:
                test_dataset = torchvision.datasets.ImageFolder(
                    os.path.join(data_dir, 'Imagenette', 'val'),
                    transform=test_transform,
                )

            if args.truncate_testset:
                test_dataset.data = test_dataset.data[:1]
        else:
            test_dataset = None

    return train_dataset, test_dataset


def imagenette_get_datasets_3x64x64(data, load_train=True, load_test=True,
                                    input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=64, target_channel=3)


def imagenette_get_datasets_48x64x64(data, load_train=True, load_test=True,
                                     input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=64, target_channel=48)


def imagenette_get_datasets_3x32x32(data, load_train=True, load_test=True,
                                    input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=32, target_channel=3)


def imagenette_get_datasets_12x32x32(data, load_train=True, load_test=True,
                                     input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=32, target_channel=12)


def imagenette_get_datasets_48x32x32(data, load_train=True, load_test=True,
                                     input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=32, target_channel=48)


def imagenette_get_datasets_3x112x112(data, load_train=True, load_test=True,
                                      input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=112, target_channel=3)


def imagenette_get_datasets_12x112x112(data, load_train=True, load_test=True,
                                       input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=112, target_channel=12)


def imagenette_get_datasets_48x112x112(data, load_train=True, load_test=True,
                                       input_size=224, folder=True, augment_data=True):
    return imagenette_get_datasets(data=data, load_train=load_train, load_test=load_test, input_size=input_size,
                                   folder=folder, augment_data=augment_data, target_size=112, target_channel=48)


datasets = [
    {
        'name': 'Imagenette_3x64x64',
        'input': (3, 64, 64),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_3x64x64
    },
    {
        'name': 'Imagenette_48x64x64',
        'input': (48, 64, 64),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_48x64x64
    },
    {
        'name': 'Imagenette_3x32x32',
        'input': (3, 32, 32),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_3x32x32
    },
    {
        'name': 'Imagenette_12x32x32',
        'input': (12, 32, 32),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_12x32x32
    },
    {
        'name': 'Imagenette_48x32x32',
        'input': (48, 32, 32),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_48x32x32
    },
    {
        'name': 'Imagenette_3x112x112',
        'input': (3, 112, 112),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_3x112x112
    },
    {
        'name': 'Imagenette_12x112x112',
        'input': (12, 112, 112),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_12x112x112
    },
    {
        'name': 'Imagenette_48x112x112',
        'input': (48, 112, 112),
        'output': list(map(str, range(10))),
        'loader': imagenette_get_datasets_48x112x112
    },
]
