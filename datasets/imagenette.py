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
from utils.data_reshape import data_reshape, fractional_repeat
from functools import partial

initial_image_size = 350
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
    if load_train:
        train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            data_reshape(target_size, target_channel),
            transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                 fractional_repeat((0.229, 0.224, 0.225), target_channel)),
            ai8x.normalize(args=args),
            # transforms.Resize(target_size)
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
            # transforms.Resize(int(input_size / 0.875)),  # 224/256 = 0.875
            # transforms.CenterCrop(input_size),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            data_reshape(target_size, target_channel),
            transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                 fractional_repeat((0.229, 0.224, 0.225), target_channel)),
            ai8x.normalize(args=args),
            # transforms.Resize(target_size)
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


datasets = []

for size in [32]:
    for channel in [3, 12, 48, 64]:
        dic = {}
        dic['name'] = f'Imagenette_{channel}x{size}x{size}'
        dic['input'] = (channel, size, size)
        dic['output'] = list(map(str, range(10)))
        dic['loader'] = partial(imagenette_get_datasets, load_train=True, load_test=True, input_size=initial_image_size,
                                target_size=size, target_channel=channel)
        datasets.append(dic)
