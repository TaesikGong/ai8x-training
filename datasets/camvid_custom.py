###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create CamVid dataset.
"""
import copy
import csv
import os
import sys

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from functools import partial

from PIL import Image
from datasets.camvid import CamVidDataset
import ai8x

from utils.data_reshape import DataReshape, fractional_repeat, Downsample_PIL, Downsample_Tensor
from utils.data_augmentation import DataAugmentation
from utils.coordconv import AI8XCoordConv2D


initial_image_size=352
def camvid_get_datasets_s352(data, load_train=True, load_test=True, num_classes=33, input_size=224, target_size=64, target_channel=3):
    """
    Load the CamVid dataset in 48x88x88 format which are composed of 3x352x352 images folded
    with a fold_ratio of 4.

    The dataset originally includes 33 keywords. A dataset is formed with 4 or 34 classes which
    includes 3, or 33 of the original keywords and the rest of the dataset is used to form the
    last class, i.e class of the others.

    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.
    """
    (data_dir, args) = data

    
    if num_classes == 3:
        classes = ['Building', 'Sky', 'Tree']
    elif num_classes == 33:
        classes = None
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')


    transform_list = []

    transform_list.append(transforms.Resize((input_size, input_size)))

    assert (args.data_augment + args.coordconv + args.data_reshape) <= 1, "Only one or zero variable should be True"
    if args.data_augment:
        transform_list.append(Downsample_PIL(target_size))
        transform_list.append(DataAugmentation(args.aug))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    elif args.coordconv:
        transform_list.append(transforms.ToTensor())
        transform_list.append(Downsample_Tensor(target_size))
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)))
        transform_list.append(AI8XCoordConv2D())

    elif args.data_reshape:
        transform_list.append(transforms.ToTensor())
        transform_list.append(DataReshape(target_size, target_channel, args.data_reshape_method))
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    else:  #simple downsampling
        transform_list.append(transforms.ToTensor())
        transform_list.append(Downsample_Tensor(target_size))
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)))
    transform_list.append(ai8x.normalize(args=args))

    
    if load_train:
        train_transform = transforms.Compose(transform_list)

        train_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='train',
                                      im_size=[352, 352], im_overlap=[168, 150], classes=classes,
                                      download=True, transform=train_transform)
    else:
        train_dataset = None

    
    if load_test:
        test_transform = transforms.Compose(transform_list)

        test_dataset = CamVidDataset(root_dir=os.path.join(data_dir, 'CamVid'), d_type='test',
                                     im_size=[352, 352], im_overlap=[0, 54], classes=classes,
                                     download=True, transform=test_transform)

        if args.truncate_testset:
            test_dataset.img_list = test_dataset.img_list[:1]
    else:
        test_dataset = None

    
    return train_dataset, test_dataset




datasets = []

for size in [32]:
    for channel in range(65):
        dic = {}
        dic['name'] = f'CamVid_{channel}x{size}x{size}'
        dic['input'] = (channel, size, size)
        dic['output'] = list(map(str, range(47)))
        dic['loader'] = partial(camvid_get_datasets_s352, num_classes=3, load_train=True, load_test=True, input_size=initial_image_size,
                                target_size=size, target_channel=channel)
        datasets.append(dic)
