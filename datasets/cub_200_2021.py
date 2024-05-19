
import torchvision
from functools import partial
import os
from torchvision import transforms
import ai8x
from datasets.custom_dataset import CustomSubsetDataset
from utils.data_reshape import DataReshape, fractional_repeat, DownsamplePIL, DownsampleTensor, DownsampleTensorRepeat
from utils.data_augmentation import DataAugmentation, DataRotation
from utils.coordconv import AI8XCoordConv2D
import torch

initial_image_size = 500


def cub_get_datasets(data, load_train=True, load_test=True,
                     input_size=224, target_size=64, target_channel=3):
    (data_dir, args) = data


    full_train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'CUB_200_2011', 'CUB_200_2011', 'images')
    )

    # Split the dataset into training (80%) and validation (20%)
    num_train = int(0.8 * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [num_train, num_val])


    transform_list = []

    transform_list.append(transforms.Resize((input_size, input_size)))

    assert (args.data_augment + args.coordconv + args.data_reshape) <= 1, "Only one or zero variable should be True"
    if args.data_augment:
        transform_list.append(DownsamplePIL(target_size))
        transform_list.append(DataAugmentation(args.aug))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    elif args.data_rotate:
        transform_list.append(DownsamplePIL(target_size))
        transform_list.append(DataRotation(target_channel))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    elif args.data_repeat:
        transform_list.append(transforms.ToTensor())
        transform_list.append(DownsampleTensorRepeat(target_size, target_channel))
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    elif args.coordconv:
        transform_list.append(transforms.ToTensor())
        transform_list.append(DownsampleTensor(target_size))
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)))
        transform_list.append(AI8XCoordConv2D(args.with_r))

    elif args.data_reshape:
        transform_list.append(transforms.ToTensor())
        transform_list.append(DataReshape(target_size, target_channel, args.data_reshape_method))
        transform_list.append(transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                                   fractional_repeat((0.229, 0.224, 0.225), target_channel)))
    else:  #simple downsampling
        transform_list.append(transforms.ToTensor())
        transform_list.append(DownsampleTensor(target_size))
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225)))
    transform_list.append(ai8x.normalize(args=args))

    if load_train:
        train_transform = transforms.Compose(transform_list)

        train_dataset = CustomSubsetDataset(
            train_dataset, transform=train_transform)
    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose(transform_list)

        test_dataset = CustomSubsetDataset(
            val_dataset, transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = []

for size in [32]:
    for channel in range(65):
        dic = {}
        dic['name'] = f'CUB_{channel}x{size}x{size}'
        dic['input'] = (channel, size, size)
        dic['output'] = list(map(str, range(200)))
        dic['loader'] = partial(cub_get_datasets, load_train=True, load_test=True, input_size=initial_image_size,
                                target_size=size, target_channel=channel)
        datasets.append(dic)
