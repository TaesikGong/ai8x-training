
import torchvision
from functools import partial
import os
from torchvision import transforms
import ai8x
from datasets.custom_dataset import CustomSubsetDataset
from utils.data_reshape import data_reshape, fractional_repeat
import torch

initial_image_size = 227


def pacs_get_datasets(data, load_train=True, load_test=True,
                      input_size=224, target_size=64, target_channel=3, domain='A'):
    (data_dir, args) = data

    if domain == 'P':
        domain = 'photo'
    elif domain == 'A':
        domain = 'art_painting'
    elif domain == 'C':
        domain = 'cartoon'
    elif domain == 'S':
        domain = 'sketch'
    else:
        raise NotImplementedError

    full_train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'PACS', 'pacs_data','pacs_data', domain)
    )

    # Split the dataset into training (80%) and validation (20%)
    num_train = int(0.8 * len(full_train_dataset))
    num_val = len(full_train_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [num_train, num_val])

    if load_train:
        train_transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            # transforms.RandomResizedCrop(input_size),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            data_reshape(target_size, target_channel),
            transforms.Normalize(fractional_repeat((0.485, 0.456, 0.406), target_channel),
                                 fractional_repeat((0.229, 0.224, 0.225), target_channel)),
            ai8x.normalize(args=args),
            # transforms.Resize(target_size)
        ])

        train_dataset = CustomSubsetDataset(
            train_dataset, transform=train_transform)
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

        test_dataset = CustomSubsetDataset(
            val_dataset, transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = []

for domain in ['P', 'A', 'C', 'S']:
    for size in [32]:
        for channel in [3, 12, 48, 64]:
            dic = {}
            dic['name'] = f'PACS-{domain}_{channel}x{size}x{size}'
            dic['input'] = (channel, size, size)
            dic['output'] = list(map(str, range(7)))
            dic['loader'] = partial(pacs_get_datasets, load_train=True, load_test=True, input_size=initial_image_size,
                                    target_size=size, target_channel=channel, domain=domain)
            datasets.append(dic)
