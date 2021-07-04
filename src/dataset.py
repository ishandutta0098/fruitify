import config 

import torch
import torchvision.datasets as datasets 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

train_transforms = transforms.Compose(
    [
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((32,32)),
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(config.MEAN), torch.Tensor(config.STD))
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(config.MEAN), torch.Tensor(config.STD))
    ]
)

train_dataset = datasets.ImageFolder(
    root = config.TRAIN_DATA_PATH,
    transform = train_transforms
)

test_dataset = datasets.ImageFolder(
    root = config.TEST_DATA_PATH,
    transform = test_transforms
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size = config.BATCH_SIZE,
    num_workers=4,
    shuffle = True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size = config.BATCH_SIZE,
    num_workers=4
)
