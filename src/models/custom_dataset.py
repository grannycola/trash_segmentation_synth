import os
import numpy as np
import albumentations as A
import torch
import yaml

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from albumentations.pytorch import ToTensorV2

torch.manual_seed(42)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def get_default_from_yaml(param_name):
    with open('../../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    default_value = config.get(param_name, 0)
    return default_value


def get_transform():
    transform = [
        A.Resize(height=256, width=256),
        A.Normalize(),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.ElasticTransform(p=0.3),
        A.RandomBrightnessContrast(p=0.3),

        A.Blur(p=0.1),
        ToTensorV2(),
    ]
    return A.Compose(transform)


def get_val_transform():
    transform = [
        A.Resize(height=256, width=256),
        A.Normalize(),
        ToTensorV2(),
    ]
    return A.Compose(transform)


class TacoDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image_path = os.path.join(self.image_dir, self.masks[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = np.asarray(image).astype(np.uint8)
        mask = np.asarray(mask).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


def create_dataloaders(image_dir=None,
                       mask_dir=None,
                       batch_size=None,
                       num_classes=None,
                       val_proportion=get_default_from_yaml('val_proportion'),
                       test_proportion=get_default_from_yaml('test_proportion'),
                       mixing_proportion=get_default_from_yaml('mixing_proportion'),
                       transform=get_transform(), ):

    if val_proportion + test_proportion >= 1:
        raise Exception("Sum of val and test proportions should be less than 1")

    print('Making new dataloader...')
    # Определите размеры наборов train, test и validation
    dataset = TacoDataset(image_dir, mask_dir)

    test_size = int(test_proportion * len(dataset))
    val_size = int(val_proportion * len(dataset))
    train_size = len(dataset) - test_size - val_size

    train_dataset, test_val_dataset = random_split(dataset, [train_size, test_size + val_size])
    test_dataset, val_dataset = random_split(test_val_dataset, [test_size, val_size])

    train_dataset.dataset.transform = transform
    test_dataset.dataset.transform = get_val_transform()
    val_dataset.dataset.transform = get_val_transform()

    synthetized_dataset = TacoDataset('../../data/synthetized_data/images/',
                                      '../../data/synthetized_data/masks_5_classes/',
                                      transform=transform)

    synthetized_size = int(mixing_proportion * train_size)
    if synthetized_size > 0:
        mix_train_dataset, _ = random_split(synthetized_dataset,
                                            [synthetized_size,
                                             len(synthetized_dataset) - synthetized_size])
        mix_train_dataset.dataset.transform = transform
        train_dataset = ConcatDataset([train_dataset, mix_train_dataset])

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=get_default_from_yaml('num_workers'),
                                  sampler=None)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                num_workers=get_default_from_yaml('num_workers'))

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=get_default_from_yaml('num_workers'))

    return train_dataloader, val_dataloader, test_dataloader
