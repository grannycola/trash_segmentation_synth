import os
import numpy as np
import pickle
import albumentations as A
import torch
import yaml


from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2


def get_default_from_yaml(param_name):
    with open('../../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    default_value = config.get(param_name, 0)
    return default_value


def get_transform():
    transform = [
        A.Resize(height=512, width=512),
        A.Normalize(),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.Perspective(scale=(0.05, 0.1), p=0.5),
        A.ElasticTransform(p=0.6),
        A.RandomBrightnessContrast(p=0.3),

        A.Blur(p=0.1),
        ToTensorV2(),
    ]
    return A.Compose(transform)


def get_val_transform():
    transform = [
        A.Resize(height=512, width=512),
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
                       dataloader_dir=None,
                       batch_size=None,
                       num_classes=None,
                       transform=get_transform(),):

    if not os.path.exists(dataloader_dir):
        print('Making new dataloader.pkl...')
        # Определите размеры наборов train, test и validation
        dataset = TacoDataset(image_dir, mask_dir)

        train_size = int(0.8 * len(dataset))
        test_size = int(0.1 * len(dataset))
        val_size = len(dataset) - train_size - test_size

        train_dataset, test_val_dataset = random_split(dataset, [train_size, test_size + val_size])
        test_dataset, val_dataset = random_split(test_val_dataset, [test_size, val_size])

        print(len(train_dataset))
        print(len(val_dataset))
        print(len(test_dataset))

        train_dataset.dataset.transform = transform
        test_dataset.dataset.transform = get_val_transform()
        val_dataset.dataset.transform = get_val_transform()

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

        with open(dataloader_dir, 'wb') as f:
            pickle.dump([train_dataloader,
                         val_dataloader,
                         test_dataloader], f)

        return train_dataloader, val_dataloader, test_dataloader
    else:
        print('Loading  from file...')

        with open(dataloader_dir, 'rb') as f:
            train_dataloader, \
                val_dataloader, \
                test_dataloader = pickle.load(f)
            print('Done')
            return train_dataloader, val_dataloader, test_dataloader
