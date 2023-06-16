import os
import numpy as np
import pickle
import albumentations as A
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2


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
        dataset = TacoDataset(image_dir, mask_dir, transform=transform)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # labels_count = torch.zeros(num_classes).to(device)
        # for i, (_, masks) in enumerate(dataset):
        #     print(i, end="\r")
        #     labels = torch.flatten(masks).to(device)
        #     labels_count += torch.bincount(labels, minlength=num_classes)
        #
        # class_inverse_frequencies = 1.0 / labels_count
        # class_weights = class_inverse_frequencies / torch.sum(class_inverse_frequencies)
        # class_weights[0] = 0
        # sampler = WeightedRandomSampler(class_weights, num_samples=len(dataset), replacement=True)
        # train_dataset, test_val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        # val_dataset, test_dataset = train_test_split(test_val_dataset, test_size=0.5, random_state=42)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=None)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

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
