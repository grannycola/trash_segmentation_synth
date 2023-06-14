import os
import numpy as np
import pickle
import albumentations as A


from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2


def get_transform():
    transform = [
        A.Resize(height=512, width=512),
        A.Normalize(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.ElasticTransform(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GridDistortion(p=0.5),
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

        if self.transform:
            transformed = self.transform(image=np.asarray(image).copy(), mask=np.asarray(mask).copy())
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


def create_dataloaders(image_dir='../../data/processed/images/',
                       mask_dir='../../data/processed/masks/',
                       batch_size=16, transform=get_transform()):
    file_path = '../../models/output/dataloader.pkl'

    if not os.path.exists(file_path):
        print('There is no dataloader.pkl, making new...')
        dataset = TacoDataset(image_dir, mask_dir, transform=transform)
        train_dataset, test_val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        val_dataset, test_dataset = train_test_split(test_val_dataset, test_size=0.5, random_state=42)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

        with open(file_path, 'wb') as f:
            pickle.dump([train_dataloader,
                         val_dataloader,
                         test_dataloader], f)

        return train_dataloader, val_dataloader, test_dataloader
    else:
        print('Loading dataloaders...')

        with open(file_path, 'rb') as f:
            train_dataloader, \
                val_dataloader, \
                test_dataloader = pickle.load(f)
            print('Done')
            return train_dataloader, val_dataloader, test_dataloader
