import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import visualize
import torch
import albumentations as A
import json

from pycocotools.coco import COCO


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


class ApplyTransform(Dataset):

    def __init__(self, dataset,
                 augmentation=None,
                 preprocessing=None,):
        self.dataset = dataset
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.dataset)
    
    
    
class TacoDatasetSegmentation(Dataset):
    
    # remake this because it overload CPU
    @staticmethod
    def minimal_transformations(input_shape):
        transform = [
            A.Resize(height=input_shape[0], width=input_shape[1], p=1)
        ]
        return A.Compose(transform)
    
    # Use supercategories instead classes
    def getSupercategory(self, category_id):
        f = open(self.coco_annotation_file_path)
        cat_json_file = json.load(f)
        
        for category in cat_json_file['categories']:
            if category_id == category['id']:
                f = open('data/supercategories.json')
                supcat_json_file = json.load(f)
                return supcat_json_file[category['supercategory']]
    
    def __init__(self, json_ann_file, input_size=(512, 512), classes=None,):
        super(TacoDatasetSegmentation, self).__init__()
        
        self.coco_annotation_file_path = json_ann_file
        self.coco_annotation = COCO(annotation_file=self.coco_annotation_file_path)
        
        self.img_ids = self.coco_annotation.getImgIds()
        self.input_size = input_size

    def __getitem__(self, index):
        # Get images
        img_id = self.img_ids[index]
        img_info = self.coco_annotation.loadImgs(img_id)[0]
        
        img_path = 'data/resized/' + img_info["file_name"]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get Mask
        mask_filename = img_info["file_name"].replace('images/','').replace('.jpg', '.png')
        path_to_mask = 'data/resized/masks/' + mask_filename
        mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2)
        
        return image, mask

    def __len__(self):
        return len(self.img_ids)
    
    
class TacoLoaders:

    @staticmethod
    def train_test_split(dataset, train_size_perc, ):

        train_size = int(train_size_perc * len(dataset))
        test_size = len(dataset) - train_size

        return torch.utils.data.random_split(dataset, [train_size, test_size])

    def __init__(self, preprocessing_fn, batch_size = 4, augmentation=None, resize_input=None,
                 train_size_perc=0.889):

        self.train_dataset = TacoDatasetSegmentation('data/annotations_0_train.json')
        self.valid_dataset = TacoDatasetSegmentation('data/annotations_0_val.json')
        
        # Сначала сплитим данные, потом аугментации
        self.train_dataset, self.test_dataset =TacoLoaders.train_test_split(self.train_dataset,
                                                                            train_size_perc=train_size_perc)
        
        print(len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        
        # Аугментируем только тренировочные данные
        self.train_dataset = ApplyTransform(self.train_dataset,
                                            augmentation=augmentation,
                                            preprocessing=get_preprocessing(
                                                preprocessing_fn),
                                            )
        self.valid_dataset = ApplyTransform(self.valid_dataset,
                                            augmentation=resize_input,
                                            preprocessing=get_preprocessing(
                                                preprocessing_fn),
                                            )
        self.test_dataset = ApplyTransform(self.test_dataset,
                                           augmentation=resize_input,
                                           preprocessing=get_preprocessing(
                                               preprocessing_fn),
                                           )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    def show_example(self):
        n = np.random.choice(len(self.train_dataset))
        image, mask = self.train_dataset[n]
        visualize(image=image.transpose((1, 2, 0)),
                  mask=mask.transpose((1, 2, 0)) * 255)    