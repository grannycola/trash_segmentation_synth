# trash_segmentation_synth
This is my garbage segmentation project. In this work, I used 5 supercategories from [TACO Dataset](http://tacodataset.org/). There are 1093 images in the dataset. Split into train/val/test - 80/10/10. 
To train the model, you can use the ```python src/models/train_model.py``` file in the console using the CLI. The default settings for model training are specified in the config.yaml file. To test the model, use the file ```python src/models/eval_model.py```. Baseline model deeplabv3_mobilenet_v3_large from Torchvision with IoU: ~0.51.

# Classes

- "Plastic bag & wrapper": 1,
- "Bottle": 2,
- "Carton": 3,
- "Can": 4,
- "Cup": 5

# Prediction Examples
![image](https://github.com/grannycola/trash_segmentation_synth/assets/54438026/24e0a73b-c776-4461-a7a3-e71b321ca32f)
![1](https://github.com/grannycola/trash_segmentation_synth/assets/54438026/9a3f41a1-c195-42b6-b610-199055e93162)


# Confusion Matrix
![CM](https://github.com/grannycola/trash_segmentation_synth/assets/54438026/b2a27b05-61f1-423d-b1c4-d97fe853094b)


# Requirements
- torchmetrics~=0.11.4
- click~=8.1.3
- pyyaml~=6.0
- torchvision~=0.14.1
- tqdm~=4.65.0
- numpy~=1.23.5
- albumentations~=1.3.0
- pillow~=9.4.0
- scikit-learn~=1.2.1
- setuptools~=65.6.3
- torch~=1.13.1
- tensorboardX~=2.6
