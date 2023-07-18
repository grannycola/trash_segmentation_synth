# trash_segmentation_synth
This is my garbage segmentation project. In this work, I used 5 supercategories from [TACO Dataset](http://tacodataset.org/). There are 1093 images in the dataset. Split into train/val/test - 80/10/10. 
To train the model, you can use the ```python src/models/train_model.py``` file in the console using the CLI. The default settings for model training are specified in the config.yaml file. To test the model, use the file ```python src/models/eval_model.py```. Baseline model deeplabv3_mobilenet_v3_large from Torchvision with IoU: ~0.27.

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


# Dockerfile

Required:
- Nvidia GPU
- Nvidia Driver.
- Cuda 12.0.0
- nvidia-container-runtime

Check file content ```/etc/docker/daemon.json```
It should look like:
```
{
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```
Build image:
```
sudo docker build -t trash_segmentation .
```
Run container:
```
sudo docker run --gpus all -it -v $PWD:/app trash_segmentation
```
Run training:

```
make train
```
Tested on Nvidia RTX 3090Ti with Cuda driver 12.0.0
