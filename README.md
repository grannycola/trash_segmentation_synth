# trash_segmentation_synth
This is my garbage segmentation project. In this work, I used 5 supercategories from [TACO Dataset](http://tacodataset.org/). There are 1093 images in the dataset. Split into train/val/test - 80/10/10. 
To train the model, you can use the ```python src/models/train_model.py``` file in the console using the CLI. The default settings for model training are specified in the config.yaml file. To test the model, use the file ```python src/models/eval_model.py```. Baseline model deeplabv3_mobilenet_v3_large from Torchvision with IoU: ~0.27. When mixing data (with mixing_proportion 0.25 - 0.5), the quality increases to ~0.33 IoU.

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
- Installed Nvidia Driver.
- Cuda 12.0.0
- Installed nvidia-container-runtime (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  
At first set ```num_workers: 0``` in ```config.yaml```

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

Run dockerd (if it's not running):
```
sudo dockerd
```

Make sure that "nvidia" is in "Runtimes" list:
```
$ docker info|grep -i runtime
 Runtimes: nvidia runc
 Default Runtime: runc
```

Build image:
```
sudo docker build --no-cache -t trash_segmentation .
```
Run container:
```
sudo docker run --memory=16g --gpus all -it -v $PWD/data:/app/data gaofen_segmentation
```
Run training:
```
make train
```
Tested on Nvidia RTX 3090Ti with Cuda driver 12.0.0
