import torch
import click
import yaml
import os

from torchvision.models.segmentation import lraspp_mobilenet_v3_large as model_type
from custom_dataset import create_dataloaders
from metrics import IoU


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def get_default_from_yaml(param_name):
    with open('../../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    default_value = config.get(param_name, 0)
    return default_value


@click.command()
@click.option('--model_path', default=get_default_from_yaml('model_path'))
@click.option('--num_classes', default=int(get_default_from_yaml('num_classes')))
def get_cli_params_for_eval(model_path, num_classes):
    eval_model(model_path, num_classes)


def eval_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_type(num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(model_path)
    print('Model has been loaded!')

    # Set dataloaders
    print('Creating dataloaders...')
    _, _, test_dataloader = create_dataloaders()

    model.eval()
    with torch.no_grad():
        epoch_iou = 0
        for n, (images, masks) in enumerate(test_dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            epoch_iou += IoU(preds, masks, num_classes)

        epoch_iou /= len(test_dataloader)
        print(f'IoU: {epoch_iou}')
    return model


if __name__ == '__main__':
    get_cli_params_for_eval()
