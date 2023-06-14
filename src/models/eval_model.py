import torch
import click

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from custom_dataset import create_dataloaders
from metrics import IoU


@click.command()
@click.option('--model_path', default='../../models/output/model.pth')
@click.option('--dataloader_dir', default='../../models/output/dataloader.pkl')
@click.option('--num_classes', default=21)
def get_cli_params_for_eval(model_path, dataloader_dir, num_classes):
    eval_model(model_path, dataloader_dir, num_classes)


def eval_model(model_path, dataloader_dir, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print('Model has been loaded!')

    # Set dataloaders
    print('Creating dataloaders...')
    _, _, test_dataloader = create_dataloaders(dataloader_dir=dataloader_dir)

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


if __name__ == '__main__':
    get_cli_params_for_eval()
