import torch
import click
import os
import sys
import yaml

from tqdm import tqdm
from torchvision.models.segmentation import lraspp_mobilenet_v3_large as model_type
from checkpoint import ModelCheckpoint
from metrics import IoU
from custom_dataset import create_dataloaders

src_path = os.path.join(os.getcwd(), 'src')
sys.path.append(src_path)


def get_default_from_yaml(param_name):
    with open('../../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    default_value = config.get(param_name, 0)
    return default_value


@click.command()
@click.option('--model_path', default=get_default_from_yaml('model_path'))
@click.option('--image_dir', default=get_default_from_yaml('image_dir'))
@click.option('--mask_dir', default=get_default_from_yaml('mask_dir'))
@click.option('--dataloader_dir', default=get_default_from_yaml('dataloader_dir'))
@click.option('--batch_size', default=get_default_from_yaml('batch_size'))
@click.option('--num_classes', default=get_default_from_yaml('num_classes'))
@click.option('--num_epochs', default=get_default_from_yaml('num_epochs'))
def get_cli_params_for_training(model_path,
                                image_dir,
                                mask_dir,
                                dataloader_dir,
                                batch_size,
                                num_classes,
                                num_epochs,):
    train_model(model_path,
                image_dir,
                mask_dir,
                dataloader_dir,
                num_classes,
                batch_size,
                num_epochs,)


def train_model(model_path,
                image_dir,
                mask_dir,
                dataloader_dir,
                num_classes,
                batch_size,
                num_epochs,):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_type(num_classes=num_classes)
    model = model.to(device)
    checkpoint = ModelCheckpoint(model, model_path)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('Model has been loaded!')

    # Set dataloaders
    print('Creating dataloaders...')
    train_dataloader, \
        val_dataloader, \
        _ = create_dataloaders(image_dir=image_dir,
                               mask_dir=mask_dir,
                               dataloader_dir=dataloader_dir,
                               batch_size=batch_size,
                               num_classes=num_classes)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    print('Training model...')
    # tqdm pbar
    pbar = tqdm(range(num_epochs), position=0)
    pbar_desc_train = tqdm(total=0, position=1, bar_format='{desc}')
    pbar_desc_val = tqdm(total=0, position=2, bar_format='{desc}')
    interrupt_message = tqdm(total=0, position=3, bar_format='{desc}')

    try:
        
        for epoch in pbar:

            model.train()
            running_loss = 0.
            epoch_iou = 0.

            # Training on one epoch
            for i, (images, masks) in enumerate(train_dataloader):
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)['out']
                preds = torch.argmax(outputs, dim=1)

                epoch_iou += IoU(preds, masks, num_classes)
                loss_value = loss(outputs, masks.squeeze(1).long())
                loss_value.backward()

                optimizer.step()
                running_loss += loss_value

            last_loss = running_loss / len(train_dataloader)
            epoch_iou /= len(train_dataloader)

            # tqdm desc-string for train
            train_desc_str = f'Train values - Loss: {round(float(last_loss), 2)} IoU: {round(float(epoch_iou), 2)}'

            # Validation on one epoch
            model.eval()
            running_loss = 0.
            epoch_iou = 0.

            with torch.no_grad():
                for images, masks in val_dataloader:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)['out']
                    preds = torch.argmax(outputs, dim=1)
                    loss_value = loss(outputs, masks.squeeze(1).long())

                    running_loss += loss_value
                    epoch_iou += IoU(preds, masks, num_classes)

            val_loss = running_loss / len(val_dataloader)
            epoch_iou /= len(val_dataloader)

            # tqdm desc-string for val
            val_desc_str = f'Val values - Loss: {round(float(val_loss), 2)} IoU: {round(float(epoch_iou), 2)}'

            # Set value then clear strings
            pbar_desc_train.set_description_str(train_desc_str, refresh=True)
            pbar_desc_val.set_description_str(val_desc_str, refresh=True)

            if checkpoint:
                checkpoint(epoch + 1, val_loss)

        return model

    except KeyboardInterrupt:
        interrupt_message.set_description_str('Training Interrupted by User!!!', refresh=True)
        return model


if __name__ == '__main__':
    get_cli_params_for_training()
