import torch
import click
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from model.checkpoint import ModelCheckpoint
from utils.metrics import IoU
from datasets.custom_dataset import create_dataloaders
from tqdm import tqdm

transform = A.Compose([
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
])


@click.command()
@click.option('--model_path', default='model/output/model.pth')
@click.option('--image_dir', default='data/processed/images/')
@click.option('--mask_dir', default='data/processed/masks/')
@click.option('--batch_size', default=16)
@click.option('--num_classes', default=21)
@click.option('--num_epochs', default=100)
def get_cli_params_for_training(model_path,
                                num_classes,
                                batch_size,
                                num_epochs,
                                image_dir,
                                mask_dir):

    train_model(model_path,
                num_classes,
                batch_size,
                num_epochs,
                image_dir, mask_dir)


def train_model(model_path,
                num_classes,
                batch_size,
                num_epochs,
                image_dir, mask_dir):

    checkpoint = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
    model = model.to(device)

    if model_path:
        checkpoint = ModelCheckpoint(model, model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('Model has been loaded!')

    # Set dataloaders
    print('Creating dataloaders...')
    train_dataloader, \
        val_dataloader, \
        test_dataloader = create_dataloaders(image_dir, mask_dir,
                                             batch_size, transform=transform)

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
        train_desc_str = ''
        val_desc_str = ''

        for epoch in pbar:

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
            train_desc_str += f'Train values - Loss: {last_loss} IoU: {epoch_iou}'

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
            val_desc_str += f'Val values - Loss: {val_loss} IoU: {epoch_iou}'

            # Set value then clear strings
            pbar_desc_train.set_description_str(train_desc_str, refresh=True)
            pbar_desc_val.set_description_str(val_desc_str, refresh=True)

            train_desc_str = ''
            val_desc_str = ''

            if checkpoint:
                checkpoint(epoch + 1, val_loss)

        return model

    except KeyboardInterrupt:
        interrupt_message.set_description_str('Training Interrupted by User!!!!', refresh=True)
        return model


if __name__ == '__main__':
    get_cli_params_for_training()
