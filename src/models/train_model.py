import torch
import click
import os
import yaml
import inspect
import mlflow

from tqdm import tqdm
from torchvision.models.segmentation import lraspp_mobilenet_v3_large as model_type
from checkpoint import ModelCheckpoint
from metrics import IoU
from custom_dataset import create_dataloaders
from src.utils.make_report import make_report


def get_default_from_yaml(param_name):
    with open('../../config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    default_value = config.get(param_name, 0)
    return default_value


@click.command()
@click.option('--model_path', default=None, type=click.Path())
@click.option('--image_dir', default=get_default_from_yaml('image_dir'), type=click.Path(exists=True))
@click.option('--mask_dir', default=get_default_from_yaml('mask_dir'), type=click.Path(exists=True))
@click.option('--logs_dir', default=get_default_from_yaml('logs_dir'), type=click.Path())
@click.option('--batch_size', default=get_default_from_yaml('batch_size'), help='Batch size', type=click.INT)
@click.option('--num_classes',
              default=get_default_from_yaml('num_classes'),
              help='Number of classes including background class',
              type=click.INT)
@click.option('--num_epochs',
              default=get_default_from_yaml('num_epochs'),
              help='Number of epochs for training',
              type=click.INT)
@click.option('--mixing_proportion', default=get_default_from_yaml('mixing_proportion'), type=click.FLOAT)
def get_cli_params_for_training(model_path,
                                image_dir,
                                mask_dir,
                                logs_dir,
                                batch_size,
                                num_classes,
                                num_epochs,
                                mixing_proportion):
    train_model(model_path,
                image_dir,
                mask_dir,
                logs_dir,
                num_classes,
                batch_size,
                num_epochs,
                mixing_proportion)


def train_model(model_path: str,
                image_dir: str,
                mask_dir: str,
                logs_dir: str,
                num_classes: int,
                batch_size: int,
                num_epochs: int,
                mixing_proportion: float):
    """
    :param mixing_proportion:
    :param model_path: Path to model directory
    :param image_dir: Path to image directory
    :param mask_dir: Path to mask directory
    :param logs_dir: Path to logs directory
    :param num_classes: Number of classes
    :param batch_size: Batch size
    :param num_epochs: Number of epochs
    :return:
    """

    params = locals()
    mlflow.start_run()
    mlflow.log_params(params)

    args = inspect.signature(train_model).parameters
    arg_names = [param for param in args]
    arg_values = []
    for param in arg_names:
        arg_values.append(locals()[param])

    arg_n_values = zip(arg_names, arg_values)

    # Set dataloaders
    print('Creating dataloaders...')
    train_dataloader, val_dataloader, _ = \
        create_dataloaders(image_dir=image_dir,
                           mask_dir=mask_dir,
                           batch_size=batch_size,
                           num_classes=num_classes,
                           mixing_proportion=mixing_proportion)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_type(num_classes=num_classes)
    model = model.to(device)

    checkpoint = None
    if model_path:
        checkpoint = ModelCheckpoint(model, model_path)
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state['model'])
            print('Model has been loaded!')

            checkpoint = ModelCheckpoint(model,
                                         model_path,
                                         state['best_loss'],
                                         state['best_metric'],)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    print('Training model...')
    # custom verbose
    pbar = tqdm(range(num_epochs),
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    pbar_batches = tqdm(total=len(train_dataloader), position=1,
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    pbar_desc_train = tqdm(total=0, position=2, bar_format='{desc}')
    pbar_desc_val = tqdm(total=0, position=3, bar_format='{desc}')
    interrupt_message = tqdm(total=0, position=4, bar_format='{desc}')

    train_loss_list = []
    val_loss_list = []
    train_metric_list = []
    val_metric_list = []

    try:
        for epoch in pbar:
            model.train()
            running_loss = 0.
            train_iou = 0.

            pbar_batches.reset()

            # Training on one epoch
            for i, (images, masks) in enumerate(train_dataloader):
                images = images.to(device)
                masks = masks.to(device)

                pbar_batches.update(1)

                optimizer.zero_grad()
                outputs = model(images)['out']
                preds = torch.argmax(outputs, dim=1)

                train_iou += IoU(preds, masks, num_classes)
                loss_value = loss(outputs, masks.squeeze(1).long())
                loss_value.backward()

                optimizer.step()
                running_loss += loss_value

            train_loss = running_loss / len(train_dataloader)
            train_iou /= len(train_dataloader)

            # tqdm desc-string for train
            train_desc_str = f'Train values - Loss: {round(float(train_loss), 2)} IoU: {round(float(train_iou), 2)}'

            # Validation on one epoch
            model.eval()
            running_loss = 0.
            val_iou = 0.

            with torch.no_grad():
                for images, masks in val_dataloader:
                    images = images.to(device)
                    masks = masks.to(device)

                    outputs = model(images)['out']
                    preds = torch.argmax(outputs, dim=1)
                    loss_value = loss(outputs, masks.squeeze(1).long())

                    running_loss += loss_value
                    val_iou += IoU(preds, masks, num_classes)

            val_loss = running_loss / len(val_dataloader)
            val_iou /= len(val_dataloader)

            train_loss_list.append(float(train_loss))
            val_loss_list.append(float(val_loss))

            train_metric_list.append(float(train_iou))
            val_metric_list.append(float(val_iou))

            mlflow.log_metric("Train Loss", train_loss)
            mlflow.log_metric("Val loss", val_loss)
            mlflow.log_metric("Train IoU", train_iou)
            mlflow.log_metric("Val IoU", val_iou)

            # tqdm desc-string for val
            val_desc_str = f'Val values - Loss: {round(float(val_loss), 2)} IoU: {round(float(val_iou), 2)}'

            # Set value then clear strings
            pbar_desc_train.set_description_str(train_desc_str, refresh=True)
            pbar_desc_val.set_description_str(val_desc_str, refresh=True)

            if checkpoint:
                checkpoint(epoch, val_loss, val_iou)

        mlflow.log_metric("Best Train Loss", min(train_loss_list))
        mlflow.log_metric("Best Val loss", min(val_loss_list))
        mlflow.log_metric("Best Train IoU", max(train_metric_list))
        mlflow.log_metric("Best Val IoU", max(val_metric_list))

        make_report(train_loss_list,
                    val_loss_list,
                    train_metric_list,
                    val_metric_list,
                    arg_n_values)

        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()
        return model

    except KeyboardInterrupt:
        make_report(train_loss_list,
                    val_loss_list,
                    train_metric_list,
                    val_metric_list,
                    arg_n_values)
        interrupt_message.set_description_str('Training interrupted by user!',
                                              refresh=True)
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()
        return model


if __name__ == '__main__':
    get_cli_params_for_training()
