import torch
import os
import inspect
import mlflow

from typing import Optional, Tuple
from tqdm import tqdm
from torchvision.models.segmentation import lraspp_mobilenet_v3_large as model_type

from src.models.checkpoint import ModelCheckpoint
from src.models.metrics import IoU
from src.models.custom_dataset import create_dataloaders
from src.utils.make_report import make_report


def run_iteration(images: torch.Tensor,
                  masks: torch.Tensor,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[torch.Tensor, torch.Tensor]:

    outputs = model(images)['out']
    preds = torch.argmax(outputs, dim=1)
    loss_value = loss_fn(outputs, masks.squeeze(1).long())

    if optimizer:
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    return loss_value, preds


def train_model(model_path: str,
                image_dir: str,
                mask_dir: str,
                logs_dir: str,
                num_classes: int,
                batch_size: int,
                num_epochs: int,
                mixing_proportion: float):
    params = locals()

    mlflow.set_tracking_uri('../../reports/mlruns/')
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

    train_loss_list = []
    val_loss_list = []
    train_metric_list = []
    val_metric_list = []

    try:
        pbar_epochs = tqdm(range(num_epochs), desc="Epochs", ncols=150)
        for epoch in pbar_epochs:
            model.train()
            running_loss = 0.
            train_iou = 0.

            pbar_batches = tqdm(train_dataloader, desc="Training", leave=False, ncols=150)
            for images, masks in pbar_batches:
                images = images.to(device)
                masks = masks.to(device)

                loss_value, preds = run_iteration(images, masks, model, loss, optimizer)
                running_loss += loss_value
                train_iou += IoU(preds, masks, num_classes)

                # tqdm desc-string for train
                train_desc_str = (f'Train values - Loss: {round(float(running_loss / len(train_dataloader)), 2)} '
                                  f'IoU: {round(float(train_iou / len(train_dataloader)), 2)}')

                pbar_batches.set_postfix_str(train_desc_str)

            train_loss = running_loss / len(train_dataloader)
            train_iou /= len(train_dataloader)

            # Validation on one epoch
            model.eval()
            running_loss = 0.
            val_iou = 0.

            with torch.no_grad():
                for images, masks in val_dataloader:
                    images = images.to(device)
                    masks = masks.to(device)
                    loss_value, preds = run_iteration(images, masks, model, loss)
                    running_loss += loss_value
                    val_iou += IoU(preds, masks, num_classes)

            val_loss = running_loss / len(val_dataloader)
            val_iou /= len(val_dataloader)

            # tqdm desc-string for val
            val_desc_str = f'Val values - Loss: {round(float(val_loss), 2)} IoU: {round(float(val_iou), 2)}'
            pbar_epochs.set_postfix_str(val_desc_str)

            train_loss_list.append(float(train_loss))
            val_loss_list.append(float(val_loss))
            train_metric_list.append(float(train_iou))
            val_metric_list.append(float(val_iou))

            metrics = {
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Train IoU": train_iou,
                "Val IoU": val_iou
            }

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if checkpoint:
                checkpoint(epoch, val_loss, val_iou)

    except KeyboardInterrupt:
        print('Training stopped by user!')
    finally:
        make_report(train_loss_list,
                    val_loss_list,
                    train_metric_list,
                    val_metric_list,
                    arg_n_values)

        best_metrics_to_log = {
            "Best Train Loss": min(train_loss_list),
            "Best Val Loss": min(val_loss_list),
            "Best Train IoU": max(train_metric_list),
            "Best Val IoU": max(val_metric_list)
        }

        for metric_name, metric_value in best_metrics_to_log.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()
        print('End of training!')
        return model
