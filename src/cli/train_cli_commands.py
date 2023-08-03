import click
from src.models.train_model import train_model
from typing import Optional, Tuple, Dict, Any
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_from_config(config: Dict[str, Any], param_name: str) -> Any:
    return config.get(param_name, 0)


config_path = '../../config.yaml'
config_params = load_config(config_path)


@click.command()
@click.option('--model_path', default=get_default_from_config(config_params, 'model_path'), type=click.Path())
@click.option('--image_dir', default=get_default_from_config(config_params, 'image_dir'), type=click.Path(exists=True))
@click.option('--mask_dir', default=get_default_from_config(config_params, 'mask_dir'), type=click.Path(exists=True))
@click.option('--logs_dir', default=get_default_from_config(config_params, 'logs_dir'), type=click.Path())
@click.option('--batch_size', default=get_default_from_config(config_params, 'batch_size'), help='Batch size', type=click.INT)
@click.option('--num_classes', default=get_default_from_config(config_params, 'num_classes'), help='Number of classes including background class', type=click.INT)
@click.option('--num_epochs', default=get_default_from_config(config_params, 'num_epochs'), help='Number of epochs for training', type=click.INT)
@click.option('--mixing_proportion', default=get_default_from_config(config_params, 'mixing_proportion'), type=click.FLOAT)
def get_cli_params_for_training(model_path,
                                image_dir,
                                mask_dir,
                                logs_dir,
                                batch_size,
                                num_classes,
                                num_epochs,
                                mixing_proportion):
    params = locals()
    train_model(**params)


if __name__ == '__main__':
    get_cli_params_for_training()
