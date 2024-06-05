import os
from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import List
from strictyaml import load, YAML


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = Path(os.path.join(PACKAGE_ROOT, 'datasets'))
SANTANDER_DATA_DIR = Path(os.path.join(DATASET_DIR, 'santander-data'))
CONFIG_FILE_PATH = Path(os.path.join(PACKAGE_ROOT, 'config.yml'))
TRAINED_MODEL_DIR = Path(os.path.join(PACKAGE_ROOT, 'trained_models'))
ENCODER_DIR = Path(os.path.join(PACKAGE_ROOT, 'encoders'))


class AppConfig(BaseModel):
    """
    application-level config
    """
    training_data: str
    test_data: str
    model_file_name: str
    user_encoder: str
    service_encoder: str


class ModelConfig(BaseModel):
    """
    all configuration relevant to model
    training and feature engineering
    """

    engineered_variable_1: str
    engineered_variable_2: str
    feature_dict: dict[str, int]
    n_epochs: int
    lr_all: float
    reg_all: float


class Config(BaseModel):
    """master config object"""

    apps_config: AppConfig
    models_config: ModelConfig


def find_config_file() -> Path:
    """locate the configuration file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """parse the YAML containing the package configuration"""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """run validation on config values"""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type
    _config = Config(
        apps_config=AppConfig(**parsed_config.data),
        models_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()

# # to test our config works correctly
# if __name__ == "__main__":
#     print(f"config: {config.models_config.engineered_variable_2}")