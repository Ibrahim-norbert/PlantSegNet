import sys
import yaml


def get_config_yaml(path):
    with open(path, "r") as f:
        hparams = yaml.safe_load(f)
    return hparams