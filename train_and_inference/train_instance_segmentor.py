import pytorch_lightning as pl
import argparse
import json
import os
import shutil
import sys

sys.path.append("..")
sys.path.append("D:\\DevPython\\PlantSegNet\\")
sys.path.append("D:\\DevPython\\PlantSegNet\\data\\")

#from models.nn_models import *
from models.nn_models import SorghumPartNetInstance

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_args():
    parser = argparse.ArgumentParser(
        description="SorghumPartNet Instance segmentation training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--hparam",
        help="The path to the hyperparameters json file. ",
        metavar="hparam",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to the directory in which the model checkpoints will be saved. ",
        metavar="output",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Whether to run it in the debuge mode or not",
        action="store_true",
    )

    parser.add_argument(
        "-f",
        "--force",
        help="Whether to forcefully overwrite the experiment if exist or not. ",
        action="store_true",
    )

    return parser.parse_args()


def get_hparam(path):
    with open(path, "r") as f:
        hparams = json.load(f)
    return hparams


def train():
    #args = get_args()
    #hparams = get_hparam(args.hparam)

    hparams = {
        "epochs": 200,
        "dgcnn_k": 100,
        "input_dim": 2,
        "batch_size": 1,
        "train_data": "D:\\DevPython\\PlantSegNet\\datasets\\npcs\\train\\",
        "val_data": "D:\\DevPython\\PlantSegNet\\datasets\\npcs\\val\\",
        "test_data": "D:\\DevPython\\PlantSegNet\\datasets\\npcs\\test\\",
        "patience": 20,
        'batch_size': 1,
        'lr': 0.001,
        'weight_decay': 0.0,
        'lr_decay': 0.5,
        'decay_step': 3e5,
        'bn_momentum': 0.5,
        'bnm_decay': 0.5,
        'output': 'checkpoint/',
        'force': True,
        'debug': False,
        'version': "D:\\DevPython\\PlantSegNet",
        "leaf_space_threshold": 0.5,
        "debug_feature_space": True,
    }
    version_name = os.path.basename(os.path.normpath(hparams["version"])).replace(".json", "")

    if "dataset" in hparams and (
        hparams["dataset"] == "TPN" or hparams["dataset"] == "PN"
    ):
        chkpt_path = os.path.join(
            hparams["output"], hparams["dataset"], "SorghumPartNetInstance"
        )
    else:
        chkpt_path = os.path.join(hparams["output"], "SorghumPartNetInstance")

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    if os.path.exists(os.path.join(chkpt_path, version_name)):
        if hparams["force"]:
            shutil.rmtree(os.path.join(chkpt_path, version_name))
        else:
            print(
                ":: There is a folder for this version of experiments. Please use -f to overwrite this experiment."
            )
            return

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_leaf_loss",
        mode="min",
    )

    tensorboard_callback = TensorBoardLogger(
        save_dir=chkpt_path, name="", default_hp_metric=False, version=version_name
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_leaf_loss", mode="min", patience=hparams["patience"]
    )

    if hparams["debug"]:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices="auto",
            max_epochs=hparams["epochs"],
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=tensorboard_callback,
        )
        segmentor = SorghumPartNetInstance(hparams, True).cuda()
    else:
        trainer = pl.Trainer(
            default_root_dir=chkpt_path,
            accelerator="gpu",
            devices="auto",
            max_epochs=hparams["epochs"],
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=tensorboard_callback,
        )
        segmentor = SorghumPartNetInstance(hparams, False).cuda()

    trainer.fit(segmentor, segmentor.train_dataloader(), segmentor.val_dataloader())


if __name__ == "__main__":
    train()

#python train_and_inference/train_instance_segmentor.py --hparam parameters.json --output checkpoint/