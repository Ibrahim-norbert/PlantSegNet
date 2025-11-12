import argparse
import os
import h5py
import numpy as np
import random
import json
import torch
import open3d as o3d
import sys
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ProjectRoot import change_wd_to_project_root
change_wd_to_project_root()

from data.load_raw_data import load_real_ply_with_labels_smlm
from models.nn_models import SorghumPartNetInstance
from models.utils import LeafMetrics, ClusterBasedMetrics
from data.utils import create_csv_smlm
from train_and_inference.test_set_instance_inference import save_results, run_inference 



class InferenceEngine:
    def __init__(self, params_dict):
        self.params_dict = params_dict
        self.set_parameters()
        
        
        
    def set_parameters(self):
        self.best_params = self.get_best_param(self.params_dict["param"])
        self.output_dir = self._setup_output_dir()
        self.model = self.load_model(self.params_dict["model"])
        self.pointwise_metric_calculator = LeafMetrics()
        self.clusterbased_metric_calculator = ClusterBasedMetrics([0.25, 0.5, 0.75], "cpu")

    def _setup_output_dir(self):
        output_dir = os.path.join(self.params_dict["output"], self.params_dict["dataset"])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def get_best_param(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def load_model(self, path):
        model = eval(self.params_dict.get("model", "SorghumPartNetInstance")).load_from_checkpoint(path)
        model.eval()
        return model

    def load_data_h5(self, path, point_key, label_key):
        with h5py.File(path) as f:
            data = np.array(f[point_key])
            label = np.array(f[label_key])
        return data, label

    def load_data_directory(self, path):
        data, labels, min_shape = [], [], sys.maxsize
        for p in os.listdir(path):
            file_path = os.path.join(path, p)
            points, instance_labels, semantic_labels = load_real_ply_with_labels_smlm(file_path)
            instance_points = points[semantic_labels == 1]
            instance_labels = instance_labels[semantic_labels == 1]
            data.append(instance_points)
            labels.append(instance_labels)
            min_shape = min(min_shape, instance_labels.shape[0])

        resized_data, resized_labels = [], []
        for datum, label in zip(data, labels):
            downsample_indexes = random.sample(np.arange(0, datum.shape[0]).tolist(), min_shape)
            resized_data.append(datum[downsample_indexes])
            resized_labels.append(label[downsample_indexes])

        return np.stack(resized_data), np.stack(resized_labels)

    def load_data(self):
        dataset = self.params_dict["dataset"]
        path = self.params_dict["input"]
        
        if dataset == "SPNS":
            return self.load_data_h5(path, "points", "labels")
        elif dataset in ["SPNR", "SMLM"]:
            return self.load_data_directory(path)
        elif dataset == "PN":
            return self.load_data_h5(path, "pts", "label")
        elif dataset == "TPN":
            return self.load_data_h5(path, "points", "primitive_id")
        else:
            raise ValueError(f"Incorrect dataset name: {dataset}")

    def run(self):
        data, label = self.load_data()
        print(
        f":: Starting the inference with the following parameters --> eps: {self.best_params['eps']} - minpoints: {self.best_params['minpoints']}")
        sys.stdout.flush()
        save_results(
        self.output_dir,*run_inference(self.output_dir, self.model, data, label, self.best_params))
    




def main():
    #args = get_args()
    cwd = r"C:\Users\ibrah\OneDrive\Desktop\Career\SuperRes PhD\.repo\SuperRes-Imperial-CNRS\DummyModels\PlantSegNet"
    params_dict = {
        "model": os.path.join(cwd,"checkpoint/SorghumPartNetInstance/PlantSegNet/checkpoints/epoch=23-step=2400.ckpt"),
        "dataset": "SMLM",
        "input": "datasets/npcs/test/",
        "output": os.path.join(cwd,"checkpoint/SorghumPartNetInstance/PlantSegNet/testing/"),
        "param": os.path.join(cwd,"checkpoint/SorghumPartNetInstance/PlantSegNet/hparam_tuning_logs/DBSCAN_best_param.json"),
    }

    """
    Running argument samples for all datasets:

    """

    InferenceEngine(params_dict=params_dict).run()





if __name__ == "__main__":
    main()


