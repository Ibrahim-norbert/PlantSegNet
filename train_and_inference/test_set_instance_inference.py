import argparse
import os
import h5py
import numpy as np
import random
import json
import torch
import open3d as o3d
import sys

sys.path.append("..")
sys.path.append("D:\\DevPython\\PlantSegNet\\")
sys.path.append("D:\\DevPython\\PlantSegNet\\data\\")

from data.load_raw_data import load_real_ply_with_labels_smlm
from models.nn_models import SorghumPartNetInstance
from models.utils import LeafMetrics, ClusterBasedMetrics
from data.utils import create_csv_smlm

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        description="Test set inference script. This script runs the instance segmentation model on the test sets of the given dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        help="The path to the test h5 file or directory containing the test point clouds. ",
        metavar="input",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="The path to the model checkpoint. ",
        metavar="model",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        help="The name of the dataset. It should be either SPNR (SPN Real), SPNS (SPN Synthetic), PN, TPN. ",
        metavar="dataset",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to the directory in which the output files will be saved. ",
        metavar="output",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-p",
        "--param",
        help="The path to the best params path (for DBSCAN parameters). ",
        metavar="param",
        required=True,
        type=str,
    )

    return parser.parse_args()


def get_best_param(path):
    with open(path, "r") as f:
        params_dict = json.load(f)
    return params_dict


def load_model(model, path):
    model = eval(model).load_from_checkpoint(path)
    model.eval()
    return model


def load_data_h5(path, point_key, label_key):
    with h5py.File(path) as f:
        data = np.array(f[point_key])
        label = np.array(f[label_key])
    return data, label


def load_data_directory(path):
    data = []
    labels = []
    min_shape = sys.maxsize

    for p in os.listdir(path):
        file_path = os.path.join(path, p)
        points, instance_labels, semantic_labels = load_real_ply_with_labels_smlm(file_path)
        instance_points = points[semantic_labels == 1]
        instance_labels = instance_labels[semantic_labels == 1]
        data.append(instance_points)
        labels.append(instance_labels)
        if instance_labels.shape[0] < min_shape:
            min_shape = instance_labels.shape[0]

    resized_data = []
    resized_labels = []
    for i, datum in enumerate(data):
        label = labels[i]
        downsample_indexes = random.sample(
            np.arange(0, datum.shape[0]).tolist(),
            min_shape,
        )
        datum = datum[downsample_indexes]
        label = label[downsample_indexes]
        resized_data.append(datum)
        resized_labels.append(label)

    resized_data = np.stack(resized_data)
    resized_labels = np.stack(resized_labels)

    return resized_data, resized_labels


def get_final_clusters(preds, DBSCAN_eps=1, DBSCAN_min_samples=10):
    try:
        preds = preds.cpu().detach().numpy().squeeze()
        clustering = DBSCAN(eps=DBSCAN_eps, min_samples=int(DBSCAN_min_samples)).fit(preds)
        final_clusters = clustering.labels_
        return final_clusters
    except Exception as e:
        print(e)
        return None


def run_inference(output_dir, model, data, label, best_params):
    data = torch.from_numpy(data).type(torch.FloatTensor)
    label = torch.Tensor(label).float().squeeze().cpu()

    print(f"We have the following the following input and output shapes: {data.size()} - {label.size()}")
    model = model.cpu()
    model.DGCNN_feature_space.device = "cpu"
    pointwise_metric_calculator = LeafMetrics()
    clusterbased_metric_calculator = ClusterBasedMetrics([0.25, 0.5, 0.75], "cpu")

    best_acc_value = 0
    best_acc_preds = None
    best_acc_index = -1
    worst_acc_value = 100
    worst_acc_preds = None
    worst_acc_index = -1

    pointwise_accuracies = []
    pointwise_precisions = []
    pointwise_recalls = []
    pointwise_f1s = []
    clusterbased_mean_coverages = []
    clusterbased_average_precisions = []
    clusterbased_average_recalls = []

    final_predictions = []
    preds_before_clustering = []

    for i in range(data.shape[0]):
        # if i == 10:
        #     break
        preds = model(data[i : i + 1])

        pca = PCA(n_components=3)
        preds_np = preds.detach().numpy()
        preds_np_2D = preds_np.squeeze()
        pca_res = pca.fit_transform(preds_np_2D)
        labels_np = label[i].detach().numpy().astype(int)
        colors = np.random.rand(np.max(labels_np), 3)
        _, idx, inverse_idx = np.unique(labels_np, return_index=True, return_inverse=True)
        '''result = colors[idx.argsort()[inverse_idx]]
        if pca_res.shape[1] == 2:
            plt.scatter(pca_res[:, 0], pca_res[:, 1], c= result, edgecolors='none', s=5, alpha=None, linewidths=None)
        elif pca_res.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pca_res[:, 0], pca_res[:, 1], pca_res[:, 2], c= result, edgecolors='none', s=5, alpha=None, linewidths=None)
        plt.title('PCA')
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.show()'''

        pred_clusters = get_final_clusters(
            preds, best_params["eps"], best_params["minpoints"]
        )
        if pred_clusters is None:
            continue

        pca_res = pca_res.transpose()
        pca_res = np.vstack((pca_res, labels_np))
        pca_res = np.vstack((pca_res, pred_clusters))
        pca_res = pca_res.transpose()
        np.savetxt(os.path.join(output_dir, str(i + 1) + "_pca.csv"), pca_res, delimiter=',', fmt='%s,%s,%s,%s,%s', header='x,y,z,id,prediction', comments='')

        preds_before_clustering.append(preds.detach().numpy())
        final_predictions.append(pred_clusters)
        pred_clusters = torch.tensor(pred_clusters)

        acc = pointwise_metric_calculator(
            pred_clusters.unsqueeze(0).unsqueeze(-1),
            label[i].unsqueeze(0).unsqueeze(-1),
        )

        pointwise_accuracies.append(acc)
        # pointwise_precisions.append(precison)
        # pointwise_recalls.append(recall)
        # pointwise_f1s.append(f1)

        if acc > best_acc_value:
            best_acc_value = acc
            best_acc_index = i
            best_acc_preds = pred_clusters

        if acc < worst_acc_value:
            worst_acc_value = acc
            worst_acc_index = i
            worst_acc_preds = pred_clusters

        clusterbased_metrics = clusterbased_metric_calculator(
            pred_clusters.squeeze(), label[i].squeeze()
        )

        clusterbased_mean_coverages.append(clusterbased_metrics["mean_coverage"])
        clusterbased_average_precisions.append(
            clusterbased_metrics["average_precision"]
        )
        clusterbased_average_recalls.append(clusterbased_metrics["average_recall"])

        print(
            f":: Instance {i}/{data.shape[0]}= Accuracy: {acc} - mCov: {clusterbased_metrics['mean_coverage']} - Average Precision: {clusterbased_metrics['average_precision']} - Average Recall: {clusterbased_metrics['average_recall']}"
        )

        sys.stdout.flush()

    full_results_dic = {
        "pointwise_accuracies": pointwise_accuracies,
        "clusterbased_mean_coverages": clusterbased_mean_coverages,
        "clusterbased_average_precisions": clusterbased_average_precisions,
        "clusterbased_average_recalls": clusterbased_average_recalls,
    }

    mean_results_dic = {
        "pointwise_accuracy": np.mean(pointwise_accuracies),
        "pointwise_precision": np.mean(pointwise_precisions),
        "pointwise_recall": np.mean(pointwise_recalls),
        "pointwise_f1": np.mean(pointwise_f1s),
        "clusterbased_mean_coverage": np.mean(clusterbased_mean_coverages),
        "clusterbased_average_precision": np.mean(clusterbased_average_precisions),
        "clusterbased_average_recall": np.mean(clusterbased_average_recalls),
    }

    best_example_ply = create_csv_smlm(
        data[best_acc_index].squeeze().numpy(),
        best_acc_preds.squeeze().numpy(),
    )
    worst_example_ply = create_csv_smlm(
        data[worst_acc_index].squeeze().numpy(),
        worst_acc_preds.squeeze().numpy(),
    )

    final_predictions = np.stack(final_predictions, 0)

    return (
        full_results_dic,
        mean_results_dic,
        best_example_ply,
        worst_example_ply,
        final_predictions,
    )


def save_results(
    path,
    full_results_dic,
    mean_results_dic,
    best_example_ply,
    worst_example_ply,
    final_predictions,
):
    #o3d.io.write_point_cloud(os.path.join(path, "best_example.ply"), best_example_ply)
    #o3d.io.write_point_cloud(os.path.join(path, "worst_example.ply"), worst_example_ply)
    np.savetxt(os.path.join(path, "best_example.csv"), best_example_ply, delimiter=',', fmt='%s,%s,%s', header='x,y,cluster_id', comments='')
    np.savetxt(os.path.join(path, "worst_example.csv"), worst_example_ply, delimiter=',', fmt='%s,%s,%s', header='x,y,cluster_id', comments='')
    with open(os.path.join(path, "full_results.json"), "w") as f:
        json.dump(full_results_dic, f)
    with open(os.path.join(path, "mean_results.json"), "w") as f:
        json.dump(mean_results_dic, f)
    with h5py.File(os.path.join(path, "predictions.hdf5"), "w") as f:
        f.create_dataset("preds", data=final_predictions)


def main():
    #args = get_args()
    
    params_dict = {
        "model": "D:/DevPython/PlantSegNet/checkpoint/SorghumPartNetInstance/PlantSegNet/checkpoints/epoch=23-step=2400.ckpt",
        "dataset": "SMLM",
        "input": "D:/DevPython/PlantSegNet/datasets/npcs/test/",
        "output": "D:/DevPython/PlantSegNet/checkpoint/SorghumPartNetInstance/PlantSegNet/testing/",
        "param": "D:/DevPython/PlantSegNet/checkpoint/SorghumPartNetInstance/PlantSegNet/hparam_tuning_logs/DBSCAN_best_param.json",
    }

    best_params = get_best_param(params_dict["param"])

    output_dir = os.path.join(params_dict["output"], params_dict["dataset"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = load_model("SorghumPartNetInstance", params_dict["model"])

    if params_dict["dataset"] == "SPNS":
        data, label = load_data_h5(params_dict["input"], "points", "labels")
    elif params_dict["dataset"] == "SPNR":
        data, label = load_data_directory(params_dict["input"])
    elif params_dict["dataset"] == "SMLM":
        data, label = load_data_directory(params_dict["input"])
    elif params_dict["dataset"] == "PN":
        data, label = load_data_h5(params_dict["input"], "pts", "label")
    elif params_dict["dataset"] == "TPN":
        data, label = load_data_h5(params_dict["input"], "points", "primitive_id")
    else:
        print(":: Incorrect dataset name. ")
        return

    print(
        f":: Starting the inference with the following parameters --> eps: {best_params['eps']} - minpoints: {best_params['minpoints']}"
    )
    sys.stdout.flush()

    (
        full_results_dic,
        mean_results_dic,
        best_example_ply,
        worst_example_ply,
        final_predictions,
    ) = run_inference(output_dir, model, data, label, best_params)
    save_results(
        output_dir,
        full_results_dic,
        mean_results_dic,
        best_example_ply,
        worst_example_ply,
        final_predictions,
    )


if __name__ == "__main__":
    main()

"""
Running argument samples for all datasets:

SPNS: python test_set_instance_inference.py -i /space/ariyanzarei/sorghum_segmentation/dataset/SPNS/SorghumPartNetFormat/instance_segmentation_test.hdf5 -m /space/ariyanzarei/sorghum_segmentation/results/training_logs/SorghumPartNetInstance/SPNS/EXP_02/checkpoints/epoch\=8-step\=43199.ckpt -d SPNS -o /space/ariyanzarei/sorghum_segmentation/results/testing/ -p /space/ariyanzarei/sorghum_segmentation/archive/results/hyperparameter_tuning/SPNS/DBSCAN_best_param.json
"""
