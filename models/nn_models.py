from models.dgcnn import DGCNNFeatureSpace
import torch
import numpy as np
import os
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.datasets import (
    SorghumDataset,
)
from collections import namedtuple
from models.utils import (
    BNMomentumScheduler,
    SpaceSimilarityLossV2,
    SpaceSimilarityLossV3,
    SpaceSimilarityLossV4,
    SpaceSimilarityLossV5,
    LeafMetrics,
    LeafMetricsTraining,
)
from data.load_raw_data import load_real_ply_with_labels, load_real_ply_with_labels_smlm
import matplotlib.pyplot as plt
import torchvision
from sklearn.cluster import DBSCAN
from data.utils import distinct_colors
from models.modules import KNNSpaceRegularizer
from sklearn.decomposition import PCA

class SorghumPartNetInstance(pl.LightningModule):
    def __init__(self, hparams, debug=False):
        """
        Parameters
        ----------
        hparams: hyper parameters
        """
        super(SorghumPartNetInstance, self).__init__()

        self.is_debug = debug
        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        MyStruct = namedtuple("args", "k")
        if "dgcnn_k" in self.hparams:
            args = MyStruct(k=self.hparams["dgcnn_k"])
        else:
            args = MyStruct(k=15)

        self.DGCNN_feature_space = DGCNNFeatureSpace(
            args, (3 if "input_dim" not in self.hparams else self.hparams["input_dim"])
        ).float()

        if "loss_fn" in self.hparams and self.hparams["loss_fn"] == "knn_space_mean":
            self.space_reqularizer_module = KNNSpaceRegularizer(
                self.hparams["loss_fn_param"]
            )
        else:
            self.space_reqularizer_module = None

        self.save_hyperparameters()

    def forward(self, xyz):

        # Normalization
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "min-max"
        ):
            mins, _ = torch.min(xyz, axis=1)
            maxs, _ = torch.max(xyz, axis=1)
            mins = mins.unsqueeze(1)
            maxs = maxs.unsqueeze(1)
            self.radius = 100. / (maxs[0][0][0] - mins[0][0][0])
            xyz = (xyz - mins) / (maxs - mins) - 0.5
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "mean-std"
        ):
            mean = torch.mean(xyz, axis=1)
            mean = mean.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            std = torch.std(xyz, axis=1)
            std = std.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            xyz = (xyz - mean) / std

        # Instance
        dgcnn_features = self.DGCNN_feature_space(xyz, self.radius * self.radius)

        # # Take mean of the k nearest neighbors
        if self.space_reqularizer_module is not None:
            dgcnn_features = self.space_reqularizer_module(xyz, dgcnn_features)

        return dgcnn_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.lr_clip / self.hparams["lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["bn_momentum"]
            * self.hparams["bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, shuff=True):
        dataset = SorghumDataset(ds_path)


        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=4, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["train_data"], shuff=True)

    def training_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, _, _, leaf = batch
        else:
            points, leaf = batch

        pred_leaf_features = self(points)

        if "loss_fn" not in self.hparams or self.hparams["loss_fn"] == "v2":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v3":
            criterion_cluster = SpaceSimilarityLossV3(points)
        elif self.hparams["loss_fn"] == "v4":
            criterion_cluster = SpaceSimilarityLossV4(points)
        elif self.hparams["loss_fn"] == "knn_space_mean":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v5":
            criterion_cluster = SpaceSimilarityLossV5(points)

        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetricsTraining(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "train_leaf_loss": leaf_loss,
            "train_leaf_accuracy": Acc,
            "train_leaf_precision": Prec,
            "train_leaf_recall": Rec,
            "train_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": leaf_loss, "log": tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["val_data"], shuff=False)

    def validation_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, _, _, leaf = batch
        else:
            points, leaf = batch

        pred_leaf_features = self(points)

        if "loss_fn" not in self.hparams or self.hparams["loss_fn"] == "v2":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v3":
            criterion_cluster = SpaceSimilarityLossV3(points)
        elif self.hparams["loss_fn"] == "v4":
            criterion_cluster = SpaceSimilarityLossV4(points)
        elif self.hparams["loss_fn"] == "knn_space_mean":
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v5":
            criterion_cluster = SpaceSimilarityLossV5(points)

        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetricsTraining(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "val_leaf_loss": leaf_loss,
            "val_leaf_accuracy": Acc,
            "val_leaf_precision": Prec,
            "val_leaf_recall": Rec,
            "val_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return tensorboard_logs
    
    def on_train_epoch_end(self):
        if "debug_feature_space" in self.hparams and self.hparams['debug_feature_space'] is True:
            device_name = "cpu"
            device = torch.device(device_name)

            instance_model = self.to(device)
            instance_model.DGCNN_feature_space.device = device_name
            
            file_path = "D:\\DevPython\\PlantSegNet\\datasets\\npcs\\train\\0_smlm_dataset.ply"
            points, instance_labels, semantic_labels = load_real_ply_with_labels_smlm(file_path)
            points = points[semantic_labels == 1]
            instance_labels = instance_labels[semantic_labels == 1]
            
            points = torch.tensor(points, dtype=torch.float64).to(device)
            if (
                "use_normals" in instance_model.hparams
                and instance_model.hparams["use_normals"]
            ):
                pred_instance_features = instance_model(
                    torch.unsqueeze(points, dim=0).to(device)
                )
            else:
                pred_instance_features = instance_model(
                    torch.unsqueeze(points[:, :3], dim=0).to(device).float()
                )
            
            pca = PCA(n_components=3)
            preds_np = pred_instance_features.detach().numpy()
            preds_np_2D = preds_np.squeeze()
            pca_res = pca.fit_transform(preds_np_2D)
            labels_np = instance_labels.astype(int)

            pca_res = pca_res.transpose()
            pca_res = np.vstack((pca_res, labels_np))
            pca_res = pca_res.transpose()
            filename = "0_epoch_" + str(self.current_epoch).zfill(3) + "_pca.csv"
            np.savetxt(os.path.join("D:/DevPython/PlantSegNet/checkpoint/SorghumPartNetInstance/PlantSegNet/FeatureSpace/", filename), pca_res, delimiter=',', fmt='%s,%s,%s,%s', header='x,y,z,id', comments='')
            
            instance_model = self.to(torch.device("cuda"))
            instance_model.DGCNN_feature_space.device = "cuda"

    def validation_epoch_end(self, batch):
        if "real_data" in self.hparams:
            self.validation_real_data()

    def validation_real_data(self):
        real_data_path = self.hparams["real_data"]

        device_name = "cpu"
        device = torch.device(device_name)

        instance_model = self.to(device)
        instance_model.DGCNN_feature_space.device = device_name

        files = os.listdir(real_data_path)
        accs = []
        precisions = []
        recals = []
        f1s = []
        pred_images = []

        for file in files:
            path = os.path.join(real_data_path, file)
            main_points, instance_labels, semantic_labels = load_real_ply_with_labels(
                path
            )
            points = main_points[semantic_labels == 1]
            instance_labels = instance_labels[semantic_labels == 1]

            points = torch.tensor(points, dtype=torch.float64).to(device)
            if (
                "use_normals" in instance_model.hparams
                and instance_model.hparams["use_normals"]
            ):
                pred_instance_features = instance_model(
                    torch.unsqueeze(points, dim=0).to(device)
                )
            else:
                pred_instance_features = instance_model(
                    torch.unsqueeze(points[:, :3], dim=0).to(device)
                )

            pred_instance_features = (
                pred_instance_features.cpu().detach().numpy().squeeze()
            )
            clustering = DBSCAN(eps=1, min_samples=10).fit(pred_instance_features)
            pred_final_cluster = clustering.labels_

            d_colors = distinct_colors(len(list(set(pred_final_cluster))))
            colors = np.zeros((pred_final_cluster.shape[0], 3))
            for i, l in enumerate(list(set(pred_final_cluster))):
                colors[pred_final_cluster == l, :] = d_colors[i]

            non_focal_points = main_points[semantic_labels == 2]
            ground_points = main_points[semantic_labels == 0]

            non_focal_color = [0, 0, 0.7, 0.3]
            ground_color = [0.3, 0.1, 0, 0.3]

            metric_calculator = LeafMetrics()
            acc, precison, recal, f1 = metric_calculator(
                torch.tensor(pred_final_cluster).unsqueeze(0).unsqueeze(-1),
                torch.tensor(instance_labels).unsqueeze(0).unsqueeze(-1),
            )

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, c=colors)
            ax.scatter(
                non_focal_points[:, 0],
                non_focal_points[:, 1],
                non_focal_points[:, 2],
                s=1,
                color=non_focal_color,
            )
            ax.scatter(
                ground_points[:, 0],
                ground_points[:, 1],
                ground_points[:, 2],
                s=1,
                color=ground_color,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(
                f"acc: {acc*100:.2f} - precision: {precison:.2f} - recall: {recal:.2f} - f1: {f1:.2f}"
            )
            fig.canvas.draw()
            X = (
                torch.tensor(np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3])
                .transpose(0, 2)
                .transpose(1, 2)
            )
            X = torchvision.transforms.functional.resize(X, (1000, 1000))
            plt.close(fig)
            accs.append(acc)
            precisions.append(precison)
            recals.append(recal)
            f1s.append(f1)
            pred_images.append(X)

        accs = torch.tensor(accs)
        precisions = torch.tensor(precisions)
        recals = torch.tensor(recals)
        f1s = torch.tensor(f1s)

        tensorboard_logs = {
            "test_real_acc": torch.mean(accs),
            "test_real_precision": torch.mean(precisions),
            "test_real_recal": torch.mean(recals),
            "test_real_f1": torch.mean(f1s),
        }

        grid = torch.cat(pred_images, 1)
        self.logger.experiment.add_image(
            "pred_real_data", grid, self.trainer.current_epoch
        )

        for key in tensorboard_logs:
            self.logger.experiment.add_scalar(
                key, tensorboard_logs[key], self.trainer.current_epoch
            )

        instance_model = self.to(torch.device("cuda"))
        instance_model.DGCNN_feature_space.device = "cuda"
