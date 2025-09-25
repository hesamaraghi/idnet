import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np

from star8 import StarMovement
from utils.visualize_utils import animate_events
from utils.data_utils import *
from idn.loader.loader_dsec import HarrisRecursive

import argparse
import os
import torch
import hashlib
import pickle
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import wandb


# -------------------------------
# Utilities
# -------------------------------
def knn_indices_from_pos(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k nearest neighbors (including self) from positions."""
    dist = torch.cdist(pos, pos)  # [N, N]
    knn_idx = dist.topk(k, largest=False).indices  # [N, k]
    return knn_idx


def build_knn_features(X: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    """Expand features using neighbors."""
    neighbors = X[knn_idx]  # [N, k, F]
    return neighbors.reshape(X.size(0), -1)  # [N, k*F]


# -------------------------------
# Create toy dataset
# -------------------------------
def create_toy_dataset(args, cache_dir="dataset_cache"):
    """
    Create or load cached toy dataset + knn_idx.
    """

    os.makedirs(cache_dir, exist_ok=True)

    # -------------------------------
    # 1. Build unique dataset hash
    # -------------------------------
    dataset_str = (
        f"{args.toy_dataset}_{args.total_frames}_{args.img_size}_"
        f"{args.num_points}_{args.outer_radius}_{args.inner_radius}_"
        f"{args.num_rotations}_{args.tau}_{args.filter_size}"
    )
    dataset_hash = hashlib.md5(dataset_str.encode()).hexdigest()
    dataset_path = os.path.join(cache_dir, f"{dataset_hash}_data.pt")

    # -------------------------------
    # 2. Load or compute dataset
    # -------------------------------
    if os.path.exists(dataset_path):
        print(f"🔹 Loading cached dataset from {dataset_path}")
        data = torch.load(dataset_path)
    else:
        print("⚡ Generating new dataset...")

        if args.toy_dataset == "star8":
            shape_movement = StarMovement(
                total_frames=args.total_frames,
                image_size=args.img_size,
                face_color="black",
                num_points=args.num_points,
                outer_radius=args.outer_radius,
                inner_radius=args.inner_radius,
                number_of_rotations=args.num_rotations,
            )
        else:
            raise ValueError(f"Unknown toy dataset: {args.toy_dataset}")

        data_array = shape_movement.generate_events()
        data = numpy2pyg_event_convertor(data_array)
        data["v"] = torch.tensor(np.array([data_array["v_x"], data_array["v_y"]])).T

        # Features
        harris_rec = HarrisRecursive(
            tau=args.tau, filter_size=args.filter_size, image_size=args.img_size
        )
        harris_rec(data_array)
        eig1 = harris_rec.eig1
        eig2 = harris_rec.eig2
        filter_values = harris_rec.filter_value_recursive
        data["eig"] = torch.tensor(np.array([eig1, eig2])).T
        data["filter"] = torch.tensor(filter_values).unsqueeze(1)

        torch.save(data, dataset_path)
        print(f"💾 Dataset cached at {dataset_path}")

    # -------------------------------
    # 3. Cache KNN index as well
    # -------------------------------
    knn_path = os.path.join(cache_dir, f"{dataset_hash}_k{args.k}_knn.pt")

    if os.path.exists(knn_path):
        print(f"🔹 Loading cached kNN index from {knn_path}")
        knn_idx = torch.load(knn_path)
    else:
        print(f"⚡ Computing kNN index with k={args.k} ...")
        knn_idx = knn_indices_from_pos(data.pos, args.k)
        torch.save(knn_idx, knn_path)
        print(f"💾 kNN index cached at {knn_path}")

    return data, knn_idx



# -------------------------------
# Model
# -------------------------------
class KNNMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # metrics
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        # update metrics
        self.train_mse.update(preds, y)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = self.train_mse.compute()
        self.log("train_loss", avg_loss, prog_bar=False, sync_dist=True)
        self.train_mse.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.val_mse.update(preds, y)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = self.val_mse.compute()
        self.log("val_loss", avg_loss, prog_bar=False, sync_dist=True)
        self.val_mse.reset()
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )


# -------------------------------
# Training Pipeline
# -------------------------------
def train(args):
    # -------------------
    # Create dataset
    # -------------------

    if args.test_train_split not in ["random", "temporal"]:
        raise ValueError(f"Unknown test_train_split: {args.test_train_split}")

    data, knn_idx = create_toy_dataset(args)
    if args.feature_type == "original":
        features = data.pos[:, 0:2]
    elif args.feature_type == "eig":
        features = torch.cat([data.pos[:, 0:2], data["eig"]], dim=1)
    elif args.feature_type == "filter":
        features = torch.cat([data.pos[:, 0:2], data["filter"]], dim=1)
    elif args.feature_type == "both":
        features = torch.cat([data.pos[:, 0:2], data["eig"], data["filter"]], dim=1)
    else:
        raise ValueError(f"Unknown feature_type: {args.feature_type}")

    print("features shape:", features.shape, flush=True)
    X_with_neighbors = build_knn_features(features, knn_idx)
    Y = data.v
    print("X_with_neighbors shape:", X_with_neighbors.shape, flush=True)
    if args.test_train_split == "random":
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_with_neighbors,
            Y,
            test_size=args.test_size,
            random_state=args.test_split_seed,
        )
    elif args.test_train_split == "temporal":
        split_idx = int(X_with_neighbors.size(0) * (1 - args.test_size))
        X_train, X_val = X_with_neighbors[:split_idx], X_with_neighbors[split_idx:]
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    else:
        raise ValueError(f"Unknown test_train_split: {args.test_train_split}")
    print(f"Train size: {X_train.size(0)}, Test size: {X_val.size(0)}")
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch_size)

    # -------------------
    # wandb logger
    # -------------------
    wandb_logger = pl.loggers.WandbLogger(project=args.project, config=vars(args))
    run_id = wandb_logger.experiment.id  # unique wandb run ID

    # -------------------
    # Model
    # -------------------
    model = KNNMLP(
        input_dim=X_train.size(1),
        hidden_dim=args.hidden_dim,
        output_dim=2,
        lr=args.lr,
    )
    callback_list = []
    callback_list.append(LearningRateMonitor(logging_interval='step'))
    # -------------------
    # Trainer
    # -------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        enable_progress_bar=False,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=callback_list,
    )

    # Train + validate
    trainer.fit(model, train_loader, val_loader)

    # -------------------
    # Save model locally
    # -------------------
    # os.makedirs(args.log_dir, exist_ok=True)
    # ckpt_path = os.path.join(args.log_dir, f"model_{run_id}.ckpt")
    # trainer.save_checkpoint(ckpt_path)
    # print(f"✅ Model saved locally at {ckpt_path}")


# -------------------------------
# Argparse
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="kNN + MLP regression with Lightning + wandb"
    )

    # Feature params
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors")
    parser.add_argument(
        "--feature_type",
        type=str,
        default="both",
        help="Type of node features from: original / eig / filter / both",
    )
    parser.add_argument(
        "--tau", type=float, default=1.0, help="Temoral constant for filter features"
    )
    parser.add_argument(
        "--filter_size", type=int, default=5, help="Filter size for filter features"
    )

    # Data params
    parser.add_argument(
        "--toy_dataset",
        type=str,
        default="star8",
        help="Toy dataset to use from: star8",
    )
    parser.add_argument(
        "--img_size", type=int, nargs=2, default=[256, 256], help="Image size (H, W)"
    )
    parser.add_argument(
        "--total_frames", type=int, default=2_000, help="Number of frames in the sequence"
    )
    parser.add_argument(
        "--num_points", type=int, default=5, help="Number of points in the star"
    )
    parser.add_argument(
        "--outer_radius", type=int, default=40, help="Outer radius of the star"
    )
    parser.add_argument(
        "--inner_radius", type=int, default=20, help="Inner radius of the star"
    )
    parser.add_argument(
        "--num_rotations", type=int, default=2, help="Number of rotations of the star"
    )

    # Dataset params
    parser.add_argument(
        "--test_train_split",
        type=str,
        default="temporal",
        help="Type of train/test split from: random / temporal",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Fraction of data for testing"
    )
    parser.add_argument(
        "--test_split_seed",
        type=int,
        default=42,
        help="Random seed for test/train split",
    )

    # Model hyperparams
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension of MLP"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Training params
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max training epochs"
    )

    # Logging / wandb
    parser.add_argument(
        "--project", type=str, default="knn-mlp-regression", help="wandb project name"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="knn_mlp_regression_log",
        help="Local directory to save checkpoints",
    )

    args = parser.parse_args()
    train(args)
