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
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
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


def build_relative_knn_features(
    features: torch.Tensor, knn_idx: torch.Tensor, relative_feat_indices: list
) -> torch.Tensor:
    """
    Expand features using neighbors, but make coordinates relative to the center node.

    Args:
        features: [N, F] feature matrix. First two columns must be coordinates (x, y).
        knn_idx: [N, k] indices of k nearest neighbors (including self).
        relative_feat_indices: List of feature indices to treat as coordinates for relative adjustment.

    Returns:
        Tensor [N, k*F] with relative coordinates for neighbors.
    """
    N, F = features.size()
    k = knn_idx.size(1)

    # gather neighbor features [N, k, F]
    neighbors = features[knn_idx]

    # central node coordinates [N, 1, len(relative_feat_indices)]
    center_coords = features[:, relative_feat_indices].unsqueeze(1)

    # subtract to make coordinates relative
    neighbors[:, :, relative_feat_indices] = (
        neighbors[:, :, relative_feat_indices] - center_coords
    )

    # flatten back [N, k*F]
    return neighbors.reshape(N, k * F)


# -------------------------------
# Create toy dataset
# -------------------------------
def create_toy_dataset(args, cache_dir="dataset_cache"):
    """
    Create or load cached toy dataset + knn_idx.
    """

    if args.test_train_split not in ["random", "temporal"]:
        raise ValueError(f"Unknown test_train_split: {args.test_train_split}")

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

    if args.feature_type == "original":
        features = data.pos[:, 0:2]
    elif args.feature_type == "original_random_augmented":
        random_feats = torch.randn(data.pos.size(0), 3)
        features = torch.cat([data.pos[:, 0:2], random_feats], dim=1)
    elif args.feature_type == "original_repeated_augmented":
        features = torch.cat(
            [data.pos[:, 0:2], data.pos[:, 0:2], data.pos[:, 0:1]], dim=1
        )
    elif args.feature_type == "eig":
        features = torch.cat([data.pos[:, 0:2], data["eig"]], dim=1)
    elif args.feature_type == "filter":
        features = torch.cat([data.pos[:, 0:2], data["filter"]], dim=1)
    elif args.feature_type == "both":
        features = torch.cat([data.pos[:, 0:2], data["eig"], data["filter"]], dim=1)
    elif args.feature_type == "eig_exclude_xy":
        features = data["eig"]
    elif args.feature_type == "filter_exclude_xy":
        features = data["filter"]
    elif args.feature_type == "both_exclude_xy":
        features = torch.cat([data["eig"], data["filter"]], dim=1)
    else:
        raise ValueError(f"Unknown feature_type: {args.feature_type}")

    print("features shape:", features.shape, flush=True)
    if args.relative_coordinates:
        if args.feature_type == "original":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "original_random_augmented":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "original_repeated_augmented":
            relative_feat_indices = [0, 1, 2, 3, 4]
        elif args.feature_type == "eig":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "filter":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "both":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "eig_exclude_xy":
            relative_feat_indices = []
        elif args.feature_type == "filter_exclude_xy":
            relative_feat_indices = []
        elif args.feature_type == "both_exclude_xy":
            relative_feat_indices = []
        else:
            raise ValueError(f"Unknown feature_type: {args.feature_type}")
        if len(relative_feat_indices) == 0:
            print(
                "Warning: No coordinate features found for relative adjustment. Using absolute coordinates instead."
            )
            X_with_neighbors = build_knn_features(features, knn_idx)
        else:
            print("Using relative coordinates for kNN features")
            X_with_neighbors = build_relative_knn_features(
                features, knn_idx, relative_feat_indices
            )
    else:
        print("Using absolute coordinates for kNN features")
        X_with_neighbors = build_knn_features(features, knn_idx)
    Y = data.v
    print("X_with_neighbors shape:", X_with_neighbors.shape, flush=True)
    if args.test_train_split == "random":
        indices = np.arange(X_with_neighbors.shape[0])       
        X_train, X_val, Y_train, Y_val, idx_train, idx_val = train_test_split(
            X_with_neighbors,
            Y,
            indices,
            test_size=args.test_size,
            random_state=args.test_split_seed,
        )
    elif args.test_train_split == "temporal":
        split_idx = int(X_with_neighbors.size(0) * (1 - args.test_size))
        idx_train = np.arange(0, split_idx)
        idx_val = np.arange(split_idx, X_with_neighbors.size(0))
        X_train, X_val = X_with_neighbors[:split_idx], X_with_neighbors[split_idx:]
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    else:
        raise ValueError(f"Unknown test_train_split: {args.test_train_split}")
    print(f"Train size: {X_train.size(0)}, Test size: {X_val.size(0)}")

    data_array = pyg2numpy_event_convertor(data)
    data_array_train = data_array[idx_train]
    data_array_val = data_array[idx_val]
    return X_train, Y_train, X_val, Y_val, data_array_train, data_array_val


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

    def forward(self, batch):
        if isinstance(batch, (list, tuple)):
            x, _ = batch  # ignore y
        else:
            x = batch
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

    X_train, Y_train, X_val, Y_val, _ , _ = create_toy_dataset(args)
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch_size)

    # -------------------
    # wandb logger
    # -------------------
    if not os.path.exists(os.path.join(args.log_dir, args.project)):
        os.makedirs(os.path.join(args.log_dir, args.project), exist_ok=True)
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=os.path.join(args.log_dir, args.project),
        project=args.project,
        config=vars(args),
        log_model=False,
        offline=not args.online,
    )
    run_id = wandb_logger.experiment.id  # unique wandb run ID

    print(f"🚀 Starting run {run_id} with config: {args}", flush=True)

    # -------------------
    # Seed everything. Note that this does not make training entirely
    # deterministic.
    # -------------------
    pl.seed_everything(args.test_split_seed, workers=True)

    # -------------------
    # Model
    # -------------------
    model = KNNMLP(
        input_dim=train_loader.dataset.tensors[0].shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=2,
        lr=args.lr,
    )
    callback_list = []
    callback_list.append(LearningRateMonitor(logging_interval="step"))
    callback_list.append(TQDMProgressBar(refresh_rate=500))
    # -------------------
    # Trainer
    # -------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=callback_list,
    )

    # Train + validate
    trainer.fit(model, train_loader, val_loader)

    # -------------------
    # Finish wandb run (important for offline sync)
    # -------------------
    wandb.finish()

    # -------------------
    # Save model locally
    # -------------------
    # os.makedirs(args.log_dir, exist_ok=True)
    # ckpt_path = os.path.join(args.log_dir, f"model_{run_id}.ckpt")
    # trainer.save_checkpoint(ckpt_path)
    # print(f"✅ Model saved locally at {ckpt_path}")


# -------------------------------
# Evaluation Pipeline
# -------------------------------
def evaluate_run(run_id: str, args):
    """
    Evaluate a trained wandb run on the test set and save results.

    Args:
        run_id: wandb run ID to evaluate
        args: argparse.Namespace with args used for training
        save_dir: Directory to save evaluation results
    """

    # -------------------
    # Load wandb run config
    # -------------------
    api = wandb.Api()
    run = api.run(f"{args.entity}/{args.project}/{run_id}")  # entity = user or team
    config = run.config

    # update args with wandb config
    for k, v in config.items():
        if k in args.__dict__:
            args.__dict__[k] = v

    # print args properly
    print(f"🔹 Using args:")
    for k, v in args.__dict__.items():
        print(f"    {k}: {v}")

    # -------------------
    # Create dataset
    # -------------------
    X_train, Y_train, X_val, Y_val, data_array_train, data_array_val = create_toy_dataset(args)
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=args.batch_size
    )
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch_size)

    # -------------------
    # Load model from checkpoint
    # -------------------
    model = KNNMLP(
        input_dim=X_train.shape[1],
        hidden_dim=config["hidden_dim"],
        output_dim=2,
        lr=config["lr"],
    )
    ckpt_path = os.path.join(
        args.log_dir, args.project, args.project, run_id, "checkpoints", "*.ckpt"
    )
    ckpt_path = glob(ckpt_path)[0]  # get the first checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in wandb artifact at {ckpt_path}")
    model = KNNMLP.load_from_checkpoint(ckpt_path)

    # -------------------
    # Lightning evaluation
    # -------------------
    trainer = pl.Trainer(accelerator="auto", devices="auto")
    print("🔹 Evaluating train split...")
    trainer.validate(model, dataloaders=train_loader)
    print("🔹 Evaluating val split...")
    trainer.validate(model, dataloaders=val_loader)

    # -------------------
    # Lightning prediction
    # -------------------
    preds_train = torch.cat(trainer.predict(model, dataloaders=train_loader), dim=0)
    preds_val = torch.cat(trainer.predict(model, dataloaders=val_loader), dim=0)

    # -------------------
    # Save outputs
    # -------------------
    save_dir = os.path.join(args.log_dir, args.project, "evaluations", run_id)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"predictions.pt")
    torch.save(
        {
            "train": {
                "X": train_loader.dataset.tensors[0],
                "Y": train_loader.dataset.tensors[1],
                "preds": preds_train,
                "data_array": data_array_train,
            },
            "val": {
                "X": val_loader.dataset.tensors[0],
                "Y": val_loader.dataset.tensors[1],
                "preds": preds_val,
                "data_array": data_array_val,
            },
        },
        out_path,
    )

    print(f"✅ Predictions + metrics saved at {out_path}")


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
        help="Type of node features from: original / eig / filter / both / original_random_augmented / original_repeated_augmented / eig_exclude_xy / filter_exclude_xy / both_exclude_xy",
    )
    parser.add_argument(
        "--tau", type=float, default=1.0, help="Temoral constant for filter features"
    )
    parser.add_argument(
        "--filter_size", type=int, default=5, help="Filter size for filter features"
    )
    parser.add_argument(
        "--relative_coordinates",
        action="store_true",
        help="Use relative coordinates for kNN features",
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
        "--total_frames",
        type=int,
        default=2_000,
        help="Number of frames in the sequence",
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

    # Training params
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max training epochs"
    )

    # Logging / wandb
    parser.add_argument(
        "--entity", type=str, default=None, help="wandb entity (user or team)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="knn-mlp-regression-relative",
        help="wandb project name",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="wandb_logs",
        help="Local directory to save wandb logs",
    )
    parser.add_argument(
        "--online", action="store_true", help="Use online mode for wandb"
    )

    # Evaluation
    parser.add_argument(
        "--eval_run_id",
        type=str,
        default=None,
        help="If set, load model from this wandb run ID and evaluate train+test.",
    )

    # Parse args
    args = parser.parse_args()

    if args.eval_run_id is not None:
        evaluate_run(args.eval_run_id, args)
    else:
        train(args)
