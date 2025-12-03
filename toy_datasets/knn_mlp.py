import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np

from omegaconf import OmegaConf
from dataset_generator import DatasetGenerator
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
import faiss

# -------------------------------
# Utilities
# -------------------------------
# def knn_indices_from_pos(pos: torch.Tensor, k: int) -> torch.Tensor:
#     """Compute k nearest neighbors (including self) from positions."""
#     dist = torch.cdist(pos, pos)  # [N, N]
#     knn_idx = dist.topk(k, largest=False).indices  # [N, k]
#     return knn_idx

def knn_indices_from_pos(pos: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k nearest neighbors (including self) from positions using FAISS.
    Automatically uses GPU if available, otherwise falls back to CPU.

    Args:
        pos (torch.Tensor): [N, D] tensor of positions.
        k (int): number of neighbors to return (including self).

    Returns:
        torch.Tensor: [N, k] tensor of neighbor indices.
    """
    
    pos_np = pos.cpu().numpy().astype('float32')
    d = pos_np.shape[1]

    # Build FAISS index (GPU if available, otherwise CPU)
    try:
        # Try GPU first
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
        print("   Using GPU for kNN computation")
    except (RuntimeError, AttributeError) as e:
        # Fall back to CPU if GPU not available
        print(f"   ⚠️  GPU not available ({type(e).__name__}), falling back to CPU for kNN computation")
        index = faiss.IndexFlatL2(d)
    
    index.add(pos_np)

    # Search k neighbors
    D, I = index.search(pos_np, k)

    return torch.from_numpy(I)

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
    # 1. Build unique dataset hash from DatasetGenerator config
    # -------------------------------
    # Use the same hash function as DatasetGenerator for consistency
    from dsec_utils import generate_dataset_hash
    
    # Build config dict with all relevant parameters
    config_dict = {
        'shape_class': 'star8',
        'total_frames': args.total_frames,
        'image_width': args.img_size[1],
        'image_height': args.img_size[0],
        'num_points': args.num_points,
        'outer_radius': args.outer_radius,
        'inner_radius': args.inner_radius,
        'number_of_rotations': args.num_rotations,
        'frame_time_us': 1000,  # Fixed for now, could be made configurable
        'event_generation_method': getattr(args, 'event_generation_method', 'synthetic'),
        # Texture parameters (if/when added)
        'foreground_texture': getattr(args, 'foreground_texture', None),
        'background_texture': getattr(args, 'background_texture', None),
        'fg_image_path': getattr(args, 'fg_image_path', None),
        'bg_image_path': getattr(args, 'bg_image_path', None),
        'use_random_dtd_texture': getattr(args, 'use_random_dtd_texture', False),
        'dtd_texture_mode': getattr(args, 'dtd_texture_mode', 'both'),
        'random_seed': getattr(args, 'random_seed', None),
        # Add feature computation parameters to hash
        'tau': args.tau,
        'filter_size': args.filter_size,
    }
    
    dataset_hash = generate_dataset_hash(**config_dict)
    dataset_path = os.path.join(cache_dir, f"{dataset_hash}_data.pt")
    
    # Print dataset configuration
    print("\n" + "="*70)
    print("📋 DATASET GENERATION CONFIG")
    print("="*70)
    print(f"🔑 Hash: {dataset_hash[:12]}...")
    print(f"\n🎯 Shape Parameters:")
    print(f"   shape_class: {config_dict['shape_class']}")
    print(f"   num_points: {config_dict['num_points']}")
    print(f"   outer_radius: {config_dict['outer_radius']}")
    print(f"   inner_radius: {config_dict['inner_radius']}")
    print(f"   number_of_rotations: {config_dict['number_of_rotations']}")
    print(f"\n🖼️  Image Parameters:")
    print(f"   width × height: {config_dict['image_width']} × {config_dict['image_height']}")
    print(f"   total_frames: {config_dict['total_frames']}")
    print(f"\n⏱️  Temporal:")
    print(f"   frame_time_us: {config_dict['frame_time_us']}")
    print(f"\n🎨 Texture:")
    print(f"   foreground: {config_dict['foreground_texture']}")
    print(f"   background: {config_dict['background_texture']}")
    print(f"\n⚡ Events & Features:")
    print(f"   event_method: {config_dict['event_generation_method']}")
    print(f"   tau: {config_dict['tau']}, filter_size: {config_dict['filter_size']}")
    print("="*70 + "\n")

    # -------------------------------
    # 2. Load or compute dataset
    # -------------------------------
    if os.path.exists(dataset_path) and not args.force_regenerate:
        print(f"🔹 Loading cached dataset from {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
    else:
        if args.force_regenerate and os.path.exists(dataset_path):
            print(f"♻️  Force regenerate: Ignoring cached dataset at {dataset_path}")
        print("⚡ Generating new dataset...")

        if args.toy_dataset == "star8":
            # Create config for DatasetGenerator
            cfg = OmegaConf.create({
                'shape_class': 'star8',
                'seq_name': f'knn_mlp_{dataset_hash}',
                'total_frames': args.total_frames,
                'image_width': args.img_size[1],
                'image_height': args.img_size[0],
                'face_color': 'black',
                'num_points': args.num_points,
                'outer_radius': args.outer_radius,
                'inner_radius': args.inner_radius,
                'number_of_rotations': args.num_rotations,
                'foreground_texture': None,
                'background_texture': None,
                'event_generation_method': getattr(args, 'event_generation_method', 'synthetic'),
                'save_step': 20,
                'frame_time_us': 1000,
                'flow_dt_us': 20000,
                'start_ts_us': 0,
                'test_size': 0.0,  # Don't split - we'll do it ourselves
                'outdir': 'toy_datasets/data',
                'sanity_check': False,
                'force_regenerate': False,
                'generate_animation': False,
                'auto_name': False,
                # v2e parameters (used only if event_generation_method='v2e')
                'v2e_pos_thres': 0.2,
                'v2e_neg_thres': 0.2,
                'v2e_sigma_thres': 0.,
                'v2e_cutoff_hz': 0,
                'v2e_leak_rate_hz': 0.0,
                'v2e_shot_noise_rate_hz': 0.0,
                'v2e_refractory_period_s': 0.0,
                'v2e_seed': args.random_seed,
                'v2e_photoreceptor_noise': False,
                'v2e_leak_jitter_fraction': 0.0,
                'v2e_noise_rate_cov_decades': 0.0,
                'v2e_fg_gamma': 2.0,
                'v2e_bg_gamma': 0.6,
                'v2e_fg_brightness': 1.0,
                'v2e_bg_brightness': 1.0,
            })
            
            # Create generator
            generator = DatasetGenerator(cfg)
            
            # Access the shape instance directly to generate events
            # This avoids generating full dataset files when we only need events
            generator._create_shape_instance()
            data_array = generator._generate_events()
        else:
            raise ValueError(f"Unknown toy dataset: {args.toy_dataset}")
        
        # Normalize time so that frame_time_us distance equals 1.0 spatial unit
        frame_time_us = cfg.frame_time_us
        data_array['t'] = data_array['t'] / frame_time_us
        print(f"⏱️  Normalized time: {frame_time_us} µs → 1.0 spatial unit")
        print(f"   Time range: [{data_array['t'].min():.2f}, {data_array['t'].max():.2f}]")
 
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

    if os.path.exists(knn_path) and not args.force_regenerate:
        print(f"🔹 Loading cached kNN index from {knn_path}")
        knn_idx = torch.load(knn_path)
    else:
        if args.force_regenerate and os.path.exists(knn_path):
            print(f"♻️  Force regenerate: Ignoring cached kNN index at {knn_path}")
        num_nodes = data.pos.size(0)
        pos_dim = data.pos.size(1)
        print(f"⚡ Computing kNN index:")
        print(f"   Number of events/nodes: {num_nodes:,}")
        print(f"   k (neighbors): {args.k}")
        print(f"   Position dimension: {pos_dim}")
        print(f"   Total distance computations: {num_nodes * args.k:,}")
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
    elif args.feature_type == "original_time_augmented":
        features = data.pos
    elif args.feature_type == "original_time_augmented_repeated_augmented":
        features = torch.cat(
            [data.pos, data.pos], dim=1
        )
    elif args.feature_type == "eig":
        features = torch.cat([data.pos[:, 0:2], data["eig"]], dim=1)
    elif args.feature_type == "filter":
        features = torch.cat([data.pos[:, 0:2], data["filter"]], dim=1)
    elif args.feature_type == "both":
        features = torch.cat([data.pos[:, 0:2], data["eig"], data["filter"]], dim=1)
    elif args.feature_type == "both_time_augmented":
        features = torch.cat([data.pos, data["eig"], data["filter"]], dim=1)
    elif args.feature_type == "eig_exclude_xy":
        features = data["eig"]
    elif args.feature_type == "filter_exclude_xy":
        features = data["filter"]
    elif args.feature_type == "both_exclude_xy":
        features = torch.cat([data["eig"], data["filter"]], dim=1)
    else:
        raise ValueError(f"Unknown feature_type: {args.feature_type}")

    print(f"📊 Feature statistics:")
    print(f"   Feature shape: {features.shape}")
    print(f"   Number of nodes: {features.size(0):,}")
    print(f"   Feature dimension: {features.size(1)}")
    print(f"   Total features (nodes × k × dim): {features.size(0) * args.k * features.size(1):,}")
    
    if args.relative_coordinates:
        if args.feature_type == "original":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "original_random_augmented":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "original_repeated_augmented":
            relative_feat_indices = [0, 1, 2, 3, 4]
        elif args.feature_type == "original_time_augmented":
            relative_feat_indices = [0, 1, 2]
        elif args.feature_type == "original_time_augmented_repeated_augmented":
            relative_feat_indices = [0, 1, 2, 3, 4, 5]
        elif args.feature_type == "eig":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "filter":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "both":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "both_time_augmented":
            relative_feat_indices = [0, 1, 2]
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
    
    # Return config_dict for wandb logging
    return X_train, Y_train, X_val, Y_val, data_array_train, data_array_val, config_dict


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

    X_train, Y_train, X_val, Y_val, _, _, dataset_config = create_toy_dataset(args)
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=args.batch_size)

    # -------------------
    # wandb logger
    # -------------------
    if not os.path.exists(os.path.join(args.log_dir, args.project)):
        os.makedirs(os.path.join(args.log_dir, args.project), exist_ok=True)
    
    # Merge dataset config into wandb config
    wandb_config = vars(args).copy()
    wandb_config['dataset_config'] = dataset_config
    
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=os.path.join(args.log_dir, args.project),
        project=args.project,
        config=wandb_config,
        log_model=False,
        offline=not args.online,
    )
    run_id = wandb_logger.experiment.id  # unique wandb run ID

    # Convert args to OmegaConf for pretty printing
    args_dict = vars(args)
    args_conf = OmegaConf.create(args_dict)
    
    print("\n" + "="*70)
    print(f"🚀 TRAINING RUN: {run_id}")
    print("="*70)
    print(OmegaConf.to_yaml(args_conf))
    print("="*70 + "\n")

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
    X_train, Y_train, X_val, Y_val, data_array_train, data_array_val, _ = create_toy_dataset(args)
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
        "--event_generation_method",
        type=str,
        default="synthetic",
        choices=["synthetic", "v2e"],
        help="Event generation method: 'synthetic' (fast) or 'v2e' (realistic DVS simulation)",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regenerate dataset and kNN index even if cached versions exist",
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
        default=None,
        help="Random seed for test/train split (defaults to --random_seed if not set)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Master random seed for reproducibility (DTD textures, v2e, train/test split if not specified separately)",
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
    
    # Use random_seed as default for other seeds if not explicitly set
    if args.test_split_seed is None:
        args.test_split_seed = args.random_seed

    if args.eval_run_id is not None:
        evaluate_run(args.eval_run_id, args)
    else:
        train(args)
