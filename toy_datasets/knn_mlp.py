import sys

sys.path.append(".")
sys.path.append("..")
import numpy as np

from omegaconf import OmegaConf
from dataset_generator import DatasetGenerator
from utils.visualize_utils import animate_events
from utils.data_utils import *
from utils.retrieve_hpc_data import ensure_local_file
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
    except Exception as e:
        # Fall back to CPU if GPU not available
        print(f"   ⚠️  GPU not available ({type(e).__name__}:{e}), falling back to CPU for kNN computation")
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
# Configuration Builder
# -------------------------------
def build_dataset_config(args, for_hash_only=False):
    """
    Build dataset configuration dictionary from args.
    
    Args:
        args: Argument namespace with dataset parameters
        for_hash_only: If True, only include parameters needed for hash computation
                      If False, include all parameters for DatasetGenerator
    
    Returns:
        OmegaConf config dict or plain dict (for hash computation)
    """
    # Handle DTD texture configuration
    # Don't pre-select textures here - let DatasetGenerator handle it with proper seeds
    foreground_texture = getattr(args, 'foreground_texture', None)
    background_texture = getattr(args, 'background_texture', None)
    fg_image_path = getattr(args, 'fg_image_path', None)
    bg_image_path = getattr(args, 'bg_image_path', None)
    
    if getattr(args, 'use_random_dtd_texture', False):
        # Set texture type to 'image' - DatasetGenerator will select the actual files
        dtd_mode = getattr(args, 'dtd_texture_mode', 'both')
        
        if dtd_mode in ['fg','foreground', 'both']:
            foreground_texture = 'image'
            # fg_image_path will be None - DatasetGenerator will select it
            
        if dtd_mode in ['bg', 'background', 'both']:
            background_texture = 'image'
            # bg_image_path will be None - DatasetGenerator will select it
    
    # Core parameters for hash computation
    hash_params = {
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
        # Texture parameters
        'foreground_texture': foreground_texture,
        'background_texture': background_texture,
        'fg_image_path': fg_image_path,
        'bg_image_path': bg_image_path,
        'use_random_dtd_texture': getattr(args, 'use_random_dtd_texture', False),
        'dtd_texture_mode': getattr(args, 'dtd_texture_mode', 'both'),
        'dtd_root': getattr(args, 'dtd_root', 'data/dtd/images'),
        'random_seed': getattr(args, 'random_seed', None),
        # Animation parameters (for consistency)
        'animation_fps': getattr(args, 'animation_fps', 10),
        'animation_frame_step': getattr(args, 'animation_frame_step', 10),
        'event_animation_fps': getattr(args, 'event_animation_fps', 10),
        'event_accumulation_ms': getattr(args, 'event_accumulation_ms', 10),
        # v2e parameters that affect the dataset
        'v2e_pos_thres': float(getattr(args, 'v2e_pos_thres', 0.2)),
        'v2e_neg_thres': float(getattr(args, 'v2e_neg_thres', 0.2)),
        'v2e_sigma_thres': float(getattr(args, 'v2e_sigma_thres', 0.0)),
        'v2e_cutoff_hz': getattr(args, 'v2e_cutoff_hz', 0),
        'v2e_leak_rate_hz': float(getattr(args, 'v2e_leak_rate_hz', 0.0)),
        'v2e_shot_noise_rate_hz': float(getattr(args, 'v2e_shot_noise_rate_hz', 0.0)),
        'v2e_refractory_period_s': float(getattr(args, 'v2e_refractory_period_s', 0.0)),
        'v2e_seed': getattr(args, 'v2e_seed', args.random_seed),
        'v2e_photoreceptor_noise': getattr(args, 'v2e_photoreceptor_noise', False),
        'v2e_leak_jitter_fraction': float(getattr(args, 'v2e_leak_jitter_fraction', 0.0)),
        'v2e_noise_rate_cov_decades': float(getattr(args, 'v2e_noise_rate_cov_decades', 0.0)),
        'v2e_fg_gamma': float(getattr(args, 'v2e_fg_gamma', 2.0)),
        'v2e_bg_gamma': float(getattr(args, 'v2e_bg_gamma', 0.6)),
        'v2e_fg_brightness': float(getattr(args, 'v2e_fg_brightness', 1.0)),
        'v2e_bg_brightness': float(getattr(args, 'v2e_bg_brightness', 1.0)),
        'v2e_temporal_filter_percent': getattr(args, 'v2e_temporal_filter_percent', None),
        # Intensity-based event generation parameters
        'intensity_pos_threshold': float(getattr(args, 'intensity_pos_threshold', 0.05)),
        'intensity_neg_threshold': float(getattr(args, 'intensity_neg_threshold', 0.05)),
        'intensity_shot_noise_rate_hz': float(getattr(args, 'intensity_shot_noise_rate_hz', 0.0)),
        # Feature computation parameters
        'tau': float(args.tau),
        'filter_size': args.filter_size,
    }
    
    if for_hash_only:
        return hash_params
    
    # Full configuration for DatasetGenerator
    from dsec_utils import generate_dataset_hash
    dataset_hash = generate_dataset_hash(**hash_params)
    
    # Build full config from hash_params and add extra parameters
    full_config = hash_params.copy()
    full_config.update({
        'seq_name': f'knn_mlp_{dataset_hash}',
        'face_color': 'black',
        'dtd_fg_seed': getattr(args, 'random_seed', None),
        'dtd_bg_seed': (getattr(args, 'random_seed', None) + 1) if getattr(args, 'random_seed', None) is not None else None,
        'save_step': None,
        'flow_dt_us': None,
        'start_ts_us': 0,
        'test_size': 0.0,
        'outdir': 'toy_datasets/data',
        'sanity_check': False,
        'force_regenerate': False,
        'generate_animation': False,
        'auto_name': False,
    })
    
    return OmegaConf.create(full_config)


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

    # Build config dict with all relevant parameters (for hashing)
    config_dict = build_dataset_config(args, for_hash_only=True)

    dataset_hash = generate_dataset_hash(**config_dict)
    config_dict['dataset_hash'] = dataset_hash
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
    if config_dict.get('use_random_dtd_texture'):
        print(f"   use_random_dtd_texture: {config_dict['use_random_dtd_texture']}")
        print(f"   dtd_texture_mode: {config_dict['dtd_texture_mode']}")
        print(f"   dtd_root: {config_dict['dtd_root']}")
        print(f"   random_seed: {config_dict['random_seed']}")
        # Note: Actual selected texture paths will be shown after dataset generation
    else:
        print(f"   foreground: {config_dict['foreground_texture']}")
        print(f"   background: {config_dict['background_texture']}")
        if config_dict.get('fg_image_path'):
            print(f"   fg_image_path: {config_dict['fg_image_path']}")
        if config_dict.get('bg_image_path'):
            print(f"   bg_image_path: {config_dict['bg_image_path']}")
    print(f"\n⚡ Events & Features:")
    print(f"   event_method: {config_dict['event_generation_method']}")
    print(f"   tau: {config_dict['tau']}, filter_size: {config_dict['filter_size']}")
    print("="*70 + "\n")

    # -------------------------------
    # 2. Load or compute dataset
    # -------------------------------
    generator = None  # Will be created if needed for animation
    frame_time_us = config_dict['frame_time_us']  # Always available from config_dict

    if os.path.exists(dataset_path) and not args.force_regenerate:
        print(f"🔹 Loading cached dataset from {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
    else:
        if args.force_regenerate and os.path.exists(dataset_path):
            print(f"♻️  Force regenerate: Ignoring cached dataset at {dataset_path}")
        print("⚡ Generating new dataset...")

        if args.toy_dataset == "star8":
            # Create config for DatasetGenerator using shared builder
            cfg = build_dataset_config(args, for_hash_only=False)

            # Create generator
            generator = DatasetGenerator(cfg)

            # Access the shape instance directly to generate events
            # This avoids generating full dataset files when we only need events
            generator._create_shape_instance()
            data_array = generator._generate_events()

            # Update config_dict with actual selected texture paths
            # After _create_shape_instance() -> _build_texture_params(), the paths are in generator.config
            if generator.config.get('fg_image_path'):
                config_dict['fg_image_path'] = generator.config.fg_image_path
            if generator.config.get('bg_image_path'):
                config_dict['bg_image_path'] = generator.config.bg_image_path

            # Print selected texture paths if DTD was used
            if config_dict.get('use_random_dtd_texture'):
                print("\n" + "="*70)
                print("🎨 SELECTED DTD TEXTURES")
                print("="*70)
                if config_dict.get('fg_image_path'):
                    print(f"   Foreground: {config_dict['fg_image_path']}")
                if config_dict.get('bg_image_path'):
                    print(f"   Background: {config_dict['bg_image_path']}")
                print("="*70 + "\n")
        else:
            raise ValueError(f"Unknown toy dataset: {args.toy_dataset}")

        data = numpy2pyg_event_convertor(data_array)
        data["v"] = torch.tensor(np.array([data_array["v_x"], data_array["v_y"]])).T     
        # Multiply tau by frame_time_us to get actual time constant in us
        # Compute Harris features
        tau_us = args.tau * frame_time_us
        harris_rec = HarrisRecursive(
            tau=tau_us, filter_size=args.filter_size, image_size=args.img_size
        )
        harris_rec(data_array)
        eig1 = harris_rec.eig1
        eig2 = harris_rec.eig2
        filter_values = harris_rec.filter_value_recursive
        grad_x = harris_rec.grad_x
        grad_y = harris_rec.grad_y
        data["eig"] = torch.tensor(np.array([eig1, eig2])).T
        data["filter"] = torch.tensor(filter_values).unsqueeze(1)
        data["grad"] = torch.tensor(np.array([grad_x, grad_y])).T

        torch.save(data, dataset_path)
        print(f"💾 Dataset cached at {dataset_path}")

    # -------------------------------
    # Save config file (always, even if dataset was cached)
    # -------------------------------
    config_save_path = os.path.join(cache_dir, f"{dataset_hash}_config.yaml")
    if not os.path.exists(config_save_path) or args.force_regenerate:
        config_to_save = OmegaConf.create(config_dict)
        OmegaConf.save(config_to_save, config_save_path)
        print(f"📋 Dataset config saved at {config_save_path}")
    else:
        print(f"📋 Config file already exists at {config_save_path}")

    # -------------------------------
    # Optionally create animation (always, even if dataset was cached)
    # -------------------------------
    if args.create_animation:
        animation_path = os.path.join(cache_dir, f"{dataset_hash}_animation.gif")
        if not os.path.exists(animation_path) or args.force_regenerate:
            print("🎬 Creating dataset animation...")
            try:
                # If generator wasn't created above (dataset was cached), create it now
                if generator is None:
                    if args.toy_dataset == "star8":
                        cfg = build_dataset_config(args, for_hash_only=False)
                        generator = DatasetGenerator(cfg)
                        generator._create_shape_instance()

                # Use texture-aware animation if textures are enabled
                if config_dict.get('foreground_texture') or config_dict.get('background_texture'):
                    print("   Creating animation with textures...")
                    frames = generator.shape_instance.create_animation_with_textures(frame_step=20, fps=10)
                    # Save frames as GIF using imageio
                    import imageio
                    imageio.mimsave(animation_path, frames, fps=10, loop=0)
                else:
                    print("   Creating animation without textures...")
                    anim = generator.shape_instance.create_animation(frame_step=20, interval=100)
                    anim.save(animation_path, writer='pillow', fps=10)
                print(f"🎞️  Animation saved at {animation_path}")
            except Exception as e:
                print(f"⚠️  Could not create animation: {e}")
        else:
            print(f"🎞️  Animation already exists at {animation_path}")

    if args.feature_type in ("grad", "grad_filter") and "grad" not in data:
        print("🔹 Cached dataset has no gradient features; recomputing Harris gradients")
        data_array_for_harris = pyg2numpy_event_convertor(data)
        tau_us = args.tau * frame_time_us
        harris_rec = HarrisRecursive(
            tau=tau_us, filter_size=args.filter_size, image_size=args.img_size
        )
        harris_rec(data_array_for_harris)
        data["eig"] = torch.tensor(np.array([harris_rec.eig1, harris_rec.eig2])).T
        data["filter"] = torch.tensor(harris_rec.filter_value_recursive).unsqueeze(1)
        data["grad"] = torch.tensor(np.array([harris_rec.grad_x, harris_rec.grad_y])).T
        torch.save(data, dataset_path)
        print(f"💾 Cached dataset updated with gradient features at {dataset_path}")

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
        print(f"   Spatial extent:  x[{data.pos[:,0].min().item():.1f}, {data.pos[:,0].max().item():.1f}]")
        print(f"                    y[{data.pos[:,1].min().item():.1f}, {data.pos[:,1].max().item():.1f}]")
        print(f"   Temporal extent: t[{data.pos[:,2].min().item():.2f}, {data.pos[:,2].max().item():.2f}]")

        # Normalize time so that frame_time_us distance equals 1.0 spatial unit
        # Use a copy to avoid modifying the original data
        pos_normalized = data.pos.clone()
        pos_normalized[:, 2] = pos_normalized[:, 2] / frame_time_us
        print("   Normalized temporal extent after scaling:", flush=True)
        print(f"                    t[{pos_normalized[:,2].min().item():.2f}, {pos_normalized[:,2].max().item():.2f}]")
        print("   Computing kNN indices (this may take a while for large datasets)...", flush=True)
        knn_idx = knn_indices_from_pos(pos_normalized, args.k)
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
        features = data.pos / torch.tensor([1.0, 1.0, frame_time_us])
    elif args.feature_type == "original_time_augmented_repeated_augmented":
        features = torch.cat(
            [
                data.pos / torch.tensor([1.0, 1.0, frame_time_us]),
                data.pos / torch.tensor([1.0, 1.0, frame_time_us]),
            ],
            dim=1,
        )
    elif args.feature_type == "eig":
        features = torch.cat([data.pos[:, 0:2], data["eig"]], dim=1)
    elif args.feature_type == "filter":
        features = torch.cat([data.pos[:, 0:2], data["filter"]], dim=1)
    elif args.feature_type == "both":
        features = torch.cat([data.pos[:, 0:2], data["eig"], data["filter"]], dim=1)
    elif args.feature_type == "grad":
        features = torch.cat([data.pos[:, 0:2], data["grad"]], dim=1)
    elif args.feature_type == "grad_filter":
        features = torch.cat([data.pos[:, 0:2], data["grad"], data["filter"]], dim=1)
    elif args.feature_type == "both_time_augmented":
        features = torch.cat(
            [
                data.pos / torch.tensor([1.0, 1.0, frame_time_us]),
                data["eig"],
                data["filter"],
            ],
            dim=1,
        )
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
        elif args.feature_type == "grad":
            relative_feat_indices = [0, 1]
        elif args.feature_type == "grad_filter":
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

    # -------------------------------
    # Filter out None values in labels
    # -------------------------------
    total_samples = Y.size(0)
    # Check for NaN or inf values in labels (vectorized)
    valid_mask = ~(torch.isnan(Y).any(dim=1) | torch.isinf(Y).any(dim=1))

    num_invalid = (~valid_mask).sum().item()

    if num_invalid > 0:
        print("\n" + "="*70)
        print("⚠️  WARNING: Found invalid (None/NaN/Inf) values in labels!")
        print("="*70)
        print(f"   Total samples: {total_samples:,}")
        print(f"   Invalid samples: {num_invalid:,}")
        print(f"   Invalid percentage: {(num_invalid/total_samples)*100:.2f}%")
        print(f"   Valid samples remaining: {valid_mask.sum().item():,}")
        print(f"   Valid percentage: {(valid_mask.sum().item()/total_samples)*100:.2f}%")
        print("="*70 + "\n")

        # Filter out invalid samples
        X_with_neighbors = X_with_neighbors[valid_mask]
        Y = Y[valid_mask]
        features = features[valid_mask]
        knn_idx = knn_idx[valid_mask]

        print(f"✅ Filtered data shapes after removing invalid labels:")
        print(f"   X_with_neighbors: {X_with_neighbors.shape}")
        print(f"   Y (labels): {Y.shape}")

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
    def __init__(self, input_dim, hidden_dim, output_dim, lr, scheduler_type="none"):
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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )

        scheduler_type = getattr(self.hparams, "scheduler_type", "none")
        if scheduler_type == "none":
            return optimizer

        # Trainer is attached by Lightning before this is called
        if getattr(self, "trainer", None) is None:
            return optimizer

        total_steps = self.trainer.estimated_stepping_batches
        if total_steps is None or total_steps == 0:
            return optimizer

        # Select scheduler based on scheduler_type
        # Each scheduler has its own interval/frequency configuration
        scheduler = None
        scheduler_config = {}
        
        if scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",      # OneCycleLR updates per batch
                "frequency": 1,
            }
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.hparams.lr * 1e-4,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",     # CosineAnnealingLR updates per epoch
                "frequency": 1,
            }
        elif scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_steps,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",     # LinearLR updates per epoch
                "frequency": 1,
            }
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

# -------------------------------
# Model (order invariant version)
# -------------------------------
class KNNMLPOrderInvariant(KNNMLP):
    """
    Order-invariant version of KNNMLP that aggregates neighbor features
    using permutation-invariant operations (mean pooling).
    
    Inherits training_step, validation_step, and configure_optimizers from KNNMLP.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, K, lr, scheduler_type="none"):
        # Call parent __init__ with dummy model that will be replaced
        super().__init__(input_dim, hidden_dim, output_dim, lr, scheduler_type)
        
        # Save K as additional hyperparameter (parent already saved the rest)
        self.hparams.K = K
        self.K = K
        assert input_dim % K == 0, "Input dimension must be divisible by K"
        self.F = input_dim // K  # Original feature dimension per neighbor
        
        # Replace the sequential model with order-invariant architecture
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.F, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )          
        self.out = nn.Linear(hidden_dim, output_dim)
        
        # Remove the simple sequential model from parent
        del self.model

    def forward(self, batch):
        if isinstance(batch, (list, tuple)):
            x, _ = batch  # ignore y
        else:
            x = batch
        # Reshape to [N, K, F]
        x_reshaped = x.view(-1, self.K, self.F)
        # Get central node and neighbor features
        x_i = x_reshaped[:, 0, :]  # Central node features [N, F]
        neighbor_feats = x_reshaped[:, 1:, :]  # Neighbor features [N, K-1, F]
        # Concatenate central node features with each neighbor
        central_expanded = x_i.unsqueeze(1).expand(-1, self.K - 1, -1)  # [N, K-1, F]
        concat_feats = torch.cat([central_expanded, neighbor_feats], dim=-1)  # [N, K-1, 2F]
        # Pass through MLP  
        mlp_out = self.mlp(concat_feats)  # [N, K-1, hidden_dim]
        # Aggregate using mean (order-invariant)
        agg = mlp_out.mean(dim=1)  # [N, hidden_dim]
        # Further processing
        agg = self.post_mlp(agg)  # [N, hidden_dim]
        # Final output layer
        out = self.out(agg)  # [N, output_dim]
        return out


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
    if args.model_variant == "knnmlporder":
        model = KNNMLPOrderInvariant(
            input_dim=train_loader.dataset.tensors[0].shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=2,
            K=args.k,
            lr=args.lr,
            scheduler_type=args.scheduler,
        )
    else:  # knnmlp (default)
        model = KNNMLP(
            input_dim=train_loader.dataset.tensors[0].shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=2,
            lr=args.lr,
            scheduler_type=args.scheduler,
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
def _setup_run_for_evaluation(run_path: str, args):
    """
    Common setup logic for evaluation functions.
    Loads wandb config, creates dataset, loads model checkpoint.
    
    Returns:
        tuple: (model, train_loader, val_loader, data_array_train, data_array_val, run)
    """
    # -------------------
    # Load wandb run config and merge with command-line args
    # -------------------
    api = wandb.Api()
    run = api.run(run_path)
    config = run.config

    # Get the set of arguments that were explicitly provided on command line
    provided_args = getattr(args, '_provided_args', set())
    
    # Update args with wandb config, but keep command-line overrides
    for k, v in config.items():
        if k == 'dataset_config':
            # Skip dataset_config - it will be regenerated from args
            continue
        if k in args.__dict__:
            # If this argument was explicitly provided on command line, keep it
            if k in provided_args:
                print(f"🔸 Using command-line value for '{k}': {getattr(args, k)}")
            else:
                # Otherwise, use the wandb config value
                args.__dict__[k] = v

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
    # Determine which model class to use based on model_variant
    if args.model_variant == "knnmlporder":
        model = KNNMLPOrderInvariant(
            input_dim=X_train.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=2,
            K=args.k,
            lr=args.lr,
            scheduler_type=args.scheduler,
        )
    else:  # knnmlp (default)
        model = KNNMLP(
            input_dim=X_train.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=2,
            lr=args.lr,
            scheduler_type=args.scheduler,
        )
    
    # Construct checkpoint directory path (relative to project root)
    ckpt_dir = os.path.join(
        args.log_dir, args.project, args.project, run.id, "checkpoints"
    )
    ckpt_pattern = os.path.join(ckpt_dir, "*.ckpt")
    
    # Try to find checkpoint locally first
    ckpt_path_list = glob(ckpt_pattern)
    
    # If not found locally, try to fetch from HPC
    if len(ckpt_path_list) == 0:
        print(f"🔹 Checkpoint not found locally at {ckpt_pattern}")
        print(f"🔹 Attempting to retrieve from HPC...")
        
        # Use ensure_local_file to fetch the checkpoint directory
        # This will handle HPC detection and rsync automatically
        ensure_local_file(ckpt_dir, verbose=True)
        
        # Try glob again after potential rsync
        ckpt_path_list = glob(ckpt_pattern)
    
    if len(ckpt_path_list) == 0:
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_pattern}. "
            f"Tried local search and HPC retrieval (if not on HPC)."
        )
    
    ckpt_path = ckpt_path_list[0]  # take the first checkpoint
    print(f"✅ Loading model checkpoint from {ckpt_path}")
    # Load from checkpoint using the appropriate model class
    if args.model_variant == "knnmlporder":
        model = KNNMLPOrderInvariant.load_from_checkpoint(ckpt_path)
    else:
        model = KNNMLP.load_from_checkpoint(ckpt_path)

    return model, train_loader, val_loader, data_array_train, data_array_val, run


def evaluate_run(run_path: str, args):
    """
    Evaluate a trained wandb run on the test set and save results.

    Args:
        run_path: wandb run path to evaluate
        args: argparse.Namespace with args used for training
    """
    # Use common setup logic
    model, train_loader, val_loader, data_array_train, data_array_val, run = _setup_run_for_evaluation(run_path, args)

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
    save_dir = os.path.join(args.log_dir, args.project, "evaluations", run.id)
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

    print(f"✅ Predictions saved at {out_path}")


def compute_metrics_for_run(run_path: str, args):
    """
    Compute metrics for a trained wandb run on train and validation sets.

    Args:
        run_path: wandb run path (e.g., "project/run_id" or "entity/project/run_id")
        args: argparse.Namespace with args used for training

    Returns:
        dict: Computed metrics for train and validation splits
    """
    from utils.evaluation_metrics import compute_vector_errors
    
    # Use common setup logic
    model, train_loader, val_loader, data_array_train, data_array_val, run = _setup_run_for_evaluation(run_path, args)

    # -------------------
    # Generate predictions
    # -------------------
    trainer = pl.Trainer(accelerator="auto", devices="auto")
    preds_train = torch.cat(trainer.predict(model, dataloaders=train_loader), dim=0)
    preds_val = torch.cat(trainer.predict(model, dataloaders=val_loader), dim=0)

    # -------------------
    # Compute metrics
    # -------------------
    # Extract ground truth labels from data loaders
    Y_train = train_loader.dataset.tensors[1]
    Y_val = val_loader.dataset.tensors[1]
    
    train_metrics = compute_vector_errors(preds_train, Y_train)
    val_metrics = compute_vector_errors(preds_val, Y_val)

    # Add run metadata to results
    results = {
        "run_id": run.id,
        "run_path": run_path,
        "run_name": run.name,
        "config": dict(run.config),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    # -------------------
    # Save outputs
    # -------------------
    save_dir = os.path.join(args.log_dir, args.project, "evaluations", run.id)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"metrics.pt")
    torch.save(results, out_path)
    
    print(f"✅ Computed metrics for run {run.id}")
    print(f"🔹 Train EPE: {train_metrics['EPE']:.4f}")
    print(f"🔹 Val EPE: {val_metrics['EPE']:.4f}")
    print(f"✅ Full metrics saved at {out_path}")
    
    return results


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
        help="Type of node features from: original / eig / filter / both / grad / grad_filter / original_random_augmented / original_repeated_augmented / eig_exclude_xy / filter_exclude_xy / both_exclude_xy",
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
        choices=["synthetic", "v2e", "intensity"],
        help="Event generation method: 'synthetic' (boundary-based), 'v2e' (realistic DVS simulator), or 'intensity' (frame intensity differences)",
    )
    parser.add_argument(
        "--intensity_pos_threshold",
        type=float,
        default=0.05,
        help="Positive intensity threshold for intensity-based events (0-1 range, e.g., 0.05 = 5%% brightness change)",
    )
    parser.add_argument(
        "--intensity_neg_threshold",
        type=float,
        default=0.05,
        help="Negative intensity threshold for intensity-based events (0-1 range, e.g., 0.05 = 5%% brightness change)",
    )
    parser.add_argument(
        "--intensity_shot_noise_rate_hz",
        type=float,
        default=0.0,
        help="Shot noise rate in Hz per pixel for intensity-based events (default: 0.0 = no noise). "
             "Only used when --event_generation_method=intensity. Adds random ON/OFF events to simulate sensor noise.",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regenerate dataset and kNN index even if cached versions exist",
    )
    parser.add_argument(
        "--create_animation",
        action="store_true",
        help="Create and save shape animation when generating new dataset",
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
    
    # Texture params
    parser.add_argument(
        "--use_random_dtd_texture",
        action="store_true",
        help="Use random textures from DTD (Describable Textures Dataset) for foreground/background",
    )
    parser.add_argument(
        "--dtd_texture_mode",
        type=str,
        default="both",
        choices=["foreground", "background", "both"],
        help="Which parts to apply DTD textures to: foreground (star), background, or both",
    )
    parser.add_argument(
        "--foreground_texture",
        type=str,
        default=None,
        help="Specific texture type for foreground (if not using random DTD)",
    )
    parser.add_argument(
        "--background_texture",
        type=str,
        default=None,
        help="Specific texture type for background (if not using random DTD)",
    )
    parser.add_argument(
        "--fg_image_path",
        type=str,
        default=None,
        help="Path to custom foreground texture image",
    )
    parser.add_argument(
        "--bg_image_path",
        type=str,
        default=None,
        help="Path to custom background texture image",
    )
    parser.add_argument(
        "--dtd_root",
        type=str,
        default="data/dtd/images",
        help="Path to DTD (Describable Textures Dataset) root directory",
    )
    parser.add_argument(
        "--v2e_temporal_filter_percent",
        type=float,
        default=None,
        help="V2E temporal filter: keep only events within [frame_time, frame_time + X%%]. "
             "For example, 4.0 keeps events in [1000us, 1040us] for frame_time_us=1000. "
             "None = no filtering (default).",
    )
    
    # V2E-specific parameters (only used when event_generation_method='v2e')
    parser.add_argument(
        "--v2e_pos_thres",
        type=float,
        default=0.2,
        help="V2E positive contrast threshold (default: 0.2)",
    )
    parser.add_argument(
        "--v2e_neg_thres",
        type=float,
        default=0.2,
        help="V2E negative contrast threshold (default: 0.2)",
    )
    parser.add_argument(
        "--v2e_sigma_thres",
        type=float,
        default=0.0,
        help="V2E threshold mismatch sigma (default: 0.0)",
    )
    parser.add_argument(
        "--v2e_cutoff_hz",
        type=float,
        default=0,
        help="V2E photoreceptor cutoff frequency in Hz (default: 0)",
    )
    parser.add_argument(
        "--v2e_leak_rate_hz",
        type=float,
        default=0.0,
        help="V2E leak event rate in Hz (default: 0.0)",
    )
    parser.add_argument(
        "--v2e_shot_noise_rate_hz",
        type=float,
        default=0.0,
        help="V2E shot noise rate in Hz (default: 0.0)",
    )
    parser.add_argument(
        "--v2e_refractory_period_s",
        type=float,
        default=0.0,
        help="V2E refractory period in seconds (default: 0.0)",
    )
    parser.add_argument(
        "--v2e_seed",
        type=int,
        default=None,
        help="V2E random seed (defaults to --random_seed if not set)",
    )
    parser.add_argument(
        "--v2e_photoreceptor_noise",
        action="store_true",
        help="V2E: use photoreceptor noise model (more realistic)",
    )
    parser.add_argument(
        "--v2e_leak_jitter_fraction",
        type=float,
        default=0.0,
        help="V2E leak event timing jitter as fraction of interval (default: 0.0)",
    )
    parser.add_argument(
        "--v2e_noise_rate_cov_decades",
        type=float,
        default=0.0,
        help="V2E spatial variation in noise rates in decades (default: 0.0)",
    )
    parser.add_argument(
        "--v2e_fg_gamma",
        type=float,
        default=2.0,
        help="V2E gamma correction for foreground (>1 darkens, <1 brightens, default: 2.0)",
    )
    parser.add_argument(
        "--v2e_bg_gamma",
        type=float,
        default=0.6,
        help="V2E gamma correction for background (>1 darkens, <1 brightens, default: 0.6)",
    )
    parser.add_argument(
        "--v2e_fg_brightness",
        type=float,
        default=1.0,
        help="V2E foreground brightness multiplier (0-1, lower=darker, default: 1.0)",
    )
    parser.add_argument(
        "--v2e_bg_brightness",
        type=float,
        default=1.0,
        help="V2E background brightness multiplier (0-1, lower=darker, default: 1.0)",
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
    parser.add_argument(
        "--model_variant",
        type=str,
        default="knnmlp",
        choices=["knnmlp", "knnmlporder"],
        help="Model variant: 'knnmlp' (standard sequential) or 'knnmlporder' (order-invariant with mean pooling)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "onecycle", "cosine", "linear"],
        help="Learning rate scheduler type: 'none' (no scheduler), 'onecycle' (OneCycleLR), 'cosine' (CosineAnnealingLR), 'linear' (LinearLR)",
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
        "--eval_run_path",
        type=str,
        default=None,
        help="If set, load model from this wandb run path and evaluate train+test.",
    )
    parser.add_argument(
        "--compute_metrics_run_path",
        type=str,
        default=None,
        help="If set, load model from this wandb run path and compute metrics only (no data saving).",
    )

    # Parse args
    args = parser.parse_args()
    
    # Use random_seed as default for other seeds if not explicitly set
    if args.test_split_seed is None:
        args.test_split_seed = args.random_seed
    if args.v2e_seed is None:
        args.v2e_seed = args.random_seed

    if args.eval_run_path is not None:
        
        # Track which arguments were explicitly provided on command line
        # This helps evaluate_run know which values to override from wandb
        import sys
        provided_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                arg_name = arg[2:].replace('-', '_')
                provided_args.add(arg_name)
        args._provided_args = provided_args
        
        evaluate_run(args.eval_run_path, args)
        
    elif args.compute_metrics_run_path is not None:
        
        # Track which arguments were explicitly provided on command line
        import sys
        provided_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                arg_name = arg[2:].replace('-', '_')
                provided_args.add(arg_name)
        args._provided_args = provided_args
        
        results = compute_metrics_for_run(args.compute_metrics_run_path, args)
        print(f"\n🎯 Results for {args.compute_metrics_run_path}:")
        print(f"   Train EPE: {results['train_metrics']['EPE']:.4f}")
        print(f"   Val EPE: {results['val_metrics']['EPE']:.4f}")
        
    else:
        train(args)
