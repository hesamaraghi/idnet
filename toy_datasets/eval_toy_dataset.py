"""
Evaluation script for toy dataset trained models.

Usage:
    python toy_datasets/eval_toy_dataset.py --run_path haraghi/toydataset-tinyIDNet-multiseed/fxd5fk65
"""

import os
import sys
import json
import argparse
from glob import glob
from datetime import datetime
import torch
import wandb
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add parent directory to path to import from idn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from idn.loader.loader_dsec import (
    assemble_dsec_sequences,
    assemble_dsec_test_set,
    train_collate,
)
from idn.model.idedeq import NanoIDEDEQIDO, TinyIDEDEQIDO, IDEDEQIDO
from idn.utils.helper_functions import move_batch_to_cuda

# Import dataset creation utilities
sys.path.append(os.path.join(os.path.dirname(__file__)))
from create_flow_from_movement import build_star8_flow_and_events


def check_and_create_dataset(data_root, seq_name, config):
    """
    Check if dataset exists, if not create it using config parameters.
    
    Args:
        data_root: Root directory where dataset should be located
        seq_name: Sequence name to check/create
        config: OmegaConf config with dataset generation parameters
        
    Returns:
        bool: True if dataset exists or was created successfully
    """
    # Check if the dataset directory exists
    flow_dir = os.path.join(data_root, "train_optical_flow", seq_name, "flow", "forward")
    events_h5 = os.path.join(data_root, "train_events", seq_name, "events", "left", "events.h5")
    
    dataset_exists = os.path.exists(flow_dir) and os.path.exists(events_h5)
    print(f"   Checking dataset at: {data_root} for sequence '{seq_name}'")

    if dataset_exists:
        print(f"   ✓ Dataset '{seq_name}' already exists at {data_root}")
        return True
    
    print(f"   ⚠️  Dataset '{seq_name}' not found at {data_root}")
    print(f"   🔄 Creating dataset from config parameters...")
    
    # Extract dataset parameters from wandb config
    # The dataset metadata is logged to wandb as "dataset_metadata_{seq_name}"
    metadata_key = f"dataset_metadata_{seq_name}"
    
    if metadata_key not in config:
        # Cannot proceed without dataset metadata - parameters are critical for correct evaluation
        raise ValueError(
            f"❌ ERROR: Dataset metadata not found in wandb config!\n"
            f"   Expected key: '{metadata_key}'\n"
            f"   Available metadata keys: {[k for k in config.keys() if 'metadata' in k.lower()]}\n\n"
            f"   This means the dataset configuration was not logged during training.\n"
            f"   Without the exact dataset parameters (total_frames, save_step, image_size, etc.),\n"
            f"   we cannot reliably recreate the dataset and evaluation results would be invalid.\n\n"
            f"   Possible solutions:\n"
            f"   1. Use a wandb run that has dataset metadata logged (runs from newer training code)\n"
            f"   2. Manually create the dataset with correct parameters before evaluation\n"
            f"   3. Update the training code to log dataset metadata to wandb"
        )
    
    # Use metadata from wandb config (logged during training)
    metadata = config[metadata_key]
    total_frames = metadata['total_frames']
    image_size = metadata['image_size']
    image_height, image_width = image_size
    save_step = metadata['save_step']
    frame_time_us = metadata['frame_time_us']
    flow_dt_us = metadata['flow_dt_us']
    start_ts_us = metadata['start_ts_us']
    test_size = metadata['test_size']
    face_color = metadata['face_color']
    print(f"   ✓ Found dataset metadata in wandb config under key '{metadata_key}'")
    
    print(f"      Dataset parameters:")
    print(f"        seq_name: {seq_name}")
    print(f"        data_root: {data_root}")
    print(f"        total_frames: {total_frames}")
    print(f"        image_size: ({image_height}, {image_width})")
    print(f"        save_step: {save_step}")
    print(f"        frame_time_us: {frame_time_us}")
    print(f"        flow_dt_us: {flow_dt_us}")
    print(f"        test_size: {test_size}")
    
    try:
        # Create the dataset
        result = build_star8_flow_and_events(
            seq_name=seq_name,
            total_frames=total_frames,
            image_size=(image_height, image_width),
            save_step=save_step,
            frame_time_us=frame_time_us,
            outdir=data_root,
            start_ts_us=start_ts_us,
            flow_dt_us=flow_dt_us,
            face_color=face_color,
            test_size=test_size,
        )
        
        print(f"   ✅ Dataset created successfully!")
        print(f"      Train flow pairs: {result['num_flow_pairs_train']}")
        print(f"      Test flow pairs: {result['num_flow_pairs_test']}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_model_from_wandb(run_path, ckpt_dir="ckpt_dir"):
    """
    Load model and config from wandb run.
    
    Args:
        run_path: wandb run path in format "entity/project/run_id"
        ckpt_dir: directory where checkpoints are stored
        
    Returns:
        model: loaded model
        config: OmegaConf config object
        run_id: wandb run id
    """
    # Parse run path
    parts = run_path.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid run_path format. Expected 'entity/project/run_id', got '{run_path}'")
    
    entity, project, run_id = parts
    
    # Load wandb run config
    print(f"🔹 Loading config from wandb run: {run_path}")
    api = wandb.Api()
    run = api.run(run_path)
    config_dict = dict(run.config)
    config = OmegaConf.create(config_dict)
    
    print(f"🔹 Config loaded:")
    print(OmegaConf.to_yaml(config))
    
    # Find checkpoint
    ckpt_pattern = os.path.join(ckpt_dir, run_id, "model.ckpt")
    if not os.path.exists(ckpt_pattern):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_pattern}")
    
    print(f"🔹 Loading checkpoint from {ckpt_pattern}")
    
    # Load model
    model_name = config.model.name
    if model_name == "TinyIDEDEQIDO":
        model = TinyIDEDEQIDO(config.model)
    elif model_name == "IDEDEQIDO":
        model = IDEDEQIDO(config.model)
    elif model_name == "NanoIDEDEQIDO":
        model = NanoIDEDEQIDO(config.model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Load checkpoint
    ckpt = torch.load(ckpt_pattern, map_location='cpu', weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    
    print(f"✅ Model loaded successfully")
    print(f"   Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    return model, config, run_id


def create_datasets(config):
    """
    Create train and validation datasets based on config.
    
    IMPORTANT: This function explicitly forces preprocessing in memory for both
    train and validation sets, regardless of the original config settings.
    This is necessary because:
    - Data may have changed since training (e.g., regenerated with different parameters)
    - Evaluation should always use fresh preprocessing to match current data
    - In-memory processing avoids conflicts with saved preprocessed files
    
    The following flags are enforced:
    - in_memory: True (all preprocessing done in memory)
    - force_preprocess: True (ignore any saved preprocessed files)
    - do_not_save_preprocessed: True (don't save new preprocessed files)
    
    Args:
        config: OmegaConf config object
        
    Returns:
        train_set: Dataset for training set
        val_set: Dataset for validation/test set
    """
    print(f"🔹 Creating dataloaders...")
    
    # Get validation sequences and config from config
    val_sequences = []
    val_dataset_config = None
    
    if "validation" in config:
        for val_config in config.validation.values():
            if "dataset" in val_config and "val" in val_config.dataset:
                val_sequences.extend(val_config.dataset.val.get("seq", []))
                # Use the validation dataset config if available
                if val_dataset_config is None:
                    val_dataset_config = val_config.dataset.val
    
    # If no validation config, try to get from dataset.val.seq directly
    if len(val_sequences) == 0 and "val" in config.dataset:
        val_sequences = config.dataset.val.get("seq", [])
        val_dataset_config = config.dataset.val
    
    print(f"   Val sequences: {val_sequences}")
    
    # Training set - load all sequences except validation ones
    # But for evaluation, we want to load only training sequences
    train_sequences = []
    if hasattr(config.dataset.train, "seq") and len(config.dataset.train.seq) > 0:
        train_sequences = config.dataset.train.seq
    else:
        # If use_all_seqs is True, we need to find all sequences in data_root
        # For now, we'll just not exclude validation sequences
        pass
    
    print(f"   Train sequences config: use_all_seqs={config.dataset.train.get('use_all_seqs', False)}")
    
    # Force reprocessing in memory for evaluation (data may have changed)
    # Create a copy of train config and explicitly set preprocessing flags
    train_eval_config = OmegaConf.create(OmegaConf.to_container(config.dataset.train, resolve=True))
    train_eval_config.in_memory = True
    train_eval_config.force_preprocess = True
    train_eval_config.do_not_save_preprocessed = True
    
    # CRITICAL FIX: Disable random transforms during evaluation
    # Random transforms (hflip, vflip, random_crop) are for training augmentation only.
    # During evaluation, we need deterministic data that matches the canonical PNG GT masks.
    # Without this fix, random transforms modify masks during forced preprocessing,
    # causing mismatch between PNG GT masks and Batch GT masks (only ~72% overlap).
    train_eval_config.horizontal_flip = None
    train_eval_config.vertical_flip = None
    train_eval_config.random_crop = None
    
    print(f"   Preprocessing config: in_memory={train_eval_config.in_memory}, "
          f"force_preprocess={train_eval_config.force_preprocess}, "
          f"do_not_save_preprocessed={train_eval_config.do_not_save_preprocessed}")
    print(f"   Transforms disabled for eval: hflip={train_eval_config.horizontal_flip}, "
          f"vflip={train_eval_config.vertical_flip}, random_crop={train_eval_config.random_crop}")
    
    # Check and create training dataset if needed
    data_root = config.dataset.common.data_root
    print(f"\n🔹 Checking training dataset at {data_root}...")
    for seq in train_sequences:
        if not check_and_create_dataset(data_root, seq, config):
            raise RuntimeError(f"Failed to ensure dataset exists for sequence '{seq}'")
    
    train_set = assemble_dsec_sequences(
        config.dataset.common.data_root,
        include_seq=set(train_sequences) if len(train_sequences) > 0 else None,
        exclude_seq=set(val_sequences) if train_eval_config.get("exclude_val", False) else None,
        require_gt=True,
        config=train_eval_config,
        representation_type=config.dataset.get("representation_type", "voxel"),
        num_bins=config.dataset.get("num_voxel_bins", 5)
    )
    
    # Validation/Test set
    if len(val_sequences) > 0:
        # Load validation sequences from data_root
        print(f"   Loading validation set from {config.dataset.common.data_root}")
        
        # Merge train config with val config for validation set
        # Start with train config and override with val-specific settings
        val_config = OmegaConf.merge(config.dataset.train, val_dataset_config)
        
        # CRITICAL: Disable ALL random transforms for deterministic evaluation
        val_config.horizontal_flip = None  # Disable hflip augmentation
        val_config.vertical_flip = None    # Disable vflip augmentation
        val_config.random_crop = None      # Disable random crop augmentation
        val_config.load_gt = True  # Make sure GT is loaded
        val_config.concat_seq = True  # Always concatenate for evaluation
        
        # Force reprocessing in memory for evaluation (data may have changed)
        val_config.in_memory = True
        val_config.force_preprocess = True
        val_config.do_not_save_preprocessed = True
        
        print(f"   Val preprocessing config: in_memory={val_config.in_memory}, "
              f"force_preprocess={val_config.force_preprocess}, "
              f"do_not_save_preprocessed={val_config.do_not_save_preprocessed}")
        print(f"   Val transforms disabled: hflip={val_config.horizontal_flip}, "
              f"vflip={val_config.vertical_flip}, random_crop={val_config.random_crop}")
        
        # Check and create validation dataset if needed
        print(f"\n🔹 Checking validation dataset at {config.dataset.common.data_root}...")
        for seq in val_sequences:
            if not check_and_create_dataset(config.dataset.common.data_root, seq, config):
                raise RuntimeError(f"Failed to ensure dataset exists for sequence '{seq}'")
        
        val_set = assemble_dsec_sequences(
            config.dataset.common.data_root,
            include_seq=set(val_sequences),
            exclude_seq=None,
            require_gt=True,
            config=val_config,
            representation_type=config.dataset.get("representation_type", "voxel"),
            num_bins=config.dataset.get("num_voxel_bins", 5)
        )
    else:
        val_set = None
    
    print(f"✅ Datasets created:")
    print(f"   Train set size: {len(train_set)} samples")
    if val_set:
        if hasattr(val_set, 'datasets'):
            print(f"   Val set has {len(val_set.datasets)} sequence(s) with {len(val_set)} samples total")
        else:
            print(f"   Val set size: {len(val_set)} samples")
    
    return train_set, val_set


def evaluate_model(model, dataset, config, gpu_id=0):
    """
    Evaluate model on a dataset and collect predictions.
    Process samples individually without batching.
    
    Args:
        model: trained model
        dataset: Dataset object (not DataLoader)
        config: OmegaConf config object
        gpu_id: GPU device id
        
    Returns:
        predictions: list of prediction tensors
        ground_truths: list of ground truth flow tensors
        valid_masks: list of valid masks
        file_indices: list of file indices
        timestamps: list of timestamps
        metrics: dictionary of metrics
    """
    model.cuda(gpu_id)
    model.eval()
    
    predictions = []
    ground_truths = []
    valid_masks = []
    file_indices = []  # Track file indices for proper PNG matching
    timestamps = []    # Track timestamps for temporal analysis
    
    print(f"🔹 Evaluating on {len(dataset)} samples (processing individually)...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            # Get single sample from dataset
            sample = dataset[idx]
            
            # Move sample to GPU - handle both tensor and list types
            sample_gpu = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample_gpu[key] = value.unsqueeze(0).cuda(gpu_id)  # Add batch dimension
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    # Handle flow_gt which is a list of [flow_tensor, mask_tensor]
                    sample_gpu[key] = [v.unsqueeze(0).cuda(gpu_id) for v in value]
                else:
                    sample_gpu[key] = value
            
            # Forward pass
            out = model(sample_gpu)
            
            # Get prediction
            if isinstance(out, dict):
                pred = out.get("final_prediction", out.get("flow_prediction"))
            else:
                pred = out
            
            # Remove batch dimension and move to CPU
            pred = pred.squeeze(0).cpu()
            
            # Get ground truth and valid mask
            gt_key = "flow_gt_event_volume_new" if "flow_gt_event_volume_new" in sample else "flow_gt"
            valid_key = "flow_gt_event_volume_new_valid_mask" if "flow_gt_event_volume_new_valid_mask" in sample else "flow_gt_valid_mask"
            
            if gt_key in sample:
                if isinstance(sample[gt_key], list):
                    gt = sample[gt_key][0]  # First element is the flow tensor
                    valid = sample[gt_key][1]  # Second element is the valid mask
                else:
                    gt = sample[gt_key]
                    valid = sample.get(valid_key)
            else:
                gt = None
                valid = None
            
            # Get file indices and timestamps for correct PNG matching and temporal tracking
            # Note: flow_gt_event_volume_new corresponds to flow_png[index+1] in the dataloader
            file_idx = sample.get("file_index")
            timestamp = sample.get("timestamp")
            
            # Store results
            predictions.append(pred)
            if gt is not None:
                ground_truths.append(gt)
            if valid is not None:
                valid_masks.append(valid)
            if file_idx is not None:
                file_indices.append(torch.tensor(file_idx))
            else:   
                raise ValueError(f"file_index is missing in sample {idx}. It is required for correct PNG matching.")
            if timestamp is not None:
                timestamps.append(torch.tensor(timestamp))
    
    # Compute metrics if ground truth is available
    metrics = {}
    if len(ground_truths) > 0:
        print(f"🔹 Computing metrics...")
        # Stack all predictions and ground truths
        all_preds = torch.stack(predictions, dim=0)  # [N, 2, H, W]
        all_gts = torch.stack(ground_truths, dim=0)  # [N, 2, H, W]
        all_valid = torch.stack(valid_masks, dim=0) if len(valid_masks) > 0 else torch.ones_like(all_gts[:, :1])  # [N, 1, H, W]
        
        # Compute L1 and L2 errors
        error = all_preds - all_gts
        error_masked = error * all_valid
        
        l1_error = torch.abs(error_masked).sum(dim=1)
        l2_error = torch.sqrt((error_masked ** 2).sum(dim=1))
        
        num_valid = all_valid.sum()
        
        metrics["l1_mean"] = (l1_error.sum() / num_valid).item()
        metrics["l2_mean"] = (l2_error.sum() / num_valid).item()
        
        # Compute EPE (End Point Error) - Euclidean distance between predicted and GT flow
        epe = torch.sqrt(((all_preds - all_gts) ** 2).sum(dim=1))  # [B, H, W]
        epe_masked = epe * all_valid.squeeze(1)
        metrics["epe_mean"] = (epe_masked.sum() / num_valid).item()
        
        # Compute Angular Error (AE) - angle between predicted and GT flow vectors
        # AE = arccos((u1*u2 + v1*v2 + 1) / sqrt((u1^2 + v1^2 + 1) * (u2^2 + v2^2 + 1)))
        # where (u1, v1) is GT flow and (u2, v2) is predicted flow
        # Adding 1 to handle the case where both flows are zero (homogeneous coordinates)
        pred_u = all_preds[:, 0, :, :]  # [N, H, W]
        pred_v = all_preds[:, 1, :, :]  # [N, H, W]
        gt_u = all_gts[:, 0, :, :]      # [N, H, W]
        gt_v = all_gts[:, 1, :, :]      # [N, H, W]
        
        # Compute dot product in homogeneous coordinates
        dot_product = pred_u * gt_u + pred_v * gt_v + 1.0
        
        # Compute magnitudes in homogeneous coordinates
        pred_magnitude = torch.sqrt(pred_u**2 + pred_v**2 + 1.0)
        gt_magnitude = torch.sqrt(gt_u**2 + gt_v**2 + 1.0)
        
        # Compute cosine of angle
        cos_angle = dot_product / (pred_magnitude * gt_magnitude + 1e-8)
        
        # Clamp to [-1, 1] to avoid numerical issues with arccos
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        # Compute angular error in degrees
        angular_error = torch.acos(cos_angle) * 180.0 / np.pi  # [N, H, W]
        angular_error_masked = angular_error * all_valid.squeeze(1)
        metrics["ae_mean"] = (angular_error_masked.sum() / num_valid).item()
        
        # Compute 1PE (1-pixel error) - percentage of pixels with EPE > 1
        outlier_mask_1px = (epe > 1.0) & (all_valid.squeeze(1) > 0)
        num_outliers_1px = outlier_mask_1px.sum().item()
        metrics["1pe"] = (num_outliers_1px / num_valid.item()) * 100.0  # percentage
        
        # Compute 3PE (3-pixel error) - percentage of pixels with EPE > 3
        outlier_mask = (epe > 3.0) & (all_valid.squeeze(1) > 0)
        num_outliers = outlier_mask.sum().item()
        metrics["3pe"] = (num_outliers / num_valid.item()) * 100.0  # percentage
        
        metrics["num_samples"] = len(predictions)
        metrics["num_valid_pixels"] = num_valid.item()
        
        print(f"   L1 Error: {metrics['l1_mean']:.4f}")
        print(f"   L2 Error: {metrics['l2_mean']:.4f}")
        print(f"   EPE (End Point Error): {metrics['epe_mean']:.4f}")
        print(f"   AE (Angular Error): {metrics['ae_mean']:.4f}°")
        print(f"   1PE (1-pixel error): {metrics['1pe']:.2f}%")
        print(f"   3PE (3-pixel error): {metrics['3pe']:.2f}%")
    
    return predictions, ground_truths, valid_masks, file_indices, timestamps, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate toy dataset trained model")
    parser.add_argument(
        "--run_path",
        type=str,
        required=True,
        help="Wandb run path in format 'entity/project/run_id' (e.g., haraghi/toydataset-tinyIDNet-multiseed/fxd5fk65)"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="ckpt_dir",
        help="Directory where checkpoints are stored"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="toy_datasets/evaluations",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device id"
    )
    
    args = parser.parse_args()
    
    # Load model and config
    model, config, run_id = load_model_from_wandb(args.run_path, args.ckpt_dir)
    
    # Create datasets (not dataloaders - we'll process samples individually)
    train_set, val_set = create_datasets(config)
    
    # Evaluate on training set
    print(f"\n{'='*60}")
    print(f"Evaluating on TRAINING set")
    print(f"{'='*60}")
    train_preds, train_gts, train_masks, train_file_indices, train_timestamps, train_metrics = evaluate_model(
        model, train_set, config, args.gpu
    )
    
    # Evaluate on validation set if available
    val_preds, val_gts, val_masks, val_file_indices, val_timestamps, val_metrics = None, None, None, None, None, {}
    if val_set is not None:
        print(f"\n{'='*60}")
        print(f"Evaluating on VALIDATION set")
        print(f"{'='*60}")
        val_preds, val_gts, val_masks, val_file_indices, val_timestamps, val_metrics = evaluate_model(
            model, val_set, config, args.gpu
        )
    
    # Save results
    entity, project, _ = args.run_path.split("/")
    save_dir = os.path.join(args.output_dir, project, run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    out_path = os.path.join(save_dir, "predictions.pt")
    
    # Collect sequence names for later visualization
    train_sequences = []
    if hasattr(config.dataset.train, "seq") and len(config.dataset.train.seq) > 0:
        train_sequences = config.dataset.train.seq
    
    val_sequences = []
    if "validation" in config:
        for val_config in config.validation.values():
            if "dataset" in val_config and "val" in val_config.dataset:
                val_sequences.extend(val_config.dataset.val.get("seq", []))
    if len(val_sequences) == 0 and "val" in config.dataset:
        val_sequences = config.dataset.val.get("seq", [])
    
    results = {
        "train": {
            "predictions": train_preds,
            "ground_truths": train_gts,
            "valid_masks": train_masks,
            "file_indices": train_file_indices,  # For correct PNG matching
            "timestamps": train_timestamps,      # For temporal analysis
            "metrics": train_metrics,
        },
        "metadata": {
            "run_path": args.run_path,
            "run_id": run_id,
            "data_root": config.dataset.common.data_root,
            "train_sequences": train_sequences,
            "val_sequences": val_sequences,
            "representation_type": config.dataset.get("representation_type", "voxel"),
            "num_voxel_bins": config.dataset.get("num_voxel_bins", 5),
        },
    }
    
    if val_preds is not None:
        results["val"] = {
            "predictions": val_preds,
            "ground_truths": val_gts,
            "valid_masks": val_masks,
            "file_indices": val_file_indices,  # For correct PNG matching
            "timestamps": val_timestamps,      # For temporal analysis
            "metrics": val_metrics,
        }
    
    torch.save(results, out_path)
    
    # Save metrics to JSON log file
    metrics_log = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "run_path": args.run_path,
        "run_id": run_id,
        "data_root": config.dataset.common.data_root,
        "train_sequences": list(train_sequences) if train_sequences else "all_sequences",
        "val_sequences": list(val_sequences) if val_sequences else [],
        "representation_type": config.dataset.get("representation_type", "voxel"),
        "num_voxel_bins": config.dataset.get("num_voxel_bins", 5),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics if val_metrics else None,
    }
    
    metrics_path = os.path.join(save_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation completed!")
    print(f"   Predictions saved to: {out_path}")
    print(f"   Metrics saved to: {metrics_path}")
    print(f"   Metadata saved: data_root={config.dataset.common.data_root}")
    print(f"   Train sequences: {train_sequences if train_sequences else 'all sequences'}")
    print(f"   Val sequences: {val_sequences}")
    print(f"\nSummary:")
    print(f"  Train metrics:")
    for key, value in train_metrics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    if val_metrics:
        print(f"  Val metrics:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
