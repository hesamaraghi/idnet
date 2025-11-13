"""
Evaluation script for toy dataset trained models.

Usage:
    python toy_datasets/eval_toy_dataset.py --run_path haraghi/toydataset-tinyIDNet-multiseed/fxd5fk65
"""

import os
import sys
import argparse
from glob import glob
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
from idn.model.idedeq import TinyIDEDEQIDO, IDEDEQIDO
from idn.utils.helper_functions import move_batch_to_cuda


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


def create_dataloaders(config):
    """
    Create train and validation dataloaders based on config.
    
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
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation/test set
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
    
    print(f"   Preprocessing config: in_memory={train_eval_config.in_memory}, "
          f"force_preprocess={train_eval_config.force_preprocess}, "
          f"do_not_save_preprocessed={train_eval_config.do_not_save_preprocessed}")
    
    train_set = assemble_dsec_sequences(
        config.dataset.common.data_root,
        include_seq=set(train_sequences) if len(train_sequences) > 0 else None,
        exclude_seq=set(val_sequences) if train_eval_config.get("exclude_val", False) else None,
        require_gt=True,
        config=train_eval_config,
        representation_type=config.dataset.get("representation_type", "voxel"),
        num_bins=config.dataset.get("num_voxel_bins", 5)
    )
    
    batch_size = config.data_loader.train.args.batch_size
    train_loader = DataLoader(
        train_set,
        collate_fn=train_collate,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        num_workers=0,
    )
    
    # Validation/Test set
    if len(val_sequences) > 0:
        # Load validation sequences from data_root
        print(f"   Loading validation set from {config.dataset.common.data_root}")
        
        # Merge train config with val config for validation set
        # Start with train config and override with val-specific settings
        val_config = OmegaConf.merge(config.dataset.train, val_dataset_config)
        val_config.horizontal_flip = 0.0  # Disable augmentation for validation
        val_config.vertical_flip = 0.0
        val_config.load_gt = True  # Make sure GT is loaded
        val_config.concat_seq = True  # Always concatenate for evaluation
        
        # Force reprocessing in memory for evaluation (data may have changed)
        val_config.in_memory = True
        val_config.force_preprocess = True
        val_config.do_not_save_preprocessed = True
        
        print(f"   Val preprocessing config: in_memory={val_config.in_memory}, "
              f"force_preprocess={val_config.force_preprocess}, "
              f"do_not_save_preprocessed={val_config.do_not_save_preprocessed}")
        
        val_set = assemble_dsec_sequences(
            config.dataset.common.data_root,
            include_seq=set(val_sequences),
            exclude_seq=None,
            require_gt=True,
            config=val_config,
            representation_type=config.dataset.get("representation_type", "voxel"),
            num_bins=config.dataset.get("num_voxel_bins", 5)
        )
        
        val_batch_size = config.data_loader.get("val", config.data_loader.train).args.batch_size
        val_loader = DataLoader(
            val_set,
            collate_fn=train_collate,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_loader = None
    
    print(f"✅ Dataloaders created:")
    print(f"   Train set size: {len(train_set)} samples")
    if val_loader:
        if hasattr(val_set, 'datasets'):
            print(f"   Val set has {len(val_set.datasets)} sequence(s) with {len(val_set)} samples total")
        else:
            print(f"   Val set size: {len(val_set)} samples")
    
    return train_loader, val_loader


def evaluate_model(model, data_loader, config, gpu_id=0):
    """
    Evaluate model on a dataset and collect predictions.
    
    Args:
        model: trained model
        data_loader: DataLoader to evaluate on
        config: OmegaConf config object
        gpu_id: GPU device id
        
    Returns:
        predictions: list of prediction tensors
        ground_truths: list of ground truth flow tensors
        valid_masks: list of valid masks
        metrics: dictionary of metrics
    """
    model.cuda(gpu_id)
    model.eval()
    
    predictions = []
    ground_truths = []
    valid_masks = []
    
    print(f"🔹 Evaluating on {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = move_batch_to_cuda(batch, gpu_id)
            
            # Forward pass
            out = model(batch)
            
            # Get prediction
            if isinstance(out, dict):
                pred = out.get("final_prediction", out.get("flow_prediction"))
            else:
                pred = out
            
            # Get ground truth and valid mask
            gt = batch.get("flow_gt_event_volume_new", batch.get("flow_gt"))
            valid = batch.get("flow_gt_event_volume_new_valid_mask", batch.get("flow_gt_valid_mask"))
            
            # Store results
            predictions.append(pred.cpu())
            if gt is not None:
                ground_truths.append(gt.cpu())
            if valid is not None:
                valid_masks.append(valid.cpu())
    
    # Compute metrics if ground truth is available
    metrics = {}
    if len(ground_truths) > 0:
        print(f"🔹 Computing metrics...")
        all_preds = torch.cat(predictions, dim=0)
        all_gts = torch.cat(ground_truths, dim=0)
        all_valid = torch.cat(valid_masks, dim=0) if len(valid_masks) > 0 else torch.ones_like(all_gts[:, :1])
        
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
        print(f"   1PE (1-pixel error): {metrics['1pe']:.2f}%")
        print(f"   3PE (3-pixel error): {metrics['3pe']:.2f}%")
    
    return predictions, ground_truths, valid_masks, metrics


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
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Evaluate on training set
    print(f"\n{'='*60}")
    print(f"Evaluating on TRAINING set")
    print(f"{'='*60}")
    train_preds, train_gts, train_masks, train_metrics = evaluate_model(
        model, train_loader, config, args.gpu
    )
    
    # Evaluate on validation set if available
    val_preds, val_gts, val_masks, val_metrics = None, None, None, {}
    if val_loader is not None:
        print(f"\n{'='*60}")
        print(f"Evaluating on VALIDATION set")
        print(f"{'='*60}")
        val_preds, val_gts, val_masks, val_metrics = evaluate_model(
            model, val_loader, config, args.gpu
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
            "metrics": val_metrics,
        }
    
    torch.save(results, out_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Evaluation completed!")
    print(f"   Results saved to: {out_path}")
    print(f"   Metadata saved: data_root={config.dataset.common.data_root}")
    print(f"   Train sequences: {train_sequences if train_sequences else 'all sequences'}")
    print(f"   Val sequences: {val_sequences}")
    print(f"\nSummary:")
    print(f"  Train metrics: {train_metrics}")
    if val_metrics:
        print(f"  Val metrics: {val_metrics}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
