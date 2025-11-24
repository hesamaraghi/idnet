#!/usr/bin/env python3
"""
Wrapper script for wandb sweeps that generates a dataset with specific parameters
and then trains on it.

Usage in wandb sweep config:
  program: toy_datasets/generate_and_train.py
  parameters:
    dataset_save_step: {values: [20, 40, 60]}
    dataset_total_frames: {values: [1000, 2000]}
    training_delta_t_ms: {values: [40, 100]}
"""

import os
import sys
import subprocess
import argparse

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Generate dataset and train model")
    
    # Dataset generation script selection
    parser.add_argument("--shape_class", "--shape-class", type=str, default="star8",
                       dest="shape_class", 
                       choices=['star8', 'triangle', 'lissajous', 'multi_lissajous'],
                       help="Shape movement class (uses create_dataset_generic.py)")
    
    # Dataset generation parameters
    # Note: Using underscores to match wandb's parameter format
    parser.add_argument("--dataset_save_step", "--dataset-save-step", type=int, default=40, 
                       dest="dataset_save_step", help="Dataset save_step parameter")
    parser.add_argument("--dataset_total_frames", "--dataset-total-frames", type=int, default=2000,
                       dest="dataset_total_frames", help="Dataset total_frames parameter")
    parser.add_argument("--dataset_test_size", "--dataset-test-size", type=float, default=0.2,
                       dest="dataset_test_size", help="Dataset test_size parameter")
    parser.add_argument("--dataset_base_name", "--dataset-base-name", default="star8",
                       dest="dataset_base_name", help="Base name for dataset")
    parser.add_argument("--dataset_outdir", "--dataset-outdir", default="toy_datasets/data",
                       dest="dataset_outdir", help="Dataset output directory (flat structure)")
    
    # Training parameters (these will be passed through to training script)
    # NOTE: training_delta_t_ms is automatically calculated from dataset_save_step
    # Only specify it if you want to override the automatic calculation
    parser.add_argument("--training_delta_t_ms", "--training-delta-t-ms", type=int, default=None,
                       dest="training_delta_t_ms", 
                       help="Training delta_t_ms parameter (auto-calculated from save_step if not specified)")
    parser.add_argument("--training_num_bins", "--training-num-bins", type=int, default=None,
                       dest="training_num_bins", help="Training num_bins parameter")
    parser.add_argument("--training_epochs", "--training-epochs", type=int, default=None,
                       dest="training_epochs", help="Number of training epochs")
    parser.add_argument("--training_seed", "--training-seed", type=int, default=None,
                       dest="training_seed", help="Random seed for reproducibility")
    
    # Model architecture parameters
    parser.add_argument("--model_name", "--model-name", type=str, default=None,
                       dest="model_name", help="Model architecture (TinyIDEDEQIDO or NanoIDEDEQIDO)")
    parser.add_argument("--model_hidden_dim", "--model-hidden-dim", type=int, default=None,
                       dest="model_hidden_dim", help="Model hidden dimension (e.g., 8, 32)")
    parser.add_argument("--model_input_dim", "--model-input-dim", type=int, default=None,
                       dest="model_input_dim", help="Model input dimension for feature encoder (e.g., 4, 8)")
    parser.add_argument("--model_mask_channels", "--model-mask-channels", type=int, default=None,
                       dest="model_mask_channels", help="Number of channels in mask convolution (e.g., 8, 16, 32)")
    
    # Feature extraction hyperparameters
    parser.add_argument("--training_add_eigenvalues", "--training-add-eigenvalues", type=lambda x: x.lower() == 'true', default=None,
                       dest="training_add_eigenvalues", help="Add eigenvalue features (true/false)")
    parser.add_argument("--training_add_filter_values", "--training-add-filter-values", type=lambda x: x.lower() == 'true', default=None,
                       dest="training_add_filter_values", help="Add filter value features (true/false)")
    parser.add_argument("--training_filter_size", "--training-filter-size", type=int, default=None,
                       dest="training_filter_size", help="Harris corner detector filter size (e.g., 5, 7)")
    parser.add_argument("--training_normalize_voxel", "--training-normalize-voxel", type=lambda x: x.lower() == 'true' if x else None, default=None,
                       dest="training_normalize_voxel", help="Normalize voxel grid (true/false)")
    parser.add_argument("--training_tau", "--training-tau", type=int, default=None,
                       dest="training_tau", help="Time constant for recursive filtering in microseconds (e.g., 1000, 15000, 30000)")
    
    # Learning and optimization hyperparameters
    parser.add_argument("--training_lr", "--training-lr", type=float, default=None,
                       dest="training_lr", help="Learning rate (e.g., 1e-4, 1e-5)")
    parser.add_argument("--training_optimizer", "--training-optimizer", type=str, default=None,
                       dest="training_optimizer", help="Optimizer type (adam, adamw)")
    parser.add_argument("--training_batch_size", "--training-batch-size", type=int, default=None,
                       dest="training_batch_size", help="Training batch size (e.g., 3, 6, 16)")
    parser.add_argument("--training_random_crop", "--training-random-crop", type=str, default=None,
                       dest="training_random_crop", help="Random crop size (e.g., 'none', '192x192')")
    
    # Shape-specific parameters
    # Lissajous parameters
    parser.add_argument("--shape_type", "--shape-type", type=str, default="star",
                       dest="shape_type", choices=['circle', 'square', 'star', 'hexagon'],
                       help="Shape type for Lissajous")
    parser.add_argument("--freq_ratio_a", "--freq-ratio-a", type=int, default=3,
                       dest="freq_ratio_a", help="Lissajous frequency ratio numerator")
    parser.add_argument("--freq_ratio_b", "--freq-ratio-b", type=int, default=2,
                       dest="freq_ratio_b", help="Lissajous frequency ratio denominator")
    parser.add_argument("--rotation_speed", "--rotation-speed", type=float, default=2.0,
                       dest="rotation_speed", help="Rotation speed for Lissajous")
    
    args = parser.parse_args()
    
    # Auto-calculate training_delta_t_ms from dataset parameters if not specified
    # This ensures delta_t_ms matches the flow timestamp intervals
    # Assumes frame_time_us = 1000 (default in create_flow_from_movement.py)
    DEFAULT_FRAME_TIME_US = 1000
    if args.training_delta_t_ms is None:
        args.training_delta_t_ms = int(args.dataset_save_step * DEFAULT_FRAME_TIME_US / 1000)
        print(f"ℹ️  Auto-calculated training_delta_t_ms = {args.training_delta_t_ms} ms "
              f"(from save_step={args.dataset_save_step})")
    
    # Get script location for proper path resolution
    import pathlib
    script_dir = pathlib.Path(__file__).parent
    parent_dir = script_dir.parent
    
    print("="*80)
    print("STEP 1: Generating Dataset")
    print("="*80)
    
    # Generate dataset with auto-naming to avoid conflicts
    # Use create_dataset_generic.py (supports all shape types including star8)
    create_dataset_script = script_dir / "create_dataset_generic.py"
    dataset_cmd = [
        sys.executable, str(create_dataset_script),
        "--shape-class", args.shape_class,
        "--seq-name", args.dataset_base_name,
        "--auto-name",  # This will append a config hash
        "--total-frames", str(args.dataset_total_frames),
        "--save-step", str(args.dataset_save_step),
        "--test-size", str(args.dataset_test_size),
        "--outdir", args.dataset_outdir,
        "--sanity-check",  # Generate sanity check visualizations
    ]
    
    # Add shape-specific parameters
    if args.shape_class == 'lissajous':
        dataset_cmd.extend([
            "--shape-type", args.shape_type,
            "--freq-ratio-a", str(args.freq_ratio_a),
            "--freq-ratio-b", str(args.freq_ratio_b),
            "--rotation-speed", str(args.rotation_speed),
        ])
    
    print(f"Running: {' '.join(dataset_cmd)}")
    result = subprocess.run(dataset_cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check if dataset generation was skipped (already exists)
    if "skip] Dataset already exists" in result.stdout:
        print("\n✓ Using existing dataset (generation was skipped)")
        # Still need to extract the variant hash for training
        import re
        for line in result.stdout.split('\n'):
            if "Auto-naming enabled: Using variant directory" in line:
                match = re.search(r"Using variant directory '([^']+)'", line)
                if match:
                    config_hash = match.group(1)
                    break
        else:
            print("ERROR: Could not determine dataset variant directory")
            sys.exit(1)
        
        dataset_variant = f"variant_{config_hash}"
    else:
        # Dataset was generated, extract hash as before
        config_hash = None
        import re
        for line in result.stdout.split('\n'):
            if "Auto-naming enabled: Using variant directory" in line:
                match = re.search(r"Using variant directory '([^']+)'", line)
                if match:
                    config_hash = match.group(1)
                    break
        
        if not config_hash:
            print("ERROR: Could not determine dataset variant directory")
            print("Make sure the dataset generation script completed successfully")
            sys.exit(1)
        
        dataset_variant = f"variant_{config_hash}"

    
    # Sequence name remains the base name (no hash)
    seq_name = args.dataset_base_name
    seq_name_test = f"{seq_name}_test"
    
    # Compute full data_root with variant subdirectory
    data_root = os.path.join(args.dataset_outdir, dataset_variant)
    
    print(f"\n✓ Dataset generated:")
    print(f"   Variant: {dataset_variant} (hash: {config_hash})")
    print(f"   Data root: {data_root}")
    print(f"   Train sequence: {seq_name}")
    print(f"   Test sequence: {seq_name_test}")
    
    print("\n" + "="*80)
    print("STEP 2: Training Model")
    print("="*80)
    
    # Need to run training from the parent directory where idn module is available
    # (script_dir and parent_dir already defined at the top)
    
    # Build training command with overrides
    training_cmd = [
        sys.executable, "-m", "idn.train_toy_dataset",
    ]
    
    # Add training parameter overrides
    overrides = []
    
    # Override data_root to point to the variant directory
    overrides.append(f"dataset.common.data_root={data_root}")
    
    # Set wandb run name to include variant hash for easy identification
    run_name_suffix = f"-ds{args.dataset_save_step}-f{args.dataset_total_frames}-v{config_hash[:6]}"
    
    # Note: We'll log dataset generation params via wandb config in the Trainer
    # For now, pass them as environment variables
    os.environ['DATASET_GEN_SAVE_STEP'] = str(args.dataset_save_step)
    os.environ['DATASET_GEN_TOTAL_FRAMES'] = str(args.dataset_total_frames)
    os.environ['DATASET_GEN_TEST_SIZE'] = str(args.dataset_test_size)
    os.environ['DATASET_GEN_VARIANT_HASH'] = config_hash
    
    # training_delta_t_ms is always set (either from args or auto-calculated)
    overrides.append(f"dataset.train.delta_t_ms={args.training_delta_t_ms}")
    
    # Model architecture parameters
    if args.model_name is not None:
        overrides.append(f"model.name={args.model_name}")
    if args.model_hidden_dim is not None:
        overrides.append(f"model.hidden_dim={args.model_hidden_dim}")
    if args.model_input_dim is not None:
        overrides.append(f"model.input_dim={args.model_input_dim}")
    if args.model_mask_channels is not None:
        overrides.append(f"model.mask_channels={args.model_mask_channels}")
    
    if args.training_num_bins is not None:
        overrides.append(f"dataset.num_voxel_bins={args.training_num_bins}")
        overrides.append(f"data_loader.common.num_voxel_bins={args.training_num_bins}")
    if args.training_epochs is not None:
        overrides.append(f"num_epoch={args.training_epochs}")
    
    # Feature extraction hyperparameters
    # Note: These automatically propagate to model and validation configs via references
    # e.g., model.add_eigenvalues: ${dataset.train.add_eigenvalues}
    if args.training_add_eigenvalues is not None:
        overrides.append(f"dataset.train.add_eigenvalues={str(args.training_add_eigenvalues).lower()}")
    if args.training_add_filter_values is not None:
        overrides.append(f"dataset.train.add_filter_values={str(args.training_add_filter_values).lower()}")
    if args.training_filter_size is not None:
        overrides.append(f"dataset.train.filter_size={args.training_filter_size}")
    if args.training_normalize_voxel is not None:
        overrides.append(f"dataset.train.normalize_voxel={str(args.training_normalize_voxel).lower()}")
    if args.training_tau is not None:
        overrides.append(f"dataset.train.tau={args.training_tau}")
    
    # Learning and optimization hyperparameters
    if args.training_lr is not None:
        overrides.append(f"optim.lr={args.training_lr}")
    if args.training_optimizer is not None:
        overrides.append(f"optim.optimizer={args.training_optimizer}")
    if args.training_batch_size is not None:
        overrides.append(f"data_loader.train.args.batch_size={args.training_batch_size}")
    if args.training_random_crop is not None:
        if args.training_random_crop.lower() == "none":
            overrides.append(f"dataset.train.random_crop=null")
        else:
            # Parse format like "192x192" to [192, 192]
            crop_size = args.training_random_crop.split('x')
            if len(crop_size) == 2:
                overrides.append(f"dataset.train.random_crop=[{crop_size[0]},{crop_size[1]}]")
    
    # Add seed to run name for identification (actual seeding handled by PyTorch defaults)
    # Note: To implement actual seeding, would need to modify the training script
    if args.training_seed is not None:
        run_name_suffix = f"{run_name_suffix}-seed{args.training_seed}"
        # Store seed in environment for potential use in training script
        os.environ['TRAINING_SEED'] = str(args.training_seed)
    
    # Add the final run name (after seed has been appended if applicable)
    overrides.append(f"wandb.run_name={run_name_suffix}")
    
    # Override validation sequences to use the generated dataset
    # IMPORTANT: Only override the seq lists, not the entire config sections
    # This preserves other settings like delta_t_ms, in_memory, etc.
    overrides.append(f"validation.nonrec.dataset.train.seq=[{seq_name}]")
    overrides.append(f"validation.nonrec.dataset.val.seq=[{seq_name}_test]")
    
    # Also set the main dataset sequences
    overrides.append(f"dataset.train.seq=[{seq_name}]")
    overrides.append(f"dataset.val.seq=[{seq_name}_test]")
    
    training_cmd.extend(overrides)
    
    print(f"Running: {' '.join(training_cmd)}")
    print(f"Overrides: {overrides}")
    print(f"Working directory: {parent_dir}")
    
    # Run training and show output in real-time
    # Note: Not using capture_output to allow real-time output
    # Change to parent directory so idn module can be imported
    result = subprocess.run(training_cmd, check=True, cwd=parent_dir)
    
    print("\n" + "="*80)
    print("✓ Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
