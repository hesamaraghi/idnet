#!/usr/bin/env python3
"""
Hydra-based wrapper for wandb sweeps that generates a dataset and trains on it.

This is the Hydra equivalent of generate_and_train.py, using Hydra's config
management instead of argparse. It provides cleaner syntax and better config
composition for sweeps.

Usage in wandb sweep config:
  program: toy_datasets/generate_and_train_hydra.py
  parameters:
    dataset.save_step: {values: [20, 40, 60]}
    dataset.total_frames: {values: [1000, 2000]}
    training.delta_t_ms: {values: [40, 100]}

Direct usage:
    # Use default config and override parameters
    python toy_datasets/generate_and_train_hydra.py \
      dataset.save_step=40 dataset.total_frames=2000 \
      training.add_eigenvalues=true training.filter_size=7
    
    # Use specific shape config
    python toy_datasets/generate_and_train_hydra.py \
      --config-name=lissajous \
      dataset.freq_ratio_a=7 dataset.freq_ratio_b=5
    
    # Override model architecture
    python toy_datasets/generate_and_train_hydra.py \
      model.name=NanoIDEDEQIDO model.hidden_dim=8 model.input_dim=4
"""

import os
import sys
import subprocess
import hydra
from omegaconf import OmegaConf, DictConfig
from pathlib import Path

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toy_datasets.dataset_generator import DatasetGenerator


@hydra.main(version_base=None, config_path="config", config_name="generate_and_train")
def main(cfg: DictConfig):
    """Main function for dataset generation + training pipeline using Hydra.
    
    The @hydra.main decorator automatically:
    - Loads config from YAML with proper composition
    - Merges CLI overrides (nested keys supported: dataset.save_step=40)
    - Handles config groups for different shapes/models
    
    Args:
        cfg: Hydra DictConfig with 'dataset' and 'training' sections
    """
    
    # Disable struct mode for flexibility
    OmegaConf.set_struct(cfg, False)
    
    # Print configuration
    print("="*80)
    print("Generate & Train Pipeline Configuration")
    print("="*80)
    print(OmegaConf.to_yaml(cfg))
    print("="*80)
    
    # Auto-calculate training_delta_t_ms if not specified
    if cfg.training.get('delta_t_ms') is None:
        frame_time_us = cfg.dataset.get('frame_time_us', 1000)
        cfg.training.delta_t_ms = int(cfg.dataset.save_step * frame_time_us / 1000)
        print(f"\nℹ️  Auto-calculated training.delta_t_ms = {cfg.training.delta_t_ms} ms "
              f"(from dataset.save_step={cfg.dataset.save_step}, frame_time_us={frame_time_us})")
    
    # ========================================================================
    # STEP 1: Generate Dataset
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Generating Dataset")
    print("="*80)
    
    # Override seq_name with base_name for consistent naming
    if cfg.dataset.get('base_name'):
        cfg.dataset.seq_name = cfg.dataset.base_name
    
    # Generate dataset directly using DatasetGenerator (same as generate_dataset_hydra.py)
    try:
        generator = DatasetGenerator(cfg.dataset)
        result = generator.generate()
        result = generator.generate_animations(result)
        result = generator.generate_sanity_checks(result)
    except Exception as e:
        print(f"\n❌ Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Extract dataset information from result
    if result.get('skipped', False):
        print("\n✓ Using existing dataset (generation was skipped)")
    
    # Get paths from result
    seq_name = cfg.dataset.base_name
    seq_name_test = f"{seq_name}_test"
    
    # Determine data_root from the generated flow directory path
    # result['flow_dir_train'] looks like: outdir/variant_hash/train_optical_flow/seq_name/flow/forward
    # We need: outdir/variant_hash (4 levels up from forward)
    flow_dir_train = result['flow_dir_train']
    data_root = str(Path(flow_dir_train).parent.parent.parent.parent)
    dataset_variant = Path(data_root).name
    config_hash = dataset_variant.replace('variant_', '')
    
    print(f"\n✓ Dataset ready:")
    print(f"   Variant: {dataset_variant} (hash: {config_hash})")
    print(f"   Data root: {data_root}")
    print(f"   Train sequence: {seq_name}")
    print(f"   Test sequence: {seq_name_test}")
    print(f"   Train flow pairs: {result['num_flow_pairs_train']}")
    if result['num_flow_pairs_test'] > 0:
        print(f"   Test flow pairs: {result['num_flow_pairs_test']}")
    
    # ========================================================================
    # STEP 2: Train Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Training Model")
    print("="*80)
    
    # Get parent directory for training
    parent_dir = Path(__file__).parent.parent
    
    # Build training command with Hydra overrides
    training_cmd = [sys.executable, "-m", "idn.train_toy_dataset"]
    
    training_overrides = []
    
    # Override data_root to point to the variant directory
    training_overrides.append(f"dataset.common.data_root={data_root}")
    
    # Set wandb run name to include variant hash
    run_name_suffix = f"-ds{cfg.dataset.save_step}-f{cfg.dataset.total_frames}-v{config_hash[:6]}"
    
    # Pass dataset generation params as environment variables for logging
    os.environ['DATASET_GEN_SAVE_STEP'] = str(cfg.dataset.save_step)
    os.environ['DATASET_GEN_TOTAL_FRAMES'] = str(cfg.dataset.total_frames)
    os.environ['DATASET_GEN_TEST_SIZE'] = str(cfg.dataset.test_size)
    os.environ['DATASET_GEN_VARIANT_HASH'] = config_hash
    
    # Training delta_t_ms (always set)
    training_overrides.append(f"dataset.train.delta_t_ms={cfg.training.delta_t_ms}")
    
    # Model architecture parameters
    if cfg.model.get('name'):
        training_overrides.append(f"model.name={cfg.model.name}")
    if cfg.model.get('hidden_dim') is not None:
        training_overrides.append(f"model.hidden_dim={cfg.model.hidden_dim}")
    if cfg.model.get('input_dim') is not None:
        training_overrides.append(f"model.input_dim={cfg.model.input_dim}")
    if cfg.model.get('mask_channels') is not None:
        training_overrides.append(f"model.mask_channels={cfg.model.mask_channels}")
    
    # Training hyperparameters
    if cfg.training.get('num_bins') is not None:
        training_overrides.append(f"dataset.num_voxel_bins={cfg.training.num_bins}")
        training_overrides.append(f"data_loader.common.num_voxel_bins={cfg.training.num_bins}")
    if cfg.training.get('epochs') is not None:
        training_overrides.append(f"num_epoch={cfg.training.epochs}")
    
    # Feature extraction hyperparameters
    if cfg.training.get('add_eigenvalues') is not None:
        training_overrides.append(f"dataset.train.add_eigenvalues={str(cfg.training.add_eigenvalues).lower()}")
    if cfg.training.get('add_filter_values') is not None:
        training_overrides.append(f"dataset.train.add_filter_values={str(cfg.training.add_filter_values).lower()}")
    if cfg.training.get('filter_size') is not None:
        training_overrides.append(f"dataset.train.filter_size={cfg.training.filter_size}")
    if cfg.training.get('normalize_voxel') is not None:
        training_overrides.append(f"dataset.train.normalize_voxel={str(cfg.training.normalize_voxel).lower()}")
    if cfg.training.get('tau') is not None:
        training_overrides.append(f"dataset.train.tau={cfg.training.tau}")
    
    # Optimization hyperparameters
    if cfg.training.get('lr') is not None:
        training_overrides.append(f"optim.lr={cfg.training.lr}")
    if cfg.training.get('optimizer'):
        training_overrides.append(f"optim.optimizer={cfg.training.optimizer}")
    if cfg.training.get('batch_size') is not None:
        training_overrides.append(f"data_loader.train.args.batch_size={cfg.training.batch_size}")
    
    # Random crop
    if cfg.training.get('random_crop'):
        if cfg.training.random_crop.lower() == "none":
            training_overrides.append(f"dataset.train.random_crop=null")
        else:
            # Parse "192x192" to [192, 192]
            crop_size = cfg.training.random_crop.split('x')
            if len(crop_size) == 2:
                training_overrides.append(f"dataset.train.random_crop=[{crop_size[0]},{crop_size[1]}]")
    
    # Seed handling
    if cfg.training.get('seed') is not None:
        run_name_suffix = f"{run_name_suffix}-seed{cfg.training.seed}"
        os.environ['TRAINING_SEED'] = str(cfg.training.seed)
    
    # Set final wandb run name
    training_overrides.append(f"wandb.run_name={run_name_suffix}")
    
    # Override validation sequences
    training_overrides.append(f"validation.nonrec.dataset.train.seq=[{seq_name}]")
    training_overrides.append(f"validation.nonrec.dataset.val.seq=[{seq_name_test}]")
    training_overrides.append(f"dataset.train.seq=[{seq_name}]")
    training_overrides.append(f"dataset.val.seq=[{seq_name_test}]")
    
    training_cmd.extend(training_overrides)
    
    print(f"\nRunning: {' '.join(training_cmd)}")
    print(f"Overrides: {training_overrides}")
    print(f"Working directory: {parent_dir}")
    
    # Run training with real-time output
    result = subprocess.run(training_cmd, check=True, cwd=parent_dir)
    
    print("\n" + "="*80)
    print("✓ Training Complete!")
    print("="*80)
    
    return {
        'dataset_variant': dataset_variant,
        'config_hash': config_hash,
        'data_root': data_root,
        'seq_name': seq_name,
        'seq_name_test': seq_name_test,
    }


if __name__ == "__main__":
    main()
