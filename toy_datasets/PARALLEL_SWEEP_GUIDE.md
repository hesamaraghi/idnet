# Parallel Dataset Generation & Training Guide

This guide explains how to run wandb sweeps that generate datasets with different parameters and train models on them, all in parallel without conflicts.

## The Problem

When running parallel experiments with different dataset parameters (e.g., varying `save_step`, `total_frames`, feature extraction settings), multiple processes would try to write to the same directories, causing:
- Dataset overwrites
- Metadata conflicts
- Incorrect training/validation splits
- Race conditions in parallel runs

## The Solution

### Automatic Variant Directories with Config Hashing

The `--auto-name` flag automatically creates a unique **variant directory** based on configuration hash:

```bash
python toy_datasets/create_flow_from_movement.py \
    --seq-name star8 \
    --auto-name \
    --save-step 40 \
    --total-frames 2000
```

This generates a dataset in `variant_a3f5b2c1/star8/` (where `a3f5b2c1` is the config hash).

**Key features:**
- ✅ Different configs → Different variant directories (no conflicts)
- ✅ Same configs → Same variant directory (automatic reuse)
- ✅ Parallel-safe: Checks if dataset exists before regenerating
- ✅ Hash based on: `total_frames`, `image_size`, `save_step`, `frame_time_us`, `flow_dt_us`, `test_size`, `face_color`
- ✅ Portable: No hardcoded absolute paths (works on any system/cluster)

## Usage Patterns

### Pattern 1: Manual Dataset Generation + Training

```bash
# Generate dataset
python toy_datasets/create_flow_from_movement.py \
    --seq-name star8 \
    --auto-name \
    --save-step 40

# This creates: toy_datasets/data/star8/variant_a3f5b2c1/star8/
#           and toy_datasets/data/star8/variant_a3f5b2c1/star8_test/

# Train on it
python -m idn.train_toy_dataset \
    dataset.common.data_root=toy_datasets/data/star8/variant_a3f5b2c1 \
    dataset.train.seq=[star8] \
    dataset.val.seq=[star8_test]
```

### Pattern 2: Automated with Wrapper Script (Recommended)

```bash
python toy_datasets/generate_and_train.py \
    --dataset-save-step 40 \
    --dataset-total-frames 2000 \
    --training-num-bins 15 \
    --training-add-eigenvalues true \
    --training-filter-size 7 \
    --training-tau 15000
```

This script:
1. Generates dataset with `--auto-name` (creates unique variant directory)
2. Captures the variant hash from output
3. Automatically configures training to use `variant_{hash}` as data_root
4. Training `delta_t_ms` is **auto-calculated** from `save_step` (no need to specify separately)
5. Handles all config overrides via OmegaConf interpolation

### Pattern 3: Wandb Sweep (Parallel Experiments)

**Step 1: Create sweep configuration** (use provided templates):

```yaml
# toy_datasets/sweep_config_example.yaml
program: toy_datasets/generate_and_train.py
method: grid
metric:
  name: val/epe
  goal: minimize

parameters:
  dataset_save_step:
    values: [20, 40, 60, 80, 100]
  
  dataset_total_frames:
    value: 2000
  
  # NOTE: training_delta_t_ms is AUTO-CALCULATED from dataset_save_step
  # No need to specify it as a separate parameter!
  
  training_num_bins:
    values: [5, 10, 15]
  
  training_seed:
    values: [42, 123, 456]  # Multiple seeds for statistical robustness
  
  # Feature extraction hyperparameters
  training_add_eigenvalues:
    values: [true, false]
  
  training_add_filter_values:
    values: [true, false]
  
  training_filter_size:
    values: [5, 7]
  
  training_tau:
    values: [1000, 15000, 30000]
```

**Available sweep templates:**
- `sweep_config_example.yaml` - Full grid search (2,160 experiments)
- `sweep_config_features.yaml` - Feature-focused (24 experiments)
- `sweep_config_minimal.yaml` - Quick testing (8 experiments)

**Step 2: Initialize sweep:**
```bash
wandb sweep sweep_config.yaml
# Returns: wandb: Created sweep with ID: abc123def
```

**Step 3: Run multiple agents in parallel:**
```bash
# Terminal 1
wandb agent your-entity/your-project/abc123def

# Terminal 2
wandb agent your-entity/your-project/abc123def

# Terminal 3
wandb agent your-entity/your-project/abc123def

# ... as many as your cluster can handle
```

Each agent will:
1. Get a unique parameter combination from wandb
2. Check if dataset already exists (via metadata file)
3. Skip regeneration if exists, or generate new dataset in `variant_{hash}` directory
4. Train on that dataset with auto-configured paths
5. Log results to wandb with comprehensive tags

**No conflicts!** Each parameter combination:
- Generates a unique variant directory
- Skips regeneration if already exists (parallel-safe)
- Runs independently without interfering with other agents

## How Config Hashing Works

The hash is computed from dataset generation parameters:

```python
config_dict = {
    'total_frames': 2000,
    'image_size': (256, 256),
    'save_step': 40,
    'frame_time_us': 1000,
    'flow_dt_us': 40000,  # Auto-calculated from save_step
    'test_size': 0.2,
    'start_ts_us': 0,
    'face_color': 'black',
}
# Hash: e450c505
```

**Examples:**

| Parameters | Hash | Variant Directory | Data Root |
|---|---|---|---|
| `total_frames=2000, save_step=40` | `e450c505` | `variant_e450c505` | `toy_datasets/data/star8/variant_e450c505` |
| `total_frames=2000, save_step=60` | `d7e9f4a2` | `variant_d7e9f4a2` | `toy_datasets/data/star8/variant_d7e9f4a2` |
| `total_frames=1000, save_step=40` | `f1c2d3e4` | `variant_f1c2d3e4` | `toy_datasets/data/star8/variant_f1c2d3e4` |

**Important:** Training parameters (like `add_eigenvalues`, `filter_size`, `tau`) do NOT affect the dataset hash. Multiple training runs with different hyperparameters can share the same dataset variant!

## Dataset Metadata & Parallel Safety

Each generated dataset includes `dataset_metadata.json` with all parameters:

```json
{
  "seq_name": "star8",
  "total_frames": 2000,
  "save_step": 40,
  "image_size": [256, 256],
  "flow_dt_us": 40000,
  "frame_time_us": 1000,
  "test_size": 0.2,
  "split_start_frame": 1600,
  "num_flow_pairs_train": 80,
  "num_flow_pairs_test": 20,
  ...
}
```

### Parallel Run Safety

The system prevents conflicts through metadata checking:

1. **Before generation**: Check if `dataset_metadata.json` exists
2. **If exists**: Skip regeneration, reuse existing dataset
3. **If missing or incomplete**: Clean up and regenerate

This means:
- ✅ Multiple parallel agents can request the same dataset configuration
- ✅ First agent generates it, subsequent agents reuse it
- ✅ No race conditions or overwrites
- ✅ Efficient: Dataset generated only once per unique configuration

Metadata is automatically:
- ✅ Saved during dataset generation
- ✅ Used for existence checking
- ✅ Logged to wandb via environment variables for full reproducibility

## Training Configuration

### For In-Memory Training (Recommended for Sweeps)

```yaml
dataset:
  train:
    in_memory: true
    force_preprocess: true
    do_not_save_preprocessed: true
```

**Why?**
- Each training process keeps data in its own memory
- No disk I/O conflicts
- Faster training (after initial preprocessing)
- Clean: no leftover preprocessed files

### For Disk-Cached Training

If you want to cache preprocessed data (e.g., for reusing across runs):

```yaml
dataset:
  train:
    in_memory: false
    force_preprocess: false
    do_not_save_preprocessed: false
```

**Note:** With unique dataset names, preprocessed files won't conflict anyway!

## Directory Structure Example

After running a sweep with 3 different dataset configurations:

```
toy_datasets/data/star8/
├── variant_e450c505/         # save_step=40, total_frames=2000
│   ├── train_optical_flow/
│   │   ├── star8/
│   │   │   ├── dataset_metadata.json
│   │   │   └── flow/
│   │   │       ├── forward/
│   │   │       │   ├── 000000.png
│   │   │       │   ├── 000001.png
│   │   │       │   └── ...
│   │   │       ├── forward_timestamps.txt
│   │   │       └── sanity/         # Sanity check visualizations
│   │   │           ├── sanity_000000.png
│   │   │           └── ...
│   │   └── star8_test/
│   │       ├── dataset_metadata.json
│   │       └── flow/
│   └── train_events/
│       ├── star8/
│       │   └── events/
│       │       └── left/
│       │           ├── events.h5
│       │           └── rectify_map.h5
│       └── star8_test/
│           └── events/
│               └── left/
├── variant_d7e9f4a2/         # save_step=60, total_frames=2000
│   ├── train_optical_flow/
│   ├── train_events/
│   └── ...
└── variant_f1c2d3e4/         # save_step=40, total_frames=1000
    ├── train_optical_flow/
    ├── train_events/
    └── ...
```

**Key points:**
- Each variant is self-contained in its own directory
- Sequence names stay consistent (`star8`, `star8_test`)
- Training script only needs to know the variant directory
- Sanity check visualizations included for verification

## Cleanup

To remove all generated datasets for a sweep:

```bash
# Remove all variant directories
rm -rf toy_datasets/data/star8/variant_*

# Remove specific variant
rm -rf toy_datasets/data/star8/variant_e450c505

# Keep specific variants, remove others
cd toy_datasets/data/star8
ls -d variant_* | grep -v "variant_e450c505\|variant_d7e9f4a2" | xargs rm -rf
```

## Troubleshooting

### Issue: "Sequence not found during training"

**Cause:** The training config points to wrong `data_root` or variant directory.

**Solution:** The `generate_and_train.py` wrapper script automatically handles this. If you're running manually, make sure to set the correct data_root:

```bash
python -m idn.train_toy_dataset \
    dataset.common.data_root=toy_datasets/data/star8/variant_e450c505 \
    dataset.train.seq=[star8] \
    dataset.val.seq=[star8_test]
```

### Issue: "Dataset already exists" message

**Cause:** You're trying to generate a dataset that already exists (same parameters).

**Solution:** This is **expected behavior** and a **feature**! The system:
- ✅ Detects existing dataset via `dataset_metadata.json`
- ✅ Skips regeneration (saves time)
- ✅ Reuses existing data (efficient for parallel runs)

To force regeneration, manually delete the variant directory first.

### Issue: "Path doubling" (e.g., `toy_datasets/toy_datasets/data/star8`)

**Cause:** Running script from `toy_datasets/` directory with default relative paths.

**Solution:** Fixed in latest version! The script now:
- Uses relative paths from script location
- No hardcoded absolute paths
- Works on any system/cluster

### Issue: "Out of disk space"

**Cause:** Many sweep runs generate many variant directories.

**Solution:** 
1. Use `in_memory=True` with `do_not_save_preprocessed=True` to avoid preprocessed file bloat
2. Clean up old variant directories between sweeps
3. Multiple training runs can share the same variant (training hyperparameters don't affect dataset hash)

### Issue: "training_delta_t_ms doesn't match dataset"

**Cause:** Manually specified `delta_t_ms` doesn't match `save_step`.

**Solution:** **Don't specify `training_delta_t_ms` in sweep config!** It's now auto-calculated:
```python
delta_t_ms = save_step * frame_time_us / 1000
# For save_step=40: delta_t_ms=40
# For save_step=60: delta_t_ms=60
```

This ensures consistency between dataset generation and training.

## Best Practices

1. **Always use `--auto-name` for sweeps** - Creates unique variant directories
2. **Don't specify `training_delta_t_ms`** - Auto-calculated from `save_step` for consistency
3. **Use in-memory training** (`in_memory=True`) for faster parallel execution
4. **Let parallel agents share datasets** - Same config → same variant → automatic reuse
5. **Monitor disk usage** - Clean up old variants between sweeps
6. **Use provided sweep templates** - Start with `sweep_config_minimal.yaml` for testing
7. **Leverage sanity checks** - Verify flow generation visually before long training runs
8. **Tag your runs** - Auto-generated tags make filtering in wandb easy

## Important Configuration Notes

### Auto-Calculated Parameters

These parameters are **automatically calculated** and should NOT be set independently:

1. **`training_delta_t_ms`**: Calculated as `save_step * frame_time_us / 1000`
   - Ensures training event windows match flow timestamp intervals
   - Prevents "time window exceeds available data" errors

2. **`flow_dt_us`**: Calculated as `save_step * frame_time_us`
   - Ensures flow timestamps align with event data timestamps
   - Auto-calculated if you change `save_step` but leave `flow_dt_us` at default

### Config Interpolation

The base config uses OmegaConf interpolation for automatic propagation:
```yaml
model:
  add_eigenvalues: ${dataset.train.add_eigenvalues}
  add_filter_values: ${dataset.train.add_filter_values}

validation:
  nonrec:
    dataset:
      train:
        add_eigenvalues: ${dataset.train.add_eigenvalues}
        filter_size: ${dataset.train.filter_size}
        tau: ${dataset.train.tau}
```

This means you only need to override `dataset.train.*` parameters - they automatically propagate to model and validation configs!

## Advanced: Reusing Datasets

### Sharing Datasets Across Training Runs

Since training hyperparameters (eigenvalues, filter_size, tau, etc.) don't affect the dataset hash, you can:

```bash
# Generate dataset once
python toy_datasets/create_flow_from_movement.py \
    --seq-name star8 \
    --auto-name \
    --save-step 40 \
    --total-frames 2000
# Creates: variant_e450c505

# Train multiple times with different hyperparameters
python toy_datasets/generate_and_train.py \
    --dataset-save-step 40 \
    --dataset-total-frames 2000 \
    --training-add-eigenvalues true \
    --training-filter-size 5

python toy_datasets/generate_and_train.py \
    --dataset-save-step 40 \
    --dataset-total-frames 2000 \
    --training-add-eigenvalues false \
    --training-filter-size 7
```

Both runs use `variant_e450c505` (same dataset), but train with different feature extraction settings!

### Manual Training on Existing Variant

```bash
python -m idn.train_toy_dataset \
    dataset.common.data_root=toy_datasets/data/star8/variant_e450c505 \
    dataset.train.seq=[star8] \
    dataset.val.seq=[star8_test] \
    dataset.train.add_eigenvalues=true \
    dataset.train.filter_size=7 \
    model.hidden_dim=64
```

### Finding Variant Hash for Specific Config

```bash
# Check existing variants
ls -la toy_datasets/data/star8/

# View metadata
cat toy_datasets/data/star8/variant_e450c505/train_optical_flow/star8/dataset_metadata.json
```

## Summary

The parallel sweep system provides:

✅ **Conflict-free parallel execution** - Variant directories prevent overwrites  
✅ **Efficient dataset reuse** - Same config automatically shares dataset  
✅ **Portable paths** - Works on any system without hardcoded paths  
✅ **Auto-calculated consistency** - `delta_t_ms` and `flow_dt_us` stay synchronized  
✅ **Rich hyperparameter space** - Tune dataset, training, and feature extraction parameters  
✅ **Full reproducibility** - Metadata and wandb logging capture everything  
✅ **Safety checks** - Sanity visualizations verify correctness  

Happy sweeping! 🎯
