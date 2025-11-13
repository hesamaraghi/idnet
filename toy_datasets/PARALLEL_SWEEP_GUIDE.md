# Parallel Dataset Generation & Training Guide

This guide explains how to run wandb sweeps that generate datasets with different parameters and train models on them, all in parallel without conflicts.

## The Problem

When running parallel experiments with different dataset parameters (e.g., varying `save_step`, `total_frames`, `delta_t_ms`), multiple processes would try to write to the same directories, causing:
- Dataset overwrites
- Metadata conflicts
- Incorrect training/validation splits

## The Solution

### Automatic Dataset Naming with Config Hashing

The `--auto-name` flag automatically appends a configuration hash to the sequence name:

```bash
python toy_datasets/create_flow_from_movement.py \
    --seq-name star8 \
    --auto-name \
    --save-step 40 \
    --total-frames 2000
```

This generates a dataset named `star8_a3f5b2c1` (where `a3f5b2c1` is the config hash).

**Key features:**
- ✅ Different configs → Different names (no conflicts)
- ✅ Same configs → Same names (can reuse if desired)
- ✅ Hash based on: `total_frames`, `image_size`, `save_step`, `frame_time_us`, `flow_dt_us`, `test_size`

## Usage Patterns

### Pattern 1: Manual Dataset Generation + Training

```bash
# Generate dataset
python toy_datasets/create_flow_from_movement.py \
    --seq-name star8 \
    --auto-name \
    --save-step 40

# This creates: star8_a3f5b2c1 and star8_a3f5b2c1_test

# Train on it
python -m idn.train_toy_dataset \
    validation.nonrec.dataset.train.seq=[star8_a3f5b2c1] \
    validation.nonrec.dataset.val.seq=[star8_a3f5b2c1_test]
```

### Pattern 2: Automated with Wrapper Script

```bash
python toy_datasets/generate_and_train.py \
    --dataset-save-step 40 \
    --dataset-total-frames 2000 \
    --training-delta-t-ms 100 \
    --training-num-bins 15
```

This script:
1. Generates dataset with `--auto-name`
2. Captures the generated sequence name
3. Automatically configures training to use that dataset

### Pattern 3: Wandb Sweep (Parallel Experiments)

**Step 1: Create sweep configuration** (`sweep_config.yaml`):

```yaml
program: toy_datasets/generate_and_train.py
method: grid
metric:
  name: val/epe
  goal: minimize

parameters:
  dataset_save_step:
    values: [20, 40, 60]
  
  dataset_total_frames:
    values: [1000, 2000]
  
  training_delta_t_ms:
    values: [40, 100]
  
  training_num_bins:
    values: [10, 15, 20]
```

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
2. Generate a unique dataset (e.g., `star8_a3f5b2c1`)
3. Train on that dataset
4. Log results to wandb

**No conflicts!** Each parameter combination generates a unique dataset name.

## How Config Hashing Works

The hash is computed from dataset generation parameters:

```python
config_dict = {
    'total_frames': 2000,
    'image_size': (256, 256),
    'save_step': 40,
    'frame_time_us': 1000,
    'flow_dt_us': 40000,
    'test_size': 0.2,
}
# Hash: a3f5b2c1
```

**Examples:**

| Parameters | Hash | Sequence Name |
|---|---|---|
| `total_frames=2000, save_step=40` | `a3f5b2c1` | `star8_a3f5b2c1` |
| `total_frames=2000, save_step=60` | `d7e9f4a2` | `star8_d7e9f4a2` |
| `total_frames=1000, save_step=40` | `f1c2d3e4` | `star8_f1c2d3e4` |

## Dataset Metadata Integration

Each generated dataset includes `dataset_metadata.json` with all parameters:

```json
{
  "seq_name": "star8_a3f5b2c1",
  "total_frames": 2000,
  "save_step": 40,
  "image_size": [256, 256],
  "flow_dt_us": 40000,
  "test_size": 0.2,
  ...
}
```

This metadata is automatically:
- ✅ Saved during dataset generation
- ✅ Loaded by training scripts
- ✅ Logged to wandb for full reproducibility

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

After running a sweep with 3 different configurations:

```
toy_datasets/data/star8/
├── train_optical_flow/
│   ├── star8_a3f5b2c1/       # save_step=40, total_frames=2000
│   │   ├── dataset_metadata.json
│   │   └── flow/
│   ├── star8_a3f5b2c1_test/
│   ├── star8_d7e9f4a2/       # save_step=60, total_frames=2000
│   │   ├── dataset_metadata.json
│   │   └── flow/
│   ├── star8_d7e9f4a2_test/
│   ├── star8_f1c2d3e4/       # save_step=40, total_frames=1000
│   │   ├── dataset_metadata.json
│   │   └── flow/
│   └── star8_f1c2d3e4_test/
└── train_events/
    ├── star8_a3f5b2c1/
    ├── star8_a3f5b2c1_test/
    ├── star8_d7e9f4a2/
    ├── star8_d7e9f4a2_test/
    ├── star8_f1c2d3e4/
    └── star8_f1c2d3e4_test/
```

## Cleanup

To remove all generated datasets for a sweep:

```bash
# Remove all datasets matching a pattern
rm -rf toy_datasets/data/star8/train_optical_flow/star8_*
rm -rf toy_datasets/data/star8/train_events/star8_*

# Or keep specific ones
# (only remove datasets with hash, keep original star8)
```

## Troubleshooting

### Issue: "Sequence not found during training"

**Cause:** The training config still references the original sequence name (e.g., `star8`) instead of the hashed name.

**Solution:** The `generate_and_train.py` wrapper script automatically handles this. If you're running manually, update your config:

```bash
python -m idn.train_toy_dataset \
    validation.nonrec.dataset.train.seq=[star8_a3f5b2c1] \
    validation.nonrec.dataset.val.seq=[star8_a3f5b2c1_test]
```

### Issue: "Multiple datasets with same hash"

**Cause:** You regenerated a dataset with identical parameters.

**Solution:** This is expected behavior! Same parameters → same hash → same directory. The old dataset will be cleaned up automatically (see `create_flow_from_movement.py` cleanup code).

### Issue: "Out of disk space"

**Cause:** Many sweep runs generate many datasets.

**Solution:** 
1. Use `in_memory=True` with `do_not_save_preprocessed=True` to avoid preprocessed file bloat
2. Clean up old datasets between sweeps
3. Consider using a shared preprocessed cache if multiple runs use the same config

## Best Practices

1. **Always use `--auto-name` for sweeps** to avoid conflicts
2. **Use in-memory training** (`in_memory=True`) for faster parallel execution
3. **Monitor disk usage** - sweep runs can accumulate many datasets
4. **Save sweep results to wandb** - easier to track which dataset was used for which run
5. **Clean up after sweeps** - remove datasets you won't reuse

## Advanced: Reusing Datasets

If you want to reuse a dataset with a specific configuration:

```bash
# Generate once
python toy_datasets/create_flow_from_movement.py \
    --seq-name star8 \
    --auto-name \
    --save-step 40 \
    --total-frames 2000
# Creates: star8_a3f5b2c1

# Train multiple times on the same dataset
python -m idn.train_toy_dataset \
    validation.nonrec.dataset.train.seq=[star8_a3f5b2c1] \
    model.hidden_dim=32

python -m idn.train_toy_dataset \
    validation.nonrec.dataset.train.seq=[star8_a3f5b2c1] \
    model.hidden_dim=64
```

Both training runs use the same dataset but different model architectures!
