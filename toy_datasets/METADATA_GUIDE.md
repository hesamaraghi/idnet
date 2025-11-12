# Dataset Metadata Guide

## Overview

Starting from the latest version, `create_flow_from_movement.py` automatically saves all generation parameters as a JSON metadata file. This eliminates the need to manually track parameters like `save_step`, `total_frames`, etc., across different scripts.

## What is Saved

When you generate a dataset, a `dataset_metadata.json` file is created in the sequence directory with the following information:

```json
{
  "seq_name": "star8",
  "total_frames": 2000,
  "image_size": [256, 256],
  "save_step": 40,
  "frame_time_us": 1000,
  "flow_dt_us": 40000,
  "start_ts_us": 0,
  "face_color": "black",
  "test_size": 0.2,
  "split_start_frame": 1600,
  "num_flow_pairs_train": 39,
  "num_flow_pairs_test": 9
}
```

## File Locations

Metadata is saved in both train and test sequence directories:
- `data/star8/train_optical_flow/{seq_name}/dataset_metadata.json` (train split)
- `data/star8/train_optical_flow/{seq_name}_test/dataset_metadata.json` (test split)

## How to Use

### Automatic Loading (Recommended)

Scripts like `visualize_predictions.py` automatically load metadata:

```bash
python toy_datasets/visualize_predictions.py \
    --predictions_path evaluations/myrun/predictions.pt \
    --data_root data/star8
```

The script will:
1. Look for `dataset_metadata.json` in the sequence directory
2. Load all parameters automatically
3. Use those parameters for visualization

### Manual Loading in Python

You can load metadata in your own scripts:

```python
from create_flow_from_movement import load_dataset_metadata

# Load metadata
metadata = load_dataset_metadata(
    data_root='toy_datasets/data/star8',
    seq_name='star8'
)

# Access parameters
save_step = metadata['save_step']
total_frames = metadata['total_frames']
image_size = metadata['image_size']
```

### Overriding Metadata

You can override metadata values with command-line arguments:

```bash
python toy_datasets/visualize_predictions.py \
    --predictions_path evaluations/myrun/predictions.pt \
    --data_root data/star8 \
    --save_step 50  # Override the metadata value
```

## Benefits

1. **No Parameter Mismatches**: Visualization and evaluation scripts use the exact same parameters as dataset generation
2. **No Manual Tracking**: Don't need to remember what `save_step` you used when generating the dataset
3. **Reproducibility**: All generation parameters are recorded automatically
4. **Easy Debugging**: Can quickly check what parameters were used for any dataset

## Backward Compatibility

If a dataset was generated before the metadata feature:
- Scripts will warn that metadata is not found
- You must provide parameters manually via command-line arguments
- Or regenerate the dataset to create metadata files

## Example Workflow

```bash
# 1. Generate dataset with custom save_step
python toy_datasets/create_flow_from_movement.py \
    --seq-name my_experiment \
    --save-step 50 \
    --total-frames 1000

# 2. Train model (uses config file)
python -m idn.train_toy_dataset

# 3. Evaluate model (parameters loaded from metadata)
python toy_datasets/eval_toy_dataset.py \
    --run_path haraghi/project/run_id \
    --ckpt_dir ckpt_dir

# 4. Visualize (parameters loaded from metadata)
python toy_datasets/visualize_predictions.py \
    --predictions_path evaluations/project/run_id/predictions.pt \
    --data_root data/star8

# No need to manually specify save_step, total_frames, etc.!
```

## Checking Metadata

To quickly check what parameters were used for a dataset:

```bash
cat data/star8/train_optical_flow/my_experiment/dataset_metadata.json
```

Or in Python:

```python
import json
from pathlib import Path

metadata_path = Path('data/star8/train_optical_flow/my_experiment/dataset_metadata.json')
metadata = json.loads(metadata_path.read_text())
print(f"Dataset generated with save_step={metadata['save_step']}")
```
