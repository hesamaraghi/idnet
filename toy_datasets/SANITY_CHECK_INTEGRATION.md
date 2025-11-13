# Sanity Check Integration

## Overview

The `create_flow_from_movement.py` script now includes **integrated sanity check functionality** to automatically verify generated optical flow datasets. This eliminates the need for the separate `sanity_check_flows.py` script and prevents parameter mismatches.

## Key Benefits

1. **No Parameter Mismatch**: Sanity checks use the exact same parameters as dataset generation
2. **Immediate Verification**: Checks run automatically after dataset creation
3. **Ground Truth Testing**: Loads and verifies the saved flow PNGs from disk
4. **Simplified Workflow**: One command generates dataset + verification
5. **Automatic Cleanup**: Removes existing dataset directories to prevent mixture of old and new data

## Features

### Automatic Cleanup
Before generating a new dataset, the script automatically removes any existing data for the same sequence name:
- Removes `train_optical_flow/{seq_name}/` directory
- Removes `train_events/{seq_name}/` directory  
- Also removes test split directories if `test_size > 0`
- Prevents accidental mixture of old and new datasets

Example output:
```
[cleanup] Removing existing data: /data/idnet/toy_datasets/data/star8/train_optical_flow/test_cleanup
[cleanup] Removing existing data: /data/idnet/toy_datasets/data/star8/train_events/test_cleanup
[cleanup] Removing existing test data: /data/idnet/toy_datasets/data/star8/train_optical_flow/test_cleanup_test
[cleanup] Removing existing test data: /data/idnet/toy_datasets/data/star8/train_events/test_cleanup_test
```

### Automatic Execution
By default, sanity checks run automatically after dataset generation:
```bash
python toy_datasets/create_flow_from_movement.py --seq-name star8
```

### Optional Disable
To skip sanity checks (faster generation):
```bash
python toy_datasets/create_flow_from_movement.py --seq-name star8 --no-sanity-check
```

### What It Does

1. **Loads GT flows** from the just-saved PNG files using `decode_flow_dsec()`
2. **Recreates star movement** with the exact parameters used for generation
3. **Generates visualizations** showing:
   - Start vertices (blue dots + polyline)
   - End vertices (red dots + polyline)
   - GT flow arrows (green) at each vertex
4. **Saves to `flow/sanity/`** directory alongside the forward flow PNGs

### Output Structure

```
data/star8/train_optical_flow/
├── star8/
│   └── flow/
│       ├── forward/
│       │   ├── 000000.png  (16-bit GT flow)
│       │   ├── 000001.png
│       │   └── ...
│       └── sanity/         <-- NEW: Automatic sanity checks
│           ├── sanity_000000.png  (visualization)
│           ├── sanity_000001.png
│           └── ...
└── star8_test/
    └── flow/
        ├── forward/
        └── sanity/         <-- Also for test split
```

## Implementation Details

### Functions Added

1. **`decode_flow_dsec(png_path)`**
   - Inverse of encoding
   - Returns `(u, v, valid)` from 16-bit PNG

2. **`get_vertices(star, frame_idx)`**
   - Extracts star vertices at specific frame
   - Uses `star.update_shape()` and `transformed_path.vertices`

3. **`generate_sanity_check_figures(...)`**
   - Main sanity check function
   - Parameters match dataset generation exactly
   - Loads GT from saved PNGs (not recomputed)
   - Creates verification visualizations

### Command-Line Arguments

```bash
--sanity-check          # Enable sanity checks (DEFAULT)
--no-sanity-check       # Disable sanity checks
```

## Example Output

```
============================================================
Running sanity checks on generated flows...
============================================================

[SANITY CHECK] Train split: star8
[sanity] Saved .../flow/sanity/sanity_000000.png (arrows: 11)
[sanity] Saved .../flow/sanity/sanity_000001.png (arrows: 11)
...
[SANITY CHECK] Generated 50 train sanity figures

[SANITY CHECK] Test split: star8_test
[sanity] Saved .../flow/sanity/sanity_000000.png (arrows: 11)
...
[SANITY CHECK] Generated 10 test sanity figures

============================================================
Sanity checks completed!
============================================================
```

## Workflow Impact

### Old Workflow (2 steps)
1. Generate dataset: `python toy_datasets/create_flow_from_movement.py`
2. Sanity check: `python toy_datasets/sanity_check_flows.py --seq-name star8`
   - **Risk**: Different default parameters might create mismatches

### New Workflow (1 step)
1. Generate dataset + sanity check: `python toy_datasets/create_flow_from_movement.py`
   - **Benefit**: Same parameters guaranteed, immediate verification

## Migration Note

The standalone `sanity_check_flows.py` script is now **optional/deprecated**. All functionality has been integrated into the dataset generator.

## Testing

Tested with:
```bash
python toy_datasets/create_flow_from_movement.py \
    --seq-name test_sanity \
    --total-frames 200 \
    --save-step 40 \
    --test-size 0.2
```

Results:
- ✅ Generated 4 train + 1 test flow pairs
- ✅ Created 4 train + 1 test sanity figures
- ✅ All visualizations show correct vertex positions and flow arrows
- ✅ Parameters match exactly between generation and verification
