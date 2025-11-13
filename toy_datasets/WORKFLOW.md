# Toy Dataset Workflow

This directory contains scripts for creating, evaluating, and visualizing synthetic toy datasets (e.g., star8) for optical flow estimation.

## Complete Workflow

### 1. Dataset Generation (with Automatic Sanity Check)
**Script:** `create_flow_from_movement.py`

Generate synthetic DSEC-format optical flow and event data:

```bash
python toy_datasets/create_flow_from_movement.py \
    --seq-name star8 \
    --total-frames 2000 \
    --test-size 0.2 \
    --save-step 20
```

**Output:** 
- `data/star8/train_optical_flow/star8/flow/forward/*.png` - Flow PNGs
- `data/star8/train_events/star8/events.h5` - Event data
- `data/star8/train_events/star8/rectify_maps.h5` - Identity rectification
- `data/star8/train_optical_flow/star8/flow/sanity/*.png` - **Automatic sanity check visualizations**
- `data/star8/train_optical_flow/star8/dataset_metadata.json` - **Dataset parameters metadata**
- Similar for `star8_test` sequence

**Sanity Check Features:**
- Automatically runs after dataset generation (enabled by default)
- Loads and verifies the saved flow PNGs
- Shows start (blue) and end (red) vertices with green arrows indicating GT flow
- Uses exact same parameters as dataset generation (no parameter mismatch)
- To disable: use `--no-sanity-check` flag

**Dataset Metadata:**
- Automatically saved as JSON file after generation
- Contains all generation parameters: `save_step`, `total_frames`, `image_size`, `flow_dt_us`, etc.
- Used by visualization and evaluation scripts to automatically load correct parameters
- No need to manually track parameters across different scripts

---

### 2. Model Training
**Script:** `idn/train_toy_dataset.py`

Train IDNet model on the toy dataset:

```bash
python -m idn.train_toy_dataset
```

**Config:** `idn/config/id_train_original_toydataset_tiny.yaml`

**Output:** 
- Model checkpoints: `ckpt_dir/{run_id}/model.ckpt`
- Wandb logs for tracking

---

### 3. Model Evaluation
**Script:** `eval_toy_dataset.py`

Evaluate trained model and save predictions:

```bash
python toy_datasets/eval_toy_dataset.py \
    --run_path haraghi/toydataset-tinyIDNet-multiseed/{run_id} \
    --ckpt_dir ckpt_dir \
    --output_dir evaluations
```

**Output:** `evaluations/{project}/{run_id}/predictions.pt`
- Contains predictions, ground truths, and metrics for train/val splits

---

### 4. Visualization & Analysis
**Script:** `visualize_predictions.py`

Compare predicted optical flow with ground truth:

```bash
python toy_datasets/visualize_predictions.py \
    --predictions_path evaluations/{project}/{run_id}/predictions.pt \
    --data_root data/star8 \
    --output_dir visualizations \
    --split both \
    --max_samples 10
```

**Note:** Parameters like `save_step`, `total_frames`, etc. are automatically loaded from the metadata file. You can override them with command-line arguments if needed:

```bash
python toy_datasets/visualize_predictions.py \
    --predictions_path evaluations/{project}/{run_id}/predictions.pt \
    --data_root data/star8 \
    --save_step 50  # Override metadata value
```

**Output:** `visualizations/star8/train/*.png` and `visualizations/star8_test/val/*.png`
- 3-panel comparison: GT (left), Predicted (middle), Overlay (right)
- Green arrows = Ground truth flow
- Orange arrows = Predicted flow
- Blue/Red polylines = Start/End object positions

---

## File Organization

```
toy_datasets/
├── create_flow_from_movement.py      # Generate DSEC-format dataset
├── sanity_check_flows.py             # Verify generated flows
├── eval_toy_dataset.py               # Evaluate trained model
├── visualize_predictions.py          # Compare predictions vs GT
├── star8.py                           # Star movement class
├── triangle.py                        # Triangle movement class
├── shape_movement.py                  # Base movement class
├── EVAL_TOY_DATASET_README.md        # Evaluation docs
├── VISUALIZE_PREDICTIONS_README.md   # Visualization docs
└── WORKFLOW.md                        # This file

data/star8/
├── train_optical_flow/
│   ├── star8/flow/forward/*.png      # Train flow PNGs
│   └── star8_test/flow/forward/*.png # Test flow PNGs
└── train_events/
    ├── star8/events.h5               # Train events
    └── star8_test/events.h5          # Test events

idn/
├── train_toy_dataset.py              # Training script
└── config/
    └── id_train_original_toydataset_tiny.yaml

ckpt_dir/
└── {run_id}/model.ckpt               # Trained model weights

evaluations/
└── {project}/{run_id}/
    └── predictions.pt                # Model predictions + metrics

visualizations/
├── star8/train/comparison_*.png      # Train visualizations
└── star8_test/val/comparison_*.png   # Val visualizations
```

## Color Coding Reference

- **Blue**: Object at start frame
- **Red**: Object at end frame  
- **Green arrows**: Ground truth optical flow
- **Orange arrows**: Predicted optical flow
- **Black background**: Original synthetic scene

## Quick Start

```bash
# 1. Generate dataset
python toy_datasets/create_flow_from_movement.py --seq-name star8

# 2. Train model (uses wandb)
python -m idn.train_toy_dataset

# 3. Evaluate model (replace {run_id})
python toy_datasets/eval_toy_dataset.py \
    --run_path haraghi/toydataset-tinyIDNet-multiseed/{run_id}

# 4. Visualize results
python toy_datasets/visualize_predictions.py \
    --predictions_path evaluations/toydataset-tinyIDNet-multiseed/{run_id}/predictions.pt \
    --data_root data/star8
```
