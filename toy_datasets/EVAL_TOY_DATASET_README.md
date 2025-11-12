# Toy Dataset Evaluation Script

This script evaluates trained IDNet models on the toy dataset (star8) and saves the predictions and metrics.

## Usage

```bash
python toy_datasets/eval_toy_dataset.py --run_path <wandb_run_path> [OPTIONS]
```

### Required Arguments

- `--run_path`: Wandb run path in format `entity/project/run_id`
  - Example: `haraghi/toydataset-tinyIDNet-multiseed/fxd5fk65`

### Optional Arguments

- `--ckpt_dir`: Directory where checkpoints are stored (default: `ckpt_dir`)
- `--output_dir`: Directory to save evaluation results (default: `evaluations`)
- `--gpu`: GPU device id (default: 0)

## Example

```bash
python toy_datasets/eval_toy_dataset.py \
    --run_path haraghi/toydataset-tinyIDNet-multiseed/fxd5fk65 \
    --ckpt_dir ckpt_dir \
    --output_dir evaluations \
    --gpu 0
```

## Output

The script will:

1. Load the model configuration from wandb
2. Load the trained model weights from `ckpt_dir/{run_id}/model.ckpt`
3. Create train and validation dataloaders based on the config
4. Run inference on both splits
5. Compute metrics: L1, L2, EPE, and 3PE
6. Save results to `{output_dir}/{project}/{run_id}/predictions.pt`

### Metrics Explained

- **L1 Error**: Mean absolute error (Manhattan distance) between predicted and ground truth flow
- **L2 Error**: Mean Euclidean distance between predicted and ground truth flow
- **EPE (End Point Error)**: Average Euclidean distance between flow endpoints (same as L2 for per-pixel flow)
- **1PE (1-pixel error)**: Percentage of pixels where EPE > 1 pixel (outlier rate)
- **3PE (3-pixel error)**: Percentage of pixels where EPE > 3 pixels (outlier rate)

### Output Structure

The saved `.pt` file contains a dictionary with the following structure:

```python
{
    'train': {
        'predictions': List[Tensor],      # List of prediction batches
        'ground_truths': List[Tensor],    # List of ground truth batches
        'valid_masks': List[Tensor],      # List of valid mask batches
        'metrics': {
            'l1_mean': float,             # Mean L1 error
            'l2_mean': float,             # Mean L2 error  
            'epe_mean': float,            # Mean End Point Error
            '1pe': float,                 # 1-pixel error (percentage)
            '3pe': float,                 # 3-pixel error (percentage)
            'num_samples': int,           # Number of batches
            'num_valid_pixels': int       # Total valid pixels
        }
    },
    'val': {
        # Same structure as 'train'
    }
}
```

Each prediction tensor has shape `[batch_size, 2, H, W]` where the 2 channels represent optical flow (u, v).

## Notes

- The script automatically loads the dataset configuration from wandb
- Data augmentation (horizontal/vertical flip) is disabled for validation
- All sequences are concatenated for evaluation
- The script requires the data to be available at the paths specified in the wandb config
