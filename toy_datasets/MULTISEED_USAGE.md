# Multi-Seed Experiments Guide

## Overview

The training pipeline now supports multi-seed experiments for statistical robustness. Seeds control random initialization of model weights, data shuffling, and other stochastic operations.

## How It Works

1. **Sweep Configuration**: Add seed parameter to your wandb sweep config
2. **Dataset Generation**: Seeds are passed via environment variable `TRAINING_SEED`
3. **Training**: Seeds are set early in the Trainer initialization to ensure reproducibility
4. **Logging**: Seeds are logged to wandb for tracking and grouping

## Usage

### 1. Wandb Sweep Configuration

Add `training_seed` to your sweep parameters:

```yaml
parameters:
  training_seed:
    values: [42, 123, 456]  # Run each configuration with these 3 seeds
```

See `sweep_config_example.yaml` for a complete example.

### 2. Manual Command Line Usage

Run with a specific seed (delta_t_ms is auto-calculated from save_step):

```bash
python generate_and_train.py \
  --dataset_save_step=40 \
  --dataset_total_frames=500 \
  --training_num_bins=5 \
  --training_epochs=10 \
  --training_seed=42
```

**Note**: `training_delta_t_ms` is automatically calculated as `save_step × frame_time_us / 1000`. For `save_step=40` and `frame_time_us=1000` (default), `delta_t_ms=40`. This ensures the training time window matches the flow timestamp intervals.

### 3. Run Name Format

The run name automatically includes the seed for easy identification:

```
-ds40-f500-v465f38-seed42
```

Where:
- `ds40`: save_step=40
- `f500`: total_frames=500
- `v465f38`: variant hash (first 6 characters)
- `seed42`: random seed=42

### 4. Tags

Seeds are automatically added as wandb tags for filtering:

```python
tags = "save_step_40,frames_500,variant_465f38,seed_42"
```

## Implementation Details

### Seed Setting

The seed is set in `idn/utils/trainer.py` during Trainer initialization:

```python
if 'TRAINING_SEED' in os.environ:
    seed = int(os.environ['TRAINING_SEED'])
    print(f"Setting random seed to {seed} for reproducibility")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
```

### Wandb Logging

The seed is logged to wandb config for tracking:

```python
if 'TRAINING_SEED' in os.environ:
    wandb.config.update({
        'training_seed': int(os.environ['TRAINING_SEED']),
    }, allow_val_change=True)
```

## Example Sweep

A sweep with 45 experiments (5 save_steps × 3 num_bins × 3 seeds):

```yaml
parameters:
  dataset_save_step:
    values: [20, 40, 60, 80, 100]
  training_num_bins:
    values: [5, 10, 15]
  training_seed:
    values: [42, 123, 456]
```

## Grouping in Wandb

Use wandb grouping/filtering features to:
- Compare performance across seeds for the same configuration
- Calculate mean ± std across seeds
- Identify outlier runs
- Filter by specific seed values using tags

## Statistical Analysis

With multiple seeds, you can:
1. Report mean performance metrics
2. Calculate confidence intervals
3. Perform statistical significance tests
4. Identify configurations that are robust across seeds

## Note on Determinism

For full determinism, you may also need to set:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

However, this can impact performance. The current implementation provides good reproducibility without these settings.
