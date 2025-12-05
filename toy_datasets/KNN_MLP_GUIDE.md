# KNN+MLP Optical Flow Prediction Guide

## Overview

`knn_mlp.py` implements a k-Nearest Neighbors + Multi-Layer Perceptron pipeline for predicting optical flow from event camera data. The approach:

1. Generates synthetic event datasets with ground truth optical flow
2. Extracts spatial-temporal features from events
3. Builds k-NN graph for each event
4. Trains MLP to predict optical flow from k-NN feature vectors
5. Evaluates predictions and saves results

---

## Quick Start

### Basic Training
```bash
# Train with default parameters (k=5, both features, temporal split)
python toy_datasets/knn_mlp.py

# Train with specific k value and feature type
python toy_datasets/knn_mlp.py --k 50 --feature_type both

# Use relative coordinates (translation-invariant)
python toy_datasets/knn_mlp.py --relative_coordinates

# Force regenerate dataset cache
python toy_datasets/knn_mlp.py --force_regenerate
```

### Evaluation
```bash
# Evaluate a trained model by W&B run ID
python toy_datasets/knn_mlp.py --eval_run_id <run_id>
```

---

## Architecture

### Pipeline Overview

```
Event Generation → Feature Extraction → k-NN Graph → MLP Training → Evaluation
     (star8)          (Harris)         (spatial)    (PyTorch)      (MSE)
```

### 1. Dataset Generation

**Shape**: Rotating star with 8 points
- Configurable radius, rotation speed, image size
- Deterministic trajectory with ground truth optical flow

**Event Generation Methods**:

| Method | Description | Speed | Realism | Use Case |
|--------|-------------|-------|---------|----------|
| `synthetic` | Simple on/off events at shape boundaries | Fast | Low | Quick prototyping, debugging |
| `v2e` | Full DVS simulator with noise, threshold mismatch | Slow | High | Realistic evaluation |

**Key Parameters**:
```bash
--total_frames 2000        # Sequence length
--img_size 256 256         # Image dimensions (H, W)
--num_points 5             # Star points
--outer_radius 40          # Star outer radius
--inner_radius 20          # Star inner radius
--num_rotations 2          # Full rotations in sequence
--event_generation_method synthetic  # or 'v2e'
```

**Dataset Caching**:
- Datasets are cached in `dataset_cache/` with hash-based filenames
- Hash includes all generation parameters for reproducibility
- Separate caches for data and k-NN indices
- Use `--force_regenerate` to rebuild cache

### 2. Feature Extraction

**Harris Corner Features**:
- **Eigenvalues** (λ₁, λ₂): Capture local structure (corners, edges)
- **Temporal Filter**: Exponentially-weighted event accumulation

**Parameters**:
```bash
--tau 1.0                  # Temporal decay constant
--filter_size 5            # Spatial filter kernel size
```

**Feature Types** (`--feature_type`):

| Feature Type | Components | Dimension | Use Case |
|--------------|-----------|-----------|----------|
| `original` | (x, y) | 2 | Baseline, purely spatial |
| `original_time_augmented` | (x, y, t) | 3 | Add temporal context |
| `eig` | (x, y, λ₁, λ₂) | 4 | Spatial + corner detection |
| `filter` | (x, y, filter_val) | 3 | Spatial + temporal activity |
| `both` | (x, y, λ₁, λ₂, filter_val) | 5 | All features (recommended) |
| `both_time_augmented` | (x, y, t, λ₁, λ₂, filter_val) | 6 | Full context |
| `eig_exclude_xy` | (λ₁, λ₂) | 2 | Pure structure, no position |
| `filter_exclude_xy` | (filter_val) | 1 | Pure activity |
| `both_exclude_xy` | (λ₁, λ₂, filter_val) | 3 | Structure + activity |

**Recommendations**:
- Start with `both` - best balance of features
- Use `both_time_augmented` if temporal patterns are important
- Try `eig_exclude_xy` to test position-invariant learning

### 3. k-NN Graph Construction

**Spatial Nearest Neighbors**:
- Computed in (x, y, t) space for each event
- Uses FAISS library (GPU-accelerated if available)
- Cached separately from dataset

**Key Parameter**:
```bash
--k 50                     # Number of neighbors (including self)
```

**Coordinate Systems**:

**Absolute Coordinates** (default):
```python
# Neighbor features use world coordinates
neighbor_features = [x₁, y₁, λ₁, λ₂, ..., xₖ, yₖ, λₖ, λₖ]
```

**Relative Coordinates** (`--relative_coordinates`):
```python
# Neighbor coordinates relative to center node
neighbor_features = [Δx₁, Δy₁, λ₁, λ₂, ..., Δxₖ, Δyₖ, λₖ, λₖ]
# where Δxᵢ = xᵢ - x_center, Δyᵢ = yᵢ - y_center
```

**Why Relative Coordinates?**
- Translation invariance: Same local pattern → same features
- Better generalization across image space
- Recommended for most use cases

**k Selection Guidelines**:
- k=5-10: Local neighborhood, fast
- k=20-50: Medium context, good balance (recommended)
- k=100+: Large receptive field, slower

### 4. Model Architecture

**KNNMLP Class**:
```python
Sequential(
    Linear(input_dim, hidden_dim),  # input_dim = k × feature_dim
    ReLU(),
    Linear(hidden_dim, hidden_dim),
    ReLU(),
    Linear(hidden_dim, 2)           # Output: (vx, vy)
)
```

**Hyperparameters**:
```bash
--hidden_dim 64            # Hidden layer size
--lr 1e-3                  # Learning rate
--batch_size 16            # Mini-batch size
--max_epochs 100           # Training epochs
```

**Loss**: Mean Squared Error (MSE) on optical flow components

**Example Dimensions**:
- k=50, feature_type=both (5D) → input_dim = 50 × 5 = 250
- hidden_dim=64
- output_dim=2 (vx, vy)

### 5. Training

**Train/Test Splitting**:

| Method | Strategy | Pros | Cons |
|--------|----------|------|------|
| `random` | Random event sampling | IID assumption | Unrealistic |
| `temporal` | Chronological split | Realistic | Harder task |

```bash
--test_train_split temporal  # or 'random'
--test_size 0.2              # Fraction for validation
--test_split_seed 42         # Reproducibility
```

**Training Framework**:
- **PyTorch Lightning**: Training loop automation
- **W&B (Weights & Biases)**: Experiment tracking
- **Metrics**: Train/validation MSE logged per epoch

**W&B Configuration**:
```bash
--project "knn-mlp-regression"     # W&B project name
--entity "your-team"                # W&B entity (user/team)
--log_dir "wandb_logs"              # Local log directory
--online                            # Enable online sync (default: offline)
```

**Reproducibility**:
```bash
--random_seed 42                    # Master seed for everything
--test_split_seed 42                # Override train/test split seed
```

---

## Command Reference

### Dataset Parameters

```bash
# Shape configuration
--toy_dataset star8                 # Dataset type (currently only star8)
--total_frames 2000                 # Sequence length
--img_size 256 256                  # Image dimensions (H W)
--num_points 5                      # Star points
--outer_radius 40                   # Outer radius (pixels)
--inner_radius 20                   # Inner radius (pixels)
--num_rotations 2                   # Complete rotations

# Event generation
--event_generation_method synthetic # synthetic or v2e
--force_regenerate                  # Ignore cache, regenerate dataset
```

### Feature Parameters

```bash
# Feature extraction
--feature_type both                 # Feature type (see table above)
--tau 1.0                           # Temporal decay constant
--filter_size 5                     # Spatial filter size

# k-NN configuration
--k 50                              # Number of neighbors
--relative_coordinates              # Use relative instead of absolute coords
```

### Training Parameters

```bash
# Model architecture
--hidden_dim 64                     # MLP hidden layer size

# Optimization
--lr 1e-3                           # Learning rate
--batch_size 16                     # Batch size
--max_epochs 100                    # Training epochs

# Data splitting
--test_train_split temporal         # random or temporal
--test_size 0.2                     # Validation fraction
--test_split_seed 42                # Split reproducibility
--random_seed 42                    # Master random seed
```

### Logging Parameters

```bash
# W&B configuration
--entity "your-username"            # W&B user or team
--project "knn-mlp-regression"      # Project name
--log_dir "wandb_logs"              # Local directory
--online                            # Enable online sync

# Evaluation
--eval_run_id "abc123xyz"           # Evaluate existing run
```

---

## Workflows

### 1. Quick Experiment

```bash
# Fast training with default parameters
python toy_datasets/knn_mlp.py \
    --max_epochs 50 \
    --batch_size 32
```

### 2. Hyperparameter Search

```bash
# Try different k values
for k in 10 20 50 100; do
    python toy_datasets/knn_mlp.py --k $k --project "knn-sweep"
done

# Try different feature types
for feat in original eig filter both; do
    python toy_datasets/knn_mlp.py --feature_type $feat --project "feature-sweep"
done
```

### 3. Realistic Evaluation

```bash
# Use v2e events with temporal split
python toy_datasets/knn_mlp.py \
    --event_generation_method v2e \
    --test_train_split temporal \
    --relative_coordinates \
    --k 50 \
    --feature_type both_time_augmented \
    --max_epochs 200 \
    --online
```

### 4. Model Evaluation

```bash
# Step 1: Train model
python toy_datasets/knn_mlp.py --project "my-experiment"
# Note the W&B run ID from output (e.g., "abc123xyz")

# Step 2: Evaluate trained model
python toy_datasets/knn_mlp.py \
    --eval_run_id "abc123xyz" \
    --project "my-experiment"

# Step 3: Visualize predictions
python toy_datasets/visualize_predictions.py \
    --predictions_path "wandb_logs/my-experiment/evaluations/abc123xyz/predictions.pt"
```

---

## Output Files

### Training Outputs

**W&B Logs** (`wandb_logs/<project>/`):
```
<project>/
├── <run_id>/
│   ├── checkpoints/
│   │   └── *.ckpt           # Model checkpoint
│   └── wandb/               # W&B run data
└── offline-run-*.wandb      # Offline runs (before sync)
```

**Cached Data** (`dataset_cache/`):
```
dataset_cache/
├── <hash>_data.pt           # Event dataset with features
├── <hash>_k<k>_knn.pt       # k-NN indices
└── <hash>_config.yaml       # Dataset configuration
```

### Evaluation Outputs

**Predictions** (`wandb_logs/<project>/evaluations/<run_id>/`):
```python
# predictions.pt structure
{
    'train': {
        'X': torch.Tensor,        # Input features [N, k*F]
        'Y': torch.Tensor,        # Ground truth flow [N, 2]
        'preds': torch.Tensor,    # Predicted flow [N, 2]
        'data_array': np.ndarray  # Original events
    },
    'val': {
        # Same structure as train
    }
}
```

---

## Advanced Topics

### Dataset Hashing

**Hash Components**:
- Shape parameters (num_points, radius, rotations)
- Image size and frame count
- Event generation method
- Feature extraction parameters (tau, filter_size)
- Texture settings (if using DTD textures)
- Random seed

**Benefits**:
- Automatic cache invalidation on parameter changes
- Reproducible datasets across runs
- Easy sharing of exact experimental conditions

### FAISS k-NN Computation

**GPU Acceleration**:
- Automatically uses GPU if available
- Falls back to CPU if CUDA unavailable
- Significant speedup for large datasets (>100k events)

**Memory Considerations**:
- k=50, N=1M events → ~200MB for indices
- Pre-compute and cache for faster iteration

### Feature Ablation Studies

**Test Feature Importance**:
```bash
# Baseline: spatial only
python toy_datasets/knn_mlp.py --feature_type original

# Add eigenvalues
python toy_datasets/knn_mlp.py --feature_type eig

# Add temporal filter
python toy_datasets/knn_mlp.py --feature_type filter

# Full feature set
python toy_datasets/knn_mlp.py --feature_type both
```

### Temporal vs Random Split

**Temporal Split** (Recommended):
- Train: Early frames (0-80%)
- Val: Late frames (80-100%)
- Tests generalization to unseen time
- More realistic evaluation

**Random Split**:
- Train/Val: Mixed frames
- Easier task (interpolation not extrapolation)
- Useful for debugging

---

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce batch size
--batch_size 8

# Use fewer neighbors
--k 10

# Reduce hidden dimension
--hidden_dim 32
```

**2. Slow Training**
```bash
# Use GPU if available
# Check: torch.cuda.is_available()

# Reduce dataset size
--total_frames 1000

# Use synthetic events instead of v2e
--event_generation_method synthetic
```

**3. Cache Issues**
```bash
# Clear cache and regenerate
--force_regenerate

# Or manually delete cache
rm -rf dataset_cache/<hash>*
```

**4. W&B Sync Failures**
```bash
# Use offline mode
# (remove --online flag)

# Manual sync later
wandb sync wandb_logs/<project>/offline-run-*.wandb
```

**5. Poor Performance**
- Try `--relative_coordinates` (often helps)
- Increase k (more context)
- Use `both_time_augmented` features
- Increase model capacity (`--hidden_dim 128`)
- Train longer (`--max_epochs 200`)

---

## Best Practices

### 1. Start Simple
```bash
# Minimal working example
python toy_datasets/knn_mlp.py \
    --k 20 \
    --feature_type both \
    --max_epochs 50
```

### 2. Use Relative Coordinates
```bash
# Generally better performance
python toy_datasets/knn_mlp.py --relative_coordinates
```

### 3. Track Everything
```bash
# Enable W&B online mode
python toy_datasets/knn_mlp.py --online
```

### 4. Test Temporal Generalization
```bash
# More realistic evaluation
python toy_datasets/knn_mlp.py --test_train_split temporal
```

### 5. Cache Wisely
- Let cache work unless parameters change
- Use `--force_regenerate` only when needed
- Share cache files for team consistency

---

## Integration with Other Tools

### Visualization

**Visualize Cached Datasets**:
```bash
# List cached datasets
python toy_datasets/visualize_cached_dataset.py --list

# Visualize specific dataset
python toy_datasets/visualize_cached_dataset.py --index 0

# Generate all visualizations
python toy_datasets/visualize_cached_dataset.py --index 0 \
    --generate_scatter \
    --generate_flow_viz \
    --frame_animation
```

**Visualize Predictions**:
```bash
python toy_datasets/visualize_predictions.py \
    --predictions_path "wandb_logs/<project>/evaluations/<run_id>/predictions.pt"
```

### Dataset Generator

**Direct Python API**:
```python
from omegaconf import OmegaConf
from dataset_generator import DatasetGenerator

cfg = OmegaConf.create({
    'shape_class': 'star8',
    'total_frames': 2000,
    # ... other params
})

generator = DatasetGenerator(cfg)
data_array = generator._generate_events()
```

---

## Performance Benchmarks

**Typical Training Times** (GPU: RTX 3090):

| Configuration | Events | k | Epochs | Time |
|--------------|--------|---|--------|------|
| Small | ~10k | 10 | 50 | ~2 min |
| Medium | ~50k | 50 | 100 | ~10 min |
| Large | ~200k | 50 | 100 | ~30 min |
| v2e Large | ~500k | 50 | 200 | ~2 hours |

**Memory Usage**:
- Small: ~500MB GPU
- Medium: ~2GB GPU
- Large: ~6GB GPU

---

## References

### Related Scripts
- `dataset_generator.py`: Core dataset generation
- `visualize_cached_dataset.py`: Dataset visualization
- `visualize_predictions.py`: Prediction analysis
- `shape_movement.py`: Shape trajectory definitions

### Related Documentation
- `DATASET_QUICK_REF.md`: Dataset generation reference
- `V2E_INTEGRATION.md`: v2e event simulator guide
- `TEXTURE_GUIDE.md`: Texture configuration (if using DTD)
- `VISUALIZATION_README.md`: Visualization tools

### Papers
- Harris Corner Detector: Harris & Stephens (1988)
- Event Cameras: Gallego et al. (2020)
- v2e Simulator: Hu et al. (2021)

---

## FAQ

**Q: What's the difference between synthetic and v2e events?**
A: Synthetic events are fast, deterministic on/off signals at shape boundaries. V2e simulates realistic DVS camera behavior with noise, threshold mismatch, and temporal dynamics. Use synthetic for prototyping, v2e for realistic evaluation.

**Q: Should I use relative or absolute coordinates?**
A: Generally use relative coordinates (`--relative_coordinates`). They provide translation invariance and usually improve performance.

**Q: How do I choose k?**
A: Start with k=50. Increase if you need more context, decrease if training is slow. Typical range: 20-100.

**Q: What feature_type should I use?**
A: Start with `both` (spatial + eigenvalues + filter). Try `both_time_augmented` if temporal patterns matter. Experiment with ablations to understand feature contributions.

**Q: Why is training slow?**
A: Common causes: (1) Large dataset, (2) High k value, (3) v2e event generation, (4) No GPU. Solutions: Reduce total_frames, decrease k, use synthetic events, enable GPU.

**Q: How do I share experiments with my team?**
A: Use W&B online mode (`--online`), share the run ID, and share the dataset_cache/ directory. Team members can evaluate your models with `--eval_run_id`.

**Q: Can I use my own shapes?**
A: Yes! Add new shape classes in `shape_movement.py`, then update the dataset generation logic in `knn_mlp.py`. See `CUSTOM_DATASET_GUIDE.md` for details.

---

## Version History

- **2024-12**: Enhanced feature extraction, relative coordinates, comprehensive caching
- **2024-11**: V2e integration, temporal filtering, texture support
- **2024-10**: Initial implementation with star8 dataset

---

## Contact & Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check existing documentation in `toy_datasets/*.md`
- Review example scripts and test cases

**Happy experimenting! 🚀**
