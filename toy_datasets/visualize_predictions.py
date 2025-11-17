"""
Visualization script for comparing predicted optical flow with ground truth.

Usage:
    python toy_datasets/visualize_predictions.py --predictions_path evaluations/toydataset-tinyIDNet-multiseed/fxd5fk65/predictions.pt
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# ensure repo root and toy_datasets on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(__file__))

from star8 import StarMovement

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_TOTAL_FRAMES = 2000
DEFAULT_SAVE_STEP = 40
DEFAULT_TEST_SIZE = 0.2


def load_dataset_metadata(data_root: str, seq_name: str):
    """Load dataset metadata from JSON file.
    
    Args:
        data_root: Root directory (e.g., 'toy_datasets/data/star8')
        seq_name: Sequence name (e.g., 'star8' or 'star8_test')
        
    Returns:
        dict: Metadata dictionary with generation parameters, or None if not found
    """
    import json
    metadata_path = os.path.join(data_root, "train_optical_flow", seq_name, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"⚠️  Warning: Metadata file not found at {metadata_path}")
        print(f"   Using default parameters. Consider regenerating the dataset to save metadata.")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata from {metadata_path}")
    return metadata


def flow_to_color(u, v, max_flow=None):
    """
    Convert optical flow (u, v) to RGB color using the standard flow color wheel.
    
    Args:
        u, v: Flow components [H, W]
        max_flow: Maximum flow magnitude for normalization (None for auto)
    
    Returns:
        rgb: RGB image [H, W, 3] in range [0, 1]
    """
    # Compute flow magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)  # Angle in radians [-pi, pi]
    
    # Normalize magnitude
    if max_flow is None:
        max_flow = np.max(mag) if np.max(mag) > 0 else 1.0
    
    mag_normalized = np.clip(mag / max_flow, 0, 1)
    
    # Convert angle to hue [0, 1] using standard flow color wheel convention:
    # 0° (right) -> hue=0 (red), 90° (down) -> hue=0.25 (yellow), 
    # 180° (left) -> hue=0.5 (cyan), 270° (up) -> hue=0.75 (blue)
    hue = (ang / (2 * np.pi)) % 1.0  # Normalize to [0, 1], wrapping negatives
    
    # Create HSV image
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.float32)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = mag_normalized  # Saturation based on magnitude
    hsv[:, :, 2] = 1.0  # Full brightness
    
    # Convert HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    
    return rgb


def create_flow_color_wheel(size=200):
    """
    Create a flow color wheel for reference.
    
    Args:
        size: Size of the wheel image (size x size)
    
    Returns:
        wheel: RGB image [size, size, 3] showing the flow color wheel
    """
    # Create coordinate grid
    y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    
    # Compute flow components (u, v) for each point
    # In image coordinates: u is horizontal (x), v is vertical (y)
    # Note: y increases downward in image space, so positive y = downward flow
    u = x
    v = y
    
    # Create mask for circular region
    radius = np.sqrt(x**2 + y**2)
    mask = radius <= 1.0
    
    # Convert to color
    wheel = flow_to_color(u, v, max_flow=1.0)
    
    # Set outside circle to white
    wheel[~mask] = 1.0
    
    return wheel


def compute_epe(u_pred, v_pred, u_gt, v_gt, valid_mask):
    """
    Compute End Point Error (EPE) between predicted and ground truth flow.
    
    Args:
        u_pred, v_pred: Predicted flow components [H, W]
        u_gt, v_gt: Ground truth flow components [H, W]
        valid_mask: Valid pixels mask [H, W]
    
    Returns:
        epe: End point error map [H, W], NaN for invalid pixels
        mean_epe: Mean EPE over valid pixels
    """
    # Compute EPE: sqrt((u_pred - u_gt)^2 + (v_pred - v_gt)^2)
    epe = np.sqrt((u_pred - u_gt) ** 2 + (v_pred - v_gt) ** 2)
    
    # Mask invalid pixels with NaN
    epe_masked = epe.copy()
    epe_masked[~valid_mask] = np.nan
    
    # Compute mean over valid pixels
    mean_epe = np.nanmean(epe_masked)
    
    return epe_masked, mean_epe


def compute_angular_error(u_pred, v_pred, u_gt, v_gt, valid_mask):
    """
    Compute Angular Error (AE) between predicted and ground truth flow.
    
    Uses the standard 3D formulation with homogeneous coordinates (u, v, 1):
    AE = arccos((u_pred*u_gt + v_pred*v_gt + 1) / sqrt((u_pred^2 + v_pred^2 + 1) * (u_gt^2 + v_gt^2 + 1)))
    
    Why (u, v, 1) instead of (u, v)?
    - Standard convention in optical flow benchmarks (Middlebury, KITTI, etc.)
    - More numerically stable for small flow magnitudes (avoids division by ~0)
    - Accounts for both direction AND magnitude relationship
    - The '1' term ensures well-defined angles even when flows are near zero
    
    Alternative 2D formulation (pure directional) would be:
    AE_2D = arccos((u_pred*u_gt + v_pred*v_gt) / (||pred|| * ||gt||))
    But this ignores magnitude and is less stable.
    
    Args:
        u_pred, v_pred: Predicted flow components [H, W]
        u_gt, v_gt: Ground truth flow components [H, W]
        valid_mask: Valid pixels mask [H, W]
    
    Returns:
        ae: Angular error map in degrees [H, W], NaN for invalid pixels
        mean_ae: Mean angular error over valid pixels in degrees
    """
    # Compute numerator: dot product of (u, v, 1) vectors in 3D homogeneous space
    numerator = u_pred * u_gt + v_pred * v_gt + 1.0
    
    # Compute denominator: product of vector magnitudes
    pred_mag = np.sqrt(u_pred ** 2 + v_pred ** 2 + 1.0)
    gt_mag = np.sqrt(u_gt ** 2 + v_gt ** 2 + 1.0)
    denominator = pred_mag * gt_mag
    
    # Compute angular error in radians, then convert to degrees
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip(numerator / (denominator + 1e-10), -1.0, 1.0)
    ae = np.arccos(cos_angle) * 180.0 / np.pi
    
    # Mask invalid pixels with NaN
    ae_masked = ae.copy()
    ae_masked[~valid_mask] = np.nan
    
    # Compute mean over valid pixels
    mean_ae = np.nanmean(ae_masked)
    
    return ae_masked, mean_ae


def get_vertices(star: StarMovement, frame_idx: int):
    """Get star vertices at a specific frame."""
    star.update_shape(frame_idx)
    return star.transformed_path.vertices


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def visualize_predictions(
    predictions_path: str,
    data_root: str,
    seq_name: str,
    output_dir: str,
    split: str = "train",
    max_samples: int = None,
    image_size: tuple = None,
    total_frames: int = None,
    save_step: int = None,
    test_size: float = None,
    require_metadata: bool = True,
    visualize_errors: bool = True,
    visualize_masks: bool = True,
):
    """
    Visualize predictions vs ground truth for a sequence.
    
    IMPORTANT: This function requires metadata to ensure visualization accuracy.
    When datasets are regenerated with different parameters, the metadata must match
    the actual dataset used for training/evaluation.
    
    Args:
        predictions_path: Path to predictions.pt file
        data_root: Root directory of the dataset (e.g., data/star8)
        seq_name: Sequence name (e.g., star8 or star8_test)
        output_dir: Directory to save visualization images
        split: 'train' or 'val'
        max_samples: Maximum number of samples to visualize (None for all)
        image_size: (H, W) image size (None to load from metadata)
        total_frames: Total frames in sequence (None to load from metadata)
        save_step: Frame step between flow pairs (None to load from metadata)
        test_size: Test split fraction (None to load from metadata)
        require_metadata: If True, fail if metadata is missing (default: True for safety)
        visualize_errors: If True, generate EPE and AE error visualizations (default: True)
        visualize_masks: If True, generate mask visualizations for GT and predictions (default: True)
    """
    # Load predictions
    print(f"🔹 Loading predictions from {predictions_path}")
    results = torch.load(predictions_path, map_location='cpu',weights_only=False)
    
    if split not in results:
        print(f"❌ Split '{split}' not found in predictions file. Available: {list(results.keys())}")
        return 0
    
    # Get run_id from predictions metadata for output path
    run_id = None
    if 'metadata' in results:
        run_id = results['metadata'].get('run_id')
        if run_id:
            print(f"   ✓ Found run_id in predictions: {run_id}")
    
    # Try to load metadata first
    metadata = load_dataset_metadata(data_root, seq_name)
    
    # Use metadata if available, otherwise use provided parameters or defaults
    if metadata:
        image_size = image_size or tuple(metadata['image_size'])
        total_frames = total_frames or metadata['total_frames']
        save_step = save_step or metadata['save_step']
        test_size = test_size if test_size is not None else metadata['test_size']
        print(f"   ✓ Using parameters from metadata:")
        print(f"     image_size: {image_size}")
        print(f"     total_frames: {total_frames}")
        print(f"     save_step: {save_step}")
        print(f"     test_size: {test_size}")
        
        # Store metadata in output for reference
        import json
        if run_id:
            metadata_out = os.path.join(output_dir, run_id, seq_name, split, "visualization_metadata.json")
        else:
            metadata_out = os.path.join(output_dir, seq_name, split, "visualization_metadata.json")
        ensure_dir(os.path.dirname(metadata_out))
        with open(metadata_out, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved visualization metadata to {metadata_out}")
    else:
        if require_metadata:
            print(f"\n{'='*60}")
            print(f"❌ ERROR: Metadata file not found for sequence '{seq_name}'")
            print(f"   Location checked: {data_root}/train_optical_flow/{seq_name}/dataset_metadata.json")
            print(f"\n   This is critical because:")
            print(f"   - Dataset parameters (save_step, total_frames, etc.) affect visualization")
            print(f"   - Using wrong parameters will misalign predictions with ground truth")
            print(f"   - Multiple trainings may use datasets with different parameters")
            print(f"\n   Solutions:")
            print(f"   1. Regenerate the dataset to create metadata file")
            print(f"   2. Manually provide parameters: --save_step X --total_frames Y --image_width W --image_height H")
            print(f"   3. Set --no-require-metadata flag (NOT RECOMMENDED)")
            print(f"{'='*60}\n")
            raise FileNotFoundError(f"Metadata file required but not found for sequence '{seq_name}'")
        
        # Fall back to provided parameters or defaults
        image_size = image_size or DEFAULT_IMAGE_SIZE
        total_frames = total_frames or DEFAULT_TOTAL_FRAMES
        save_step = save_step or DEFAULT_SAVE_STEP
        test_size = test_size if test_size is not None else DEFAULT_TEST_SIZE
        print(f"   ⚠️  WARNING: Using provided/default parameters (metadata not found):")
        print(f"     image_size: {image_size}")
        print(f"     total_frames: {total_frames}")
        print(f"     save_step: {save_step}")
        print(f"     test_size: {test_size}")
        print(f"   ⚠️  Visualization may be incorrect if these don't match the actual dataset!")
    
    H, W = image_size
    
    split_data = results[split]
    predictions = split_data['predictions']
    ground_truths = split_data['ground_truths']
    valid_masks = split_data['valid_masks']
    file_indices = split_data.get('file_indices', None)  # For PNG matching
    timestamps = split_data.get('timestamps', None)      # For frame calculation
    
    print(f"   Found {len(predictions)} samples in {split} split")
    
    # Check if we have timestamps and file_indices (new sample-based format)
    if timestamps is None or file_indices is None:
        print(f"   ⚠️  Warning: predictions.pt missing 'timestamps' or 'file_indices'")
        print(f"   This is an older batch-based format. Visualization may be less accurate.")
        print(f"   Consider re-running evaluation with updated eval_toy_dataset.py")
    
    # Create output directories with run_id if available (run_id was extracted earlier)
    if run_id:
        vis_dir = ensure_dir(os.path.join(output_dir, run_id, seq_name, split))
        comparison_dir = ensure_dir(os.path.join(output_dir, run_id, seq_name, split, "comparisons"))
    else:
        vis_dir = ensure_dir(os.path.join(output_dir, seq_name, split))
        comparison_dir = ensure_dir(os.path.join(output_dir, seq_name, split, "comparisons"))
    
    print(f"   Output directory: {vis_dir}")
    print(f"   Comparison visualization directory: {comparison_dir}")
    
    if visualize_errors:
        error_dir = ensure_dir(os.path.join(vis_dir, "errors"))
        print(f"   Error visualization directory: {error_dir}")
    
    if visualize_masks:
        mask_dir = ensure_dir(os.path.join(vis_dir, "masks"))
        print(f"   Mask visualization directory: {mask_dir}")
    
    # Create star movement object
    star = StarMovement(total_frames=total_frames, image_size=image_size, face_color="black")
    
    # Convert predictions, ground truths, and masks to tensors if they're lists
    # (they should already be individual samples, not batches)
    if isinstance(predictions, list):
        all_preds = torch.stack(predictions, dim=0)  # [N, 2, H, W]
    else:
        all_preds = predictions
        
    if isinstance(ground_truths, list):
        all_gts = torch.stack(ground_truths, dim=0)  # [N, 2, H, W]
    else:
        all_gts = ground_truths
        
    if isinstance(valid_masks, list):
        all_valid = torch.stack(valid_masks, dim=0)  # [N, 1, H, W]
    else:
        all_valid = valid_masks
    
    print(f"   Total samples: {all_preds.shape[0]}")
    
    # Get frame_time_us and flow_dt_us from metadata for timestamp-to-frame conversion
    if metadata and 'frame_time_us' in metadata:
        frame_time_us = metadata['frame_time_us']
        flow_dt_us = metadata.get('flow_dt_us', save_step * frame_time_us)
    else:
        # Fallback defaults
        frame_time_us = 1000  # 1ms per frame
        flow_dt_us = save_step * frame_time_us
        if timestamps is not None:
            print(f"   ⚠️  Warning: frame_time_us not in metadata, using defaults")
            print(f"      frame_time_us={frame_time_us} us, flow_dt_us={flow_dt_us} us")
    
    # Limit samples if requested
    num_samples = all_preds.shape[0]
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    
    saved = 0
    for idx in range(num_samples):
        # Determine frame numbers from timestamps if available
        split_start_frame = int(np.floor(total_frames * (1.0 - test_size))) if test_size > 0 else total_frames
        if split == "train":
            base_from = 0
        else:
            base_from = split_start_frame
        
        if timestamps is not None and idx < len(timestamps):
            # Get the timestamp for this sample (start of interval)
            ts_start = timestamps[idx]
            if isinstance(ts_start, torch.Tensor):
                ts_start = ts_start.item()
            ts_start += base_from * frame_time_us
            
            # Calculate frame numbers from timestamp
            # timestamps[idx] is the start of the interval, duration is flow_dt_us
            # So the flow represents motion from ts_start to (ts_start + flow_dt_us)
            ts_from = ts_start
            ts_to = ts_start + flow_dt_us
            
            frame_from = int(ts_from / frame_time_us)
            frame_to = int(ts_to / frame_time_us)
        else:
            # Fallback: use sequential frame numbers based on split and save_step
            # This is less accurate but works if timestamps are missing
            # Give warnning if the code is using fallback
            if timestamps is None:
                print(f"   ⚠️  Warning: Missing timestamps, using fallback frame calculation for sample {idx}")
            
            frame_from = base_from + idx * save_step
            frame_to = frame_from + save_step
        
        if frame_to > total_frames:
            print(f"   [skip] Sample {idx}: frame_to={frame_to} exceeds total_frames={total_frames}")
            break
        
        # Get vertices at start and end frames
        verts_start = get_vertices(star, frame_from)
        verts_end = get_vertices(star, frame_to)
        
        # Get ground truth flow from predictions.pt
        gt_flow = all_gts[idx].numpy()  # [2, H, W]
        u_gt = gt_flow[0]
        v_gt = gt_flow[1]
        
        # Get GT valid mask from predictions.pt
        valid_gt = all_valid[idx, 0].numpy() > 0.5  # [H, W]
        
        # Get predicted flow
        pred_flow = all_preds[idx].numpy()  # [2, H, W]
        u_pred = pred_flow[0]
        v_pred = pred_flow[1]
        
        # Get valid mask (same as GT valid mask from predictions.pt)
        valid_mask = valid_gt  # Both GT and pred use the same valid mask
        
        # Create figure with 3 subplots: left=GT, middle=Pred, right=Comparison
        fig = plt.figure(figsize=(18, 6), dpi=150)
        
        # --- Subplot 1: Ground Truth Flow ---
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_xlim(0, W)
        ax1.set_ylim(0, H)
        ax1.set_aspect("equal")
        ax1.invert_yaxis()
        ax1.set_title(f"Ground Truth Flow\nframes {frame_from}->{frame_to}")
        
        # Plot start vertices (blue) with connecting lines
        if len(verts_start) > 0:
            ax1.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
            vs = verts_start
            if not np.allclose(vs[0], vs[-1]):
                vs = np.vstack([vs, vs[:1]])
            ax1.plot(vs[:, 0], vs[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Plot end vertices (red) with connecting lines
        if len(verts_end) > 0:
            ax1.scatter(verts_end[:, 0], verts_end[:, 1], c='red', s=20, label='end vertices', zorder=5)
            ve = verts_end
            if not np.allclose(ve[0], ve[-1]):
                ve = np.vstack([ve, ve[:1]])
            ax1.plot(ve[:, 0], ve[:, 1], '-', color='red', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Draw GT flow arrows at vertices (green)
        vy_valid_gt, vx_valid_gt = np.where(valid_gt)
        has_any_valid_gt = len(vx_valid_gt) > 0
        arrow_count_gt = 0
        for vx, vy in verts_start:
            px, py = int(round(vx)), int(round(vy))
            du, dv = None, None
            if 0 <= px < W and 0 <= py < H and valid_gt[py, px]:
                du = u_gt[py, px]
                dv = v_gt[py, px]
            elif has_any_valid_gt:
                dx = vx_valid_gt.astype(np.float32) - float(px)
                dy = vy_valid_gt.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_gt[idx_min])
                px_n = int(vx_valid_gt[idx_min])
                du = u_gt[py_n, px_n]
                dv = v_gt[py_n, px_n]
            if du is not None and dv is not None:
                ax1.arrow(vx, vy, du, dv, color='green', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.9, zorder=6)
                arrow_count_gt += 1
        ax1.legend(loc='upper right', fontsize=8)
        
        # --- Subplot 2: Predicted Flow ---
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_xlim(0, W)
        ax2.set_ylim(0, H)
        ax2.set_aspect("equal")
        ax2.invert_yaxis()
        ax2.set_title(f"Predicted Flow\nframes {frame_from}->{frame_to}")
        
        # Plot start vertices (blue)
        if len(verts_start) > 0:
            ax2.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
            vs = verts_start
            if not np.allclose(vs[0], vs[-1]):
                vs = np.vstack([vs, vs[:1]])
            ax2.plot(vs[:, 0], vs[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Plot end vertices (red)
        if len(verts_end) > 0:
            ax2.scatter(verts_end[:, 0], verts_end[:, 1], c='red', s=20, label='end vertices', zorder=5)
            ve = verts_end
            if not np.allclose(ve[0], ve[-1]):
                ve = np.vstack([ve, ve[:1]])
            ax2.plot(ve[:, 0], ve[:, 1], '-', color='red', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Draw predicted flow arrows at vertices (orange/gold color for distinction)
        vy_valid_pred, vx_valid_pred = np.where(valid_mask)
        has_any_valid_pred = len(vx_valid_pred) > 0
        arrow_count_pred = 0
        for vx, vy in verts_start:
            px, py = int(round(vx)), int(round(vy))
            du, dv = None, None
            if 0 <= px < W and 0 <= py < H and valid_mask[py, px]:
                du = u_pred[py, px]
                dv = v_pred[py, px]
            elif has_any_valid_pred:
                dx = vx_valid_pred.astype(np.float32) - float(px)
                dy = vy_valid_pred.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_pred[idx_min])
                px_n = int(vx_valid_pred[idx_min])
                du = u_pred[py_n, px_n]
                dv = v_pred[py_n, px_n]
            if du is not None and dv is not None:
                ax2.arrow(vx, vy, du, dv, color='darkorange', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.9, zorder=6)
                arrow_count_pred += 1
        ax2.legend(loc='upper right', fontsize=8)
        
        # --- Subplot 3: GT + Predicted Overlay ---
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_xlim(0, W)
        ax3.set_ylim(0, H)
        ax3.set_aspect("equal")
        ax3.invert_yaxis()
        ax3.set_title(f"GT (green) vs Predicted (orange)\nframes {frame_from}->{frame_to}")
        
        # Plot start vertices (blue) - without connecting edges
        if len(verts_start) > 0:
            ax3.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
        
        # End vertices and edges removed for clarity (too crowded)
        
        # Draw both GT (green) and predicted (orange) arrows
        for vx, vy in verts_start:
            px, py = int(round(vx)), int(round(vy))
            
            # GT arrow (green)
            du_gt, dv_gt = None, None
            if 0 <= px < W and 0 <= py < H and valid_gt[py, px]:
                du_gt = u_gt[py, px]
                dv_gt = v_gt[py, px]
            elif has_any_valid_gt:
                dx = vx_valid_gt.astype(np.float32) - float(px)
                dy = vy_valid_gt.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_gt[idx_min])
                px_n = int(vx_valid_gt[idx_min])
                du_gt = u_gt[py_n, px_n]
                dv_gt = v_gt[py_n, px_n]
            if du_gt is not None and dv_gt is not None:
                ax3.arrow(vx, vy, du_gt, dv_gt, color='green', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.8, zorder=6, 
                         label='GT flow' if vx == verts_start[0, 0] else '')
            
            # Predicted arrow (orange)
            du_pred, dv_pred = None, None
            if 0 <= px < W and 0 <= py < H and valid_mask[py, px]:
                du_pred = u_pred[py, px]
                dv_pred = v_pred[py, px]
            elif has_any_valid_pred:
                dx = vx_valid_pred.astype(np.float32) - float(px)
                dy = vy_valid_pred.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_pred[idx_min])
                px_n = int(vx_valid_pred[idx_min])
                du_pred = u_pred[py_n, px_n]
                dv_pred = v_pred[py_n, px_n]
            if du_pred is not None and dv_pred is not None:
                ax3.arrow(vx, vy, du_pred, dv_pred, color='darkorange', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.8, zorder=7,
                         label='Predicted flow' if vx == verts_start[0, 0] else '')
        
        # Add legend with unique labels only
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        # Save combined figure
        out_path = os.path.join(comparison_dir, f"comparison_{idx:06d}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        
        # Save each subplot separately for individual GIFs
        # Create subdirectories for each subplot
        gt_dir = ensure_dir(os.path.join(comparison_dir, "gt_flow"))
        pred_dir = ensure_dir(os.path.join(comparison_dir, "pred_flow"))
        overlay_dir = ensure_dir(os.path.join(comparison_dir, "overlay"))
        
        # Save GT subplot
        fig1 = plt.figure(figsize=(6, 6), dpi=150)
        ax1_solo = fig1.add_subplot(1, 1, 1)
        ax1_solo.set_xlim(0, W)
        ax1_solo.set_ylim(0, H)
        ax1_solo.set_aspect("equal")
        ax1_solo.invert_yaxis()
        ax1_solo.set_title(f"Ground Truth Flow\nframes {frame_from}->{frame_to}")
        
        if len(verts_start) > 0:
            ax1_solo.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
            vs = verts_start
            if not np.allclose(vs[0], vs[-1]):
                vs = np.vstack([vs, vs[:1]])
            ax1_solo.plot(vs[:, 0], vs[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, zorder=4)
        
        if len(verts_end) > 0:
            ax1_solo.scatter(verts_end[:, 0], verts_end[:, 1], c='red', s=20, label='end vertices', zorder=5)
            ve = verts_end
            if not np.allclose(ve[0], ve[-1]):
                ve = np.vstack([ve, ve[:1]])
            ax1_solo.plot(ve[:, 0], ve[:, 1], '-', color='red', alpha=0.6, linewidth=1.5, zorder=4)
        
        for vx, vy in verts_start:
            px, py = int(round(vx)), int(round(vy))
            du, dv = None, None
            if 0 <= px < W and 0 <= py < H and valid_gt[py, px]:
                du = u_gt[py, px]
                dv = v_gt[py, px]
            elif has_any_valid_gt:
                dx = vx_valid_gt.astype(np.float32) - float(px)
                dy = vy_valid_gt.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_gt[idx_min])
                px_n = int(vx_valid_gt[idx_min])
                du = u_gt[py_n, px_n]
                dv = v_gt[py_n, px_n]
            if du is not None and dv is not None:
                ax1_solo.arrow(vx, vy, du, dv, color='green', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.9, zorder=6)
        ax1_solo.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        fig1.savefig(os.path.join(gt_dir, f"gt_{idx:06d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # Save Pred subplot
        fig2 = plt.figure(figsize=(6, 6), dpi=150)
        ax2_solo = fig2.add_subplot(1, 1, 1)
        ax2_solo.set_xlim(0, W)
        ax2_solo.set_ylim(0, H)
        ax2_solo.set_aspect("equal")
        ax2_solo.invert_yaxis()
        ax2_solo.set_title(f"Predicted Flow\nframes {frame_from}->{frame_to}")
        
        if len(verts_start) > 0:
            ax2_solo.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
            vs = verts_start
            if not np.allclose(vs[0], vs[-1]):
                vs = np.vstack([vs, vs[:1]])
            ax2_solo.plot(vs[:, 0], vs[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, zorder=4)
        
        if len(verts_end) > 0:
            ax2_solo.scatter(verts_end[:, 0], verts_end[:, 1], c='red', s=20, label='end vertices', zorder=5)
            ve = verts_end
            if not np.allclose(ve[0], ve[-1]):
                ve = np.vstack([ve, ve[:1]])
            ax2_solo.plot(ve[:, 0], ve[:, 1], '-', color='red', alpha=0.6, linewidth=1.5, zorder=4)
        
        for vx, vy in verts_start:
            px, py = int(round(vx)), int(round(vy))
            du, dv = None, None
            if 0 <= px < W and 0 <= py < H and valid_mask[py, px]:
                du = u_pred[py, px]
                dv = v_pred[py, px]
            elif has_any_valid_pred:
                dx = vx_valid_pred.astype(np.float32) - float(px)
                dy = vy_valid_pred.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_pred[idx_min])
                px_n = int(vx_valid_pred[idx_min])
                du = u_pred[py_n, px_n]
                dv = v_pred[py_n, px_n]
            if du is not None and dv is not None:
                ax2_solo.arrow(vx, vy, du, dv, color='darkorange', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.9, zorder=6)
        ax2_solo.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        fig2.savefig(os.path.join(pred_dir, f"pred_{idx:06d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # Save Overlay subplot
        fig3 = plt.figure(figsize=(6, 6), dpi=150)
        ax3_solo = fig3.add_subplot(1, 1, 1)
        ax3_solo.set_xlim(0, W)
        ax3_solo.set_ylim(0, H)
        ax3_solo.set_aspect("equal")
        ax3_solo.invert_yaxis()
        ax3_solo.set_title(f"GT (green) vs Predicted (orange)\nframes {frame_from}->{frame_to}")
        
        if len(verts_start) > 0:
            ax3_solo.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
        
        for vx, vy in verts_start:
            px, py = int(round(vx)), int(round(vy))
            
            # GT arrow (green)
            du_gt, dv_gt = None, None
            if 0 <= px < W and 0 <= py < H and valid_gt[py, px]:
                du_gt = u_gt[py, px]
                dv_gt = v_gt[py, px]
            elif has_any_valid_gt:
                dx = vx_valid_gt.astype(np.float32) - float(px)
                dy = vy_valid_gt.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_gt[idx_min])
                px_n = int(vx_valid_gt[idx_min])
                du_gt = u_gt[py_n, px_n]
                dv_gt = v_gt[py_n, px_n]
            if du_gt is not None and dv_gt is not None:
                ax3_solo.arrow(vx, vy, du_gt, dv_gt, color='green', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.8, zorder=6, 
                         label='GT flow' if vx == verts_start[0, 0] else '')
            
            # Predicted arrow (orange)
            du_pred, dv_pred = None, None
            if 0 <= px < W and 0 <= py < H and valid_mask[py, px]:
                du_pred = u_pred[py, px]
                dv_pred = v_pred[py, px]
            elif has_any_valid_pred:
                dx = vx_valid_pred.astype(np.float32) - float(px)
                dy = vy_valid_pred.astype(np.float32) - float(py)
                idx_min = np.argmin(dx * dx + dy * dy)
                py_n = int(vy_valid_pred[idx_min])
                px_n = int(vx_valid_pred[idx_min])
                du_pred = u_pred[py_n, px_n]
                dv_pred = v_pred[py_n, px_n]
            if du_pred is not None and dv_pred is not None:
                ax3_solo.arrow(vx, vy, du_pred, dv_pred, color='darkorange', width=0.5, head_width=3.0, 
                         head_length=4.0, length_includes_head=True, alpha=0.8, zorder=7,
                         label='Predicted flow' if vx == verts_start[0, 0] else '')
        
        handles, labels = ax3_solo.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3_solo.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        plt.tight_layout()
        fig3.savefig(os.path.join(overlay_dir, f"overlay_{idx:06d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        plt.close(fig)
        saved += 1
        
        # Create detailed log message
        log_msg = f"[saved] {out_path}"
        log_msg += f" | Sample {idx}: frames {frame_from}->{frame_to}"
        if timestamps is not None and idx < len(timestamps):
            ts_val = timestamps[idx]
            if isinstance(ts_val, torch.Tensor):
                ts_val = ts_val.item()
            log_msg += f" | timestamp={ts_val}us"
        if file_indices is not None and idx < len(file_indices):
            fi_val = file_indices[idx]
            if isinstance(fi_val, torch.Tensor):
                fi_val = fi_val.item()
            log_msg += f" | file_idx={fi_val}"
        log_msg += f" | GT arrows: {arrow_count_gt}, Pred arrows: {arrow_count_pred}"
        print(log_msg)
        
        # Generate error visualizations if requested
        if visualize_errors:
            # Compute EPE and Angular Error
            combined_valid = valid_gt & valid_mask
            epe_map, mean_epe = compute_epe(u_pred, v_pred, u_gt, v_gt, combined_valid)
            ae_map, mean_ae = compute_angular_error(u_pred, v_pred, u_gt, v_gt, combined_valid)
            
            # Convert flows to color for visualization
            # Use same max_flow for both to make them comparable
            max_flow = max(np.nanmax(np.sqrt(u_gt**2 + v_gt**2)), 
                          np.nanmax(np.sqrt(u_pred**2 + v_pred**2)))
            if not np.isfinite(max_flow) or max_flow == 0:
                max_flow = 1.0
            
            flow_gt_color = flow_to_color(u_gt, v_gt, max_flow=max_flow)
            flow_pred_color = flow_to_color(u_pred, v_pred, max_flow=max_flow)
            
            # Mask invalid regions (set to white)
            flow_gt_color[~valid_gt] = 1.0
            flow_pred_color[~valid_mask] = 1.0
            
            # Create flow color wheel for reference
            wheel = create_flow_color_wheel(size=150)
            
            # Create error visualization figure with 2x3 grid
            fig_err = plt.figure(figsize=(21, 12), dpi=150)
            
            # --- Row 1: Ground Truth Flow ---
            ax_gt_flow = fig_err.add_subplot(2, 3, 1)
            ax_gt_flow.imshow(flow_gt_color, interpolation='nearest')
            ax_gt_flow.set_title(f"Ground Truth Flow\nframes {frame_from}->{frame_to}\nMax: {max_flow:.2f} px/frame", 
                               fontsize=11, fontweight='bold')
            ax_gt_flow.set_xlabel("X (pixels)", fontsize=9)
            ax_gt_flow.set_ylabel("Y (pixels)", fontsize=9)
            ax_gt_flow.axis('off')
            
            # Overlay vertices
            if len(verts_start) > 0:
                ax_gt_flow.scatter(verts_start[:, 0], verts_start[:, 1], c='white', 
                                  s=20, marker='o', edgecolors='black', linewidths=1.5, 
                                  label='vertices', zorder=10, alpha=0.8)
            
            # --- Row 1: Predicted Flow ---
            ax_pred_flow = fig_err.add_subplot(2, 3, 2)
            ax_pred_flow.imshow(flow_pred_color, interpolation='nearest')
            ax_pred_flow.set_title(f"Predicted Flow\nframes {frame_from}->{frame_to}\nMax: {max_flow:.2f} px/frame", 
                                 fontsize=11, fontweight='bold')
            ax_pred_flow.set_xlabel("X (pixels)", fontsize=9)
            ax_pred_flow.set_ylabel("Y (pixels)", fontsize=9)
            ax_pred_flow.axis('off')
            
            # Overlay vertices
            if len(verts_start) > 0:
                ax_pred_flow.scatter(verts_start[:, 0], verts_start[:, 1], c='white', 
                                   s=20, marker='o', edgecolors='black', linewidths=1.5, 
                                   label='vertices', zorder=10, alpha=0.8)
            
            # --- Row 1: Flow Color Wheel Reference ---
            ax_wheel = fig_err.add_subplot(2, 3, 3)
            ax_wheel.imshow(wheel, interpolation='nearest')
            ax_wheel.set_title("Flow Color Wheel\n(Direction & Magnitude)", fontsize=11, fontweight='bold')
            ax_wheel.axis('off')
            
            # Add directional labels
            wheel_size = wheel.shape[0]
            center = wheel_size // 2
            ax_wheel.text(wheel_size - 10, center, '→', fontsize=20, ha='right', va='center', color='black')
            ax_wheel.text(10, center, '←', fontsize=20, ha='left', va='center', color='black')
            ax_wheel.text(center, 10, '↑', fontsize=20, ha='center', va='top', color='black')
            ax_wheel.text(center, wheel_size - 10, '↓', fontsize=20, ha='center', va='bottom', color='black')
            ax_wheel.text(center, wheel_size + 20, f'Max: {max_flow:.2f} px', 
                         fontsize=9, ha='center', va='top', transform=ax_wheel.transData)
            
            # --- Row 2: End Point Error (EPE) ---
            ax_epe = fig_err.add_subplot(2, 3, 4)
            
            # Plot EPE heatmap
            im_epe = ax_epe.imshow(epe_map, cmap='hot', interpolation='nearest')
            ax_epe.set_title(f"End Point Error (EPE)\nframes {frame_from}->{frame_to}\nMean EPE: {mean_epe:.3f} px", 
                           fontsize=11, fontweight='bold')
            ax_epe.set_xlabel("X (pixels)", fontsize=9)
            ax_epe.set_ylabel("Y (pixels)", fontsize=9)
            
            # Add colorbar with enhanced formatting
            cbar_epe = plt.colorbar(im_epe, ax=ax_epe, fraction=0.046, pad=0.04)
            cbar_epe.set_label("EPE (pixels)", rotation=270, labelpad=20, fontsize=9, fontweight='bold')
            cbar_epe.ax.tick_params(labelsize=8)
            
            # Overlay vertices for reference
            if len(verts_start) > 0:
                ax_epe.scatter(verts_start[:, 0], verts_start[:, 1], c='cyan', 
                              s=30, marker='o', edgecolors='blue', linewidths=1.5, 
                              label='vertices', zorder=10, alpha=0.8)
            
            ax_epe.legend(loc='upper right', fontsize=8)
            
            # --- Row 2: Angular Error (AE) ---
            ax_ae = fig_err.add_subplot(2, 3, 5)
            
            # Plot AE heatmap
            im_ae = ax_ae.imshow(ae_map, cmap='hot', interpolation='nearest')
            ax_ae.set_title(f"Angular Error (AE)\nframes {frame_from}->{frame_to}\nMean AE: {mean_ae:.3f}°", 
                          fontsize=11, fontweight='bold')
            ax_ae.set_xlabel("X (pixels)", fontsize=9)
            ax_ae.set_ylabel("Y (pixels)", fontsize=9)
            
            # Add colorbar with enhanced formatting
            cbar_ae = plt.colorbar(im_ae, ax=ax_ae, fraction=0.046, pad=0.04)
            cbar_ae.set_label("AE (degrees)", rotation=270, labelpad=20, fontsize=9, fontweight='bold')
            cbar_ae.ax.tick_params(labelsize=8)
            
            # Overlay vertices for reference
            if len(verts_start) > 0:
                ax_ae.scatter(verts_start[:, 0], verts_start[:, 1], c='cyan', 
                             s=30, marker='o', edgecolors='blue', linewidths=1.5, 
                             label='vertices', zorder=10, alpha=0.8)
            
            ax_ae.legend(loc='upper right', fontsize=8)
            
            # --- Row 2: Error Statistics Summary ---
            ax_stats = fig_err.add_subplot(2, 3, 6)
            ax_stats.axis('off')
            
            # Compute additional statistics
            valid_count = combined_valid.sum()
            total_count = combined_valid.size
            valid_pct = 100 * valid_count / total_count if total_count > 0 else 0
            
            epe_valid = epe_map[~np.isnan(epe_map)]
            ae_valid = ae_map[~np.isnan(ae_map)]
            
            stats_text = f"Error Statistics\n" + "="*40 + "\n\n"
            stats_text += f"Valid Pixels: {valid_count:,} / {total_count:,} ({valid_pct:.1f}%)\n\n"
            stats_text += f"End Point Error (EPE):\n"
            stats_text += f"  Mean:   {mean_epe:.4f} px\n"
            if len(epe_valid) > 0:
                stats_text += f"  Median: {np.median(epe_valid):.4f} px\n"
                stats_text += f"  Std:    {np.std(epe_valid):.4f} px\n"
                stats_text += f"  Min:    {np.min(epe_valid):.4f} px\n"
                stats_text += f"  Max:    {np.max(epe_valid):.4f} px\n\n"
            
            stats_text += f"Angular Error (AE):\n"
            stats_text += f"  Mean:   {mean_ae:.4f}°\n"
            if len(ae_valid) > 0:
                stats_text += f"  Median: {np.median(ae_valid):.4f}°\n"
                stats_text += f"  Std:    {np.std(ae_valid):.4f}°\n"
                stats_text += f"  Min:    {np.min(ae_valid):.4f}°\n"
                stats_text += f"  Max:    {np.max(ae_valid):.4f}°\n\n"
            
            stats_text += f"Flow Magnitudes:\n"
            stats_text += f"  GT Max:   {max_flow:.4f} px/frame\n"
            gt_mag = np.sqrt(u_gt**2 + v_gt**2)
            pred_mag = np.sqrt(u_pred**2 + v_pred**2)
            stats_text += f"  GT Mean:  {np.mean(gt_mag[valid_gt]):.4f} px/frame\n"
            stats_text += f"  Pred Mean: {np.mean(pred_mag[valid_mask]):.4f} px/frame\n"
            
            ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                         fontsize=9, verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            # Save error visualization
            error_path = os.path.join(error_dir, f"errors_{idx:06d}.png")
            plt.tight_layout()
            fig_err.savefig(error_path, dpi=150, bbox_inches='tight')
            plt.close(fig_err)
            
            print(f"[saved] {error_path} (Mean EPE: {mean_epe:.3f} px, Mean AE: {mean_ae:.3f}°)")
        
        # Generate mask visualizations if requested
        if visualize_masks:
            # Create mask visualization figure with 3 subplots
            fig_mask = plt.figure(figsize=(18, 6), dpi=150)
            
            # --- Subplot 1: GT Valid Mask ---
            ax_gt_mask = fig_mask.add_subplot(1, 3, 1)
            
            # Plot GT mask (white=valid, black=invalid)
            im_gt_mask = ax_gt_mask.imshow(valid_gt.astype(float), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax_gt_mask.set_title(f"Ground Truth Valid Mask\nframes {frame_from}->{frame_to}\nValid: {valid_gt.sum()}/{valid_gt.size} px ({100*valid_gt.sum()/valid_gt.size:.1f}%)", 
                               fontsize=11, fontweight='bold')
            ax_gt_mask.set_xlabel("X (pixels)", fontsize=9)
            ax_gt_mask.set_ylabel("Y (pixels)", fontsize=9)
            
            # Add colorbar
            cbar_gt_mask = plt.colorbar(im_gt_mask, ax=ax_gt_mask, fraction=0.046, pad=0.04)
            cbar_gt_mask.set_label("Valid (1=valid, 0=invalid)", rotation=270, labelpad=20, fontsize=9, fontweight='bold')
            cbar_gt_mask.ax.tick_params(labelsize=8)
            
            # Overlay vertices for reference
            if len(verts_start) > 0:
                ax_gt_mask.scatter(verts_start[:, 0], verts_start[:, 1], c='red', 
                                  s=30, marker='o', edgecolors='yellow', linewidths=1.5, 
                                  label='vertices', zorder=10, alpha=0.8)
            ax_gt_mask.legend(loc='upper right', fontsize=8)
            
            # --- Subplot 2: Prediction Valid Mask ---
            ax_pred_mask = fig_mask.add_subplot(1, 3, 2)
            
            # Plot prediction mask
            im_pred_mask = ax_pred_mask.imshow(valid_mask.astype(float), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            ax_pred_mask.set_title(f"Prediction Valid Mask\nframes {frame_from}->{frame_to}\nValid: {valid_mask.sum()}/{valid_mask.size} px ({100*valid_mask.sum()/valid_mask.size:.1f}%)", 
                                  fontsize=11, fontweight='bold')
            ax_pred_mask.set_xlabel("X (pixels)", fontsize=9)
            ax_pred_mask.set_ylabel("Y (pixels)", fontsize=9)
            
            # Add colorbar
            cbar_pred_mask = plt.colorbar(im_pred_mask, ax=ax_pred_mask, fraction=0.046, pad=0.04)
            cbar_pred_mask.set_label("Valid (1=valid, 0=invalid)", rotation=270, labelpad=20, fontsize=9, fontweight='bold')
            cbar_pred_mask.ax.tick_params(labelsize=8)
            
            # Overlay vertices for reference
            if len(verts_start) > 0:
                ax_pred_mask.scatter(verts_start[:, 0], verts_start[:, 1], c='red', 
                                    s=30, marker='o', edgecolors='yellow', linewidths=1.5, 
                                    label='vertices', zorder=10, alpha=0.8)
            ax_pred_mask.legend(loc='upper right', fontsize=8)
            
            # --- Subplot 3: Mask Comparison (Combined & Difference) ---
            ax_mask_diff = fig_mask.add_subplot(1, 3, 3)
            
            # Create RGB image showing mask comparison
            # Red channel: GT only (valid in GT but not in pred)
            # Green channel: Both valid (valid in both GT and pred)
            # Blue channel: Pred only (valid in pred but not in GT)
            mask_comparison = np.zeros((H, W, 3), dtype=np.float32)
            mask_comparison[:, :, 0] = valid_gt.astype(float) * (1 - valid_mask.astype(float))  # GT only (red)
            mask_comparison[:, :, 1] = valid_gt.astype(float) * valid_mask.astype(float)        # Both (green)
            mask_comparison[:, :, 2] = (1 - valid_gt.astype(float)) * valid_mask.astype(float)  # Pred only (blue)
            
            im_mask_diff = ax_mask_diff.imshow(mask_comparison, interpolation='nearest')
            
            combined_valid = (valid_gt & valid_mask).sum()
            gt_only = (valid_gt & ~valid_mask).sum()
            pred_only = (~valid_gt & valid_mask).sum()
            
            ax_mask_diff.set_title(f"Mask Comparison\nframes {frame_from}->{frame_to}\n" + 
                                  f"Green (both): {combined_valid} | Red (GT only): {gt_only} | Blue (Pred only): {pred_only}", 
                                  fontsize=11, fontweight='bold')
            ax_mask_diff.set_xlabel("X (pixels)", fontsize=9)
            ax_mask_diff.set_ylabel("Y (pixels)", fontsize=9)
            
            # Add custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', label=f'Both valid ({combined_valid} px)'),
                Patch(facecolor='red', label=f'GT only ({gt_only} px)'),
                Patch(facecolor='blue', label=f'Pred only ({pred_only} px)'),
            ]
            ax_mask_diff.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # Overlay vertices for reference
            if len(verts_start) > 0:
                ax_mask_diff.scatter(verts_start[:, 0], verts_start[:, 1], c='yellow', 
                                    s=30, marker='o', edgecolors='black', linewidths=1.5, 
                                    label='vertices', zorder=10, alpha=0.8)
            
            # Save mask visualization
            mask_path = os.path.join(mask_dir, f"masks_{idx:06d}.png")
            plt.tight_layout()
            fig_mask.savefig(mask_path, dpi=150, bbox_inches='tight')
            plt.close(fig_mask)
            
            print(f"[saved] {mask_path} (GT valid: {valid_gt.sum()}, Pred valid: {valid_mask.sum()}, Combined: {combined_valid})")
    
    return saved


def create_gifs_from_images(
    image_dir: str,
    output_path: str,
    frame_duration: int = 200,
    pattern: str = "*.png",
    loop: int = 0
):
    """
    Create a GIF animation from a directory of images.
    
    Args:
        image_dir: Directory containing the images
        output_path: Path to save the output GIF
        frame_duration: Duration of each frame in milliseconds (default: 200ms = 5fps)
        pattern: Glob pattern to match image files (default: "*.png")
        loop: Number of loops (0 = infinite loop, default: 0)
    
    Returns:
        int: Number of frames in the GIF, or 0 if no images found
    """
    from glob import glob
    
    # Get all image files
    image_files = sorted(glob(os.path.join(image_dir, pattern)))
    
    if len(image_files) == 0:
        print(f"   ⚠️  No images found matching pattern '{pattern}' in {image_dir}")
        return 0
    
    # Load all images
    images = []
    for img_path in image_files:
        img = Image.open(img_path)
        images.append(img)
    
    # Save as GIF
    if len(images) > 0:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,
            loop=loop,
            optimize=False  # Set to True for smaller file size but slower processing
        )
        print(f"   ✓ Created GIF: {output_path} ({len(images)} frames, {frame_duration}ms/frame)")
        return len(images)
    
    return 0


def create_visualization_gifs(
    vis_dir: str,
    frame_duration: int = 200
):
    """
    Create GIF animations from visualization images.
    Creates 3 separate GIFs for comparison subplots: GT, Pred, and Overlay.
    
    Args:
        vis_dir: Base visualization directory containing subdirectories with images
        frame_duration: Duration of each frame in milliseconds (default: 200ms = 5fps)
    
    Returns:
        dict: Dictionary with created GIF paths and frame counts
    """
    print(f"\n🔹 Creating GIF animations...")
    results = {}
    
    comparison_dir = os.path.join(vis_dir, "comparisons")
    if not os.path.exists(comparison_dir):
        print(f"   ⚠️  Comparison directory not found: {comparison_dir}")
        return results
    
    # Create GIF for GT flow subplot
    gt_dir = os.path.join(comparison_dir, "gt_flow")
    if os.path.exists(gt_dir):
        gif_path = os.path.join(vis_dir, "gt_flow_animation.gif")
        n_frames = create_gifs_from_images(
            gt_dir, 
            gif_path, 
            frame_duration=frame_duration,
            pattern="gt_*.png"
        )
        if n_frames > 0:
            results['gt_flow'] = {'path': gif_path, 'frames': n_frames}
    
    # Create GIF for Predicted flow subplot
    pred_dir = os.path.join(comparison_dir, "pred_flow")
    if os.path.exists(pred_dir):
        gif_path = os.path.join(vis_dir, "pred_flow_animation.gif")
        n_frames = create_gifs_from_images(
            pred_dir,
            gif_path,
            frame_duration=frame_duration,
            pattern="pred_*.png"
        )
        if n_frames > 0:
            results['pred_flow'] = {'path': gif_path, 'frames': n_frames}
    
    # Create GIF for Overlay subplot
    overlay_dir = os.path.join(comparison_dir, "overlay")
    if os.path.exists(overlay_dir):
        gif_path = os.path.join(vis_dir, "overlay_animation.gif")
        n_frames = create_gifs_from_images(
            overlay_dir,
            gif_path,
            frame_duration=frame_duration,
            pattern="overlay_*.png"
        )
        if n_frames > 0:
            results['overlay'] = {'path': gif_path, 'frames': n_frames}
    
    if len(results) > 0:
        print(f"   ✅ Created {len(results)} GIF animations in {vis_dir}")
    else:
        print(f"   ⚠️  No GIF animations created (no images found)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Visualize predicted optical flow vs ground truth")
    parser.add_argument("--predictions_path", required=True, 
                       help="Path to predictions.pt file (e.g., evaluations/project/run_id/predictions.pt)")
    parser.add_argument("--data_root", default=None,
                       help="Root directory of the dataset (default: auto-detect from predictions metadata)")
    parser.add_argument("--output_dir", default="toy_datasets/visualizations",
                       help="Directory to save visualization images")
    parser.add_argument("--seq_name_train", default="star8",
                       help="Training sequence name")
    parser.add_argument("--seq_name_val", default="star8_test",
                       help="Validation/test sequence name")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both",
                       help="Which split to visualize")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to visualize per split (None for all)")
    parser.add_argument("--image_width", type=int, default=None,
                       help="Image width (None to load from metadata)")
    parser.add_argument("--image_height", type=int, default=None,
                       help="Image height (None to load from metadata)")
    parser.add_argument("--total_frames", type=int, default=None,
                       help="Total frames (None to load from metadata)")
    parser.add_argument("--save_step", type=int, default=None,
                       help="Save step (None to load from metadata)")
    parser.add_argument("--test_size", type=float, default=None,
                       help="Test size (None to load from metadata)")
    parser.add_argument("--no-require-metadata", action="store_true",
                       help="Don't fail if metadata is missing (NOT RECOMMENDED - may produce incorrect visualizations)")
    parser.add_argument("--no-visualize-errors", action="store_true",
                       help="Skip generating EPE and AE error visualizations")
    parser.add_argument("--no-visualize-masks", action="store_true",
                       help="Skip generating mask visualizations")
    parser.add_argument("--create-gifs", action="store_true", default=False,
                       help="Create GIF animations for the 3 comparison subplots (GT, Pred, Overlay)")
    parser.add_argument("--gif-frame-duration", type=int, default=200,
                       help="Duration of each frame in GIF animations in milliseconds (default: 200ms = 5fps)")
    
    args = parser.parse_args()
    image_size = None
    if args.image_height is not None and args.image_width is not None:
        image_size = (args.image_height, args.image_width)
    
    require_metadata = not args.no_require_metadata
    visualize_errors = not args.no_visualize_errors
    visualize_masks = not args.no_visualize_masks
    
    # Try to load predictions file to get metadata
    print(f"🔹 Loading predictions file: {args.predictions_path}")
    try:
        predictions_data = torch.load(args.predictions_path, map_location='cpu', weights_only=False)
        if "metadata" in predictions_data:
            pred_metadata = predictions_data["metadata"]
            print(f"   ✓ Found evaluation metadata in predictions file:")
            print(f"     run_id: {pred_metadata.get('run_id', 'N/A')}")
            print(f"     data_root: {pred_metadata.get('data_root', 'N/A')}")
            print(f"     train_sequences: {pred_metadata.get('train_sequences', [])}")
            print(f"     val_sequences: {pred_metadata.get('val_sequences', [])}")
            
            # Use metadata from predictions if data_root not explicitly provided
            if args.data_root is None:
                data_root = pred_metadata.get('data_root')
                if data_root is None:
                    raise ValueError("No data_root found in predictions metadata and not provided via --data_root")
                print(f"   ✓ Auto-detected data_root from predictions: {data_root}")
            else:
                data_root = args.data_root
                print(f"   ✓ Using user-provided data_root: {data_root}")
            
            # Auto-detect sequence names if not overridden
            train_seqs = pred_metadata.get('train_sequences', [])
            seq_name_train = args.seq_name_train if args.seq_name_train != "star8" else (train_seqs[0] if train_seqs else args.seq_name_train)
            
            val_seqs = pred_metadata.get('val_sequences', [])
            seq_name_val = args.seq_name_val if args.seq_name_val != "star8_test" else (val_seqs[0] if val_seqs else args.seq_name_val)
            
            print(f"   ✓ Using: data_root={data_root}, train_seq={seq_name_train}, val_seq={seq_name_val}")
        else:
            print(f"   ⚠️  No metadata found in predictions file")
            if args.data_root is None:
                raise ValueError("No metadata in predictions file and no --data_root provided. Please provide --data_root manually.")
            data_root = args.data_root
            seq_name_train = args.seq_name_train
            seq_name_val = args.seq_name_val
            print(f"   ⚠️  Using command line arguments: data_root={data_root}, train_seq={seq_name_train}, val_seq={seq_name_val}")
    except Exception as e:
        print(f"   ⚠️  Could not load predictions file metadata: {e}")
        if args.data_root is None:
            raise ValueError(f"Failed to load predictions metadata and no --data_root provided: {e}")
        data_root = args.data_root
        seq_name_train = args.seq_name_train
        seq_name_val = args.seq_name_val
    
    total_saved = 0
    
    # Visualize training split
    if args.split in ("train", "both"):
        print(f"\n{'='*60}")
        print(f"Visualizing TRAINING split")
        print(f"{'='*60}")
        total_saved += visualize_predictions(
            predictions_path=args.predictions_path,
            data_root=data_root,
            seq_name=seq_name_train,
            output_dir=args.output_dir,
            split="train",
            max_samples=args.max_samples,
            image_size=image_size,
            total_frames=args.total_frames,
            save_step=args.save_step,
            test_size=args.test_size,
            require_metadata=require_metadata,
            visualize_errors=visualize_errors,
            visualize_masks=visualize_masks,
        )
    
    # Visualize validation split
    if args.split in ("val", "both"):
        print(f"\n{'='*60}")
        print(f"Visualizing VALIDATION split")
        print(f"{'='*60}")
        total_saved += visualize_predictions(
            predictions_path=args.predictions_path,
            data_root=data_root,
            seq_name=seq_name_val,
            output_dir=args.output_dir,
            split="val",
            max_samples=args.max_samples,
            image_size=image_size,
            total_frames=args.total_frames,
            save_step=args.save_step,
            test_size=args.test_size,
            require_metadata=require_metadata,
            visualize_errors=visualize_errors,
            visualize_masks=visualize_masks,
        )
    
    # Create GIF animations if requested
    if args.create_gifs:
        print(f"\n{'='*60}")
        print(f"Creating GIF animations")
        print(f"{'='*60}")
        
        # Determine output directories to create GIFs for
        gif_results = {}
        
        # Get run_id from predictions for directory structure
        try:
            predictions_data = torch.load(args.predictions_path, map_location='cpu', weights_only=False)
            run_id = None
            if "metadata" in predictions_data:
                run_id = predictions_data['metadata'].get('run_id')
        except:
            run_id = None
        
        # Create GIFs for training split
        if args.split in ("train", "both"):
            if run_id:
                train_vis_dir = os.path.join(args.output_dir, run_id, seq_name_train, "train")
            else:
                train_vis_dir = os.path.join(args.output_dir, seq_name_train, "train")
            
            if os.path.exists(train_vis_dir):
                print(f"\nCreating GIFs for training split...")
                gif_results['train'] = create_visualization_gifs(
                    train_vis_dir,
                    frame_duration=args.gif_frame_duration
                )
        
        # Create GIFs for validation split
        if args.split in ("val", "both"):
            if run_id:
                val_vis_dir = os.path.join(args.output_dir, run_id, seq_name_val, "val")
            else:
                val_vis_dir = os.path.join(args.output_dir, seq_name_val, "val")
            
            if os.path.exists(val_vis_dir):
                print(f"\nCreating GIFs for validation split...")
                gif_results['val'] = create_visualization_gifs(
                    val_vis_dir,
                    frame_duration=args.gif_frame_duration
                )
        
        # Print summary of created GIFs
        if len(gif_results) > 0:
            print(f"\n{'='*60}")
            print(f"✅ GIF animations created!")
            for split_name, split_gifs in gif_results.items():
                print(f"\n{split_name.capitalize()} split:")
                for gif_type, gif_info in split_gifs.items():
                    print(f"  - {gif_type}: {gif_info['path']} ({gif_info['frames']} frames)")
            print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Saved {total_saved} visualization images to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
