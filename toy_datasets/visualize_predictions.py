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
import imageio.v2 as imageio
import torch

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


def decode_flow_dsec(png_path: str):
    """Decode DSEC-format flow PNG."""
    I = imageio.imread(png_path, format="PNG-FI")
    u = (I[..., 0].astype(np.float32) - 2**15) / 128.0
    v = (I[..., 1].astype(np.float32) - 2**15) / 128.0
    valid = I[..., 2].astype(bool)
    return u, v, valid


def get_vertices(star: StarMovement, frame_idx: int):
    """Get star vertices at a specific frame."""
    star.update_shape(frame_idx)
    return star.transformed_path.vertices


def read_timestamp_rows(ts_path: str):
    """Read timestamp file."""
    with open(ts_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
    rows = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 2:
            rows.append((int(parts[0]), int(parts[1])))
    return rows


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
):
    """
    Visualize predictions vs ground truth for a sequence.
    
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
    """
    # Try to load metadata first
    metadata = load_dataset_metadata(data_root, seq_name)
    
    # Use metadata if available, otherwise use provided parameters or defaults
    if metadata:
        image_size = image_size or tuple(metadata['image_size'])
        total_frames = total_frames or metadata['total_frames']
        save_step = save_step or metadata['save_step']
        test_size = test_size if test_size is not None else metadata['test_size']
        print(f"   Using parameters from metadata:")
        print(f"     image_size: {image_size}")
        print(f"     total_frames: {total_frames}")
        print(f"     save_step: {save_step}")
        print(f"     test_size: {test_size}")
    else:
        # Fall back to provided parameters or defaults
        image_size = image_size or DEFAULT_IMAGE_SIZE
        total_frames = total_frames or DEFAULT_TOTAL_FRAMES
        save_step = save_step or DEFAULT_SAVE_STEP
        test_size = test_size if test_size is not None else DEFAULT_TEST_SIZE
        print(f"   Using provided/default parameters:")
        print(f"     image_size: {image_size}")
        print(f"     total_frames: {total_frames}")
        print(f"     save_step: {save_step}")
        print(f"     test_size: {test_size}")
    
    H, W = image_size
    
    # Load predictions
    print(f"🔹 Loading predictions from {predictions_path}")
    results = torch.load(predictions_path, map_location='cpu')
    
    if split not in results:
        print(f"❌ Split '{split}' not found in predictions file. Available: {list(results.keys())}")
        return 0
    
    split_data = results[split]
    predictions = split_data['predictions']
    ground_truths = split_data['ground_truths']
    valid_masks = split_data['valid_masks']
    
    print(f"   Found {len(predictions)} batches in {split} split")
    
    # Compute split info
    split_start_frame = int(np.floor(total_frames * (1.0 - test_size))) if test_size > 0 else total_frames
    
    # Setup paths for GT flow
    flow_root = os.path.join(data_root, "train_optical_flow", seq_name, "flow")
    forward_dir = os.path.join(flow_root, "forward")
    ts_path = os.path.join(flow_root, "forward_timestamps.txt")
    
    if not os.path.exists(forward_dir) or not os.path.exists(ts_path):
        print(f"❌ Missing flow assets: {forward_dir} or {ts_path}")
        return 0
    
    # Create output directory
    vis_dir = ensure_dir(os.path.join(output_dir, seq_name, split))
    print(f"   Output directory: {vis_dir}")
    
    # Create star movement object
    star = StarMovement(total_frames=total_frames, image_size=image_size, face_color="black")
    
    # Read timestamps
    ts_rows = read_timestamp_rows(ts_path)
    
    # Determine base frame based on split
    if split == "train":
        base_from = 0
    else:
        base_from = split_start_frame
    
    # Flatten predictions and ground truths
    all_preds = torch.cat(predictions, dim=0)  # [N, 2, H, W]
    all_gts = torch.cat(ground_truths, dim=0)  # [N, 2, H, W]
    all_valid = torch.cat(valid_masks, dim=0)  # [N, 1, H, W]
    
    print(f"   Total samples: {all_preds.shape[0]}")
    
    # Limit samples if requested
    num_samples = min(len(ts_rows), all_preds.shape[0])
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)
    
    saved = 0
    for idx in range(num_samples):
        frame_from = base_from + idx * save_step
        frame_to = frame_from + save_step
        if frame_to > total_frames:
            break
        
        # Get vertices at start and end
        verts_start = get_vertices(star, frame_from)
        verts_end = get_vertices(star, frame_to)
        
        # Load GT flow from PNG
        flow_png = os.path.join(forward_dir, f"{idx:06d}.png")
        if not os.path.exists(flow_png):
            print(f"[warn] Flow png missing: {flow_png}")
            continue
        u_gt, v_gt, valid_gt = decode_flow_dsec(flow_png)
        
        # Get predicted flow
        pred_flow = all_preds[idx].numpy()  # [2, H, W]
        u_pred = pred_flow[0]
        v_pred = pred_flow[1]
        
        # Get valid mask
        valid_mask = all_valid[idx, 0].numpy() > 0.5  # [H, W]
        
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
        
        # Plot start vertices (blue)
        if len(verts_start) > 0:
            ax3.scatter(verts_start[:, 0], verts_start[:, 1], c='blue', s=20, label='start vertices', zorder=5)
            vs = verts_start
            if not np.allclose(vs[0], vs[-1]):
                vs = np.vstack([vs, vs[:1]])
            ax3.plot(vs[:, 0], vs[:, 1], '-', color='blue', alpha=0.6, linewidth=1.5, zorder=4)
        
        # Plot end vertices (red)
        if len(verts_end) > 0:
            ax3.scatter(verts_end[:, 0], verts_end[:, 1], c='red', s=20, label='end vertices', zorder=5)
            ve = verts_end
            if not np.allclose(ve[0], ve[-1]):
                ve = np.vstack([ve, ve[:1]])
            ax3.plot(ve[:, 0], ve[:, 1], '-', color='red', alpha=0.6, linewidth=1.5, zorder=4)
        
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
        
        # Save figure
        out_path = os.path.join(vis_dir, f"comparison_{idx:06d}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved += 1
        print(f"[saved] {out_path} (GT arrows: {arrow_count_gt}, Pred arrows: {arrow_count_pred})")
    
    return saved


def main():
    parser = argparse.ArgumentParser(description="Visualize predicted optical flow vs ground truth")
    parser.add_argument("--predictions_path", required=True, 
                       help="Path to predictions.pt file (e.g., evaluations/project/run_id/predictions.pt)")
    parser.add_argument("--data_root", default="/data/idnet/toy_datasets/data/star8",
                       help="Root directory of the dataset")
    parser.add_argument("--output_dir", default="/data/idnet/toy_datasets/visualizations",
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
    
    args = parser.parse_args()
    image_size = None
    if args.image_height is not None and args.image_width is not None:
        image_size = (args.image_height, args.image_width)
    
    total_saved = 0
    
    # Visualize training split
    if args.split in ("train", "both"):
        print(f"\n{'='*60}")
        print(f"Visualizing TRAINING split")
        print(f"{'='*60}")
        total_saved += visualize_predictions(
            predictions_path=args.predictions_path,
            data_root=args.data_root,
            seq_name=args.seq_name_train,
            output_dir=args.output_dir,
            split="train",
            max_samples=args.max_samples,
            image_size=image_size,
            total_frames=args.total_frames,
            save_step=args.save_step,
            test_size=args.test_size,
        )
    
    # Visualize validation split
    if args.split in ("val", "both"):
        print(f"\n{'='*60}")
        print(f"Visualizing VALIDATION split")
        print(f"{'='*60}")
        total_saved += visualize_predictions(
            predictions_path=args.predictions_path,
            data_root=args.data_root,
            seq_name=args.seq_name_val,
            output_dir=args.output_dir,
            split="val",
            max_samples=args.max_samples,
            image_size=image_size,
            total_frames=args.total_frames,
            save_step=args.save_step,
            test_size=args.test_size,
        )
    
    print(f"\n{'='*60}")
    print(f"✅ Done! Saved {total_saved} visualization images to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
